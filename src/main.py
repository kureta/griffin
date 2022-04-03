from functools import partial
from pathlib import Path
from typing import Callable, List

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
import torchaudio.functional as AF
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def normalize_spectrum(s):
    return (s - np.mean(s, axis=0, keepdims=True)) / np.std(s, axis=0, keepdims=True)


def amp_to_db(s):
    return librosa.amplitude_to_db(s, amin=1e-6, top_db=96)


def unitify(s):
    return s / 1024


def pre_process_file(data_path, sample_rate, n_fft, hop_length, f):
    y, _ = librosa.load(f, mono=True, sr=sample_rate)
    s = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # remove zero offset
    s = s[1:, :]
    np.save(str(data_path / f'{str(f.stem)}.npy'), s)


def prepare_intermediate_data(folder_path: Path,
                              sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
    data_path = folder_path.parent / f'data-{sample_rate}-{n_fft}-{hop_length}'
    try:
        data_path.mkdir(exist_ok=False)
    except FileExistsError:
        print('Already pre-processed')
        return data_path
    # convert to list to see tqdm info
    files = list(folder_path.glob('*.wav'))
    # sort for reproducible ordering
    files.sort()
    func = partial(pre_process_file, data_path, sample_rate, n_fft, hop_length)
    process_map(func, files, max_workers=8)

    return data_path


class AudioDataModule(pl.LightningDataModule):
    def prepare_data(self):
        """Generate all relevant features from a folder of wav files and save them to file."""

    def setup(self):
        """Instantiate train and validation datasets and setup any transforms."""

    def train_dataloader(self):
        """Create and return training dataloader."""

    def val_dataloader(self):
        """Create and return validation dataloader."""


class AudioDataset(Dataset):
    def __init__(self, folder_path: Path,
                 input_transforms: List[Callable] = None,
                 output_transforms: List[Callable] = None,
                 sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        data_path = prepare_intermediate_data(folder_path, sample_rate, n_fft, hop_length)

        spectra = []
        files = list(data_path.glob('*.npy'))
        files.sort()
        for f in tqdm(files):
            spectra.append(np.load(f))

        self.dataset = np.concatenate(spectra, axis=1)
        self.input_transforms = [] if input_transforms is None else input_transforms
        self.output_transforms = [] if output_transforms is None else output_transforms

    def __len__(self):
        return self.dataset.shape[1]

    def __getitem__(self, index):
        x = self.dataset[:, index]
        for t in self.input_transforms:
            x = t(x)
        y = self.dataset[:, index]
        for t in self.output_transforms:
            y = t(y)
        return x, y


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer_sizes = [1024, 512, 256, 128, 64]
        layers = []
        for in_, out_ in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_, out_),
                nn.InstanceNorm1d(out_),
                nn.LeakyReLU(0.2)
            ))
        self.enc = nn.Sequential(*layers)
        self.complication = nn.Sequential(*[nn.Linear(64, 64)] * 4)
        self.mean = nn.Linear(64, 64)
        self.log_var = nn.Linear(64, 64)

    def forward(self, x):
        z = self.enc(x)
        w = self.complication(z)
        mean = self.mean(w)
        log_var = self.log_var(w)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.complication = nn.Sequential(*[nn.Linear(64, 64)] * 4)

        layer_sizes = [64, 128, 256, 512]
        layers = []
        for in_, out_ in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_, out_),
                nn.InstanceNorm1d(out_),
                nn.LeakyReLU(0.2)
            ))
        layers.append(nn.Linear(512, 1024))
        self.dec = nn.Sequential(*layers)

    def forward(self, z):
        w = self.complication(z)
        x_hat = self.dec(w)
        x_hat = torch.sigmoid(x_hat)
        # TODO: try DDSP's modified sigmoid
        return x_hat


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.log_scale = nn.Parameter(torch.Tensor([1e-6]))

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(self.log_scale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=1)

    @staticmethod
    def kl_divergence(z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def forward(self, x):
        mean, log_var = self.encoder(x)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()

        y = self.decoder(z)

        return y, mean, std, z


class LitVAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vae = VAE()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, mean, std, z = self.vae(x)

        # both_y_hats = torch.cat([AF.amplitude_to_DB(y_hat, 20., 1e-6, 1., 96.), y_hat], dim=1)
        # both_xs = torch.cat([AF.amplitude_to_DB(y, 20., 1e-6, 1., 96.), x], dim=1)

        recon_loss = self.vae.gaussian_likelihood(y_hat, y)

        kl = self.vae.kl_divergence(z, mean, std)

        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl_loss': kl.mean(),
            'recon_loss': recon_loss.mean(),
        })

        return elbo

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer


def main():
    dataset = AudioDataset(Path('/home/kureta/Music/violin/Violin Samples'),
                           input_transforms=[unitify],
                           output_transforms=[unitify])
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
    vae = LitVAE()

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model=vae, train_dataloaders=loader)


if __name__ == '__main__':
    main()
