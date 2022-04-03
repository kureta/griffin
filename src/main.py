from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


def amp_to_db(s):
    return AF.amplitude_to_DB(s, 20., 1e-6, 1., 96.)


def pre_process_file(data_path, sample_rate, n_fft, hop_length, f):
    y, _ = librosa.load(f, mono=True, sr=sample_rate)
    # calculate spectrum
    s = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    # normalize (assuming hann window)
    s = s / (n_fft / 2)
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
    process_map(func, files, max_workers=8, chunksize=10)

    return data_path


class AudioDataset(Dataset):
    def __init__(self, folder_path: Path,
                 sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        data_path = prepare_intermediate_data(folder_path, sample_rate, n_fft, hop_length)

        spectra = []
        files = list(data_path.glob('*.npy'))
        files.sort()
        for f in tqdm(files):
            spectra.append(np.load(f))

        self.dataset = np.concatenate(spectra, axis=1)

    def __len__(self):
        return self.dataset.shape[1]

    def __getitem__(self, index):
        return self.dataset[:, index]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer_sizes = [1024, 512, 256, 128, 64]
        layers = []
        for in_, out_ in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_, out_),
                # nn.InstanceNorm1d(out_),
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

        layer_sizes = [64, 128, 256, 512, 1024]
        layers = []
        for in_, out_ in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Sequential(
                nn.Linear(in_, out_),
                # nn.InstanceNorm1d(out_),
                nn.LeakyReLU(0.2)
            ))

        self.dec = nn.Sequential(*layers)
        self.mean = nn.Linear(1024, 1024)

    def forward(self, z):
        w = self.complication(z)
        x_hat = self.dec(w)
        mean = self.mean(x_hat)

        return torch.sigmoid(mean)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.log_scale = nn.Parameter(torch.Tensor([1e-7]))

    def likelihood(self, mean, x):
        scale = torch.exp(self.log_scale)
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
        mu, log_var = self.encoder(x)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        x_hat = self.decoder(z)

        return x_hat, mu, std, z


class LitVAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vae = VAE()

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat, mean, std, z = self.vae(x)

        recon_loss = self.vae.likelihood(x_hat, x)

        kl = self.vae.kl_divergence(z, mean, std)

        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl_loss': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'log_scale': self.vae.log_scale,
        })

        return elbo

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def main():
    dataset = AudioDataset(Path('/home/kureta/Music/cello/Cello Samples'))
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
    vae = LitVAE()

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model=vae, train_dataloaders=loader)


if __name__ == '__main__':
    main()
