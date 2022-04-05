from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchvision.transforms.functional as tvf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from torchcrepe import predict
from torchcrepe.decode import argmax

CREPE_SAMPLE_RATE = 16000


def next_power_of_2(n):
    n = int(np.ceil(n))
    if n and not (n & (n - 1)):
        return n

    p = 1
    while p < n:
        p <<= 1

    return p


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


def prepare_pitches(folder_path: Path, sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
    crepe_hop_length = next_power_of_2(hop_length * CREPE_SAMPLE_RATE / sample_rate)
    data_path = folder_path.parent / f'pitch-{sample_rate}-{n_fft}-{hop_length}'
    try:
        data_path.mkdir(exist_ok=False)
    except FileExistsError:
        print('Already pre-processed')
        return data_path

    def do_single_file(f):
        y, sr = librosa.load(f, mono=True, sr=sample_rate)
        assert sr == sample_rate
        _, _, probs = predict(torch.from_numpy(y).unsqueeze(0), sample_rate=sample_rate, hop_length=crepe_hop_length,
                              return_periodicity=True, device='cuda', decoder=argmax, batch_size=512)
        probs = probs.argmax(dim=1)[0].cpu().numpy()
        np.save(str(data_path / f'{str(f.stem)}.npy'), probs)

    # convert to list to see tqdm info
    files = list(folder_path.glob('*.wav'))
    # sort for reproducible ordering
    files.sort()
    for f in tqdm(files):
        do_single_file(f)

    return data_path


class AudioDataset(Dataset):
    def __init__(self, folder_path: Path,
                 sample_rate: int = 44100, n_fft: int = 2048, hop_length: int = 512):
        data_path = prepare_intermediate_data(folder_path, sample_rate, n_fft, hop_length)
        pitch_path = prepare_pitches(folder_path, sample_rate, n_fft, hop_length)

        spectra = []
        pitches = []
        spec_files = list(data_path.glob('*.npy'))
        spec_files.sort()
        pitch_files = list(pitch_path.glob('*.npy'))
        pitch_files.sort()
        for s, p in tqdm(zip(spec_files, pitch_files)):
            s_array = torch.from_numpy(np.load(s))
            array = torch.from_numpy(np.load(p))
            array = tvf.resize(array.unsqueeze(0).unsqueeze(0), [1, s_array.shape[1]]).squeeze(1)

            s_array = s_array.unfold(1, 128, 64).transpose(0, 1)
            spectra.append(s_array)

            array = array.unfold(1, 128, 64).transpose(0, 1)
            pitches.append(array)

        self.dataset = torch.cat(spectra, dim=0)
        self.pitchset = torch.cat(pitches, dim=0)

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return self.dataset[index], F.one_hot(self.pitchset[index].squeeze(0), num_classes=360).T


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        layer_sizes = [1024 + 360, 512, 256, 128, 64]
        layers = []
        for in_, out_ in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Sequential(
                nn.Conv1d(in_, out_, 5, padding='same', padding_mode='replicate'),
                nn.GroupNorm(out_, out_),
                nn.LeakyReLU(0.2)
            ))
        self.enc = nn.Sequential(*layers)
        self.complication = nn.Sequential(*[nn.Conv1d(64, 64, 1, padding='same', padding_mode='replicate')] * 4)
        self.mean = nn.Conv1d(64, 64, 1, padding='same', padding_mode='replicate')
        self.log_var = nn.Conv1d(64, 64, 1, padding='same', padding_mode='replicate')

    def forward(self, x):
        z = self.enc(x)
        w = self.complication(z)
        mean = self.mean(w)
        log_var = self.log_var(w)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.complication = nn.Sequential(
            *[nn.Conv1d(64 + 360, 64 + 360, 1, padding='same', padding_mode='replicate')] * 4)

        layer_sizes = [64 + 360, 128, 256, 512, 1024]
        layers = []
        for in_, out_ in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Sequential(
                nn.Conv1d(in_, out_, 5, padding='same', padding_mode='replicate'),
                nn.GroupNorm(out_, out_),
                nn.LeakyReLU(0.2)
            ))

        self.dec = nn.Sequential(*layers)
        self.mean = nn.Conv1d(1024, 1024, 1, padding='same', padding_mode='replicate')

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
        return log_pxz.sum(dim=(1, 2))

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
        kl = kl.sum(dim=(1, 2))
        return kl

    def forward(self, x, c):
        feat = torch.cat([x, c], dim=1)
        mu, log_var = self.encoder(feat)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        z_feat = torch.cat([z, c], dim=1)
        x_hat = self.decoder(z_feat)

        return x_hat, mu, std, z


class LitVAE(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vae = VAE()

    def training_step(self, batch, batch_idx):
        x, c = batch
        x_hat, mean, std, z = self.vae(x, c)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    dataset = AudioDataset(Path('/home/kureta/Music/cello/Cello Samples'))
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8, pin_memory=True)
    vae = LitVAE()

    trainer = pl.Trainer(gpus=1)
    trainer.fit(model=vae, train_dataloaders=loader)


if __name__ == '__main__':
    main()
