import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __getitem__(self, index):
        pass


class AudioDataModule(pl.LightningDataModule):
    def prepare_data(self):
        """Generate all relevant features from a folder of wav files and save them to file."""

    def setup(self, **kwargs) -> None:
        """Instantiate train and validation datasets and setup any transforms."""

    def train_dataloader(self):
        """Create and return training dataloader."""

    def val_dataloader(self):
        """Create and return validation dataloader."""
