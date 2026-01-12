import os
import random
import lmdb
import six
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import string

class WrappedDataModule(pl.LightningDataModule):
    def __init__(self, data_config, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.config = data_config
        self.batch_size = data_config.batch_size

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = LmdbTextDataset(self.config.train)
            self.val = LmdbTextDataset(self.config.validation)
        if stage == "test" or stage == "predict":
            self.val = LmdbTextDataset(self.config.test)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

def random_patch(img, L):
    # img: PIL Image or Tensor, shape (C, H, W) or (H, W, C)
    W, H = img.size  # PIL Image: (width, height)
    if L <= 2:
        min_ratio = 0.5
        max_ratio = 0.5
    else:
        min_ratio = 1.0 / L
        max_ratio = (L - 1.0) / L
    ratio_w = random.uniform(min_ratio, max_ratio)
    pw = max(1, int(W * ratio_w))
    left = random.randint(0, W - pw)
    patch = img.crop((left, 0, left + pw, H))
    return patch

def split_and_shuffle(img, L):
    W, H = img.size
    if L < 2:
        n_split = 2
    else:
        n_split = random.randint(2, L)
    n_split = min(n_split, W)
    split_widths = [W // n_split] * n_split
    for i in range(W % n_split):
        split_widths[i] += 1
    x = 0
    splits = []
    for w in split_widths:
        splits.append(img.crop((x, 0, x + w, H)))
        x += w
    idxs = list(range(n_split))
    random.shuffle(idxs)
    shuffled = [splits[i] for i in idxs]
    new_img = Image.new('RGB', (W, H))
    x = 0
    for patch in shuffled:
        new_img.paste(patch, (x, 0))
        x += patch.size[0]
    return new_img


def augment_img(i_s, L):
    if random.random() < 0.7:
        i_s = split_and_shuffle(i_s, L)
    i_s = random_patch(i_s, L)
    return i_s


class LmdbTextDataset(Dataset):
    def __init__(self, config):
        """
        lmdb_paths: list of lmdb directory paths
        size: output image size (int)
        transform: torchvision transform for img (default: resize+totensor)
        """
        super().__init__()
        lmdb_paths = config.lmdb_paths
        if isinstance(lmdb_paths, str):
            lmdb_paths = [lmdb_paths]
        self.envs = []
        self.num_samples = []
        self.cum_samples = []
        total = 0
        for path in lmdb_paths:
            env = lmdb.open(
                path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=32,
            )
            with env.begin(write=False) as txn:
                n = int(txn.get('num-samples'.encode()))
            self.envs.append(env)
            self.num_samples.append(n)
            total += n
            self.cum_samples.append(total)
        self.total_samples = total

        self.size = config.size
        self.transform = T.Compose([
            T.Resize((self.size[0], self.size[1])),
            T.ToTensor(),
        ])

    def __len__(self):
        return self.total_samples

    def _get_env_and_idx(self, index):
        # index: global index
        for env_idx, cum in enumerate(self.cum_samples):
            if index < cum:
                if env_idx == 0:
                    local_idx = index
                else:
                    local_idx = index - self.cum_samples[env_idx-1]
                return self.envs[env_idx], local_idx + 1  # lmdb keys start from 1
        raise IndexError

    def __getitem__(self, index):
        env, idx = self._get_env_and_idx(index)
        with env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % idx
            label_key = 'label-%09d'.encode() % idx
            imgbuf = txn.get(img_key)
            label = txn.get(label_key)
            if label is not None:
                label = label.decode('utf-8')
            else:
                label = ""
        img = Image.open(six.BytesIO(imgbuf)).convert('RGB')
        w, h = img.size
        if h > 2 * w:
            img = img.rotate(90, expand=True)
        i_s = augment_img(img, max(1, len(label)))
        img = self.transform(img)  # [C,H,W], float32
        i_s = self.transform(i_s)  # [C,H,W], float32
        return dict(img=img, texts=label, hint=i_s)
