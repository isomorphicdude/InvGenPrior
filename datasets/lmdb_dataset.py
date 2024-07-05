"""Implements base class for lmdb datasets."""


import os
import io
from abc import ABC, abstractmethod

import torch.utils.data as data
from torchvision import transforms
import numpy as np
import lmdb
from PIL import Image

__LMDB_DATASETS__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __LMDB_DATASETS__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __LMDB_DATASETS__[name] = cls
        return cls

    return wrapper


def get_dataset(name: str, root, split, transform=None, is_encoded=False, **kwargs):
    if __LMDB_DATASETS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __LMDB_DATASETS__[name](root, split, transform, is_encoded, **kwargs)



class LMDBDataset(data.Dataset):
    def __init__(self, root, split="val", transform=None, is_encoded=False):
        self.transform = transform
        self.split = split
        self.root = root
        if self.split == "train":
            lmdb_path = os.path.join(self.root, "train.lmdb")
        elif self.split == "val":
            lmdb_path = os.path.join(self.root, "validation.lmdb")
        else:
            lmdb_path = os.path.join(f"{self.root}.lmdb")
        
        self.data_lmdb = lmdb.Environment(
            path=lmdb_path,
            readonly=True,
            max_readers=1,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = 0
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                # assume data is a byte array
                img = Image.open(io.BytesIO(data))
                img = img.convert("RGB")
            else:
                # assume data is a numpy array
                img = np.asarray(data, dtype=np.uint8)
                
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target, {"index": index}

    def __len__(self):
        if hasattr(self, "length"):
            return self.length
        else:
            with self.data_lmdb.begin() as txn:
                self.length = txn.stat()["entries"]
            return self.length
        
        

# celebrity face 256x256
@register_dataset(name="celeba")
class CelebADataset(LMDBDataset):
    def __init__(self, root, split="val", transform=transforms.ToTensor(), is_encoded=False):
        super().__init__(root, split, transform, is_encoded)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
            ]
        )
    # we may want to use a subset of the celeba dataset
    # require selecting the subset beforehand 
    

# cat face 
@register_dataset(name="afhq")
class AFHQDataset(LMDBDataset):
    def __init__(self, root, split="val", transform=transforms.ToTensor(), is_encoded=False):
        super().__init__(root, split, transform, is_encoded)
        
        


def num_samples(dataset, train):
    if dataset == "celeba":
        return 27000 if train else 3000
    elif dataset == "celeba64":
        return 162770 if train else 19867
    elif dataset == "imagenet-oord":
        return 1281147 if train else 50000
    elif dataset == "ffhq":
        return 63000 if train else 7000
    else:
        raise NotImplementedError("dataset %s is unknown" % dataset)
    
    
# def get_dataset(config, uniform_dequantization=False, evaluation=False):
#     """Create data loaders for training and evaluation.

#     Args:
#       - config: A ml_collection.ConfigDict parsed from config files.
#       - uniform_dequantization: If `True`, add uniform dequantization to images.
#       - evaluation: If `True`, fix number of epochs to 1.

#     Returns:  
#       - train_ds, eval_ds, dataset_builder.
#     """  
    
    
#     pass