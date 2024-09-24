"""Implements base class for lmdb datasets."""

import os
import os.path as osp
import io

import torch
import torch.utils
import torch.utils.data
import torchvision
import numpy as np
import lmdb
import pyarrow
import pickle
from PIL import Image

__LMDB_DATASETS__ = {}


def register_dataset(name: str):
    def wrapper(cls):
        if __LMDB_DATASETS__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __LMDB_DATASETS__[name] = cls
        return cls

    return wrapper


def get_dataset(name: str, db_path, transform=None, target_transform=None):
    if __LMDB_DATASETS__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __LMDB_DATASETS__[name](db_path, transform, target_transform)


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


class LMDBDataset(torch.utils.data.Dataset):
    """
    Implements base class for lmdb datasets.
    Assumes there is no label.

    Attributes:
      - db_path: str, path to lmdb database
      - transform: callable, data augmentation
    """

    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.transform = transform
        self.target_transform = target_transform

        env = lmdb.open(
            self.db_path,
            subdir=osp.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b"__len__"))
            self.keys = pickle.loads(txn.get(b"__keys__"))

    def open_lmdb(self):
        self.env = lmdb.open(
            self.db_path,
            subdir=osp.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False, buffers=True)
        self.length = pickle.loads(self.txn.get(b"__len__"))
        self.keys = pickle.loads(self.txn.get(b"__keys__"))

    def __getitem__(self, index):
        if not hasattr(self, "txn"):
            self.open_lmdb()

        img, target = None, None
        byteflow = self.txn.get(self.keys[index])
        unpacked = pickle.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = io.BytesIO()
        buf.write(imgbuf[0])
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + " (" + self.db_path + ")"


# celebrity face 256x256
@register_dataset(name="celeba")
class CelebADataset(LMDBDataset):
    def __init__(
        self,
        db_path,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
    ):
        super().__init__(db_path, transform, target_transform)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor(),
            ]
        )

    # we may want to use a subset of the celeba dataset
    # require selecting the subset beforehand
    
    
@register_dataset(name="ffhq256")
class FFHQ256Dataset(LMDBDataset):
    def __init__(
        self,
        db_path,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
    ):
        super().__init__(db_path, transform, target_transform)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor(),
            ]
        )


# cat face
@register_dataset(name="afhq")
class AFHQDataset(LMDBDataset):
    def __init__(
        self,
        db_path,
        transform=torchvision.transforms.ToTensor(),
        target_transform=None,
    ):
        super().__init__(db_path, transform, target_transform)
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.ToTensor(),
            ]
        )


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


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def raw_reader(path):
    with open(path, "rb") as f:
        bin_data = f.read()
    return bin_data


def dump_pickle(obj):
    """
    Serialize an object.

    Returns :
        The pickled representation of the object obj as a bytes object
    """
    return pickle.dumps(obj)


def folder2lmdb(dpath, name="train_images", write_frequency=5000, num_workers=0):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = torchvision.datasets.ImageFolder(directory, loader=raw_reader)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generating LMDB to %s" % lmdb_path)
    map_size = 30737418240  # this should be adjusted based on OS/db size
    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        map_size=map_size,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, (data, label) in enumerate(data_loader):
        # print(type(data), data)
        image = data
        label = label.numpy()
        txn.put("{}".format(idx).encode("ascii"), dump_pickle((image, label)))
        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = ["{}".format(k).encode("ascii") for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dump_pickle(keys))
        txn.put(b"__len__", dump_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()


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
# class LMDBDataset(data.Dataset):
#     def __init__(self, root, split="val", transform=None, is_encoded=False):
#         self.transform = transform
#         self.split = split
#         self.root = root
#         if self.split == "train":
#             lmdb_path = os.path.join(self.root, "train.lmdb")
#         elif self.split == "val":
#             lmdb_path = os.path.join(self.root, "validation.lmdb")
#         else:
#             lmdb_path = os.path.join(f"{self.root}.lmdb")

#         self.data_lmdb = lmdb.Environment(
#             path=lmdb_path,
#             readonly=True,
#             max_readers=1,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )
#         self.is_encoded = is_encoded

#     def __getitem__(self, index):
#         target = 0
#         with self.data_lmdb.begin(write=False, buffers=True) as txn:
#             data = txn.get(str(index).encode())
#             if self.is_encoded:
#                 # assume data is a byte array
#                 img = Image.open(io.BytesIO(data))
#                 img = img.convert("RGB")
#             else:
#                 # assume data is a numpy array
#                 img = np.asarray(data, dtype=np.uint8)

#                 # assume data is RGB
#                 size = int(np.sqrt(len(img) / 3))
#                 img = np.reshape(img, (size, size, 3))
#                 img = Image.fromarray(img, mode="RGB")

#         if self.transform is not None:
#             img = self.transform(img)

#         return img, target, {"index": index}

#     def __len__(self):
#         if hasattr(self, "length"):
#             return self.length
#         else:
#             with self.data_lmdb.begin() as txn:
#                 self.length = txn.stat()["entries"]
#             return self.length
