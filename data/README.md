# Setting up the data

## CelebA-HQ

- Download the CelebA-HQ dataset (256 x 256) and put the validation set into the `data/celeba-hq/val` directory.

- Convert to `lmdb` format by running `python InvGenPrior/data/convert_to_lmdb.py --dpath InvGenPrior/data/celeba-hq --split val` in the project root directory.  



