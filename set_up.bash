#!/bin/bash

git clone https://github.com/isomorphicdude/InvGenPrior.git
cd InvGenPrior

pip install -r requirements.txt

pip install gdown

# download the models
# celebA
gdown https://drive.google.com/uc?id=1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF -O "checkpoints/celebA_ckpt.pth"
# afhq-cats
gdown https://drive.google.com/uc?id=1nI2bCF-y7AjuPGyIwtqtppt8CPtWM7JM -O "checkpoints/afhq_cats_ckpt.pth"

# download the data
# celebA
gdown https://drive.google.com/uc?id=1-GLh0oJjPLUdffjtVI3vv0RlR_YI-fsz -O "data/celebA.zip"
unzip data/celebA.zip -d data/
mv data/content/stargan-v2/data/celeba_hq data/celeba-hq
python data/convert_to_lmdb.py --dpath data/celeba-hq --split val

# afhq-cats
gdown https://drive.google.com/uc?id=1-EHyIZgJYkAAmhg2Y1uBslCuxIQyp9GV -O "data/afhq_cats.zip"
unzip data/afhq_cats.zip -d data/
mkdir -p data/afhq/val/cat
mv data/all_cats/val/* data/afhq/val/cat
python data/convert_to_lmdb.py --dpath data/afhq --split val


