# Code for MSc Project 2024

## Abstract
> Flow-based generative models trained with simulation-free approaches have become strong contenders to diffusion models in various tasks. However, training-free guidance methods for controlled generation for flow models are not as well developed compared to diffusion. In this work, we propose Tweedie Moment Projected Flows (TMPF) model for zero-shot controlled generation using a pre-trained flow model. We demonstrate the effectiveness of the proposed method on linear inverse problems on both synthetic and real-world data such as high-resolution images. 

This repository is based the Rectified Flow implementation [(Liu et al. 2023)](https://arxiv.org/abs/2209.03003).

## Results on AFHQ
Some qualitative results are shown below on the AFHQ-cats dataset [(Choi et al. 2019)](https://arxiv.org/abs/1912.01865):  

<img src="https://github.com/isomorphicdude/InvGenPrior/blob/main/assests/afhq_qualitative.png">

<!-- 
Some qualitative results are shown below on the AFHQ-cats dataset [(Choi et al. 2019)](https://arxiv.org/abs/1912.01865):
| Guided Samplers    |
| ------ |

| DPS [(Chung et al. 2023)](https://arxiv.org/abs/2209.14687) | <img src="https://github.com/isomorphicdude/InvGenPrior/blob/main/assets/afhq_dps.gif" width="150" height="150" /> 

| PiGDM [(Song et al. 2023)](https://openreview.net/forum?id=9_gsMA8MRKQ)  | <img src="https://github.com/isomorphicdude/InvGenPrior/blob/main/assets/afhq_pgdm.gif" width="150" height="150" />

| TMPD [(Boys et al. 2023)](https://arxiv.org/abs/2310.06721)  | <img src="https://github.com/isomorphicdude/InvGenPrior/blob/main/assets/afhq_tmpd.gif" width="150" height="150" />

| REDdiff [(Mardani et al. 2023)](https://arxiv.org/abs/2305.04391)| <img src="https://github.com/isomorphicdude/InvGenPrior/blob/main/assets/afhq_reddiff.gif" width="150" height="150" /> -->


<!-- | Guided Samplers    |
| ------ |

| DPS | <img src="assets/afhq_dps.gif" width="150" height="150"/>

| PiGDM  | <img src="assets/afhq_pgdm.gif" width="150" height="150" />

| TMPD   | <img src="assets/afhq_tmpd.gif" width="150" height="150" />

| REDdiff| <img src="assets/afhq_reddiff.gif" width="150" height="150" /> -->


<!-- DPS
 <img src="assets/afhq_dps.gif" width="150" height="150"/>  

PiGDM
 <img src="assets/afhq_pgdm.gif" width="150" height="150" />  

TMPD
 <img src="assets/afhq_tmpd.gif" width="150" height="150" />

REDdiff
 <img src="assets/afhq_reddiff.gif" width="150" height="150" /> -->
