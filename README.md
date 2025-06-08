# (ICML 2025) TabNAT: A Continuous-Discrete Joint Generative Framework for Tabular Data
<!-- <p align="center">
  <a href="https://openreview.net/forum?id=4Ay23yeuz0">
    <img alt="Openreview" src="https://img.shields.io/badge/review-OpenReview-red">
  </a>
  <a href="https://arxiv.org/abs/2310.09656">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2310.09656-blue">
  </a>
</p> -->

This repository contains the implementation of the paper:
> **TabNAT: A Continuous-Discrete Joint Generative Framework for Tabular Data**  <br>
> Forty-Second International Conference on Machine Learning<br>
> Hengrui Zhang<sup>*</sup>,  Liancheng Fang<sup>*</sup>,  Qitian Wu,  Philip S. Yu<br>

## Installing Dependencies

Python version: 3.10

Create environment

```
conda create -n tabnat python=3.10
conda activate tabnat
```

Install pytorch
```
pip install torch torchvision torchaudio
```

or via conda
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other dependencies

```
pip install -r requirements.txt
```

## Preparing Datasets

### Using the datasets adopted in the paper

Download and process dataset:

```
python download_and_process.py
```

## Train, Sample and evaluate

```
bash run.sh
```







