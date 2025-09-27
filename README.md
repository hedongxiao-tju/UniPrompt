# One Prompt Fits All: Universal Graph Adaptation for Pretrained Models

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://neurips.cc/" alt="Conference">
        <img src="https://img.shields.io/badge/MIT-License-Yellow" /></a>
    <a href="https://neurips.cc/" alt="Conference">
        <img src="https://img.shields.io/badge/NeurIPS'25-purple" /></a>
<!--     <img src="https://img.shields.io/pypi/l/torch-rechub"> -->
</p>


The official source code for **One Prompt Fits All: Universal Graph Adaptation for Pretrained Models** at NeurIPS 2025.

Part of code is referenced from [*ProG: A Unified Python Library for Graph Prompting*](https://github.com/sheldonresearch/ProG) and [*PyGCL: A PyTorch Library for Graph Contrastive Learning*](https://github.com/PyGCL/PyGCL), and we include the code for pre-trained models [*DGI*](https://github.com/PetarV-/DGI), [*GRACE*](https://github.com/CRIPAC-DIG/GRACE) and [*GraphMAE*](https://github.com/THUDM/GraphMAE).

## Requirements

- python=3.9.21
- torch=2.4.0 (with CUDA 12.4 support)
- torch-geometric=2.6.1
- torch_sparse=0.6.18
- scikit-learn=1.6.1
- numpy=2.0.1
- scipy=1.13.1
- tqdm=4.67.1


## How to run

To get started, unzip the datasets (can be found in ./datasets), and then, use the script that corresponds to the dataset you're working with.
```
bash uniprompt.sh  # you could also run the code manually
```

### Citation  

```BibTex

```
