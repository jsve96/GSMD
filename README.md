# GSMD
This repository is built upon GSE (Group-Wise Sparse and Explainable Attack) and implements a structured adversarial attack using Group Structured Mirror Descent (GSMD).

## Setup
Dependencies: `numpy`, `torch`, `torchvision`, `natsort`, `pandas`, `matplotlib`, `skimage`
  
The NIPS2017 data set can be found at https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/data.

Set

`
IMAGENET_PATH = "path_to_ImageNet"
NIPS_PATH = "path_to_nips2017_dataset"
CIFAR_PATH = "path_to_CIFAR10"
`

to actual path of datasets.


## Run main
The `main.py` file contains the code for the main experiments. It can split the data set into chunks for 'embarrassingly parallel' execution.
For example to run a targeted test for GSE and a ResNet20 on images 1000-1999 of the CIFAR10 test set with a batch size of 500, execute
  `python main.py --dataset 'CIFAR10' --model 'ResNet20' --numchunks 10 --chunk 1 --batchsize 500 --attack 'GSE' --targeted 1`

Run in repo directory:

`nohup python3 -u main.py --dataset ImageNet --model ResNet50 --attack GSMD --cuda 7 --batchsize 64 --targeted 1 --numchunks 5 --chunk 0 > logs/GSMD_Imagenet_targeted.out 2>&1 &`

repeat this for --cunk $\in [0,1,2,3,4]$ to obtain results over all 10k images from ImageNet.

If specified log will be saved into /logs, and outputs into outputs. When all experiments are finished, execute `process_results.py` to combine the results corresponding to the same experiment.


