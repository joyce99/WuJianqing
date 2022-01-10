# RALG
[Relational-aware Line Graph Neural Networks for Cross-lingual Entity Alignment](https://github.com/joyce99/MREA-master/)

## Datasets
Please download the initial representation file of the entity from [here](https://github.com/StephanieWyt/RDGCN) and extract them into the corresponding dataset folder.

## Environment

```
apex
pytorch
torch_geometric
```

## Running

For entity alignment, use:
```
CUDA_VISIBLE_DEVICES=0 python train.py --data data/DBP15K --lang zh_en


All the experiments are conducted on a PC with a GeForce GTX TITAN X GPU (12GB) and 64GB memory
