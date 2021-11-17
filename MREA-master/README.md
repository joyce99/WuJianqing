# MREA
Multi-Relation Aware method for Cross-language
entity alignment with Directional Line Graphs

## Datasets
Please download the datasets [here](https://drive.google.com/file/d/1uJ2omzIs0NCtJsGQsyFCBHCXUhoK1mkO/view?usp=sharing) and extract them into root directory.

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

