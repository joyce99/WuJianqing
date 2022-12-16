# SREA
Structure-aware Relation Representation Learning in Cross-Lingual Entity Alignment

## Datasets
please download the zip file from [here](http://nlp.stanford.edu/data/glove.6B.zip) and choose "glove.6B.300d.txt" as the word vectors.

Initial dataset DBP15K are from [JAPE](https://github.com/nju-websoft/JAPE).

Initial dataset SRPRS are from [RSN](https://github.com/nju-websoft/RSN).


## Datasets

* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;
* lg_merge1.npy: adjacency structure line graph from source KG;
* lg_merge2.npy: adjacency structure line graph from target KG;
* lg_triangular1.npy: ring structure line graph from source KG;
* lg_triangular2.npy: ring structure line graph from target KG;

## Environment

* apex
* tqdm
* Numpy
* python=3.7.1
* pytorch=1.8.1
* torch_geometric=2.0.1

## Running

For pretrained word embeddings, run:
```
generate_word_embedding.py
```

For supervised entity alignment, run:
```
train.py
```

For unsupervised entity alignment, run:
```
train_unsupervised_extend.py
```
