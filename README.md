The source code of the paper "IDGCN: Individual aware Diversified Graph
Convolutional Network for Recommendation"

Environment:
  python = 3.6
  pandas = 1.1.5
  torch = 1.9.1+cu111
  numpy = 1.19.5
  tqdm = 4.63.0


Directory:

IDGCN
- config
  config file

- data

  Beauty, download from https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews

  ml-10m, download from https://grouplens.org/datasets/movielens/10m/

  Music, download from http://www.cp.jku.at/datasets/Music4All-Onion
- src
  - models
    - IDGCN.py: proposed model
    - LightGCN.py: LightGCN model
    - MF.py: mf mode
  - trainer
    - base_trainer.py: trainer module
    - idgcn_trainer.py: idgcn_trainer module
  - util
    - spmm.py: sparse matrix multiplication function
  - metrics.py: some metric functions like recall, ndcg, and our proposed IHC 
  - data_generator.py: date generator module

- main_mf.py: main file for conducting MF model
- main_idgcn.py : main file for conducting IDGCN model
- main_lightgcn.py:main file for conducting LightGCN model


Run the Codes:
python main_mf.py  or python main_ligtgcn.py or python main_idgcn.py

you can change hyperparameters by resetting the config file.

other baseline methods can be found on the link:

DivMF https://github.com/snudatalab/DivMF. 

ALGCN https://github.com/AllminerLab/Code-for-ALGCN-master

DGCN https://github.com/tsinghua-fib-lab/DGCN.

DGRec https://github.com/YangLiangwei/DGRec.

Preprocessing the Beauty dataset can reference:
SIGIR 2021 Enhancing Domain-Level and User-Level Adaptivity in Diversified Recommendation
https://github.com/NLPWM-WHU/EDUA/blob/main/code/preprocess_beauty.py



