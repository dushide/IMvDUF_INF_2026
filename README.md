## An Interpretable Deep Unfolding Framework for Multi-view Representation Learning

This framework unfolds a generalized multi-view representation problem into a network. This repository contains the code and datasets used for the experiments in our paper.

### Datasets

Download the datasets from <https://drive.google.com/drive/folders/1Mv1l_CWMbN_F9LmCcrjfhntzaR2B9wBg?usp=sharing> and place the `.mat` files in `data/`. `ORL.mat`, `NUS-WIDE.mat`, and `NUSWIDEWG.mat` are included as examples.

### Requirements

- Python 3.7
- torch 1.7.1
- numpy 1.17.3
- scipy 1.3.1
- scikit-learn 0.21.3
- PyYAML 5.1.2
- tqdm 4.38.0

```bash
pip install -r requirements.txt
```


Clustering:

```bash
python clustering.py
```

Classification:

```bash
python semiClassifier.py
```

- `fusion_type` is `trust`, `average`, `weight`, or `attention`. `active` is `l1` or `l21`. `input_type` is `feature` or `similarity`. The current default values ​​are `trust`, `l1`, and `feature`. Both default in `args.py` and apply to clustering and classification.

- At the bottom of each script, `RUN_ALL = True` runs every dataset in `ALL_DATASETS`. Set `RUN_ALL = False` to run only `SINGLE_DATASET`.


### Contact

If you have any questions, please feel free to contact dushidems@gmail.com. Thanks.
