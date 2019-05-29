# SuperTML for HiggsML
Attempting to reproduce the results of [SuperTML - Sun et al., 2019](https://arxiv.org/abs/1903.06246) for the [Higgs ML Kaggle Challenge](https://www.kaggle.com/c/higgs-boson)
]

## Process
- Tabula data is pre-processed to:
    - Fix event orientation in phi, x, and z
    - Convert 3-momenta to Cartesian coordinates
- Events converted to 224x224 images by printing feature values as floats (3 d.p. precision) as text on black backgrounds
- CNN classifier constructed using Resnet34 pretrained on ImageNet as a backbone
- CNN is refined on training data in two stages:
    - 1st stage only trains final two dense layers
    - 2nd stage unfreezes and trains entire network (with discriminative learning rates)
