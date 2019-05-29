import numpy as np
import pandas
import math
import os
import types
import h5py
import pickle
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import multiprocessing as mp
import json
from functools import partial

from sklearn.metrics import roc_auc_score

from torch.tensor import Tensor
import torch

from lumin.nn.data.fold_yielder import FoldYielder
from lumin.plotting.data_viewing import *
from lumin.utils.misc import *
from lumin.optimisation.threshold import binary_class_cut
from lumin.evaluation.ams import calc_ams
from lumin.plotting.results import *
from lumin.plotting.plot_settings import PlotSettings

import seaborn as sns
import matplotlib.pyplot as plt


sns.set_style("whitegrid")

DATA_PATH     = Path("../data/")
IMG_PATH      = Path("/home/giles/Documents/kaggle/higgsml")
RESULTS_PATH  = Path("../results/")
plot_settings = PlotSettings(cat_palette='tab10', savepath=Path('.'), format='.pdf')


def multiclass2binary(x:Tensor) -> Tensor:
    preds = torch.max(x, dim=1)
    preds[0][preds[1] == 0] = 1-preds[0][preds[1] == 0]
    return preds[0]


def score_test_df(df:pd.DataFrame, cut:float):
    accept = (df.pred >= cut)
    signal = (df.gen_target == 1)
    bkg = (df.gen_target == 0)
    public = (df.private == 0)
    private = (df.private == 1)

    public_ams = calc_ams(np.sum(df.loc[accept & public & signal, 'gen_weight']),
                          np.sum(df.loc[accept & public & bkg, 'gen_weight']))

    private_ams = calc_ams(np.sum(df.loc[accept & private & signal, 'gen_weight']),
                           np.sum(df.loc[accept & private & bkg, 'gen_weight']))

    print("Public:Private AMS: {} : {}".format(public_ams, private_ams))    
    return public_ams, private_ams
