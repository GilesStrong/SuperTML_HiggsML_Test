import pandas
import pickle
import numpy as np
import optparse
import os
import h5py
from pathlib import Path
from collections import OrderedDict

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.cluster import k_means

from lumin.data_processing.hep_proc import *
from lumin.data_processing.pre_proc import *
from lumin.data_processing.file_proc import *
from lumin.utils.misc import *


def import_data(data_path=Path("../data/"),
                rotate=False, flip_y=False, flip_z=False, cartesian=True,
                mode='OpenData',
                val_size=0.2, seed=None, cat_feats=[]):
    '''Import and split data from CSV(s)'''
    if mode == 'OpenData':  # If using data from CERN Open Access
        data = pandas.read_csv(data_path/'atlas-higgs-challenge-2014-v2.csv')
        data.rename(index=str, columns={"KaggleWeight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        data.drop(columns=['Weight'], inplace=True)
        training_data = pandas.DataFrame(data.loc[data.KaggleSet == 't'])
        training_data.drop(columns=['KaggleSet'], inplace=True)
        
        test = pandas.DataFrame(data.loc[(data.KaggleSet == 'b') | (data.KaggleSet == 'v')])
        test['private'] = 0
        test.loc[(data.KaggleSet == 'v'), 'private'] = 1
        test['gen_target'] = 0
        test.loc[test.Label == 's', 'gen_target'] = 1
        test.drop(columns=['KaggleSet', 'Label'], inplace=True)

    else:  # If using data from Kaggle
        training_data = pandas.read_csv(data_path/'training.csv')
        training_data.rename(index=str, columns={"Weight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        test = pandas.read_csv(data_path/'test.csv')
        test.rename(index=str, columns={'PRI_met': 'PRI_met_pt'}, inplace=True)

    proc_event(training_data, fix_phi=rotate, fix_y=flip_y, fix_z=flip_z, use_cartesian=cartesian, ref_vec_0='PRI_lep', ref_vec_1='PRI_tau', default_vals=[-999.0], keep_feats=['PRI_met_pt'])
    proc_event(test, fix_phi=rotate, fix_y=flip_y, fix_z=flip_z, use_cartesian=cartesian, ref_vec_0='PRI_lep', ref_vec_1='PRI_tau', default_vals=[-999.0], keep_feats=['PRI_met_pt'])

    training_data['gen_target'] = 0
    training_data.loc[training_data.Label == 's', 'gen_target'] = 1
    training_data.drop(columns=['Label'], inplace=True)
    training_data['gen_weight_original'] = training_data['gen_weight']  # gen_weight might be renormalised

    training_data['gen_strat_key'] = training_data['gen_target'] if len(cat_feats) == 0 else ids2unique(training_data[['gen_target'] + cat_feats].values)
    
    train_feats = [x for x in training_data.columns if 'gen' not in x and x != 'EventId' and 'kaggle' not in x.lower()]
    train, val = train_test_split(training_data, test_size=val_size, random_state=seed, stratify=training_data.gen_strat_key)

    print('Training on {} datapoints and validating on {}, using {} feats:\n{}'.format(len(train), len(val), len(train_feats), [x for x in train_feats]))

    return {'train': train[train_feats + ['gen_target', 'gen_weight', 'gen_weight_original', 'gen_strat_key']], 
            'val': val[train_feats + ['gen_target', 'gen_weight', 'gen_weight_original', 'gen_strat_key']],
            'test': test,
            'feats': train_feats}


def save_fold(in_data, n, input_pipe, out_file, norm_weights, mode, cont_feats, cat_feats, multi):
    '''Save fold into hdf5 file'''
    grp = out_file.create_group(f'fold_{n}')
    
    x = np.hstack((input_pipe.transform(in_data[cont_feats].values.astype('float32')),
                   in_data[cat_feats].values.astype('float32')))
    save_to_grp(x, grp, 'inputs')
    
    if mode != 'testing':
        if norm_weights:
            if multi:
                for c in set(in_data.gen_sample):
                    in_data.loc[in_data.gen_sample == c, 'gen_weight'] /= np.sum(in_data.loc[in_data.gen_sample == c, 'gen_weight'])
            else:
                in_data.loc[in_data.gen_target == 0, 'gen_weight'] /= np.sum(in_data.loc[in_data.gen_target == 0, 'gen_weight'])
                in_data.loc[in_data.gen_target == 1, 'gen_weight'] /= np.sum(in_data.loc[in_data.gen_target == 1, 'gen_weight'])

        if multi:
            y = in_data['gen_sample'].values.astype('int')
        else:
            y = in_data['gen_target'].values.astype('int')
        save_to_grp(y, grp, 'targets')

        if multi:
            y = in_data['gen_target'].values.astype('int')
            save_to_grp(y, grp, 'orig_targets')

        w = in_data['gen_weight'].values.astype('float32')  # if not (multi and norm_weights) else weight
        save_to_grp(w, grp, 'weights')

        w = in_data['gen_weight_original'].values.astype('float32')
        save_to_grp(w, grp, 'orig_weights')
    
    else:
        x = in_data['EventId'].values.astype('int')
        save_to_grp(x, grp, 'EventId')

        if 'private' in in_data.columns:
            w = in_data['gen_weight'].values.astype('float32')
            save_to_grp(w, grp, 'weights')

            s = in_data['private'].values.astype('int')
            save_to_grp(s, grp, 'private')

            y = in_data['gen_target'].values.astype('int')
            save_to_grp(y, grp, 'targets')


def prepare_sample(in_data, mode, input_pipe, norm_weights, N, cont_feats, cat_feats, data_path, multi):
    '''Split data sample into folds and save to hdf5'''
    print("Running", mode)
    os.system(f'rm {data_path/mode}.hdf5')
    out_file = h5py.File(f'{data_path/mode}.hdf5', "w")

    if mode != 'testing':
        kf = StratifiedKFold(n_splits=N, shuffle=True)
        folds = kf.split(in_data, in_data['gen_strat_key'])
    else:
        kf = KFold(n_splits=N, shuffle=True)
        folds = kf.split(in_data)

    for i, (_, fold) in enumerate(folds):
        print("Saving fold:", i, "of", len(fold), "events")
        save_fold(in_data.iloc[fold].copy(), i, input_pipe, out_file, norm_weights, mode, cont_feats, cat_feats, multi)


def proc_targets(data):
    cluster = k_means(data['train'].loc[data['train'].gen_target == 0, 'gen_weight'].values[:, None], 3)
    data['train'].loc[data['train'].gen_target == 0, 'gen_sample'] = cluster[1]
    data['train'].loc[data['train'].gen_target == 1, 'gen_sample'] = 3
    data['val'].loc[data['val'].gen_target == 0, 'gen_sample'] = abs(data['val'].loc[data['val'].gen_target == 0, 'gen_weight'][None, :] - cluster[0][:, None]).argmin(axis=0)[0]
    data['val'].loc[data['val'].gen_target == 1, 'gen_sample'] = 3


def run_data_import(data_path, rotate, flip_y, flip_z, cartesian, mode, val_size, seed, n_folds, cat_feats, multi):
    '''Run through all the stages to save the data into files for training, validation, and testing'''
    # Get Data
    data = import_data(data_path, rotate, flip_y, flip_z, cartesian, mode, val_size, seed, cat_feats)

    cont_feats = [x for x in data['feats'] if x not in cat_feats]
    input_pipe = fit_input_pipe(data['train'], cont_feats, data_path/'input_pipe')
    cat_maps, cat_szs = proc_cats(data['train'], cat_feats, data['val'], data['test'])
    if multi:
        proc_targets(data)

    prepare_sample(data['train'], 'train', input_pipe, True, n_folds, cont_feats, cat_feats, data_path, multi)
    prepare_sample(data['val'], 'val', input_pipe, False, n_folds, cont_feats, cat_feats, data_path, multi)
    prepare_sample(data['test'], 'testing', input_pipe, False, n_folds, cont_feats, cat_feats, data_path, multi)

    with open(data_path/'feats.pkl', 'wb') as fout:
        pickle.dump({'cont_feats': cont_feats, 'cat_feats': cat_feats, 'cat_maps': cat_maps, 'cat_szs': cat_szs}, fout)


def parse_cats(string):
    return [x.strip() for x in string.split(',')] if string is not None else []
        

if __name__ == '__main__':
    parser = optparse.OptionParser(usage=__doc__)
    parser.add_option("-d", "--data_path", dest="data_path", action="store", default="./data/", help="Data folder location")
    parser.add_option("-r", "--rotate", dest="rotate", action="store", default=False, help="Rotate events in phi to have common alignment")
    parser.add_option("-y", "--flipy", dest="flip_y", action="store", default=False, help="Flip events in y to have common alignment")
    parser.add_option("-z", "--flipz", dest="flip_z", action="store", default=False, help="Flip events in z to have common alignment")
    parser.add_option("-c", "--cartesian", dest="cartesian", action="store", default=True, help="Convert to Cartesian system")
    parser.add_option("-m", "--mode", dest="mode", action="store", default="OpenData", help="Using open data or Kaggle data")
    parser.add_option("-v", "--val_size", dest="val_size", action="store", default=0.2, help="Fraction of data to use for validation")
    parser.add_option("-s", "--seed", dest="seed", action="store", default=1337, help="Seed for train/val split")
    parser.add_option("-n", "--n_folds", dest="n_folds", action="store", default=10, help="Number of folds to split data")
    parser.add_option("-f", "--cat_feats", dest="cat_feats", action="store", default=None, help="Comma-separated list of features to be treated as categorical")
    parser.add_option("--multi", dest="multi", action="store", default=False, help="Use multiclass classification")    
    opts, args = parser.parse_args()



    run_data_import(Path(opts.data_path),
                    str2bool(opts.rotate), str2bool(opts.flip_y), str2bool(opts.flip_z), str2bool(opts.cartesian),
                    opts.mode, opts.val_size, int(opts.seed), opts.n_folds, parse_cats(opts.cat_feats), str2bool(opts.multi))
    