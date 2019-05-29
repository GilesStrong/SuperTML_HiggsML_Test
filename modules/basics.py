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

from lumin.nn.training.fold_train import *
from lumin.nn.models.model_builder import ModelBuilder
from lumin.nn.data.fold_yielder import *
from lumin.nn.ensemble.ensemble import Ensemble
from lumin.nn.metrics.class_eval import *
from lumin.plotting.data_viewing import *
from lumin.utils.misc import *
from lumin.optimisation.threshold import binary_class_cut
from lumin.optimisation.hyper_param import fold_lr_find
from lumin.evaluation.ams import calc_ams
from lumin.nn.callbacks.cyclic_callbacks import *
from lumin.nn.callbacks.model_callbacks import *
from lumin.nn.callbacks.data_callbacks import *
from lumin.nn.callbacks.loss_callbacks import *
from lumin.plotting.results import *
from lumin.plotting.plot_settings import PlotSettings
from lumin.plotting.interpretation import *
from lumin.nn.losses.basic_weighted import *
from lumin.nn.models.helpers import Embedder

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

DATA_PATH     = Path("../data/")
IMG_PATH      = Path("/home/giles/Documents/kaggle/higgsml")
RESULTS_PATH  = Path("../results/")
plot_settings = PlotSettings(cat_palette='tab10', savepath=Path('.'), format='.pdf')


def multiclass2binary(x:Tensor) -> Tensor:
    preds = torch.max(x, dim=1)
    preds[0][preds[1]==0] = 1-preds[0][preds[1]==0]
    return preds[0]


class Experiment():
    def __init__(self, exp_name:str, machine:str, path:Optional[Path]=None):
        self.exp_name,self.machine,self.path = exp_name,machine,path
        self.device = self.lookup_machine(self.machine)
        self.results = {}
        self.seed = self.lookup_seed(self.machine)
    
    def __repr__(self) -> str:
        rep = f'Experiment:\t{self.exp_name}\nMachine:\t{self.machine}\nDevice:\t{self.device}'
        for r in self.results: rep += f'\n{r}\t{self[r]}'
        return rep

    def __getitem__(self, idx:str) -> Any: return self.results[idx]

    def __setitem__(self, idx:str, val:Any) -> None: self.results[idx] = val   
        
    def save(self, path:Optional[Path]=None) -> None:
        path = path if path is not None else self.path
        assert path is not None, 'Path is not set'
        os.makedirs(path, exist_ok=True)
        with open(path/f'{self.exp_name}_{self.machine}.json', 'w') as fout:
            json.dump({'exp_name':self.exp_name, 'machine':self.machine, 'results':self.results},  fout)
    
    @classmethod
    def from_json(cls, fname:str):
        with open(fname) as fin: data = json.load(fin)
        if not isinstance(fname, str): fname = str(fname)
        e = cls(data['exp_name'], data['machine'], Path(fname[:fname.rfind('/')]))
        e.results = data['results']
        return e
            
    @staticmethod
    def lookup_machine(machine:str) -> str:
        if machine == 'helios_cuda': return 'Nvidia GeForce GTX 1080 Ti GPU'
        if machine == 'helios_cpu':  return 'Intel Core i7-8700K CPU @ 3.7 GHz (6x2)'
        if machine == 'mbp':         return 'Intel Core i7-8559U CPU @ 2.7 GHz (4x2)'
        if machine == 'daedalus':    return 'Intel Xenon Skylake CPU @ 2.2 GHz (4x1)'
        if machine == 'icarus':      return 'Intel Xenon Skylake CPU @ 2.2 GHz (4x1)'
        if machine == 'morpheus':    return 'Intel Xenon Skylake CPU @ 2.2 GHz (2x2)'
        return 'Unknown'
    
    @staticmethod
    def lookup_seed(machine:str) -> str:
        if machine == 'helios_cuda': return 1111
        if machine == 'helios_cpu':  return 2222
        if machine == 'mbp':         return 3333
        if machine == 'daedalus':    return 4444
        if machine == 'icarus':      return 1337
        if machine == 'morpheus':    return 5555
        return 'Unknown'


class Result():
    def __init__(self, exp_name:str, path:Path, is_blind:bool=True, settings:PlotSettings=plot_settings,
                 machines:List[str]=['helios_cuda', 'helios_cpu', 'mbp', 'daedalus', 'icarus', 'morpheus']):
        self.exp_name,self.path,self.is_blind,self.settings,self.machines = exp_name,path,is_blind,settings,machines
        self.load_exps()
        self.devices = sorted(set([e.device for e in self.exps]))
        self.proc_results()
        
    def __repr__(self) -> str:
        rep = f'Experiment:\t{self.exp_name}\nMachines:'
        for m in self.get_machines(): rep += f'\n\t{m}'
        return rep
    
    def __getitem__(self, idx:str) -> Any:
        matches = [e for e in self.exps if e.machine == idx]
        return matches[0] if len(matches) == 1 else None
            
    def load_exps(self) -> None: self.exps = [Experiment.from_json(self.path/f'{self.exp_name}_{m}.json') for m in self.machines]
        
    def get_machines(self) -> List[str]: return sorted([e.machine for e in self.exps])
        
    def proc_results(self) -> None:
        self.metrics = set([r for e in self.exps for r in e.results if not ('private' in r and self.is_blind)])
        self.results = {}
        for m in self.metrics:
            if 'time' in m:
                df = pd.DataFrame([{'device':e.device, 'time':e.results[m]} for e in self.exps])
                agg = df.groupby('device').agg({'time':[np.mean, partial(np.std, ddof=1), 'count']})['time']
                self.results[m] = {agg.index[i]:(agg['mean'][i], agg['std'][i]/np.sqrt(agg['count'][i])) for i in range(len(agg))}
                
            else:
                vals = [e.results[m] for e in self.exps]
                mean = np.mean(vals, axis=0)
                std  = np.std(vals, axis=0, ddof=1)/np.sqrt(len(vals))
                self.results[m] = np.array((mean, std)) if 'mean' not in m else np.array((mean[0], std[0]))
    
    def print_results(self) -> None:
        for r in reversed(sorted(self.results)):
            res = f'{r}:'
            if isinstance(self.results[r], dict):
                for d in sorted(self.results[r]):
                    val = uncert_round(self.results[r][d][0], self.results[r][d][1])
                    res += f'\n\t{d}\t{val[0]}±{val[1]}'
            else:
                val = uncert_round(self.results[r][0], self.results[r][1])
                res += f'\t{val[0]}±{val[1]}'
            print(res)

    def print_table_row(self, base_result) -> None:
        row = r''
        for r in ['val_ams_max', 'val_ams_smooth', 'test_public_ams_mean']:
            val = uncert_round(self.results[r][0], self.results[r][1])
            row += fr'${val[0]}\pm{val[1]}$ & '
                    
        for r in ['train_time', 'test_time']:
            bases,times = [],[]
            for d in sorted(self.results[r]):
                times.append(self.results[r][d][0])
                bases.append(base_result.results[r][d][0])
            bases,times = np.array(bases),np.array(times)
            deltas = (times-bases)/bases
            val = uncert_round(np.mean(deltas), np.std(deltas, ddof=1)/np.sqrt(len(deltas)))
            row += fr'${val[0]}\pm{val[1]}$ & '
        row = row[:-3] + r'\\'
        print(row)
            
    def plot(self) -> None:
        metrics = list(reversed(sorted([m for m in self.metrics if 'ams' in m])))
        with sns.axes_style(self.settings.style), sns.color_palette(self.settings.cat_palette, len(metrics)):
            plt.figure(figsize=(self.settings.w_mid, self.settings.h_mid))
            for m in metrics:
                if isinstance(self.exps[0].results[m], list):
                    plt.plot([e.machine for e in self.exps], [e.results[m][0] for e in self.exps], label=m)
                else:
                    plt.plot([e.machine for e in self.exps], [e.results[m] for e in self.exps], label=m)
            plt.legend(loc=self.settings.leg_loc, fontsize=self.settings.leg_sz)
            plt.xlabel("Machine", fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            plt.ylabel("AMS", fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            plt.xticks(fontsize=self.settings.tk_sz, color=self.settings.tk_col)
            plt.yticks(fontsize=self.settings.tk_sz, color=self.settings.tk_col)
            plt.title(self.settings.title, fontsize=self.settings.title_sz, color=self.settings.title_col, loc=self.settings.title_loc)
            plt.show()
            
    def compare_to(self, comp, frac_comp:bool=True) -> None:
        for r in reversed(sorted(self.results)):
            if r not in comp.results: continue
            res = f'{r}:'
            if isinstance(self.results[r], dict):
                for d in sorted(self.results[r]):
                    diff = self.results[r][d][0]-comp.results[r][d][0]
                    if frac_comp: diff /= 0.01*comp.results[r][d][0]
                    diff_unc = np.sqrt(np.square(comp.results[r][d][1])+np.square(self.results[r][d][1]))
                    if frac_comp: diff_unc = np.abs(diff*np.sqrt(np.square(diff_unc/diff)+np.square(comp.results[r][d][1]/comp.results[r][d][0])))
                    val = uncert_round(diff, diff_unc)
                    res += f'\n\t{d}\t{val[0]}±{val[1]}'
                    if frac_comp: res += ' %'
            else:
                diff = self.results[r][0]-comp.results[r][0]
                if frac_comp: diff /= 0.01*comp.results[r][0]
                diff_unc = np.sqrt(np.square(comp.results[r][1])+np.square(self.results[r][1]))
                if frac_comp: diff_unc = np.abs(diff*np.sqrt(np.square(diff_unc/diff)+np.square(comp.results[r][1]/comp.results[r][0])))
                val = uncert_round(diff, diff_unc)
                res += f'\t{val[0]}±{val[1]}'
                if frac_comp: res += ' %'
            print(res)


class ExpComp():
    def __init__(self, results:List[Result], settings:PlotSettings=plot_settings):
        self.results,self.settings = results,settings
        self.ams_metrics  = list(reversed(sorted(set(m for r in self.results for m in r.metrics if 'ams'  in m and '_no_tta' not in m and '_4' not in m and '_8' not in m and '_16' not in m and '_32' not in m))))
        self.time_metrics = list(reversed(sorted(set(m for r in self.results for m in r.metrics if 'time' in m and '_no_tta' not in m))))
        self.exps = [self.lookup_exp(r.exp_name) for r in self.results]

    @staticmethod
    def lookup_metric(metric:str) -> None:
        if metric == 'val_ams_smooth':        return 'Val. AMS at cut'
        if metric == 'val_ams_max':           return 'Maximum Val. AMS'
        if metric == 'test_public_ams_mean':  return 'Mean Public AMS'
        if metric == 'test_public_ams':       return 'Overall Public AMS'
        if metric == 'test_private_ams_mean': return 'Mean Private AMS'
        if metric == 'test_private_ams':      return 'Overall Private AMS'
        if metric == 'val_time':              return 'Validation time'
        if metric == 'test_time':             return 'Test time'
        if metric == 'train_time':            return 'Train time'
        
        return metric
    
    @staticmethod
    def lookup_exp(exp:str) -> None:
        split = [s.capitalize() for s in exp.split('_')][1:]
        exp = '\n'.join(split)
        exp = exp.replace('Relu', 'ReLU')
        exp = exp.replace('Swa', 'SWA')
        exp = exp.replace('Sr', 'SR')
        return exp

    def _time_delta(self, vals:List[float], errs:List[float], frac:bool=False) -> Tuple[List[float], List[float]]:
        vals, errs = np.array(vals), np.array(errs)

        delta_vals = vals[1:]-vals[0]
        if frac: delta_vals /= vals[0]

        delta_errs = np.sqrt((errs[1:]**2)+(errs[0]**2))
        if frac: delta_errs = delta_vals*np.sqrt(((delta_errs/(vals[1:]-vals[0]))**2)+((errs[0]/vals[0])**2))

        return ([0]+list(delta_vals), list([errs[0]])+list(delta_errs))
    
    def plot_ams(self, plot_comp_scores:bool=True, savename:Optional[str]=None) -> None:
        with sns.axes_style(self.settings.style), sns.color_palette(self.settings.cat_palette):
            plt.figure(figsize=(self.settings.w_mid, self.settings.h_mid))
            for m in self.ams_metrics:
                if m == 'test_private_ams_mean': continue
                plt.errorbar(self.exps, [r.results[m][0] for r in self.results], yerr=[r.results[m][1] for r in self.results], label=self.lookup_metric(m))
            if plot_comp_scores:
                plt.plot(self.exps, 3.80581*np.ones(len(self.exps)), label='1st = 3.80581', linestyle='--')
                plt.plot(self.exps, 3.78912*np.ones(len(self.exps)), label='2nd = 3.78912', linestyle='--')
                plt.plot(self.exps, 3.78682*np.ones(len(self.exps)), label='3rd = 3.78682', linestyle='--')
            plt.legend(loc=self.settings.leg_loc, fontsize=self.settings.leg_sz)
            plt.xlabel("Solution", fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            plt.ylabel("Metric", fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
            plt.xticks(fontsize=self.settings.tk_sz, color=self.settings.tk_col)
            plt.yticks(fontsize=self.settings.tk_sz, color=self.settings.tk_col)
            plt.title(self.settings.title, fontsize=self.settings.title_sz, color=self.settings.title_col, loc=self.settings.title_loc)
            if savename is not None: plt.savefig(self.settings.savepath/f'{savename}{self.settings.format}', bbox_inches='tight')
            plt.show()
            
    def plot_time(self, delta:bool=False, frac:bool=False, savename:Optional[str]=None  ) -> None:
        if delta and frac: lbl = "Time increase / Initial time"
        elif delta:        lbl = "Time increase [$s$]"
        else:              lbl = "Time [$s$]"
        with sns.axes_style(self.settings.style), sns.color_palette(self.settings.cat_palette):
            for m in self.time_metrics:
                plt.figure(figsize=(self.settings.w_mid, self.settings.h_mid))
                for d in self.results[0].devices:
                    try:
                        vals, errs = [r.results[m][d][0] for r in self.results], [r.results[m][d][1] for r in self.results]
                        if delta: vals, errs = self._time_delta(vals, errs, frac)
                        plt.errorbar(self.exps, vals, yerr=errs, label=d)
                    except KeyError: pass
                plt.legend(loc=self.settings.leg_loc, fontsize=self.settings.leg_sz)
                plt.xlabel("Solution", fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
                plt.ylabel(lbl, fontsize=self.settings.lbl_sz, color=self.settings.lbl_col)
                plt.xticks(fontsize=self.settings.tk_sz, color=self.settings.tk_col)
                plt.yticks(fontsize=self.settings.tk_sz, color=self.settings.tk_col)
                plt.title(f'{self.settings.title} {self.lookup_metric(m)}', fontsize=self.settings.title_sz, color=self.settings.title_col, loc=self.settings.title_loc)
                if savename is not None: plt.savefig(self.settings.savepath/f'{savename}_{m}{self.settings.format}', bbox_inches='tight')
                plt.show()


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


def score_test_data(test_fy, cut, pred_name='pred', zero_preds=['pred_0', 'pred_1', 'pred_2'], one_preds=['pred_3']):
    data = pandas.DataFrame()
    pred = test_fy.get_column(pred_name)
    if len(pred.shape) > 1:
        for p in range(pred.shape[-1]):
            data[f'pred_{p}'] = pred[:, p]
        to_binary_class(data, zero_preds, one_preds)
    else:
        data['pred'] = pred
    data['gen_weight'] = test_fy.get_column('weights')
    data['gen_target'] = test_fy.get_column('targets')
    data['private'] = test_fy.get_column('private')

    accept = (data.pred >= cut)
    signal = (data.gen_target == 1)
    bkg = (data.gen_target == 0)
    public = (data.private == 0)
    private = (data.private == 1)

    public_ams = calc_ams(np.sum(data.loc[accept & public & signal, 'gen_weight']),
                          np.sum(data.loc[accept & public & bkg, 'gen_weight']))

    private_ams = calc_ams(np.sum(data.loc[accept & private & signal, 'gen_weight']),
                           np.sum(data.loc[accept & private & bkg, 'gen_weight']))

    print("Public:Private AMS: {} : {}".format(public_ams, private_ams))    
    return public_ams, private_ams


def score_test_data_per_fold(test_fy, cut, pred_name='pred', zero_preds=['pred_0', 'pred_1', 'pred_2'], one_preds=['pred_3']):
    private = test_fy.get_column('private')
    n_tot_pub, n_tot_pri = len(private[private == 0]), len(private[private == 1])
    public_ams, private_ams = [], []
    
    for i in range(test_fy.n_folds):
        data = pandas.DataFrame()
        pred = test_fy.get_column(pred_name, 1, i)
        if len(pred.shape) > 1:
            for p in range(pred.shape[-1]):
                data[f'pred_{p}'] = pred[:, p]
            to_binary_class(data, zero_preds, one_preds)
        else:
            data['pred'] = pred
        data['gen_weight'] = test_fy.get_column('weights', 1, i)
        data['gen_target'] = test_fy.get_column('targets', 1, i)
        data['private'] = test_fy.get_column('private', 1, i)
        
        data.loc[data.private == 1, 'gen_weight'] *= n_tot_pri/len(data[data.private == 1])
        data.loc[data.private == 0, 'gen_weight'] *= n_tot_pub/len(data[data.private == 0])

        accept = (data.pred >= cut)
        signal = (data.gen_target == 1)
        bkg = (data.gen_target == 0)
        public = (data.private == 0)
        private = (data.private == 1)

        public_ams.append(calc_ams(np.sum(data.loc[accept & public & signal, 'gen_weight']),
                                   np.sum(data.loc[accept & public & bkg, 'gen_weight'])))

        private_ams.append(calc_ams(np.sum(data.loc[accept & private & signal, 'gen_weight']),
                                    np.sum(data.loc[accept & private & bkg, 'gen_weight'])))
    
    public_mean, public_std = np.mean(public_ams),  np.std(public_ams,  ddof=1)/np.sqrt(test_fy.n_folds)
    private_mean, private_std = np.mean(private_ams), np.std(private_ams, ddof=1)/np.sqrt(test_fy.n_folds)

    public  = uncert_round(public_mean, public_std)
    private = uncert_round(private_mean, private_std)
    
    print(f"Mean Public:Private AMS: {public[0]}±{public[1]} : {private[0]}±{private[1]}")    
    return (public_mean, public_std), (private_mean, private_std)


def bs_ams(args:Dict[str,Any], out_q:Optional[mp.Queue]=None) -> [Dict[str,Any]]:
    out_dict, ams = {}, []
    name    = ''   if 'name'    not in args else args['name']
    weights = None if 'weights' not in args else args['weights']
    if 'n' not in args: args['n'] = 100
    df = args['df']
    cut = args['cut']
    len_df = len(df)
    wgt_total = df.gen_weight.sum()

    np.random.seed()
    for i in range(args['n']):
        idxs = np.random.choice(np.arange(len_df), len_df, replace=True, p=weights)
        tmp = df.iloc[idxs]
        tmp.loc[:, 'gen_weight'] = tmp.loc[:, 'gen_weight']*wgt_total/tmp.loc[:, 'gen_weight'].sum()
        ams.append(calc_ams(np.sum(tmp.loc[(tmp.pred >= cut) & (tmp.gen_target == 1), 'gen_weight']),
                            np.sum(tmp.loc[(tmp.pred >= cut) & (tmp.gen_target == 0), 'gen_weight'])))

    out_dict[f'{name}_ams'] = ams
    if out_q is not None: out_q.put(out_dict)
    else: return out_dict


def bootstrap_score_test_data(test_fy, cut, n, pred_name='pred', zero_preds=['pred_0', 'pred_1', 'pred_2'], one_preds=['pred_3']):
    private = test_fy.get_column('private')
    data = pandas.DataFrame()
    pred = test_fy.get_column(pred_name)
    if len(pred.shape) > 1:
        for p in range(pred.shape[-1]):
            data[f'pred_{p}'] = pred[:, p]
        to_binary_class(data, zero_preds, one_preds)
    else:
        data['pred'] = pred
    data['gen_weight'] = test_fy.get_column('weights')
    data['gen_target'] = test_fy.get_column('targets')
    data['private'] = test_fy.get_column('private')

    amss = mp_run([{'n':n, 'df':data[(data.private == 0)], 'cut':cut, 'name':'public'},
                   {'n':n, 'df':data[(data.private == 1)], 'cut':cut, 'name':'private'}], bs_ams)    
    public  = uncert_round(np.mean(amss['public_ams']),  np.std(amss['public_ams'],  ddof=1))
    private = uncert_round(np.mean(amss['private_ams']), np.std(amss['private_ams'], ddof=1))
    
    print(f"Public:Private AMS: {public[0]}±{public[1]} : {private[0]}±{private[1]}")
    return public, private   


def export_test_to_csv(cut, name, data_path=DATA_PATH):
    test_data = h5py.File(data_path + 'testing.hdf5', "r+")

    data = pandas.DataFrame()
    data['EventId'] = get_feature('EventId', test_data)
    data['pred']    = get_feature('pred', test_data)

    data['Class'] = 'b'
    data.loc[data.pred >= cut, 'Class'] = 's'

    data.sort_values(by=['pred'], inplace=True)
    data['RankOrder'] = range(1, len(data) + 1)
    data.sort_values(by=['EventId'], inplace=True)

    print(data_path + name + '_test.csv')
    data.to_csv(data_path + name + '_test.csv', columns=['EventId', 'RankOrder', 'Class'], index=False)


def convert_to_df(datafile, pred_name='pred', n_load=-1, set_fold=-1):
    data = pandas.DataFrame()
    data['gen_target'] = get_feature('targets', datafile, n_load, set_fold=set_fold)
    data['gen_weight'] = get_feature('weights', datafile, n_load, set_fold=set_fold)
    data['pred']       = get_feature(pred_name, datafile, n_load, set_fold=set_fold)
    print(len(data), "candidates loaded")
    return data
