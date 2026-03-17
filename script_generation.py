#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn

import ot as pot

import time
import os
import csv
import json
import pickle as pkl
import copy
from datetime import datetime

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from utils import *
from data_loaders import dataset_loader, EFVFM_DATASETS, get_efvfm_splits
from sklearn.model_selection import train_test_split
import argparse

import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'Python-Package/base-ForestDiffusion'))

from ForestDiffusion import ForestDiffusionModel
from metrics import test_on_multiple_models, compute_coverage, test_imputation_regression, test_on_multiple_models_classifier
from vfm_metrics import TabMetrics
from STaSy.stasy import STaSy_model
from sdv.single_table import GaussianCopulaSynthesizer, TVAESynthesizer, CTGANSynthesizer, CopulaGANSynthesizer
from sdv.metadata import SingleTableMetadata
import miceforest as mf
from missforest import MissForest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--out_path', type=str, default='jolicoea/tabular_generation_results.csv',
                    help='filename for the results')

parser.add_argument("--restore_from_name", type=str2bool, default=False, help="if True, restore session based on name")
parser.add_argument("--name", type=str, default='my_exp', help="used when restoring from crashed instances")
parser.add_argument("--results_dir", type=str, default='experiments/results', help="if set, save generated samples and detailed metrics (samples.csv, all_results.json, shapes/trends, density_plots.png when --plot_density)")
parser.add_argument("--plot_density", type=str2bool, default=True, help="if True, plot real vs synthetic density per column and save (requires --results_dir or saves to results_dir)")

parser.add_argument("--methods", type=str, nargs='+', default=['oracle', 'CTGAN', 'GaussianCopula', 'TVAE', 'CopulaGAN', 'CTABGAN', 'stasy', 'TabDDPM', 'forest_diffusion'], help="oracle, CTGAN, GaussianCopula, TVAE, CopulaGAN, CTABGAN, stasy, TabDDPM, forest_diffusion")
parser.add_argument('--nexp', type=int, default=1,
                    help='number of experiences per parameter setting')
parser.add_argument('--ngen', type=int, default=5,
                    help='number of generations per method')
parser.add_argument('--n_tries', type=int, default=1,
                    help='number of models trained with different seeds in the metrics')
# parser.add_argument('--datasets', nargs='+', type=str, default=['iris', 'wine', 'parkinsons', 'climate_model_crashes', 'concrete_compression', 'yacht_hydrodynamics', 'airfoil_self_noise', 'connectionist_bench_sonar', 'ionosphere', 'qsar_biodegradation', 'seeds', 'glass', 'ecoli', 'yeast', 'libras', 'planning_relax', 'blood_transfusion', 'breast_cancer_diagnostic', 'connectionist_bench_vowel', 'concrete_slump', 'wine_quality_red', 'wine_quality_white', 'california', 'bean', 'tictactoe','congress','car'],
#                     help='datasets on which to run the experiments')
parser.add_argument('--datasets', nargs='+', type=str, default=['credit-g', 'default', 'adult', 'news', 'beijing', 'shoppers', 'magic'],
                    help='datasets on which to run the experiments')

# Setting for Missingness if used
parser.add_argument('--add_missing_data', type=str2bool, default=False)
parser.add_argument('--p', type=float, default=0.2, help='Proportion of missing')
parser.add_argument('--imputation_method', type=str, default='MissForest', help='miceforest or MissForest or none (MissForest is better and the one used in the paper for the non-ForestDiffusion methods; ForestDiffusion is the only method that can handle none)')

# Quantile / imputation pre-processing (optional, to mimic ef-vfm)
parser.add_argument('--use_quantile', type=str2bool, default=False, help='If True, apply QuantileTransformer on numerical columns before training/generation and inverse-transform fake samples before evaluation.')
parser.add_argument('--n_quantiles', type=int, default=None, help='Number of quantiles for QuantileTransformer (None uses ef-vfm heuristic).')
parser.add_argument('--efvfm_style_impute', type=str2bool, default=False, help='If True, impute missing values like ef-vfm (num mean, cat most_frequent) on X/y before training.')

# Forest hyperparameters
parser.add_argument('--forest_model', type=str, default='xgboost', help='xgboost, random_forest, lgbm, catboost')
parser.add_argument('--diffusion_type', type=str, default='vp', help='flow (flow-matching), vp (Variance-Preserving diffusion)')
parser.add_argument('--n_t', type=int, default=50, help='number of times t in [0,1]')
parser.add_argument('--n_t_sampling', type=int, default=0, help='number of times t in [0,1] for sampling  (0 will uses n_t steps; ignore this parameter honestly, its worth changing)')
parser.add_argument('--n_t_sampling_list', type=str, default='', help='Comma-separated n_t for sampling sweep, e.g. 10,20,30,50. When set, train once then generate+eval per value with --n_t_sampling_repeats; saves curve JSON/PNG under run_dir. Empty disables.')
parser.add_argument('--n_t_sampling_repeats', type=int, default=3, help='Number of generate+eval repeats per n_t_sampling when --n_t_sampling_list is set (metrics averaged).')
parser.add_argument('--max_depth', type=int, default=7, help='max tree depth (xgboost, random_forest)')
parser.add_argument('--num_leaves', type=int, default=31, help='max number of leaves (lgbm)')
parser.add_argument('--n_estimators', type=int, default=100, help='number of trees (xgboost, random_forest, lgbm)')
parser.add_argument('--eta', type=float, default=0.3, help='lr (xgboost, random_forest, lgbm)')
parser.add_argument('--duplicate_K', type=int, default=100, help='number of times to duplicate the data for improved performanced of forests')
parser.add_argument('--gpu_hist', type=str2bool, default=False, help='If True, xgboost use the GPU')
parser.add_argument('--ycond', type=str2bool, default=True, help='If True, make a different forest model per label (when its not regression obviously)')
parser.add_argument('--eps', type=float, default=1e-3, help='')
parser.add_argument('--beta_min', type=float, default=0.1, help='')
parser.add_argument('--beta_max', type=float, default=8, help='')
parser.add_argument('--n_jobs', type=int, default=-1, help='')
parser.add_argument('--n_batch', type=int, default=1, help='If >0 use the data iterator with the specified number of batches (supported for flow/vp/mixed-flow)')
parser.add_argument('--use_ot', type=str2bool, default=False, help='If True, use Minibatch Optimal Transport coupling (group-normalized L2²) for straighter flow paths')

# stasy hyperparameters
parser.add_argument('--act', type=str, default='elu', help='')
parser.add_argument('--layer_type', type=str, default='concatsquash', help='')
parser.add_argument('--sde', type=str, default='vesde', help='')
parser.add_argument('--lr', type=float, default=2e-3, help='')
parser.add_argument('--num_scales', type=int, default=50, help='')

args = parser.parse_args()


def _build_cat_labels_from_train(Xy_train, bin_indexes, cat_indexes, cat_y, bin_y):
    """从训练矩阵中按 factorize 顺序得到各分类列的标签列表，供保存 samples 时还原。适用于非 ef-vfm 及无 cat_labels 时。"""
    cat_col_indices = list(bin_indexes) if bin_indexes else []
    if cat_indexes is not None:
        cat_col_indices = cat_col_indices + list(cat_indexes)
    if cat_y or bin_y:
        cat_col_indices.append(Xy_train.shape[1] - 1)
    cat_col_indices = sorted(set(cat_col_indices))
    cat_labels = {}
    for col_idx in cat_col_indices:
        if col_idx >= Xy_train.shape[1]:
            continue
        _, uniques = pd.factorize(Xy_train[:, col_idx])
        cat_labels[col_idx] = uniques.tolist()
    return cat_labels


def _samples_df_for_save(df_fake, info, cat_labels=None):
    """若 info 含 idx_name_mapping/column_names 则列名映射回原名；若 cat_labels 存在则分类列编码还原为原始标签（与 ef-vfm 一致）。"""
    out = df_fake.copy()
    # 1) 还原分类列：编码 0,1,2,... -> 原始标签
    if cat_labels:
        for col_idx, labels in cat_labels.items():
            if col_idx not in out.columns:
                continue
            arr = out[col_idx]
            n = len(labels)
            # 支持 float 编码值，取整后裁剪到 [0, n-1]；NaN 视为 0
            idx = np.asarray(arr, dtype=float)
            idx = np.nan_to_num(idx, nan=0.0)
            idx = np.round(idx).astype(int)
            idx = np.clip(idx, 0, n - 1)
            out[col_idx] = np.array(labels, dtype=object)[idx]
    # 2) 列名映射回原名
    name_mapping = info.get('idx_name_mapping')
    if name_mapping is None and info.get('column_names'):
        name_mapping = {str(i): info['column_names'][i] for i in range(len(info['column_names']))}
    if name_mapping:
        name_mapping = {str(k): v for k, v in name_mapping.items()}
        out.columns = [name_mapping.get(str(i), i) for i in out.columns]
    return out


def _remap_efvfm_info_to_xy_layout(base_info, feature_indices, target_idx):
    """
    将 ef-vfm 原始列索引语义映射到当前 Xy 布局：
    当前布局为 [feature_indices..., target]，其中 feature_indices 已去除 target。
    """
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(feature_indices)}
    old_to_new[target_idx] = len(feature_indices)

    def _remap_idx_list(old_idx_list):
        return [old_to_new[idx] for idx in old_idx_list if idx in old_to_new]

    remapped_info = {
        "num_col_idx": _remap_idx_list(base_info["num_col_idx"]),
        "cat_col_idx": _remap_idx_list(base_info["cat_col_idx"]),
        "target_col_idx": _remap_idx_list(base_info["target_col_idx"]),
        "task_type": base_info["task_type"],
        "metadata": {"columns": {}},
    }

    metadata_cols = (base_info.get("metadata") or {}).get("columns", {})
    for old_idx, new_idx in old_to_new.items():
        col_meta = metadata_cols.get(str(old_idx))
        if col_meta is None:
            col_meta = metadata_cols.get(old_idx)
        if col_meta is not None:
            remapped_info["metadata"]["columns"][str(new_idx)] = col_meta

    idx_name_mapping = base_info.get("idx_name_mapping")
    if idx_name_mapping:
        remapped_info["idx_name_mapping"] = {
            str(old_to_new[int(old_idx)]): name
            for old_idx, name in idx_name_mapping.items()
            if int(old_idx) in old_to_new
        }

    column_names = base_info.get("column_names")
    if column_names:
        reordered_names = [None] * (len(feature_indices) + 1)
        for old_idx, new_idx in old_to_new.items():
            if 0 <= old_idx < len(column_names):
                reordered_names[new_idx] = column_names[old_idx]
        remapped_info["column_names"] = reordered_names

    return remapped_info, old_to_new


def _remap_cat_labels_to_xy_layout(cat_labels, old_to_new):
    if not cat_labels:
        return None
    return {
        old_to_new[int(old_idx)]: labels
        for old_idx, labels in cat_labels.items()
        if int(old_idx) in old_to_new
    }


# Parse n_t_sampling sweep list (e.g. "10,20,30,50")
if getattr(args, 'n_t_sampling_list', None) and str(args.n_t_sampling_list).strip():
    args.n_t_sampling_list_parsed = [
        int(x.strip()) for x in str(args.n_t_sampling_list).split(',') if x.strip()
    ]
else:
    args.n_t_sampling_list_parsed = []
if args.plot_density and not args.results_dir:
    args.results_dir = 'results'
if getattr(args, 'results_dir', None):
    args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":

    if args.n_t_sampling == 0:
        args.n_t_sampling = args.n_t

    OTLIM = 5000

    dataset_index_start = 0
    method_index_start = 0
    if args.restore_from_name:
        if os.path.isfile(args.name):
            with open(args.name, 'r') as f: # where we track where we are, to restore sessions after crashes
                dataset_index_start, method_index_start = f.read().split('&')
                dataset_index_start = int(dataset_index_start)
                method_index_start = int(method_index_start)

    for dataset_index in range(dataset_index_start, len(args.datasets)):

        dataset = args.datasets[dataset_index]
        if dataset == 'ecoli' and args.add_missing_data:
            print("Ecoli with missing data can causes problems in the metrics due to near-constant variables, skipping it")
            continue
        print(dataset)
        X, bin_x, cat_x, int_x, y, bin_y, cat_y, int_y, column_names = dataset_loader(dataset)
        # 保证所有数据集都有列名（保存 samples 时映射回原名）
        if column_names is None or len(column_names) != X.shape[1] + 1:
            column_names = [f'feature_{i}' for i in range(X.shape[1])] + ['target']

        # For ef-vfm datasets we use the predefined train/test splits
        efvfm_split = None
        efvfm_info_xy = None
        efvfm_cat_labels_xy = None
        if dataset in EFVFM_DATASETS:
            efvfm_split = get_efvfm_splits(dataset)
            if efvfm_split is not None:
                efvfm_info_xy, efvfm_old_to_new = _remap_efvfm_info_to_xy_layout(
                    efvfm_split["info"],
                    efvfm_split["feature_indices"],
                    efvfm_split["target_idx"],
                )
                efvfm_cat_labels_xy = _remap_cat_labels_to_xy_layout(
                    efvfm_split.get("cat_labels"),
                    efvfm_old_to_new,
                )

        # Binary
        bin_indexes = []
        if bin_x is not None:
            bin_indexes = bin_indexes + bin_x
        bin_indexes_no_y = copy.deepcopy(bin_indexes)
        if bin_y:
            bin_indexes.append(X.shape[1])

        # Categorical (>=2 classes)
        cat_indexes = []
        if cat_x is not None:
            cat_indexes = cat_indexes + cat_x
        cat_indexes_no_y = copy.deepcopy(cat_indexes)
        if cat_y:
            cat_indexes.append(X.shape[1])

        # Integers
        int_indexes = []
        if int_x is not None:
            int_indexes = int_indexes + int_x
        int_indexes_no_y = copy.deepcopy(int_indexes)
        if int_y:
            int_indexes.append(X.shape[1])

        score_W1_train = {}
        score_W1_test = {}
        coverage = {}
        coverage_test = {}
        time_taken = {}
        percent_bias = {}
        coverage_rate = {}
        AW = {}
        f1_class = {}
        
        # New metrics
        density_val = {}
        mle_val = {}
        c2st_val = {}

        for method in args.methods:
            score_W1_test[method] = 0.0
            score_W1_train[method] = 0.0
            coverage[method] = 0.0
            coverage_test[method] = 0.0
            time_taken[method] = 0.0
            percent_bias[method] = 0.0
            coverage_rate[method] = 0.0
            AW[method] = 0.0
            f1_class[method] = []
            
            density_val[method] = {'Shape': 0.0, 'Trend': 0.0, 'Overall': 0.0}
            mle_val[method] = 0.0
            c2st_val[method] = 0.0

        R2 = {}
        f1 = {}
        for method in args.methods:
            R2[method] = {'real': {}, 'fake': {}, 'both': {}}
            f1[method] = {'real': {}, 'fake': {}, 'both': {}}
            for test_type in ['real','fake','both']:
                for test_type2 in ['mean','lin','linboost', 'tree', 'treeboost']:
                    R2[method][test_type][test_type2] = 0.0
                    f1[method][test_type][test_type2] = 0.0

        for method_index in range(method_index_start, len(args.methods)):
            
            method = args.methods[method_index]
            print(f'method={method}')

            with open(args.name, 'w') as f: # where we track where we are, to restore sessions after crashes
                f.write(f'{dataset_index}&{method_index}')

            for n in range(args.nexp):

                print(n)

                # Need to train/test split for evaluating the linear regression performance and for W1 based on test
                if efvfm_split is not None:
                    X_train = efvfm_split['X_train']
                    X_test = efvfm_split['X_test']
                    y_train = efvfm_split['y_train']
                    y_test = efvfm_split['y_test']
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=0.2,
                        random_state=n,
                        stratify=y if bin_y or cat_y else None,
                    )

                # Build joint matrices and keep an untouched copy for evaluation
                Xy_train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
                Xy_test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)
                X_train_orig = X_train.copy()
                X_test_orig = X_test.copy()
                y_train_orig = y_train.copy()
                y_test_orig = y_test.copy()
                Xy_train_orig = Xy_train.copy()
                Xy_test_orig = Xy_test.copy()

                # Optional quantile preprocessing on numerical columns
                qt = None
                num_col_idx = None
                if args.use_quantile:
                    all_indices = set(range(Xy_train.shape[1]))
                    cat_bin_set = set(bin_indexes + cat_indexes)
                    num_col_idx = sorted(list(all_indices - cat_bin_set))
                    if len(num_col_idx) > 0:
                        Xy_train, Xy_test, qt = apply_quantile_fit_transform(
                            Xy_train,
                            Xy_test,
                            num_col_idx,
                            n_quantiles=args.n_quantiles,
                            random_state=n,
                        )
                        # Update X_train / X_test / y_train / y_test for training/generation
                        X_train = Xy_train[:, :-1]
                        y_train = Xy_train[:, -1]
                        X_test = Xy_test[:, :-1]
                        y_test = Xy_test[:, -1]

                # Optional ef-vfm-style missing value imputation (on original X/y space)
                if args.efvfm_style_impute:
                    all_indices = set(range(Xy_train.shape[1]))
                    cat_indices_set = set(bin_indexes + cat_indexes)
                    target_idx = Xy_train.shape[1] - 1
                    num_cols_impute = sorted(list(all_indices - cat_indices_set - {target_idx}))
                    cat_cols_impute = sorted(list(cat_indices_set - {target_idx}))
                    if len(num_cols_impute) > 0 or len(cat_cols_impute) > 0:
                        Xy_train, Xy_test = efvfm_style_impute_train_test(
                            Xy_train,
                            Xy_test,
                            num_cols_impute,
                            cat_cols_impute,
                        )
                        X_train = Xy_train[:, :-1]
                        y_train = Xy_train[:, -1]
                        X_test = Xy_test[:, :-1]
                        y_test = Xy_test[:, -1]

                if args.add_missing_data:
                    print("Adding missing data")

                    torch.manual_seed(n)
                    np.random.seed(n)

                    if torch.cuda.is_available():
                        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
                    else:
                        torch.set_default_tensor_type('torch.DoubleTensor')

                    ### Each entry from the second axis has a probability p of being NA 
                    X_true = torch.tensor(X_train)
                    mask_x = (torch.rand(X_true.shape) < args.p).double()

                    # Now adding the outcome Y without missing data
                    Xy_true = torch.tensor(Xy_train)
                    mask = torch.zeros_like(Xy_true)
                    mask[:, :-1] = mask_x

                    Xy_nas = Xy_true.clone()
                    Xy_nas[mask.bool()] = np.nan # torch data
                    data_nas = Xy_nas.cpu().numpy() # numpy data
                    M = mask.sum(1) > 0

                    mask_np = mask.detach().cpu().numpy()
                    M_np = M.detach().cpu().numpy()

                    # We must impute the data first
                    if args.imputation_method == 'MissForest':
                        data_nas_ = copy.deepcopy(data_nas)
                        imputer = MissForest(random_state=0)
                        Xy_train_used = imputer.fit_transform(data_nas_, cat_vars=bin_indexes + cat_indexes if len(bin_indexes + cat_indexes) > 0 else None)
                    elif args.imputation_method == 'miceforest':
                        # Convert to Pandas
                        data_pd = pd.DataFrame(data_nas, columns = [str(i) for i in range(data_nas.shape[1])])
                        # indicate which column is categorical so that they are handled properly
                        for column_k in bin_indexes + cat_indexes:
                            data_pd[str(column_k)] = data_pd[str(column_k)].astype('category') 
                        kds = mf.ImputationKernel(data_pd, save_all_iterations=False, datasets=1, random_state=n)
                        kds.mice(5) # 5 iterations is the default and should be enough
                        Xy_train_used = kds.complete_data(dataset=0).to_numpy()
                    elif args.imputation_method == "none":
                        assert method == "forest_diffusion"
                        Xy_train_used = data_nas
                    else:
                        raise NotImplementedError("imputation_method must be MissForest or miceforest")

                else: # no missing data
                    Xy_train_used = Xy_train

                if method in ['TabDDPM','TVAE'] or not torch.cuda.is_available():
                    torch.set_default_tensor_type('torch.FloatTensor')
                else:
                    torch.set_default_tensor_type('torch.cuda.FloatTensor')

                start = time.time()

                if method == 'oracle':
                    Xy_fake = np.tile(np.expand_dims(Xy_train, axis=0), reps=(args.ngen, 1, 1)) # [ngen, n, p]

                elif method == 'CTGAN':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = CTGANSynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'GaussianCopula':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = GaussianCopulaSynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'TVAE':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = TVAESynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'CopulaGAN':

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    # indicate which column is categorical so that they are handled properly (only used for the metadata)
                    for column_k in bin_indexes + cat_indexes:
                        data_pd[str(column_k)] = data_pd[str(column_k)].astype('category')

                    metadata = SingleTableMetadata()
                    metadata.detect_from_dataframe(data=data_pd)
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])]) # category dtype cause problems so we remove it

                    synthesizer = CopulaGANSynthesizer(metadata)
                    synthesizer.fit(data_pd)

                    def my_synthesizer():
                        synthetic_data = synthesizer.sample(num_rows=Xy_train_used.shape[0])
                        return synthetic_data.to_numpy()
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'CTABGAN': # CTABGAN+
                    from CTABGANPlus.ctabgan import CTABGAN

                    # Convert to Pandas
                    data_pd = pd.DataFrame(Xy_train_used, columns = [str(i) for i in range(Xy_train_used.shape[1])])
                    synthesizer =  CTABGAN(pd_data = data_pd,
                                     categorical_columns = [str(i) for i in cat_indexes + bin_indexes],  
                                     general_columns= [str(i) for i in range(Xy_train_used.shape[1]) if i not in cat_indexes + bin_indexes + int_indexes],
                                     integer_columns = [str(i) for i in int_indexes]) 
                    synthesizer.fit()

                    def my_synthesizer():
                        synthetic_data = synthesizer.generate_samples()
                        return synthetic_data.to_numpy().astype('float')
                    Xy_fake = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                    for i_gen in range(args.ngen-1):
                        Xy_fake_new = np.expand_dims(try_until_all_classes_found(y=y_test, synthesizer=my_synthesizer, cat=bin_y or cat_y), axis=0)
                        Xy_fake = np.concatenate((Xy_fake, Xy_fake_new), axis=0) # [ngen, n, p]

                elif method == 'stasy':
                    Xy_fake = STaSy_model(Xy_train_used,
                       categorical_columns=cat_indexes + bin_indexes, 
                       ordinal_columns=int_indexes, 
                       seed=n, 
                       epochs=10000,
                       ngen=args.ngen,
                       activation = args.act, layer_type = args.layer_type, sde = args.sde, lr = args.lr, num_scales = args.num_scales)
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]

                elif method == 'TabDDPM': # TabDDPM
                    from TabDDPM.scripts.pipeline import main_fn as tab_ddpm_fn
                    from TabDDPM.lib.dataset_prep import my_data_prep

                    # Prep the data, will be save in the format that TabDDPM wants
                    columns = [str(i) for i in range(Xy_train_used.shape[1])]
                    data_pd = pd.DataFrame(Xy_train_used, columns = columns)
                    X_pd = data_pd[columns[0:-1]]
                    y_pd = data_pd[columns[-1]]
                    cat_ind = [str(i) for i in range(X_pd.shape[1]) if i in cat_indexes+bin_indexes]
                    noncat_ind = [str(i) for i in range(X_pd.shape[1]) if i not in cat_indexes+bin_indexes]
                    if cat_y:
                        task_type='multiclass'
                    elif bin_y:
                        task_type='binclass'
                    else:
                        task_type='regression'
                    my_data_prep(X_pd, y_pd, task=task_type, cat_ind=cat_ind, noncat_ind=noncat_ind)
                    synthetic_data = tab_ddpm_fn(config='TabDDPM/config/config.toml', 
                        cat_indexes=cat_indexes+bin_indexes, num_classes=len(np.unique(y_pd)) if cat_y or bin_y else 0, 
                        num_samples=Xy_train_used.shape[0], num_numerical_features=len(noncat_ind), seed=n, ngen=args.ngen)
                    Xy_fake = synthetic_data.astype('float')
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]

                elif method == 'forest_diffusion':

                    if args.ycond and (bin_y or cat_y):
                        forest_model = ForestDiffusionModel(X=Xy_train_used[:,:-1],
                            label_y=Xy_train_used[:,-1],
                            n_t=args.n_t,
                            model=args.forest_model, # in random_forest, xgboost, lgbm
                            diffusion_type=args.diffusion_type, # vp, flow
                            max_depth = args.max_depth, n_estimators = args.n_estimators, # random_forest and xgboost hyperparameters
                            eta=args.eta, # xgboost hyperparameters
                            num_leaves=args.num_leaves, # lgbm hyperparameters
                            gpu_hist=args.gpu_hist,
                            duplicate_K=args.duplicate_K,
                            cat_indexes=cat_indexes_no_y,
                            bin_indexes=bin_indexes_no_y,
                            int_indexes=int_indexes_no_y,
                            n_jobs=args.n_jobs,
                            n_batch=args.n_batch,
                            use_ot=args.use_ot,
                            eps=args.eps, beta_min=args.beta_min, beta_max=args.beta_max,
                            seed=n)
                    else:
                        forest_model = ForestDiffusionModel(X=Xy_train_used,
                            n_t=args.n_t,
                            model=args.forest_model, # in random_forest, xgboost, lgbm
                            diffusion_type=args.diffusion_type, # vp, flow
                            max_depth = args.max_depth, n_estimators = args.n_estimators, # random_forest and xgboost hyperparameters
                            eta=args.eta, # xgboost hyperparameters
                            num_leaves=args.num_leaves, # lgbm hyperparameters
                            gpu_hist=args.gpu_hist,
                            duplicate_K=args.duplicate_K,
                            cat_indexes=cat_indexes,
                            bin_indexes=bin_indexes,
                            int_indexes=int_indexes,
                            n_jobs=args.n_jobs,
                            n_batch=args.n_batch,
                            use_ot=args.use_ot,
                            eps=args.eps, beta_min=args.beta_min, beta_max=args.beta_max,
                            seed=n)
                    sweep_list = getattr(args, 'n_t_sampling_list_parsed', None) or []
                    if sweep_list:
                        # sweep n_t_sampling with repeats; train once, multiple generate+eval (align with ForestDiffusion)
                        run_dir_base = None
                        if getattr(args, 'results_dir', None):
                            method_safe = method.replace(' ', '_')
                            run_dir_base = os.path.join(
                                args.results_dir, dataset, method_safe, f"run_{args.run_id}"
                            )
                            os.makedirs(run_dir_base, exist_ok=True)
                            args_path = os.path.join(run_dir_base, "args.json")
                            if not os.path.isfile(args_path):
                                args_dict = {
                                    k: v for k, v in vars(args).items()
                                    if isinstance(v, (str, int, float, bool, list, dict, type(None)))
                                }
                                with open(args_path, "w", encoding="utf-8") as f:
                                    json.dump(args_dict, f, indent=2, ensure_ascii=False)
                        else:
                            run_dir_base = os.path.join(
                                args.results_dir if getattr(args, 'results_dir', None) else 'results',
                                dataset, method.replace(' ', '_'), f"run_{args.run_id}"
                            )
                            os.makedirs(run_dir_base, exist_ok=True)

                        # 对 ef-vfm 数据集，直接使用 ef-vfm 的 info，避免重复推断导致列类型不一致
                        if efvfm_split is not None:
                            base_info_sw = efvfm_info_xy
                            info_sw = {
                                "num_col_idx": base_info_sw["num_col_idx"],
                                "cat_col_idx": base_info_sw["cat_col_idx"],
                                "target_col_idx": base_info_sw["target_col_idx"],
                                "task_type": base_info_sw["task_type"],
                                "metadata": base_info_sw["metadata"],
                            }
                            num_col_idx_sw = base_info_sw["num_col_idx"]
                            cat_col_idx_sw = base_info_sw["cat_col_idx"]
                            target_idx_sw = base_info_sw["target_col_idx"][0]
                        else:
                            # 非 ef-vfm 数据集保持原来的自动推断逻辑
                            all_indices_sw = set(range(Xy_train.shape[1]))
                            cat_indices_set_sw = set(bin_indexes + cat_indexes)
                            target_idx_sw = Xy_train.shape[1] - 1
                            num_col_idx_sw = sorted(list(all_indices_sw - cat_indices_set_sw - {target_idx_sw}))
                            cat_col_idx_sw = sorted(list(cat_indices_set_sw - {target_idx_sw}))
                            data_pd_sw = pd.DataFrame(
                                Xy_train_orig if args.use_quantile else Xy_train,
                                columns=[str(i) for i in range(Xy_train.shape[1])],
                            )
                            for column_k in bin_indexes + cat_indexes:
                                if str(column_k) in data_pd_sw.columns:
                                    data_pd_sw[str(column_k)] = data_pd_sw[str(column_k)].astype("category")
                            metadata_sw = SingleTableMetadata()
                            metadata_sw.detect_from_dataframe(data=data_pd_sw)
                            info_metadata_sw = metadata_sw.to_dict()
                            if cat_y:
                                task_type_sw = "multiclass"
                            elif bin_y:
                                task_type_sw = "binclass"
                            else:
                                task_type_sw = "regression"
                            info_sw = {
                                "num_col_idx": num_col_idx_sw,
                                "cat_col_idx": cat_col_idx_sw,
                                "target_col_idx": [target_idx_sw],
                                "task_type": task_type_sw,
                                "metadata": info_metadata_sw,
                                "column_names": column_names,
                            }
                        df_train_sw = pd.DataFrame(
                            Xy_train_orig if args.use_quantile else Xy_train,
                            columns=range(Xy_train.shape[1]),
                        )
                        df_test_sw = pd.DataFrame(
                            Xy_test_orig if args.use_quantile else Xy_test,
                            columns=range(Xy_test.shape[1]),
                        )

                        curve_rows = []
                        repeats = max(1, int(getattr(args, 'n_t_sampling_repeats', 3)))
                        for n_ts in sweep_list:
                            reps_metrics = []
                            sweep_dir = os.path.join(
                                run_dir_base, 'n_t_sampling_sweep', f'n_ts_{n_ts}'
                            )
                            for rep in range(repeats):
                                np.random.seed(n + rep * 1000 + n_ts)
                                Xy_flat = forest_model.generate(
                                    batch_size=args.ngen * Xy_train_used.shape[0],
                                    n_t=n_ts,
                                )
                                Xy_fake_sw = Xy_flat.reshape(
                                    args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]
                                )
                                # 注意：quantile 拟合时使用的是 num_col_idx（可能包含数值型 target），
                                # 因此在 sweep 阶段做 inverse_transform 也必须用同一组列索引，
                                # 不能直接用 info/base_info_sw 里的 num_col_idx_sw（该索引不含 target），
                                # 否则会出现 “X has d features, but QuantileTransformer is expecting d+1” 的错误。
                                if (
                                    args.use_quantile
                                    and qt is not None
                                    and num_col_idx is not None
                                    and len(num_col_idx) > 0
                                ):
                                    Xy_fake_sw[0] = apply_quantile_inverse_transform(
                                        Xy_fake_sw[0], num_col_idx, qt
                                    )
                                # Skip int inverse (rint) for eval/samples to align with ef-vfm (continuous int cols).
                                # if args.use_quantile and int_indexes is not None and len(int_indexes) > 0:
                                #     Xy_fake_sw[0] = apply_int_inverse_transform(
                                #         Xy_fake_sw[0], int_indexes
                                #     )
                                Xy_fake_i_sw = Xy_fake_sw[0]
                                df_fake_sw = pd.DataFrame(
                                    Xy_fake_i_sw, columns=range(Xy_fake_i_sw.shape[1])
                                )
                                try:
                                    tab_metrics_sw = TabMetrics(
                                        real_data=df_train_sw,
                                        test_data=df_test_sw,
                                        val_data=None,
                                        info=info_sw,
                                        metric_list=['density', 'mle', 'c2st'],
                                    )
                                    metrics_res_sw, extras_sw = tab_metrics_sw.evaluate(
                                        df_fake_sw
                                    )
                                    reps_metrics.append(metrics_res_sw)
                                    rep_dir = os.path.join(sweep_dir, f'rep_{rep}')
                                    os.makedirs(rep_dir, exist_ok=True)
                                    _samples_df_for_save(
                                        df_fake_sw,
                                        base_info_sw if efvfm_split is not None else info_sw,
                                        cat_labels=(
                                            efvfm_cat_labels_xy
                                            if efvfm_split is not None
                                            else _build_cat_labels_from_train(
                                                Xy_train, bin_indexes, cat_indexes, cat_y, bin_y
                                            )
                                        ),
                                    ).to_csv(
                                        os.path.join(rep_dir, 'samples.csv'), index=False
                                    )
                                    with open(
                                        os.path.join(rep_dir, 'all_results.json'), 'w'
                                    ) as f:
                                        json.dump(metrics_res_sw, f, indent=4)
                                    for name, extra in extras_sw.items():
                                        if isinstance(extra, pd.DataFrame):
                                            extra.to_csv(
                                                os.path.join(rep_dir, f'{name}.csv'),
                                                index=False,
                                            )
                                        elif isinstance(extra, dict):
                                            with open(
                                                os.path.join(rep_dir, f'{name}.json'), 'w'
                                            ) as f:
                                                json.dump(extra, f, indent=4)
                                    if getattr(args, 'plot_density', False):
                                        img = tab_metrics_sw.plot_density(df_fake_sw)
                                        img.save(
                                            os.path.join(rep_dir, 'density_plots.png')
                                        )
                                    print(
                                        f"n_t_sampling sweep n_ts={n_ts} rep={rep} -> {rep_dir}"
                                    )
                                except Exception as e:
                                    print(
                                        f"n_t_sampling sweep n_ts={n_ts} rep={rep} error: {e}"
                                    )
                            if reps_metrics:
                                all_keys = set()
                                for m in reps_metrics:
                                    all_keys.update(m.keys())
                                mean_metrics = {}
                                std_metrics = {}
                                for key in sorted(all_keys):
                                    vals = []
                                    for m in reps_metrics:
                                        if key in m and isinstance(
                                            m[key], (int, float, np.floating)
                                        ):
                                            vals.append(float(m[key]))
                                    if vals:
                                        mean_metrics[key] = float(np.mean(vals))
                                        std_metrics[key] = (
                                            float(np.std(vals)) if len(vals) > 1 else 0.0
                                        )
                                curve_rows.append({
                                    'n_t_sampling': n_ts,
                                    'mean': mean_metrics,
                                    'std': std_metrics,
                                    'repeats': repeats,
                                })
                        curve_path = os.path.join(run_dir_base, 'n_t_sampling_curve.json')
                        with open(curve_path, 'w', encoding='utf-8') as f:
                            json.dump(curve_rows, f, indent=2, ensure_ascii=False)
                        print(f"n_t_sampling curve saved to {curve_path}")

                        if curve_rows and plt is not None:
                            try:
                                xs = [r['n_t_sampling'] for r in curve_rows]
                                first_mean = curve_rows[0].get('mean', {})
                                metric_keys = [
                                    k for k in first_mean.keys()
                                    if isinstance(first_mean[k], (int, float, np.floating))
                                ]
                                n_plots = len(metric_keys)
                                if n_plots > 0:
                                    fig, axes = plt.subplots(
                                        n_plots,
                                        1,
                                        figsize=(8, 2.5 * n_plots),
                                        squeeze=False,
                                    )
                                    for idx, key in enumerate(metric_keys):
                                        ax = axes[idx, 0]
                                        ys = [r['mean'].get(key, np.nan) for r in curve_rows]
                                        yerr = [r['std'].get(key, 0) for r in curve_rows]
                                        ax.errorbar(xs, ys, yerr=yerr, marker='o', capsize=3)
                                        ax.set_xlabel('n_t_sampling')
                                        ax.set_ylabel(key)
                                        ax.set_title(key)
                                        ax.grid(True, alpha=0.3)
                                    plt.tight_layout()
                                    plot_path = os.path.join(
                                        run_dir_base, 'n_t_sampling_curve.png'
                                    )
                                    plt.savefig(plot_path, dpi=150)
                                    plt.close()
                                    print(f"n_t_sampling curve plot saved to {plot_path}")
                                else:
                                    print("n_t_sampling curve plot skipped: no numeric metrics in mean.")
                            except Exception as e:
                                err = globals().get('_matplotlib_import_error', '')
                                print(f"n_t_sampling curve plot failed: {e} {err}")
                        elif curve_rows and plt is None:
                            err = globals().get('_matplotlib_import_error', '')
                            print(f"n_t_sampling curve plot skipped: matplotlib not available. {err}")

                        end = time.time()
                        time_taken[method] += (end - start) / args.nexp
                        continue

                    Xy_fake = forest_model.generate(batch_size=args.ngen*Xy_train_used.shape[0], n_t=args.n_t_sampling)
                    Xy_fake = Xy_fake.reshape(args.ngen, Xy_train_used.shape[0], Xy_train_used.shape[1]) # [ngen, n, p]
                end = time.time()
                time_taken[method] += (end - start) / args.nexp
                assert Xy_fake.shape[0] == args.ngen and Xy_fake.shape[1] == Xy_train_used.shape[0] and Xy_fake.shape[2] == Xy_train_used.shape[1]

                # If we used quantile preprocessing, bring fake samples back to the original space.
                # Align with ef-vfm for evaluation: do NOT rint integer columns (ef-vfm with dequant_dist='none'
                # keeps integer columns continuous), so shape/KS is computed as continuous synth vs discrete real.
                if args.use_quantile and qt is not None and num_col_idx is not None and len(num_col_idx) > 0:
                    for gen_i in range(args.ngen):
                        Xy_fake[gen_i] = apply_quantile_inverse_transform(Xy_fake[gen_i], num_col_idx, qt)
                # Skip int inverse (rint) so integer columns stay continuous for eval and saved samples (ef-vfm-aligned).
                # if args.use_quantile and int_indexes is not None and len(int_indexes) > 0:
                #     for gen_i in range(args.ngen):
                #         Xy_fake[gen_i] = apply_int_inverse_transform(Xy_fake[gen_i], int_indexes)

                for gen_i in range(args.ngen):

                    #np.set_printoptions(threshold=np.inf)
                    #print(Xy_train[0:150])
                    #print(Xy_fake[gen_i][0:150])

                    Xy_fake_i = Xy_fake[gen_i]

                    # New metrics calculation (density, mle, c2st)
                    # Construct info object
                    if gen_i == 0:  # Do it once per experiment to save time on metadata detection
                        if efvfm_split is not None:
                            # 对 ef-vfm 数据集，直接复用其 info（含 idx_name_mapping/column_names 以便保存 samples 时列名映射回原名）
                            base_info_eval = efvfm_info_xy
                            info = {
                                "num_col_idx": base_info_eval["num_col_idx"],
                                "cat_col_idx": base_info_eval["cat_col_idx"],
                                "target_col_idx": base_info_eval["target_col_idx"],
                                "task_type": base_info_eval["task_type"],
                                "metadata": base_info_eval["metadata"],
                            }
                            if "idx_name_mapping" in base_info_eval:
                                info["idx_name_mapping"] = base_info_eval["idx_name_mapping"]
                            if "column_names" in base_info_eval:
                                info["column_names"] = base_info_eval["column_names"]
                        else:
                            # 非 ef-vfm 数据集保持自动推断逻辑
                            data_pd = pd.DataFrame(
                                Xy_train_orig if args.use_quantile else Xy_train,
                                columns=[str(i) for i in range(Xy_train.shape[1])],
                            )
                            # indicate which column is categorical
                            for column_k in bin_indexes + cat_indexes:
                                if str(column_k) in data_pd.columns:
                                    data_pd[str(column_k)] = data_pd[str(column_k)].astype("category")
                            metadata = SingleTableMetadata()
                            metadata.detect_from_dataframe(data=data_pd)
                            info_metadata = metadata.to_dict()

                            # Determine task type
                            if cat_y:
                                task_type = "multiclass"
                            elif bin_y:
                                task_type = "binclass"
                            else:
                                task_type = "regression"

                            # Indices
                            all_indices = set(range(Xy_train.shape[1]))
                            cat_indices_set = set(bin_indexes + cat_indexes)
                            target_idx = Xy_train.shape[1] - 1

                            # Remove target from feature indices lists
                            num_col_idx = list(all_indices - cat_indices_set - {target_idx})
                            cat_col_idx = list(cat_indices_set - {target_idx})
                            target_col_idx = [target_idx]

                            info = {
                                "num_col_idx": sorted(num_col_idx),
                                "cat_col_idx": sorted(cat_col_idx),
                                "target_col_idx": target_col_idx,
                                "task_type": task_type,
                                "metadata": info_metadata,
                                "column_names": column_names,
                            }

                        # Prepare DataFrames for TabMetrics（两种分支公用）
                        df_train = pd.DataFrame(
                            Xy_train_orig if args.use_quantile else Xy_train,
                            columns=range(Xy_train.shape[1]),
                        )
                        df_test = pd.DataFrame(
                            Xy_test_orig if args.use_quantile else Xy_test,
                            columns=range(Xy_test.shape[1]),
                        )

                    # Evaluate
                    # Xy_fake_i is numpy array
                    df_fake = pd.DataFrame(Xy_fake_i, columns=range(Xy_fake_i.shape[1]))
                    
                    try:
                        tab_metrics = TabMetrics(real_data=df_train, test_data=df_test, val_data=None, info=info, metric_list=['density', 'mle', 'c2st'])
                        metrics_res, extras = tab_metrics.evaluate(df_fake)
                        
                        density_val[method]['Shape'] += metrics_res.get('density/Shape', 0.0) / (args.nexp*args.ngen)
                        density_val[method]['Trend'] += metrics_res.get('density/Trend', 0.0) / (args.nexp*args.ngen)
                        density_val[method]['Overall'] += metrics_res.get('density/Overall', 0.0) / (args.nexp*args.ngen)
                        mle_val[method] += metrics_res.get('mle', 0.0) / (args.nexp*args.ngen)
                        c2st_val[method] += metrics_res.get('c2st', 0.0) / (args.nexp*args.ngen)

                        # Save generated data and detailed metrics (reference: ef-vfm-dev-mixed)
                        # 使用 run_id 子目录，不覆盖之前运行的结果；并 dump args 配置
                        if getattr(args, 'results_dir', None):
                            method_safe = method.replace(' ', '_')
                            run_dir = os.path.join(args.results_dir, dataset, method_safe, f"run_{args.run_id}")
                            save_path = os.path.join(run_dir, f"exp_{n}", f"gen_{gen_i}")
                            os.makedirs(save_path, exist_ok=True)
                            # 每个 run 只写一次 args 配置
                            args_path = os.path.join(run_dir, "args.json")
                            if not os.path.isfile(args_path):
                                args_dict = {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                                with open(args_path, "w", encoding="utf-8") as f:
                                    json.dump(args_dict, f, indent=2, ensure_ascii=False)
                            _samples_df_for_save(
                                df_fake,
                                info,
                                cat_labels=(
                                    efvfm_cat_labels_xy
                                    if efvfm_split is not None
                                    else _build_cat_labels_from_train(
                                        Xy_train, bin_indexes, cat_indexes, cat_y, bin_y
                                    )
                                ),
                            ).to_csv(
                                os.path.join(save_path, "samples.csv"), index=False
                            )
                            with open(os.path.join(save_path, "all_results.json"), "w") as f:
                                json.dump(metrics_res, f, indent=4)
                            for name, extra in extras.items():
                                if isinstance(extra, pd.DataFrame):
                                    extra.to_csv(os.path.join(save_path, f"{name}.csv"), index=False)
                                elif isinstance(extra, dict):
                                    with open(os.path.join(save_path, f"{name}.json"), "w") as f:
                                        json.dump(extra, f, indent=4)
                            if getattr(args, 'plot_density', False):
                                img = tab_metrics.plot_density(df_fake)
                                img.save(os.path.join(save_path, "density_plots.png"))
                                print(f"Density plots saved to {os.path.join(save_path, 'density_plots.png')}")
                            print(f"Results saved to {save_path}")
                    except Exception as e:
                        print(f"Error calculating vfm metrics: {e}")

                    # Mixed data is tricky, nearest neighboors (for the coverage) and Wasserstein distance (based on L2) are not scale invariant
                    # To ensure that the scaling between variables is relatively uniformized, we take inspiration from the Gower distance used in mixed-data KNNs: https://medium.com/analytics-vidhya/the-ultimate-guide-for-clustering-mixed-data-1eefa0b4743b
                    # Continuous: we do min-max normalization (to use Gower |x1-x2|/(max-min) as distance)
                    # Categorical: We one-hot and then divide by 2 (e.g., 0 0 0.5 with 0.5 0 0 will have distance 0.5 + 0.5 = 1)
                    # After these transformations, taking the L1 (City-block / Manhattan distance) norm distance will give the Gower distance
                    base_train = Xy_train_orig if args.use_quantile else Xy_train
                    base_test = Xy_test_orig if args.use_quantile else Xy_test
                    Xy_train_scaled, Xy_fake_scaled, _, _, _ = minmax_scale_dummy(base_train, Xy_fake_i, cat_indexes, divide_by=2)
                    _, Xy_test_scaled, _, _, _ = minmax_scale_dummy(base_train, base_test, cat_indexes, divide_by=2)

                    assert Xy_train_scaled.shape[1] == Xy_fake_scaled.shape[1] == Xy_test_scaled.shape[1], f"Xy_train_scaled.shape: {Xy_train_scaled.shape}, Xy_fake_scaled.shape: {Xy_fake_scaled.shape}, Xy_test_scaled.shape: {Xy_test_scaled.shape}"

                    # Wasserstein-1 based on L1 cost (after scaling)
                    if Xy_train.shape[0] < OTLIM:
                        score_W1_train[method] += pot.emd2(pot.unif(Xy_train_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_train_scaled, Xy_fake_scaled, metric='cityblock')) / (args.nexp*args.ngen)
                        score_W1_test[method] += pot.emd2(pot.unif(Xy_test_scaled.shape[0]), pot.unif(Xy_fake_scaled.shape[0]), M = pot.dist(Xy_test_scaled, Xy_fake_scaled, metric='cityblock')) / (args.nexp*args.ngen)

                    X_fake, y_fake = Xy_fake_i[:,:-1], Xy_fake_i[:,-1]

                    # Trained on real data (always original space)
                    base_X_train = X_train_orig if args.use_quantile else X_train
                    base_y_train = y_train_orig if args.use_quantile else y_train
                    base_X_test = X_test_orig if args.use_quantile else X_test
                    base_y_test = y_test_orig if args.use_quantile else y_test

                    f1_real, R2_real = test_on_multiple_models(base_X_train, base_y_train, base_X_test, base_y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)

                    # Trained on fake data
                    f1_fake, R2_fake = test_on_multiple_models(X_fake, y_fake, base_X_test, base_y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)

                    # Trained on real data and fake data
                    X_both = np.concatenate((base_X_train, X_fake), axis=0)
                    y_both = np.concatenate((base_y_train, y_fake))
                    f1_both, R2_both = test_on_multiple_models(X_both, y_both, base_X_test, base_y_test, classifier=cat_y or bin_y, cat_indexes=cat_indexes_no_y, nexp=args.n_tries)
                    
                    for key in['mean', 'lin', 'linboost', 'tree', 'treeboost']:
                        f1[method]['real'][key] += f1_real[key] / (args.nexp*args.ngen)
                        f1[method]['fake'][key] += f1_fake[key] / (args.nexp*args.ngen)
                        f1[method]['both'][key] += f1_both[key] / (args.nexp*args.ngen)
                        R2[method]['real'][key] += R2_real[key] / (args.nexp*args.ngen)
                        R2[method]['fake'][key] += R2_fake[key] / (args.nexp*args.ngen)
                        R2[method]['both'][key] += R2_both[key] / (args.nexp*args.ngen)

                    # Get another different fake data for use as test fake-data
                    Xy_fake_j = Xy_fake[(gen_i + 1) % args.ngen] # 0 -> 1, 1-> 2, n -> 0
                    # Classifier comparing real to fake data, the less it classify fake data as fake = the better
                    Xy_train_real = Xy_train_orig if args.use_quantile else Xy_train
                    f1_class[method] += [test_on_multiple_models_classifier(X_train_real=Xy_train_real, X_train_fake=Xy_fake_i, X_test_fake=Xy_fake_j, cat_indexes=cat_indexes, nexp=args.n_tries)]

                    # coverage based on L1 cost (after scaling)
                    coverage[method] += compute_coverage(Xy_train_scaled, Xy_fake_scaled, None) / (args.nexp*args.ngen)
                    coverage_test[method] += compute_coverage(Xy_test_scaled, Xy_fake_scaled, None) / (args.nexp*args.ngen)

                # Statistical measures
                X_fake = []
                y_fake = []
                for gen_i in range(args.ngen):
                    X_fake.append(np.expand_dims(Xy_fake[gen_i][:,:-1], axis=0))
                    y_fake.append(np.expand_dims(Xy_fake[gen_i][:,-1], axis=0))
                X_fake = np.concatenate(X_fake, axis=0) # [nimp, n, p-1]
                y_fake = np.concatenate(y_fake, axis=0) # [nimp, n, 1]
                # Too unstable with classification due toquasi-seperation with logistic regression
                # dataset=ecoli with missing data is removed because it has near-constant variables, and the non-constant parts can be lost when adding missing data making it perfectly multicorrelated which will give regression errors
                if not cat_y and not bin_y and not (dataset == 'ecoli' and args.add_missing_data):
                    base_X_train = X_train_orig if args.use_quantile else X_train
                    base_y_train = y_train_orig if args.use_quantile else y_train
                    percent_bias_, coverage_rate_, AW_ = test_imputation_regression(base_X_train, base_y_train, X_fake, y_fake,
                        cat_indexes=cat_indexes_no_y, type_model='regression')
                else: 
                    percent_bias_, coverage_rate_, AW_ = 0.0, 0.0, 0.0
                percent_bias[method] += percent_bias_ / args.nexp
                coverage_rate[method] += coverage_rate_ / args.nexp
                AW[method] += AW_ / args.nexp

            # Write results in csv file
            # Columns: dataset , method , score_W1_train , score_W1_test , R2_real , R2_fake , f1_real , f1_fake, coverage
            if method == 'forest_diffusion':
                method_str = f"{method} n_t={args.n_t} n_t_sampling={args.n_t_sampling} model={args.forest_model} diffusion={args.diffusion_type} duplicate_K={args.duplicate_K} ycond={args.ycond} "
                if args.forest_model == 'xgboost':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} eta={args.eta} "
                elif args.forest_model == 'random_forest':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} "
                elif args.forest_model == 'catboost':
                    method_str += f"depth={args.max_depth} n_trees={args.n_estimators} "
                elif args.forest_model == 'lgbm':
                    method_str += f"num_leaves={args.num_leaves} n_trees={args.n_estimators} lr={args.eta} "
            else:
                method_str = f"{method} "
            # Mark whether quantile / ef-vfm-style preprocessing was used
            method_str += f" quantile={args.use_quantile} "
            method_str += f" efvfm_impute={args.efvfm_style_impute} "
            method_str += f" use_ot={args.use_ot} "
            header_cols = ["dataset"]
            if args.add_missing_data:
                header_cols.append("mask")
            header_cols += [
                "method",
                "score_W1_train",
                "score_W1_test",
                "R2_real_mean",
                "R2_fake_mean",
                "R2_both_mean",
                "f1_real_mean",
                "f1_fake_mean",
                "f1_both_mean",
                "coverage_train",
                "coverage_test",
                "percent_bias",
                "coverage_rate",
                "AW",
                "f1_class",
                "time_taken",
                "density_shape",
                "density_trend",
                "density_overall",
                "mle",
                "c2st",
            ]
            for key in ['lin', 'linboost', 'tree', 'treeboost']:
                header_cols += [
                    f"R2_real_{key}",
                    f"R2_fake_{key}",
                    f"R2_both_{key}",
                    f"f1_real_{key}",
                    f"f1_fake_{key}",
                    f"f1_both_{key}",
                ]
            row = []
            row.append(dataset)
            if args.add_missing_data:
                mask_str = f"MCAR({args.p} {args.imputation_method}) "
                row += [mask_str, method_str]
            else:
                row.append(method_str)
            row += [
                score_W1_train[method],
                score_W1_test[method],
                R2[method]['real']['mean'],
                R2[method]['fake']['mean'],
                R2[method]['both']['mean'],
                f1[method]['real']['mean'],
                f1[method]['fake']['mean'],
                f1[method]['both']['mean'],
                coverage[method],
                coverage_test[method],
                percent_bias[method],
                coverage_rate[method],
                AW[method],
                f1_class[method],
                time_taken[method],
                density_val[method]['Shape'],
                density_val[method]['Trend'],
                density_val[method]['Overall'],
                mle_val[method],
                c2st_val[method],
            ]
            for key in ['lin', 'linboost', 'tree', 'treeboost']:
                row += [
                    R2[method]['real'][key],
                    R2[method]['fake'][key],
                    R2[method]['both'][key],
                    f1[method]['real'][key],
                    f1[method]['fake'][key],
                    f1[method]['both'][key],
                ]
            print(row)
            write_header = (not os.path.isfile(args.out_path)) or os.path.getsize(args.out_path) == 0
            with open(args.out_path, 'a+', newline='') as f: # where we keep track of the results
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header_cols)
                writer.writerow(row)
        method_index_start = 0 #  so we loop back again