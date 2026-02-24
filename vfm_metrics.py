from copy import deepcopy
import traceback
import numpy as np
import pandas as pd
# Metrics
from mle_utils import get_evaluator
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.single_table import LogisticDetection

class TabMetrics(object):
    def __init__(self, real_data, test_data, val_data, info, metric_list) -> None:
        self.real_data = real_data # DataFrame
        self.test_data = test_data # DataFrame
        self.val_data = val_data # DataFrame (optional)
        self.info = info
        self.metric_list = metric_list

    def evaluate(self, syn_data):
        # syn_data is DataFrame
        metrics, extras = {}, {}
        syn_data_cp = deepcopy(syn_data)
        for metric in self.metric_list:
            try:
                func = getattr(self, f"evaluate_{metric}")
                print(f"Evaluating {metric}")
                out_metrics, out_extras = func(syn_data_cp)
                metrics.update(out_metrics)
                extras.update(out_extras)
            except Exception as e:
                print(f"[EVAL ERROR] metric '{metric}' failed: {e}")
                traceback.print_exc()
        return metrics, extras
    
    def evaluate_density(self, syn_data):
        real_data = self.real_data.copy()
        real_data.columns = [str(i) for i in range(len(real_data.columns))]
        syn_data.columns = [str(i) for i in range(len(syn_data.columns))]
        
        info = deepcopy(self.info)
        
        y_only = len(syn_data.columns)==1
        if y_only:
             # Logic for y_only if needed, but likely not for this use case
             pass

        metadata = info['metadata']
        # Ensure keys are integers for sdv metadata
        # metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()} 
        # Actually sdmetrics expects strings as column names usually, but if we renamed columns to range(len), we need keys to match.
        # But wait, we renamed columns to integers 0, 1, 2...
        # So metadata keys should be integers? sdv metadata uses strings usually.
        # Let's align metadata with dataframe columns.
        
        # In reorder function, it constructs new metadata.
        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        qual_report = QualityReport()
        qual_report.generate(new_real_data, new_syn_data, metadata)

        diag_report = DiagnosticReport()
        diag_report.generate(new_real_data, new_syn_data, metadata)

        quality =  qual_report.get_properties()
        # diag = diag_report.get_properties()

        Shape = quality['Score'][0]
        Trend = quality['Score'][1]

        Overall = (Shape + Trend) / 2

        shape_details = qual_report.get_details(property_name='Column Shapes')
        trend_details = qual_report.get_details(property_name='Column Pair Trends')

        out_metrics = {
            "density/Shape": Shape,
            "density/Trend": Trend,
            "density/Overall": Overall,
        }
        out_extras = {
            "shapes": shape_details,
            "trends": trend_details
        }
        return out_metrics, out_extras
    
    def evaluate_mle(self, syn_data):
        train = syn_data.to_numpy()
        test = self.test_data.to_numpy()
        val = self.val_data.to_numpy() if self.val_data is not None else None
        
        info = deepcopy(self.info)

        task_type = info['task_type']

        evaluator = get_evaluator(task_type)

        if task_type == 'regression':
            best_r2_scores, best_rmse_scores = evaluator(train, test, info, val=val)
            
            overall_scores = {}
            for score_name in ['best_r2_scores', 'best_rmse_scores']:
                overall_scores[score_name] = {}
                
                # scores = eval(score_name) # eval is dangerous and relies on locals
                scores = locals()[score_name]
                for method in scores:
                    name = method['name']  
                    method_cp = method.copy()
                    method_cp.pop('name')
                    overall_scores[score_name][name] = method_cp 

        else:
            best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info, val=val)

            overall_scores = {}
            for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
                overall_scores[score_name] = {}
                
                # scores = eval(score_name)
                scores = locals()[score_name]
                for method in scores:
                    name = method['name']  
                    method_cp = method.copy()
                    method_cp.pop('name')
                    overall_scores[score_name][name] = method_cp
                    
        mle_score = overall_scores['best_rmse_scores']['XGBRegressor']['RMSE'] if task_type == 'regression' else overall_scores['best_auroc_scores']['XGBClassifier']['roc_auc']
        out_metrics = {
            "mle": mle_score,
        }
        out_extras = {
            "mle": overall_scores,
        }
        return out_metrics, out_extras
    
    def evaluate_c2st(self, syn_data):
        info = deepcopy(self.info)
        real_data = self.real_data.copy()

        real_data.columns = [str(i) for i in range(len(real_data.columns))]
        syn_data.columns = [str(i) for i in range(len(syn_data.columns))]

        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        score = LogisticDetection.compute(
            real_data=new_real_data,
            synthetic_data=new_syn_data,
            metadata=metadata
        )
        
        out_metrics = {
            "c2st": score,
        }
        out_extras = {}
        return out_metrics, out_extras


def reorder(real_data, syn_data, info):
    num_col_idx = deepcopy(info['num_col_idx']) 
    cat_col_idx = deepcopy(info['cat_col_idx'])
    target_col_idx = deepcopy(info['target_col_idx'])

    task_type = info['task_type']
    # If regression, target is numerical. If classification, target is categorical (usually).
    # But wait, script_generation.py handles cat_indexes separately from y.
    # We should assume target_col_idx is already classified as num or cat in info construction?
    # No, reorder function logic in ef-vfm-dev-mixed handles this.
    
    # Check if target is already in num/cat idxs?
    # In ef-vfm-dev-mixed logic:
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[[str(i) for i in num_col_idx]]
    real_cat_data = real_data[[str(i) for i in cat_col_idx]]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = [str(i) for i in range(len(new_real_data.columns))]

    syn_num_data = syn_data[[str(i) for i in num_col_idx]]
    syn_cat_data = syn_data[[str(i) for i in cat_col_idx]]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = [str(i) for i in range(len(new_syn_data.columns))]

    
    metadata = deepcopy(info['metadata'])

    columns = metadata['columns'] # This is dict of col_idx -> info or col_name -> info?
    # In ForestDiffusion/script_generation.py we will construct it.
    # sdv metadata.to_dict()['columns'] is dict of col_name -> info.
    # But here we are using integer indices as column names for DataFrame.
    
    metadata['columns'] = {}

    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            # Map back to original column info
            original_idx = num_col_idx[i]
            # metadata['columns'][i] = columns[str(original_idx)] # Assuming columns keyed by str(idx)
            metadata['columns'][str(i)] = columns.get(str(original_idx)) or columns.get(original_idx)
        else:
            original_idx = cat_col_idx[i-len(num_col_idx)]
            metadata['columns'][str(i)] = columns.get(str(original_idx)) or columns.get(original_idx)
    
    return new_real_data, new_syn_data, metadata
