import os
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random
from torch_geometric.loader import NeighborLoader
from torch_geometric.loader import DataLoader
import re
import inspect
from sklearn.metrics import roc_curve, roc_auc_score
# Use a backend que não precisa de interface gráfica
import matplotlib.pyplot as plt
from collections import OrderedDict
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity

try:
    from models3_complete_cindex_new import GraphDataset_featsnorml_hetero, compute_mean_std_hetero, test_model_on_loader,verify_post_normalization, reset_weights, model_init, HeteroGAT_survival,GraphDataset_featsnorml,plot_attention_histograms,compute_mean_std,initialize_fold_results, initialize_repeat_results,store_fold_data, GAT_survival,prepare_fold_data,setup_data_loaders,get_class_distribution, train_accum_graddient_new_sigmoid_threshold,test_model_on_loader,verify_post_normalization,  GraphDataset_featsnorml_hyperinc, compute_mean_std_hyperinc, HyperGCN_survival, HyperGAT_survival, PatientWSIPackDataset_featsnorml, AgglomerativeGCN_Survival, AgglomerativeGAT_Survival,plot_roc_three_panels,  collect_activations_for_umap, plot_umap_per_layer_dual_and_metrics,save_layer_acts_for_fold, aggregate_umap_across_folds, build_renyi_patch_selector as _build_renyi_patch_selector_impl, summarize_patch_selector_usage, run_baseline_lda
except ImportError:
    from models3_gnn_jupiter_survival_orig_heteroconv_roc_umap_thr_final_final_lda import GraphDataset_featsnorml_hetero, compute_mean_std_hetero, test_model_on_loader,verify_post_normalization, reset_weights, model_init, HeteroGAT_survival,GraphDataset_featsnorml,plot_attention_histograms,compute_mean_std,initialize_fold_results, initialize_repeat_results,store_fold_data, GAT_survival,prepare_fold_data,setup_data_loaders,get_class_distribution, train_accum_graddient_new_sigmoid_threshold,test_model_on_loader,verify_post_normalization,  GraphDataset_featsnorml_hyperinc, compute_mean_std_hyperinc, HyperGCN_survival, HyperGAT_survival, PatientWSIPackDataset_featsnorml, AgglomerativeGCN_Survival, AgglomerativeGAT_Survival,plot_roc_three_panels,  collect_activations_for_umap, plot_umap_per_layer_dual_and_metrics,save_layer_acts_for_fold, aggregate_umap_across_folds, build_renyi_patch_selector as _build_renyi_patch_selector_impl, summarize_patch_selector_usage, run_baseline_lda

_RENYI_SELECTOR_PARAMS = inspect.signature(_build_renyi_patch_selector_impl).parameters
_RENYI_SELECTOR_SUPPORTS_PERCENTILE = "percentile" in _RENYI_SELECTOR_PARAMS
_RENYI_SELECTOR_SUPPORTS_KEEP_FRACTION = "keep_fraction" in _RENYI_SELECTOR_PARAMS


def build_renyi_patch_selector(*args, percentile=None, keep_fraction=None, **kwargs):
    """Compatibility wrapper to allow percentile-based selection regardless of backend signature."""

    if percentile is not None and not _RENYI_SELECTOR_SUPPORTS_PERCENTILE and keep_fraction is None:
        keep_fraction = percentile / 100.0
        print(
            "[Rényi] build_renyi_patch_selector lacks percentile support; "
            "falling back to keep_fraction≈percentile/100."
        )

    call_kwargs = dict(kwargs)

    if _RENYI_SELECTOR_SUPPORTS_PERCENTILE and percentile is not None:
        call_kwargs["percentile"] = percentile
    if _RENYI_SELECTOR_SUPPORTS_KEEP_FRACTION and keep_fraction is not None:
        call_kwargs["keep_fraction"] = keep_fraction

    return _build_renyi_patch_selector_impl(*args, **call_kwargs)


# Set random seed for reproducibility
seed = 47
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device selection
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('\nUsing device:', device)
auxxx = ['gcn']#,'survival''gat' ]#,'GraphSAGE_max']#, 'AdaptiveGraphSAGE','AdaptiveGraphSAGE_max'] #'survival','gat','gat','gat','GraphSAGE', 'AdaptiveGraphSAGE', 'GraphSAGE',
#'AdaptiveGraphSAGE'

for runn in range(0,6):
    ###### Variables ###################################
    # Define the number of folds
    
    num_folds = 5
    repeatt = 1
    batch_sizee = 2 ##########falta testar
    num_epochs = 100
    lr = 0.0001
    effective_batch = 64
    node_batch_size = 64 #128

    individual = 'False' # if true then wsi1-wsi2, etc... false wsi-others
    plot_weights='True'
    dataset= 'lung' #'cptac' "lung
    task = 'risk'  # '12months' (classification) or 'risk' (Cox PH survival)
                                                                                                #CPTAC           #LUNG
    all_t= 'combined_20_all_files' #combined_5_all_files                                     #yes   yes         #YES
                           #combined_20_all_files                                                  #yes   yes         #YES
                           #combined_10_all_files - combined 1 grpah per patien                    #yes  yes          #YES
                           #combined_100_all_files                                                 #yes 

                            # baseline_lda_all_files
    
                           #baseline_aglomerative_mean_all_files
                           #baseline_aglomerative_max_all_files

                            # baseline_aglomerative_mean_all_files_renyi75
    
                           #aglomerative_mean_all_files
                           #aglomerative_max_all_files
                           #all_files - with cross validation                                     #yes  yes          #YES

                           #hetero_combined_5_all_files                                          #yes yes           #YES
                           #hetero_combined_10_all_files                                          #yes yes           #YES
                           #hetero_combined_20_all_files                                          #yes  yes          #YES
                           #hetero_combined_100_all_files
                           #hetero_patients_tcga_all_files

                           #hyper_combined_5_all_files                                           #yes
                           #hyper_combined_10_all_files                                            #yes 
                           #hyper_combined_20_all_files                                           #yes
                           #hyper_combined_100_all_files                                           #yes
                           #hyper_patients_tcga_all_files
    
                           #patients_tcga_all_files #nao é partiçao tiago mas é so 1 fold tb 
                           #combined_patients_tcga_5_all_files
                           #combined_patients_tcga_10_all_files
                           #combined_patients_tcga_20_all_files
    
                           #'all_tiago' #all files but split tiago
                           #'patient_tiago' #split tiago with 1 file per patient
                           #combined_5
                           #combined_20
                           #combined_10 - combined 1 grpah per patient using knn


    def _infer_renyi_percentile(tag_value):
        tag_lower = str(tag_value).lower()
        for token, pct in (("renyi25", 25), ("renyi50", 50), ("renyi75", 75)):
            if token in tag_lower:
                return pct
        return 50

    renyi_enabled = 'renyi' in all_t.lower()
    renyi_target_percentile = _infer_renyi_percentile(all_t)
    umap_layers = False
    renyi_k = 4
    renyi_alpha = 0.85
    renyi_umap_df = None
    renyi_umap_path = None
    renyi_umap_df_by_cohort = {}

    RENYI_BASE_PATHS = {
        "cptac": "/home/ritav/patch_selection_csv/cptac_umap_all_patches.csv",
        "tcga": "/home/ritav/patch_selection_csv/tcga_umap_all_patches.csv",
    }

    def _resolve_umap_path(cohort, split=None, base_path=None):
        split = split.upper() if split else None
        cohort_upper = cohort.upper()
        candidates = []
        if split:
            candidates.extend([
                os.environ.get(f"RENYI_UMAP_CSV_{cohort_upper}_{split}"),
                os.environ.get(f"RENYI_UMAP_CSV_{split}"),
                f"csvs/{cohort}_umap_all_patches_{split.lower()}.csv",])
            if dataset.lower() == cohort:
                candidates.insert(0, os.environ.get(f"RENYI_UMAP_CSV_{dataset.upper()}_{split}"))
        candidates.append(base_path)
        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand
        return None

    def _infer_cohort_from_path(path, default):
        low = str(path).lower()
        if "tcga" in low:
            return "tcga"
        if "cptac" in low:
            return "cptac"
        return default

    def _ensure_cohort_column(df, cohort_value):
        if df is None or isinstance(df, list):
            return df
        df = df.copy()
        if "cohort" in df.columns:
            df["cohort"] = df["cohort"].fillna(cohort_value)
        elif "dataset" in df.columns:
            df["cohort"] = df["dataset"].fillna(cohort_value)
        else:
            df["cohort"] = cohort_value
        return df

    def _populate_cohort_column(df, default_cohort=None):
        """Ensure a per-row cohort label exists on the given DataFrame.

        Preference order for inference:
        1) Existing ``cohort`` column values (filled with ``default_cohort`` when missing).
        2) ``dataset`` column values (falling back to ``default_cohort`` when missing).
        3) Infer from filename-like columns using ``_infer_cohort_from_path``.
        4) Fallback to ``default_cohort`` if nothing else is available.
        """

        if df is None or isinstance(df, list) or not isinstance(df, pd.DataFrame):
            return df

        df = df.copy()
        target_default = (default_cohort or dataset).lower()

        if "cohort" in df.columns and df["cohort"].notna().all():
            return df

        def _row_cohort(row):
            if "cohort" in row and pd.notna(row["cohort"]):
                return str(row["cohort"]).lower()
            if "dataset" in row and pd.notna(row["dataset"]):
                return str(row["dataset"]).lower()

            case_id_val = None
            for case_col in ("case_id", "patient_id"):
                if case_col in row and pd.notna(row[case_col]):
                    case_id_val = str(row[case_col]).strip()
                    break

            if case_id_val:
                if case_id_val.lower().startswith("t"):
                    return "tcga"
                if case_id_val.lower().startswith("c"):
                    return "cptac"

            for col in ("image_filename", "filename", "slide_id", "slide_name"):
                if col in row and pd.notna(row[col]):
                    inferred = _infer_cohort_from_path(row[col], target_default)
                    if inferred:
                        return inferred
            return target_default

        df["cohort"] = df.apply(_row_cohort, axis=1)
        df["cohort"] = df["cohort"].fillna(target_default)
        return df

    def _load_cohort_umaps(cohort):
        env_candidates = [f"RENYI_UMAP_CSV_{cohort.upper()}"]
        if dataset.lower() == cohort:
            env_candidates.insert(0, "RENYI_UMAP_CSV")
            env_candidates.insert(1, f"RENYI_UMAP_CSV_{dataset.upper()}")

        base_path = None
        for cand in env_candidates:
            if cand and os.environ.get(cand):
                base_path = os.environ.get(cand)
                break
        base_path = base_path or RENYI_BASE_PATHS.get(cohort) or f"csvs/{cohort}_umap_all_patches.csv"

        resolved_base = _resolve_umap_path(cohort=cohort, base_path=base_path)
        if resolved_base is None or not os.path.exists(resolved_base):
            print(f"[Rényi] UMAP CSV for {cohort} not found at {resolved_base}. Skipping cohort.")
            return None

        cohort_map = {"shared": _ensure_cohort_column(pd.read_csv(resolved_base), cohort)}
        print(
            f"[Rényi] Loaded base UMAP coordinates for {cohort} from {resolved_base} with shape {cohort_map['shared'].shape}"
        )

        for split in ("train", "val", "test"):
            split_path = _resolve_umap_path(cohort=cohort, split=split, base_path=resolved_base)
            if split_path is None or split_path == resolved_base:
                continue
            if not os.path.exists(split_path):
                print(f"[Rényi] Split-specific UMAP for {cohort} {split} not found at {split_path}. Skipping.")
                continue
            split_df = _ensure_cohort_column(pd.read_csv(split_path), cohort)
            cohort_map[split] = split_df
            print(
                f"[Rényi] Loaded {cohort} {split} UMAP coordinates from {split_path} with shape {split_df.shape}"
            )

        if len(cohort_map) > 1:
            cohort_map["shared"] = pd.concat(cohort_map.values(), ignore_index=True)
            print(
                f"[Rényi] Concatenated {cohort} UMAPs → combined shape {cohort_map['shared'].shape}"
            )
        return cohort_map

    if renyi_enabled:
        for cohort in ("cptac", "tcga"):
            cohort_map = _load_cohort_umaps(cohort)
            if cohort_map is not None:
                renyi_umap_df_by_cohort[cohort] = cohort_map
        if not renyi_umap_df_by_cohort:
            raise FileNotFoundError("No UMAP CSVs for Rényi selection were found for any cohort")
        renyi_umap_df = pd.concat(
            [cohort_maps["shared"] for cohort_maps in renyi_umap_df_by_cohort.values()],
            ignore_index=True,
        )

                          
    agg_reduce = 'mean' if 'mean' in all_t else 'max'

    print('\nAll t:',all_t)
    print(f'########## ACTIVATED dataset {dataset} ###############')

    #num_folds = 2 # Define the number of folds
    early_stopping_rounds = 10 #umber of epochs to wait before early stopping
    lr_scheduler_patience = 5 # Number of epochs to wait before reducing learning rate
    lr_scheduler_factor = 0.5 # Factor by which to reduce learning rate

    # GNN parameters
    num_node_features = 1536
    hidden_channels=512
    hidden_2 =512
    hidden_3 = 512
    num_layers=3

    #runn=1
    headss= 1
    headss_2 = 1
    headss_3 = 1
    
    if runn ==1:
        # For Gat model:
        headss= 2
        headss_2 = 1
        headss_3 = 1
    elif runn ==0:
        # For Gat model:
        headss= 2
        headss_2 = 2
        headss_3 = 1
    elif runn ==4:
        # For Gat model:
        headss= 2
        headss_2 = 2
        headss_3 = 2

    calculate_threshold = 'True' #'False'True' # true if sigmoid want threshold not 0,5
    sampller = 'False'
    track='bacc'
    track_lr ='bacc'
    
    sigmoid = 'True' #false is softmax
    norm= 'True' #False
    print(f'\n\n Norm {norm} !!')
    # Variables to select
    survival =auxxx[runn]#'survival' #Model Selection: gin, gcn, gnn, gat, survivaL

    if survival == 'gat' or 'SAGE' in survival:
        batch_sizee=2
        if  headss_2 ==2: # or 'agglomerative' in all_t:
            batch_sizee=1  
        elif 'combined_100_all_files' == all_t and headss ==2:
            batch_sizee=1  
            
    
    if 'combi' in all_t:
        batch_sizee= 2
        if  headss_2 ==2:
            batch_sizee=1
            
        if '5' in all_t:
            virtual_percentage = 5
        elif '10_' in all_t:
            virtual_percentage = 10
        elif '20' in all_t:
            virtual_percentage = 20
        elif '100' in all_t:
            virtual_percentage = 100
            
            
    #elif all_t== 'all_files':
    #    batch_sizee= 2
    lrrr = 'counter'#'counter' #'reduceplateau' # 'counter' #counter if logic normal #'reduceplateau' if redeceonplateu'
    load = 'False' #True if you want to load previous model if training interrupted
    more_data = 'False'
    paralel = 'False'


    if sigmoid=='True':
        num_classes=1
    else:
        num_classes=2

    best_val_acc=[]
    best_score_matrix_v=[]
    best_cm_v=[]
    best_score_matrix_t=[]
    best_cm=[]
    best_bacc_majority_voting=[]
    best_bacc_one_dominance=[]
    best_bacc_major_voting=[]
    op_threshold_v = []

    actualtime = datetime.now().strftime("%d-%m-%Y---%H-%M-%S")

    # File paths
    if dataset == 'cptac':
            csv_paths = {
        
            'combined_10_all_files': {
                'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'                       
            },
            
            'combined_20_all_files': {
                'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'                       
            },
            
            'combined_5_all_files': {
                'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'                       
            },
    
            'combined_100_all_files': {
                'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'                       
            },
    
             'hetero_combined_5_all_files':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
            
             'hetero_combined_10_all_files':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
             
            'hetero_combined_20_all_files':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
            'hetero_combined_100_all_files':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
            
    
             'hyper_combined_5_all_files':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
            
             'hyper_combined_10_all_files':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
             
            'hyper_combined_20_all_files':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
            'hyper_combined_100_all_files':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
        
            'all_files': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },

            'aglomerative_mean_all_files': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'aglomerative_max_all_files': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_aglomerative_mean_all_files':{
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_aglomerative_max_all_files':{
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_lda_all_files': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_lda_all_files_renyi25': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_lda_all_files_renyi50': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_lda_all_files_renyi75': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'aglomerative_mean_all_files_renyi25': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'aglomerative_mean_all_files_renyi50': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },

            'aglomerative_mean_all_files_renyi75': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_aglomerative_mean_all_files_renyi25':{
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_aglomerative_mean_all_files_renyi50':{
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'baseline_aglomerative_mean_all_files_renyi75':{
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'all_files_renyi50': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'all_files_renyi75': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },
            'all_files_renyi25': {
                'all': 'csvs/all_filesencoded_three_updated_with_filenames_cptac_FULL.csv',
            },

            'combined_5_all_files_renyi25': {
                'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'                       
            },
            'combined_5_all_files_renyi50': {
                'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'                       
            },
            'combined_5_all_files_renyi75': {
                'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'                       
            },
             'hyper_combined_5_all_files_renyi25':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
             'hyper_combined_5_all_files_renyi50':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
             'hyper_combined_5_all_files_renyi75':{
                 'all': 'csvs/allpatients_combined_allpatients_data_knn.csv'   
            },
            
            # Partiçoes tcga train  e val cptac
            'patients_tcga_all_files': {
                'all': "csvs/train_tcga.csv",
                'test': "csvs/val_cptac.csv"
            },
            
            'combined_patients_tcga_5_all_files': {
                'all': "csvs/train_tcga_knn.csv",
                'test': "csvs/val_cptac_knn.csv"
            },
    
            'combined_patients_tcga_10_all_files': {
                'all': "csvs/train_tcga_knn.csv",
                'test': "csvs/val_cptac_knn.csv"
            },
    
            'combined_patients_tcga_20_all_files': {
                'all': "csvs/train_tcga_knn.csv",
                'test': "csvs/val_cptac_knn.csv"
            },
    
            ## partiçoes tiago
            'all_tiago': {
                'train': "csvs/encoded_train_allfiles.csv",
                'val': "/home/ritav/Graphs/cptac_233_adj_selfloop/encoded_test_allfiles.csv"
            },
            'baselinepatient_tiago': {
                'train': 'csvs/allpatients_mile_baseline_train.csv',
                'val': 'csvs/allpatients_mile_baseline_test.csv'
            },
            'patient_tiago': {
                'train': "csvs/filtered_combined_case_ids_train.csv",
                'val': "csvs/filtered_combined_case_ids_test.csv"
            },
            'combined_5': {
                'train': "csvs/allpatients_train_split_combinedgnn_complete-up.csv",
                'val': "csvs/allpatient_test_split_combinedgnn_complete-up.csv" 
            },
            
            'combined': {
                'train': "csvs/allpatients_train_split_combinedgnn-up.csv",
                'val': "csvs/allpatient_test_split_combinedgnn-up.csv"
            },
            'combined_10': {
                'train': "csvs/allpatients_train_split_combinedgnn_complete-up.csv",
                'val': "csvs/allpatient_test_split_combinedgnn_complete-up.csv" 
            }
        }
    elif dataset =='lung':
           csv_paths = {
            # csvs/updated_allpatients_combined_allpatients_data_knn_lung.csv
            'combined_10_all_files': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'                       
            },
            
            'combined_20_all_files': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'                       
            },
            
            'combined_5_all_files': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'                       
            },
    
            'combined_100_all_files': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'                       
            },
    
             'hetero_combined_5_all_files':{
                 'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'   
            },
            
             'hetero_combined_10_all_files':{
                 'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'   
            },
             
            'hetero_combined_20_all_files':{
                 'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'   
            },
            'hetero_combined_100_all_files':{
                 'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'   
            },

            'hyper_combined_5_all_files':{
                 'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'   
            },
            
             'hyper_combined_10_all_files':{
                 'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'   
            },
             
            'hyper_combined_20_all_files':{
                 'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'   
            },
            'hyper_combined_100_all_files':{
                 'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12_knn.csv'   
            },

            #csvs/updated_all_filesencoded_three_updated_with_filenames_lung_FULL.csv
            'all_files': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },
               
            'aglomerative_mean_all_files': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },
            'aglomerative_max_all_files': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },
            'baseline_aglomerative_mean_all_files':{
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },
            'baseline_aglomerative_max_all_files':{
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },      
            'baseline_lda_all_files': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },
            'baseline_lda_all_files_renyi25': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },
            'baseline_lda_all_files_renyi50': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },
            'baseline_lda_all_files_renyi75': {
                'all': 'csvs/up_lung_cptac_tcga_merged_cox_vital12.csv',
            },
                
         }
        
        
    def _normalize_all_t_key(key):
        if key in csv_paths:
            return key

        stripped = re.sub(r"_renyi\d+$", "", key)
        stripped = re.sub(r"(_all_files)+", "_all_files", stripped)

        if stripped in csv_paths:
            print(
                f"[csv_paths] Normalized all_t='{key}' to '{stripped}' for CSV lookup."
            )
            return stripped

        return None

    def _get_csv_path(key, split):
        resolved_key = _normalize_all_t_key(key)
        if resolved_key is None:
            available = ", ".join(sorted(csv_paths.keys()))
            raise KeyError(
                f"No CSV mapping found for all_t='{key}'. Available keys: {available}"
            )

        try:
            return csv_paths[resolved_key][split]
        except KeyError as exc:
            raise KeyError(
                f"Split '{split}' not configured for all_t='{resolved_key}'"
            ) from exc

    def _load_patient_metadata(path):
        default_cohort = _infer_cohort_from_path(path, dataset)
        return _ensure_cohort_column(pd.read_csv(path), default_cohort)

    # Load and prepare dataset based on `all_t` value
    if 'all_files' in all_t or 'baseline_lda' in all_t:
        df = _load_patient_metadata(_get_csv_path(all_t, 'all'))
        df_train = []
        df_val = []
        df_test =  []
        if 'tcga' in all_t:
            df_test = _load_patient_metadata(_get_csv_path(all_t, 'test'))
            print(df_test)

    else:
        df_train = _load_patient_metadata(_get_csv_path(all_t, 'train'))
        df_val = _load_patient_metadata(_get_csv_path(all_t, 'val'))
        df = pd.concat([df_train, df_val], ignore_index=True)
        #num_folds = 0

    if task == 'risk':
        if 'days_to_death' not in df.columns:
            raise ValueError("task='risk' requires column 'days_to_death' in the metadata CSV.")
        df['event'] = df['days_to_death'].notna().astype(int)
        t_censor = df['days_to_death'].max()
        if pd.isna(t_censor):
            raise ValueError("task='risk' could not compute censoring time: all days_to_death values are NaN.")
        df['time'] = df['days_to_death'].fillna(t_censor).astype(float)

    # Initialize StratifiedKFold and dataset statistics
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    #mean_features, std_dev = compute_mean_std(df_train, "/home/ritav/Graphs/cptac_233_adj_selfloop", device)


    # Initialize fold results and CSV paths
    fold_results = initialize_fold_results(num_folds=num_folds)


    if batch_sizee == effective_batch:
        grad_accum = 'False'
        print('\n###  :(  Deactivated grad_accum')
    else:
        grad_accum = 'True'
        print(effective_batch)
        print(batch_sizee)
        print('\n###  :) Activated grad_accum size: ', effective_batch // batch_sizee)


    # Group by 'case_id' and get the first filename for each group
    unique_patients = df.groupby('case_id')['image_filename'].first().reset_index()

    #print('\nunique patients',unique_patients)
    print('\ntotal unique_patients',len(unique_patients))
    # Get labels for each unique patient
    labels = df.groupby('case_id')['vital_status_12'].first() if task == '12months' else df.groupby('case_id')['event'].first()

    if 'all_files' not in all_t:
        num_folds = 0
        num_foldss= 1
        
    # Open a log file to write fold completion status
    # Save metrics to file
    save_path = f'logs_fold_{dataset}/logs_fold_{all_t}_univ2/{actualtime}_Model_{survival}_hidden{hidden_channels}_lr{lr}_batch{batch_sizee}_heads{headss}{headss_2}{headss_3}_calculate_threshold_{calculate_threshold}_individual{individual}'
    os.makedirs(save_path, exist_ok=True)
    #with open(f"{save_path}/metrics_per_fold_and_repeats.txt", "w") as file:
    #print(df_train)
    for fold, (train_index, val_index) in enumerate(skf.split(unique_patients['case_id'], labels)):
        with open(f"{save_path}/metrics_per_fold_{fold}_and_repeats.txt", "w") as file:
            '''if (fold == 1 or fold ==0) and survival == 'gcn':
                print(f'\n\n##### Fold {fold} already done. Skipping...')
                continue  # Skip to the next fold'''
                
            # Repeat the training and validation process 5 times for the current fold
            generator = torch.Generator().manual_seed(47)
            for repeat in range(repeatt):
                print(f"\nStarting Fold {fold + 1}, Repeat {repeat + 1}...")

                train_df, val_df, test_df,num_folds,train_patients,val_patients, test_patients = prepare_fold_data(df_train,df_val,df, df_test,all_t,unique_patients, task,train_index, val_index)

                default_cohort = dataset.lower()
                train_df = _populate_cohort_column(train_df, default_cohort)
                val_df = _populate_cohort_column(val_df, default_cohort)
                test_df = _populate_cohort_column(test_df, default_cohort)

                patch_selector = None
                if renyi_enabled and isinstance(train_df, pd.DataFrame):
                    pid_col = "case_id" if "case_id" in train_df.columns else "patient_id"

                    def _get_patient_ids(frame, cohort_value):
                        if not isinstance(frame, pd.DataFrame) or getattr(frame, 'empty', True):
                            return []
                        df_local = frame
                        if "cohort" in df_local.columns:
                            df_local = df_local[df_local["cohort"] == cohort_value]
                        return df_local[pid_col].astype(str).unique()

                    def _accumulate_stats(target, stats):
                        target["total"] += stats.get("total_patches", 0)
                        target["selected"] += stats.get("selected_patches", 0)

                    patch_selector = {}
                    agg_stats = {split: {"total": 0, "selected": 0} for split in ("train", "val", "test")}
                    used_thresholds = {}

                    cohort_groups = (
                        train_df.groupby("cohort") if "cohort" in train_df.columns else [(dataset.lower(), train_df)]
                    )

                    selector_supports_percentile = "percentile" in inspect.signature(
                        build_renyi_patch_selector
                    ).parameters

                    if not selector_supports_percentile:
                        print(
                            "[Rényi] build_renyi_patch_selector lacks percentile support; "
                            "falling back to keep_fraction≈percentile/100."
                        )

                    for cohort_name, cohort_train_df in cohort_groups:
                        cohort_key = str(cohort_name).lower() if cohort_name is not None else dataset.lower()
                        umap_sources = renyi_umap_df_by_cohort.get(cohort_key)
                        if umap_sources is None:
                            print(f"[Rényi] No UMAP data for cohort {cohort_key}. Skipping patch selection for this cohort.")
                            continue

                        train_ids = cohort_train_df[pid_col].astype(str).unique()
                        apply_ids = list(train_ids)
                        apply_ids.extend(_get_patient_ids(val_df, cohort_name))
                        apply_ids.extend(_get_patient_ids(test_df, cohort_name))
                        apply_ids = list(pd.unique(pd.Series(apply_ids).dropna()))

                        if len(train_ids) == 0 or len(apply_ids) == 0:
                            continue

                        percentile_used = renyi_target_percentile

                        train_umap_source = umap_sources.get("train", umap_sources["shared"])
                        apply_umap_source = umap_sources.get("shared")
                        cohort_selector, final_threshold, final_fraction = build_renyi_patch_selector(
                            train_umap_source,
                            train_patient_ids=train_ids,
                            apply_patient_ids=apply_ids,
                            keep_fraction=None,
                            threshold=None,
                            percentile=percentile_used,
                            k=renyi_k,
                            alpha=renyi_alpha,
                            apply_umap_df=apply_umap_source,
                        )
                        patch_selector.update(cohort_selector)
                        used_thresholds[cohort_key] = (final_threshold, percentile_used, final_fraction)

                        train_sel_stats = summarize_patch_selector_usage(apply_umap_source, cohort_selector, train_ids)
                        val_sel_stats = summarize_patch_selector_usage(apply_umap_source, cohort_selector, _get_patient_ids(val_df, cohort_name))
                        test_sel_stats = summarize_patch_selector_usage(
                            apply_umap_source,
                            cohort_selector,
                            _get_patient_ids(test_df, cohort_name),
                        )

                        _accumulate_stats(agg_stats["train"], train_sel_stats)
                        _accumulate_stats(agg_stats["val"], val_sel_stats)
                        _accumulate_stats(agg_stats["test"], test_sel_stats)

                    if patch_selector:
                        thr_desc = []
                        for cohort_key, (thr, pct, frac) in used_thresholds.items():
                            pct_label = f"p{pct}" if pct is not None else "unknown"
                            if thr is None:
                                thr_desc.append(f"{cohort_key}: computed via {pct_label} (kept≈{frac:.2f})")
                            else:
                                thr_desc.append(f"{cohort_key}: {thr:.4f} ({pct_label}, kept≈{frac:.2f})")
                        thr_desc_txt = "; ".join(thr_desc) if thr_desc else "None"
                        print(
                            f"[Rényi] Selector built for {len(patch_selector)} WSIs (thresholds={thr_desc_txt})."
                        )

                        def _log_stats(name, stats):
                            total = stats["total"]
                            selected = stats["selected"]
                            frac = selected / total if total else float("nan")
                            print(
                                f"[Rényi] {name} patches: kept {selected} / {total} ({frac:.3f})."
                            )

                        _log_stats("Train", agg_stats["train"])
                        _log_stats("Val  ", agg_stats["val"])
                        _log_stats("Test ", agg_stats["test"])
                    else:
                        patch_selector = None
                else:
                    patch_selector = None
                    if not renyi_enabled:
                        print("[Rényi] Skipping patch selection because 'renyi' was not requested in all_t.")

                if 'hetero' in all_t:
                    if individual == 'True':
                        if 'combined' in all_t:
                            mean_features, std_dev= compute_mean_std_hetero(train_df, f"{dataset}_univ2_combinedknn_hetero_selfloop_names_individualTrue", device)
                        else:
                            print('\n ########### YOU SHOULD SELECT INDIVIDUAL = fALSE THERE IS NO TRUE AND NOT COMBINED !!!!!!!!!!!!   ##############')
                            mean_features, std_dev= compute_mean_std(train_df, f"{dataset}_univ2_patchgraphadj_selfloop", device) 
                            virtual_percentage = 0
                    else:
                        if 'combined' in all_t:
                            mean_features, std_dev= compute_mean_std_hetero(train_df, f"{dataset}_univ2_combinedknn_hetero_selfloop_names", device)
                        else:
                            print('\n ########### YOU SHOULD SELECT INDIVIDUAL = fALSE THERE IS NO TRUE AND NOT COMBINED !!!!!!!!!!!!   ##############')
                            mean_features, std_dev= compute_mean_std(train_df, f"{dataset}_univ2_patchgraphadj_selfloop", device) 
                
                elif 'hyper' in all_t:
                    if individual == 'True':
                        if 'combined' in all_t:
                            mean_features, std_dev= compute_mean_std_hyperinc(train_df, f"{dataset}_univ2_combinedknn_hypergraph_gpu_selfloop_names_individualTrue", device)
                        else:
                            print('\n ########### YOU SHOULD SELECT INDIVIDUAL = fALSE THERE IS NO TRUE AND NOT COMBINED !!!!!!!!!!!!   ##############')
                            mean_features, std_dev= compute_mean_std(train_df, f"{dataset}_univ2_patchgraphadj_selfloop", device) 
                            virtual_percentage = 0
                    else:
                        if 'combined' in all_t:
                            mean_features, std_dev= compute_mean_std_hyperinc(train_df, f"{dataset}_univ2_combinedknn_hypergraph_gpu_selfloop_names", device)
                        else:
                            print('\n ########### YOU SHOULD SELECT INDIVIDUAL = fALSE THERE IS NO TRUE AND NOT COMBINED !!!!!!!!!!!!   ##############')
                            mean_features, std_dev= compute_mean_std(train_df, f"{dataset}_univ2_patchgraphadj_selfloop", device) 
                    
                #homo            
                else:
                    if individual == 'True':
                        if 'combined' in all_t:
                            mean_features, std_dev= compute_mean_std(train_df, f"{dataset}_univ2_combinedknn_homo_selfloop_names_individualTrue", device)
                        else:
                            print('\n ########### YOU SHOULD SELECT INDIVIDUAL = fALSE THERE IS NO TRUE AND NOT COMBINED !!!!!!!!!!!!   ##############')
                            mean_features, std_dev= compute_mean_std(train_df, f"{dataset}_univ2_patchgraphadj_selfloop", device) 
                            virtual_percentage = 0
                    else:
                        if 'combined' in all_t:
                            mean_features, std_dev= compute_mean_std(train_df, f"{dataset}_univ2_combinedknn_selfloop_names", device)
                        else:
                            mean_features, std_dev= compute_mean_std(train_df, f"{dataset}_univ2_patchgraphadj_selfloop", device) 
                            virtual_percentage = 0

                if norm=='True':
                    # Data loaders
                    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset,num_neighbors,node_batch_size, = setup_data_loaders(all_t,train_df, val_df, test_df, mean_features, std_dev, batch_sizee, device,sampller,survival,node_batch_size,virtual_percentage,individual,dataset, patch_selector=patch_selector, task=task)
                    print(train_loader)
                    print(val_loader)
                    print(test_loader)

                    '''print('\ntrain')
                    verify_post_normalization(train_dataset)
                    print('val')
                    verify_post_normalization(val_dataset)
                    print('test')
                    verify_post_normalization(test_dataset)
                    
                    train_means, train_stds, train_x = collect_feature_stats(train_loader)
                    val_means, val_stds, val_x = collect_feature_stats(val_loader)
                    test_means, test_stds, test_x = collect_feature_stats(test_loader)
                    
                    print("\n\nTrain mean (per feature):", train_means.mean().item(), "±", train_means.std().item())
                    print("Val mean  (per feature):", val_means.mean().item(), "±", val_means.std().item())
                    print("Test mean  (per feature):", test_means.mean().item(), "±", test_means.std().item())
                    
                    print("\nTrain std (per feature):", train_stds.mean().item(), "±", train_stds.std().item())
                    print("Val std (per feature):", val_stds.mean().item(), "±", val_stds.std().item())
                    print("Test std  (per feature):", test_stds.mean().item(), "±", test_stds.std().item())'''

                
                elif norm =='False':
                    # Data loaders
                    train_loader, val_loader, train_dataset, val_dataset,num_neighbors,node_batch_size = setup_data_loaders_nonorm(all_t,train_df, val_df, mean_features, std_dev,
                                                                                         batch_sizee, device,sampller,node_batch_size, patch_selector=patch_selector )


                ##################### Prints ###############################################################################################
                # Calculate the number of 0s and 1s per patient in the training set
                label_col = 'vital_status_12' if task == '12months' else 'event'
                train_0s_per_patient = train_df[train_df[label_col] == 0].groupby('case_id').size()
                train_1s_per_patient = train_df[train_df[label_col] == 1].groupby('case_id').size()
                
                print('\ntrain patients 0s',train_0s_per_patient)
                   
                # Calculate the number of 0s and 1s per patient in the validation set
                val_0s_per_patient = val_df[val_df[label_col] == 0].groupby('case_id').size()
                val_1s_per_patient = val_df[val_df[label_col] == 1].groupby('case_id').size()
                print('val patients 0s',val_0s_per_patient)
                
                print(f"\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Fold {fold+1} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>:")
                #print(f"Total training sets {train_dataset} with {train_patients} patients")
                #print(f"Total validation set: {val_dataset} with {val_patients} patients")
                print("\nTraining set:")
                print("Number of 0s per patient:", len(train_0s_per_patient))
                print("Number of 1s per patient:", len(train_1s_per_patient))
                print("\nValidation set:")
                print("Number of 0s per patient:", len(val_0s_per_patient))
                print("Number of 1s per patient:", len(val_1s_per_patient))

                if 'baseline_lda' in all_t:
                    fold_result = run_baseline_lda(
                        train_loader,
                        val_loader,
                        test_loader,
                        all_t=all_t,
                        calculate_threshold=calculate_threshold,
                    )

                    y_true_val = np.array(fold_result[18])
                    val_probs = np.array(fold_result[19])
                    val_auc = roc_auc_score(y_true_val, val_probs) if len(np.unique(y_true_val)) > 1 else 0.0

                    log_path = os.path.join(
                        save_path,
                        f"baseline_lda_fold{fold}_repeat{repeat}.txt",
                    )
                    with open(log_path, "w") as lda_log:
                        lda_log.write(f"Baseline LDA fold {fold} repeat {repeat}\n")
                        lda_log.write(f"Balanced Accuracy: {fold_result[0]:.4f}\n")
                        lda_log.write(f"Recall: {fold_result[11]:.4f}\n")
                        lda_log.write(f"Specificity: {fold_result[12]:.4f}\n")
                        lda_log.write(f"AUC: {val_auc:.4f}\n")
                        lda_log.write(f"Threshold: {fold_result[17]:.4f}\n")
                        lda_log.write(f"Confusion Matrix:\n{fold_result[2]}\n")
                        lda_log.write(f"Classification Report:\n{fold_result[1]}\n")

                    print(f"\n[Baseline LDA] Fold {fold + 1}, Repeat {repeat + 1}")
                    print(f"Val Balanced Accuracy: {fold_result[0]:.4f}")
                    print(f"Val Recall: {fold_result[11]:.4f}")
                    print(f"Val Specificity: {fold_result[12]:.4f}")
                    print(f"Val AUC: {val_auc:.4f}")
                    print(f"Optimal Threshold: {fold_result[17]:.4f}\n")

                    continue
                
                # to create the training and validation datasets for each fold.
                print('\n\n Train Data per batch :( :\n ')
                auxs=0
                for batch in train_loader:
                    if auxs <= 8:
                      print(batch)
                      auxs = auxs + 1
                print('\n')
                ############################################################################################################################
                    
                train_class_dist = get_class_distribution(train_df, label_col=label_col)
                val_class_dist = get_class_distribution(val_df, label_col=label_col)
                print(f"Fold {fold + 1}: Train 0s per patient: {train_class_dist[0]}, Train 1s per patient: {train_class_dist[1]}")
                print(f"Val 0s per patient: {val_class_dist[0]}, Val 1s per patient: {val_class_dist[1]}")

                # Store dataset sizes, patient IDs, and class distributions for the current fold
                #store_fold_data(fold_results, train_dataset, val_dataset, train_df, val_df)
                if 'SAGE' in survival:
                    best_model_path_init= f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage/init'
                elif 'hetero' in all_t:
                    best_model_path_init= f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hetero/init'
                elif 'hyper' in all_t:
                    best_model_path_init= f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hyper/init'
                elif 'aglomerative' in all_t:
                    if 'baseline' in all_t:
                        best_model_path_init= f'Checkpoints-univ2/Checkpoint_{dataset}_adj_aglomerative_baseline/init'
                    else:
                        best_model_path_init= f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_aglomerative/init'
                else:
                    best_model_path_init= f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}/init'

                
                model, best_model_path, best_model_path_init = model_init(
                    survival, num_node_features, num_epochs, hidden_channels, hidden_2, hidden_3, lr, batch_sizee, more_data, num_classes,
                    headss, headss_2, headss_3, paralel, device, fold, all_t,
                    best_model_path_init,sigmoid,repeat,individual,calculate_threshold,virtual_percentage,dataset
                )
                model.to(device)

                

                writer = SummaryWriter( f"check-tensorboarduniv2/logs_fold_{all_t}_{dataset}/{actualtime}_Model_{survival}_epoch{num_epochs}_hidden{hidden_channels}_lr{lr}_fold{fold}_batch{batch_sizee}_heads{headss}{headss_2}{headss_3}_repeat_{repeat}")

                #optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                optimizer = torch.optim.AdamW(
                                                filter(lambda p: p.requires_grad, model.parameters()),
                                                lr=lr,
                                                weight_decay=5e-4,
                                                betas=(0.85, 0.999)

                                            )


                #optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

                # Train and validate for the current fold
                total_samples = train_dataset.n_pos + train_dataset.n_neg
                print('Died samples', train_dataset.n_neg)
                print('Survived samples', train_dataset.n_pos)

                weight_pos = total_samples / (2 * train_dataset.n_pos) 
                weight_neg = total_samples / (2 * train_dataset.n_neg)

                
                class_weights = torch.tensor([weight_neg, weight_pos]).to(device)
                print(class_weights)

                weight_tensor = class_weights
                '''
                criterion = torch.nn.BCEWithLogitsLoss(reduction='sum',pos_weight=class_weights[0])'''

                total_samples_calc = train_dataset.n_pos + train_dataset.n_neg # 1972.0
                weight_pos = torch.tensor(weight_pos).to(device)
                criterion = torch.nn.BCEWithLogitsLoss(reduction='sum',pos_weight=weight_pos) if task == '12months' else None

                fold_result = train_accum_graddient_new_sigmoid_threshold(
                train_loader, val_loader, model, criterion, optimizer, device, early_stopping_rounds,
                lr_scheduler_patience, lr_scheduler_factor, writer, best_model_path, weight_tensor,num_epochs, hidden_channels,lr,num_node_features,class_weights, best_val_acc,best_score_matrix_v,best_cm_v,best_score_matrix_t, best_cm,fold,
                lrrr, effective_batch // batch_sizee, load, survival, all_t, actualtime, effective_batch,calculate_threshold,track,track_lr ,repeat,num_neighbors,node_batch_size,batch_sizee,dataset, task=task
            )


                ###########################
                # Append fold results
                print(f"\n----------------- FOLD {fold + 1} repeat {repeat+1}: Evaluation of the best model for Fold {fold + 1} c deterministic dropout norm{norm} weight decay effectivebatch {effective_batch}  batch{batch_sizee} adam sampler e sigmoid thresholdl {calculate_threshold} sampller{sampller} track{track} {lrrr} -----------------")
                print(f"Total training sets {len(train_dataset)} with {len(train_df['case_id'].unique())} patients, 0s:{len(train_class_dist[0])}, 1s:{len(train_class_dist[1])}")
                print(f"Total validation set: {len(val_dataset)} with {len(val_df['case_id'].unique())} patients, 0s:{len(val_class_dist[0])}, 1s:{len(val_class_dist[1])}. \nOptimal Threshold val: {fold_result[17]}")
                print("Best Confusion Matrix (Training):\n", fold_result[4])
                print("\nBest Score Matrix (Training):\n", fold_result[3])
                print(f"\nBest Validation Balanced Accuracy: {fold_result[0]}")
                print("Recall:", fold_result[11])
                print("SP:", fold_result[12])
                print("Best Confusion Matrix (Validation):\n", fold_result[2])
                print("Best Score Matrix (Validation):\n", fold_result[1])
                print("\n #########  Major Voting Strategy #############")
                print("Balanced Accuracy for Majority Voting:", fold_result[5])
                print("Recall for Majority Voting:", fold_result[13])
                print("SP for Majority Voting:", fold_result[14])
                print('Classification Report:\n', fold_result[6])
                print("\n #########  1-Dominance #############")
                print("Balanced Accuracy for 1-Dominance:", fold_result[8])
                print("Classification Report - 1-Dominance Strategy:\n", fold_result[9])
                print("Recall for Majority Voting:", fold_result[15])
                print("SP for Majority Voting:", fold_result[16])
                print("y_true_wsi_level",fold_result[18])
                print("probs_wsi_level",fold_result[19])

                optimal_threshold = fold_result[17]
                op_threshold_v.append(optimal_threshold)
                
                y_true_val_wsi = fold_result[18]
                probs_val_wsi =fold_result[19]
                
                # put model in eval mode once before collecting activations
                model.eval()
                if umap_layers:
                    # --------------------------
                    # UMAP on TRAIN
                    # --------------------------
                    umap_dir_train = os.path.join(save_path, f"umap_fold{fold}_repeat{repeat}_train")

                    # Collect activations for training data
                    layer_acts_train, layer_domains_train = collect_activations_for_umap(
                        model=model, loader=train_loader, device=device, max_nodes_per_layer=50000
                    )

                    # Plot UMAPs and compute metrics for training
                    plot_umap_per_layer_dual_and_metrics(
                        layer_acts=layer_acts_train,
                        layer_acts_domain=layer_domains_train,
                        save_dir=umap_dir_train,
                        prefix="train",
                        umap_n_neighbors=15,
                        umap_min_dist=0.5,
                        umap_metric="cosine",
                        renyi_alpha=1.5,
                        renyi_k=5,
                        renyi_repeats=30
                    )

                    # Persist per-fold arrays (for later aggregation)
                    save_layer_acts_for_fold(layer_acts_train, layer_domains_train, umap_dir_train, fold=fold, split_tag="train")

                    # --------------------------
                    # UMAP on VAL
                    # --------------------------
                    umap_dir_val = os.path.join(save_path, f"umap_fold{fold}_repeat{repeat}")

                    # Collect activations for validation data
                    layer_acts_val, layer_domains_val = collect_activations_for_umap(
                        model=model, loader=val_loader, device=device, max_nodes_per_layer=50000
                    )

                    # Plot UMAPs and compute metrics for validation
                    plot_umap_per_layer_dual_and_metrics(
                        layer_acts=layer_acts_val,
                        layer_acts_domain=layer_domains_val,
                        save_dir=umap_dir_val,
                        prefix="val",
                        umap_n_neighbors=15,
                        umap_min_dist=0.5,
                        umap_metric="cosine",
                        renyi_alpha=1.5,
                        renyi_k=5,
                        renyi_repeats=30
                    )

                    # Persist per-fold arrays (for later aggregation)
                    save_layer_acts_for_fold(layer_acts_val, layer_domains_val, umap_dir_val, fold=fold, split_tag="val")

                
                '''
                # ==========================
                # UMAP on TRAIN
                # ==========================
                umap_dir_train = os.path.join(
                    save_path,
                    f"umap_fold{fold}_repeat{repeat}_train"
                )
                
                layer_acts_train = collect_activations_for_umap(
                    model=model,
                    loader=train_loader,
                    device=device,
                    max_nodes_per_layer=50000
                )
                
                plot_umap_per_layer(
                    layer_acts=layer_acts_train,
                    save_dir=umap_dir_train,
                    prefix=f"fold{fold}_repeat{repeat}_train"
                )

                
                # ==========================
                # UMAP on VAL
                # ==========================
                umap_dir = os.path.join(
                    save_path,
                    f"umap_fold{fold}_repeat{repeat}"
                )
                
                model.eval()
                layer_acts_val = collect_activations_for_umap(
                    model=model,
                    loader=val_loader,
                    device=device,
                    max_nodes_per_layer=50000
                )
                
                plot_umap_per_layer(
                    layer_acts=layer_acts_val,
                    save_dir=umap_dir,
                    prefix=f"fold{fold}_repeat{repeat}_val"
                )'''

                
                if all_t == "all_files":
                    y_true_patient_level = fold_result[20]
                    patient_mv_scores =  fold_result[21]
                    patient_1d_scores =  fold_result[22]
                    
                    np.save(os.path.join(save_path, f"y_true_wsi_all_files_fold{fold}.npy"), y_true_val_wsi)
                    np.save(os.path.join(save_path, f"probs_wsi_all_files_fold{fold}.npy"), probs_val_wsi)
                
                    np.save(os.path.join(save_path, f"y_true_patient_all_files_fold{fold}.npy"), y_true_patient_level)
                    np.save(os.path.join(save_path, f"scores_mv_patient_all_files_fold{fold}.npy"), patient_mv_scores)
                    np.save(os.path.join(save_path, f"scores_1d_patient_all_files_fold{fold}.npy"), patient_1d_scores)

                np.save(os.path.join(save_path, f"y_true_fold{fold}.npy"), y_true_val_wsi)
                np.save(os.path.join(save_path, f"probs_fold{fold}.npy"), probs_val_wsi)


                ################# Attention Weights #####################
                if plot_weights == 'True' and survival == 'gat':
                    if 'hyper' in all_t.lower():
                        print('[weights] Skipping: HypergraphConv does not return attention weights in this build.')
                    else:
                        interperct = f'inter_{virtual_percentage}'
                        # Instantiate the model
                        if 'hetero' in all_t:
                            print('\n\n Selected the HETERO GAT model')
                            model = HeteroGAT_survival(
                                num_node_features, hidden_channels, hidden_2, hidden_3,
                                num_classes,interperct, headss, headss_2, headss_3, return_attn=True).to(device)
                       
                        elif 'aglomerative' in all_t and 'baseline' not in all_t:
                            model = AgglomerativeGAT_Survival(in_ch=num_node_features,h1=hidden_channels, h2=hidden_2, h3=hidden_3,out_ch=num_classes,heads1=headss, heads2=headss_2, heads3=headss_3,reduce=agg_reduce,dropout=0.3,return_attn=True).to(device)
                        else:
                            print('\n\n Selected the GAT model')
                            model = GAT_survival(
                                num_node_features, hidden_channels, hidden_2, hidden_3,
                                num_classes, headss, headss_2, headss_3, return_attn=True
                            ).to(device)
                    
                        print('\n', model)
                    
                        # Load the trained model weights
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                        model.eval()
                        valll_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False
    )
                        # Get one batch from validation loader
                        data = next(iter(valll_loader)).to(device)
                    
                        # Run inference and extract attention
                        with torch.no_grad():
                            if 'hetero' in all_t:
                                _, attentions = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                            #elif 'hyper' in all_t:
                               # _, attentions = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                            else:
                                _, attentions = model(data.x, data.edge_index, data.batch)
                    
                        # Plot histogram of attention weights
                        plot_attention_histograms(attentions,
    save_dir=f'{save_path}/attn_plots_{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3_{hidden_3}_lr{lr}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{calculate_threshold}_individual{individual}',
                            prefix='val_'
                        )
                        
                    #plot_attention_weights(attentions, save_dir='{save_path}/attn_plots_{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3_{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}', prefix='val_')
                    
                
                if 'tcga' in all_t:# Load best model
                    model.load_state_dict(torch.load(best_model_path))
                    model.to(device)
                    
                    # Run test
                    if calculate_threshold == 'False':
                        testtt= 0.5
                    elif calculate_threshold == 'True':
                        testtt=fold_result[17]
                    print('\n Threshold test')
                    results = test_model_on_loader(test_loader, model, device, threshold=testtt, criterion=criterion)

                    print("\n--- TEST RESULTS ---")
                    print("Score Matrix (Test):\n", results[1])
                    print(f"Balanced Accuracy: {results[0]}")
                    print("Recall:", results[4])
                    print("Specificity:", results[5])
                    print("Confusion Matrix:\n", results[2])
                    print("AUC:", results[3])
                    
                    print("\n #########  Majority Voting #############")
                    print("Balanced Accuracy:", results[6])
                    print("Recall:", results[7])
                    print("Specificity:", results[8])
                    print("Classification Report:\n", results[9])
                    print("Confusion Matrix:\n", results[10])
                    
                    print("\n #########  1-Dominance #############")
                    print("Balanced Accuracy:", results[11])
                    print("Recall:", results[12])
                    print("Specificity:", results[13])
                    print("Classification Report:\n", results[14])
                    print("Confusion Matrix:\n", results[15])

                     # Save test results to file
                    with open(f"{save_path}/test_results_fold{fold}_repeat{repeat}.txt", "w") as test_file:
                        test_file.write("--- TEST RESULTS ---\n")
                        test_file.write(f'val threshold:{fold_result[17]} \n')
                        test_file.write(f"Score Matrix (Test):\n{results[1]}\n")
                        test_file.write(f"Balanced Accuracy: {results[0]}\n")
                        test_file.write(f"Recall: {results[4]}\n")
                        test_file.write(f"Specificity: {results[5]}\n")
                        test_file.write(f"Confusion Matrix:\n{results[2]}\n")
                        test_file.write(f"AUC: {results[3]}\n\n")
                        test_file.write("######### Majority Voting #########\n")
                        test_file.write(f"Balanced Accuracy: {results[6]}\n")
                        test_file.write(f"Recall: {results[7]}\n")
                        test_file.write(f"Specificity: {results[8]}\n")
                        test_file.write(f"Classification Report:\n{results[9]}\n")
                        test_file.write(f"Confusion Matrix:\n{results[10]}\n\n")
                        test_file.write("######### 1-Dominance #########\n")
                        test_file.write(f"Balanced Accuracy: {results[11]}\n")
                        test_file.write(f"Recall: {results[12]}\n")
                        test_file.write(f"Specificity: {results[13]}\n")
                        test_file.write(f"Classification Report:\n{results[14]}\n")
                        test_file.write(f"Confusion Matrix:\n{results[15]}\n")
     
                  

                # Append results for the current repeat and fold
                repeat_results['best_val_acc'][fold].append(fold_result[0].cpu().item() if torch.is_tensor(fold_result[0]) else fold_result[0])

                # Append score matrices
                repeat_results['best_score_matrix_v'][fold].append(
                    fold_result[1].cpu().numpy() if isinstance(fold_result[1], torch.Tensor) else fold_result[1]
                )
                repeat_results['best_score_matrix_t'][fold].append(
                    fold_result[3].cpu().numpy() if isinstance(fold_result[3], torch.Tensor) else fold_result[3]
                )

                # Append confusion matrices
                repeat_results['best_cm'][fold].append(
                    fold_result[4].cpu().numpy() if isinstance(fold_result[4], torch.Tensor) else fold_result[4]
                )
                repeat_results['best_cm_v'][fold].append(
                    fold_result[2].cpu().numpy() if isinstance(fold_result[2], torch.Tensor) else fold_result[2]
                )

                # Append balanced accuracies
                repeat_results['best_bacc_major_voting'][fold].append(fold_result[5].cpu().item() if torch.is_tensor(fold_result[5]) else fold_result[5])
                repeat_results['best_bacc_one_dominance'][fold].append(fold_result[8].cpu().item() if torch.is_tensor(fold_result[8]) else fold_result[8])

                # Append confusion matrices for major voting and 1-dominance strategies
                repeat_results['best_conf_matrix_major_voting'][fold].append(
                    fold_result[7].cpu().numpy() if isinstance(fold_result[7], torch.Tensor) else fold_result[7]
                )
                repeat_results['best_conf_matrix_one_dominance'][fold].append(
                    fold_result[10].cpu().numpy() if isinstance(fold_result[10], torch.Tensor) else fold_result[10]
                )
                
                # Append confusion matrices for major voting and 1-dominance strategies
                repeat_results['best_classification_reports_major'][fold].append(
                    fold_result[6].cpu().numpy() if isinstance(fold_result[6], torch.Tensor) else fold_result[6]
                )
                repeat_results['best_classification_reports_one_dominance'][fold].append(
                    fold_result[9].cpu().numpy() if isinstance(fold_result[9], torch.Tensor) else fold_result[9]
                )

            
                repeat_results['best_recall'][fold].append(fold_result[11].cpu().item() if torch.is_tensor(fold_result[11]) else fold_result[11])
                repeat_results['best_specificity'][fold].append(fold_result[12].cpu().item() if torch.is_tensor(fold_result[12]) else fold_result[12])

                repeat_results['best_recall_mj'][fold].append(fold_result[13].cpu().item() if torch.is_tensor(fold_result[13]) else fold_result[13])
                repeat_results['best_specificity_mj'][fold].append(fold_result[14].cpu().item() if torch.is_tensor(fold_result[14]) else fold_result[14])

                repeat_results['best_recall_1d'][fold].append(fold_result[15].cpu().item() if torch.is_tensor(fold_result[15]) else fold_result[15])
                repeat_results['best_specificity_1d'][fold].append(fold_result[16].cpu().item() if torch.is_tensor(fold_result[16]) else fold_result[16])               
                

            # Ensure all fold-level metrics are computed on CPU
            if repeat == repeatt - 1:
                valid_best_cm = [cm.cpu().numpy() if isinstance(cm, torch.Tensor) else cm for cm in repeat_results['best_cm'][fold] if cm is not None]
                fold_results['mean_best_cm'][fold] = np.mean(np.stack(valid_best_cm), axis=0) if valid_best_cm else np.zeros((2, 2))

                valid_best_cm_v = [cm.cpu().numpy() if isinstance(cm, torch.Tensor) else cm for cm in repeat_results['best_cm_v'][fold] if cm is not None]
                fold_results['mean_best_cm_v'][fold] = np.mean(np.stack(valid_best_cm_v), axis=0) if valid_best_cm_v else np.zeros((2, 2))

                valid_best_val_acc = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_val_acc'][fold] if val is not None
                ]
                fold_results['mean_best_val_acc'][fold] = np.mean(valid_best_val_acc) if valid_best_val_acc else 0.0
                fold_results['std_best_val_acc'][fold] = np.std(valid_best_val_acc) if valid_best_val_acc else 0.0

                valid_bacc_major_voting = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_bacc_major_voting'][fold] if val is not None
                ]
                fold_results['mean_best_bacc_major_voting'][fold] = np.mean(valid_bacc_major_voting) if valid_bacc_major_voting else 0.0
                fold_results['std_best_bacc_major_voting'][fold] = np.std(valid_bacc_major_voting) if valid_bacc_major_voting else 0.0

                valid_bacc_one_dominance = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_bacc_one_dominance'][fold] if val is not None
                ]
                fold_results['mean_best_bacc_one_dominance'][fold] = np.mean(valid_bacc_one_dominance) if valid_bacc_one_dominance else 0.0
                fold_results['std_best_bacc_one_dominance'][fold] = np.std(valid_bacc_one_dominance) if valid_bacc_one_dominance else 0.0

                valid_conf_matrix_major_voting = [
                    cm.cpu().numpy() if isinstance(cm, torch.Tensor) else cm for cm in repeat_results['best_conf_matrix_major_voting'][fold] if cm is not None
                ]
                fold_results['mean_best_conf_matrix_major_voting'][fold] = (
                    np.mean(np.stack(valid_conf_matrix_major_voting), axis=0) if valid_conf_matrix_major_voting else np.zeros((2, 2))
                )

                valid_conf_matrix_one_dominance = [
                    cm.cpu().numpy() if isinstance(cm, torch.Tensor) else cm for cm in repeat_results['best_conf_matrix_one_dominance'][fold] if cm is not None
                ]
                fold_results['mean_best_conf_matrix_one_dominance'][fold] = (
                    np.mean(np.stack(valid_conf_matrix_one_dominance), axis=0) if valid_conf_matrix_one_dominance else np.zeros((2, 2))
                )
         
                valid_recall = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_recall'][fold] if val is not None
                ]
                fold_results['mean_best_recall'][fold] = np.mean(valid_recall) if valid_recall else 0.0
                fold_results['std_best_recall'][fold] = np.std(valid_recall) if valid_recall else 0.0
                
                valid_specificity = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_specificity'][fold] if val is not None
                ]
                fold_results['mean_best_specificity'][fold] = np.mean(valid_specificity) if valid_specificity else 0.0
                fold_results['std_best_specificity'][fold] = np.std(valid_specificity) if valid_specificity else 0.0
               
            
                valid_recall_1d = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_recall_1d'][fold] if val is not None
                ]
                fold_results['mean_best_recall_1d'][fold] = np.mean(valid_recall_1d) if valid_recall_1d else 0.0
                fold_results['std_best_recall_1d'][fold] = np.std(valid_recall_1d) if valid_recall_1d else 0.0
                
                valid_specificity_1d = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_specificity_1d'][fold] if val is not None
                ]
                fold_results['mean_best_specificity_1d'][fold] = np.mean(valid_specificity_1d) if valid_specificity_1d else 0.0
                fold_results['std_best_specificity_1d'][fold] = np.std(valid_specificity_1d) if valid_specificity_1d else 0.0
                
                valid_recall_mj = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_recall_mj'][fold] if val is not None
                ]
                fold_results['mean_best_recall_mj'][fold] = np.mean(valid_recall_mj) if valid_recall_mj else 0.0
                fold_results['std_best_recall_mj'][fold] = np.std(valid_recall_mj) if valid_recall_mj else 0.0
                
                valid_specificity_mj = [
                    val.cpu().item() if torch.is_tensor(val) else val for val in repeat_results['best_specificity_mj'][fold] if val is not None
                ]
                fold_results['mean_best_specificity_mj'][fold] = np.mean(valid_specificity_mj) if valid_specificity_mj else 0.0
                fold_results['std_best_specificity_mj'][fold] = np.std(valid_specificity_mj) if valid_specificity_mj else 0.0
              

                # Print summary for the current fold and repeat
                print(f"\n----------------- FOLD {fold + 1}, REPEAT {repeat + 1} -----------------")
                print("Mean Best Confusion Matrix (Training):", fold_results['mean_best_cm'][fold])
                print(f"Mean Best Validation Accuracy: {fold_results['mean_best_val_acc'][fold]} ± {fold_results['std_best_val_acc'][fold]}")
                print(f"Mean Best recall: {fold_results['mean_best_recall'][fold]} ± {fold_results['std_best_recall'][fold]}")
                print(f"Mean Best sp: {fold_results['mean_best_specificity'][fold]} ± {fold_results['std_best_specificity'][fold]}")
                print("Mean Best Confusion Matrix (Validation):", fold_results['mean_best_cm_v'][fold])
                print("\n #########  Major Voting Strategy #############")
                print(f"Mean Balanced Accuracy for Major Voting: {fold_results['mean_best_bacc_major_voting'][fold]} ± {fold_results['std_best_bacc_major_voting'][fold]}")
                print(f"Mean Best recall: {fold_results['mean_best_recall_mj'][fold]} ± {fold_results['std_best_recall_mj'][fold]}")
                print(f"Mean Best sp: {fold_results['mean_best_specificity_mj'][fold]} ± {fold_results['std_best_specificity_mj'][fold]}")
                print("Mean Confusion Matrix (Major Voting):", fold_results['mean_best_conf_matrix_major_voting'][fold])
                print("\n #########  1-Dominance #############")
                print(f"Mean Balanced Accuracy for 1-Dominance: {fold_results['mean_best_bacc_one_dominance'][fold]} ± {fold_results['std_best_bacc_one_dominance'][fold]}")
                print(f"Mean Best recall: {fold_results['mean_best_recall_1d'][fold]} ± {fold_results['std_best_recall_1d'][fold]}")
                print(f"Mean Best sp: {fold_results['mean_best_specificity_1d'][fold]} ± {fold_results['std_best_specificity_1d'][fold]}")
                print("Mean Confusion Matrix (1-Dominance):", fold_results['mean_best_conf_matrix_one_dominance'][fold])

                #for fold in range(num_foldss):
                # Write metrics for each repeat in the current fold
                file.write(f"\n----------------- FOLD {fold + 1} -----------------\n")
                file.write(f"\n Nº patient train:{train_patients} and val patients: {val_patients}\n")
                
                
                for repeat_idx, repeat_val_acc in enumerate(repeat_results['best_val_acc'][fold]):
                    file.write(f"Repeat {repeat_idx + 1}:\n")
                    file.write(f"  Best Validation Accuracy: {repeat_val_acc:.8f}\n")
                    if repeat_results['best_recall'][fold]:
                        file.write(f"  Balanced recall: {repeat_results['best_recall'][fold][repeat_idx]:.8f}\n")
                    if repeat_results['best_specificity'][fold]:
                        file.write(f"  Balanced specificity: {repeat_results['best_specificity'][fold][repeat_idx]:.8f}\n")
                        
                    # Best Confusion Matrix (Training)
                    if repeat_results['best_cm'][fold]:
                        file.write("  Best Confusion Matrix (Training):\n")
                        file.write(np.array2string(repeat_results['best_cm'][fold][repeat_idx], separator=",") + "\n")
                    else:
                        file.write("  No training confusion matrix available.\n")

                    # Best Confusion Matrix (Validation)
                    if repeat_results['best_cm_v'][fold]:
                        file.write("  Best Confusion Matrix (Validation):\n")
                        file.write(np.array2string(repeat_results['best_cm_v'][fold][repeat_idx], separator=",") + "\n")
                    else:
                        file.write("  No validation confusion matrix available.\n")

                    # Major Voting
                    if repeat_results['best_bacc_major_voting'][fold]:
                        file.write(f"  Balanced Accuracy for Major Voting: {repeat_results['best_bacc_major_voting'][fold][repeat_idx]:.8f}\n")
                    if repeat_results['best_recall_mj'][fold]:
                        file.write(f"  Balanced recall for Major Voting: {repeat_results['best_recall_mj'][fold][repeat_idx]:.8f}\n")
                    if repeat_results['best_specificity_mj'][fold]:
                        file.write(f"  Balanced specificity mojor voting: {repeat_results['best_specificity_mj'][fold][repeat_idx]:.8f}\n")
                    if repeat_results['best_conf_matrix_major_voting'][fold]:
                        file.write("  Confusion Matrix (Major Voting):\n")
                        file.write(np.array2string(repeat_results['best_conf_matrix_major_voting'][fold][repeat_idx], separator=",") + "\n")
                    else:
                        file.write("  No major voting confusion matrix available.\n")

                    # One-Dominance
                    if repeat_results['best_bacc_one_dominance'][fold]:
                        file.write(f"  Balanced Accuracy for 1-Dominance: {repeat_results['best_bacc_one_dominance'][fold][repeat_idx]:.8f}\n")
                    # One-Dominance
                    if repeat_results['best_recall_1d'][fold]:
                        file.write(f"  Balanced recall 1-Dominance: {repeat_results['best_recall_1d'][fold][repeat_idx]:.8f}\n")
                    if repeat_results['best_specificity_1d'][fold]:
                        file.write(f"  Balanced specificity 1-Dominance: {repeat_results['best_specificity_1d'][fold][repeat_idx]:.8f}\n")
                    if repeat_results['best_conf_matrix_one_dominance'][fold]:
                        file.write("  Confusion Matrix (1-Dominance):\n")
                        file.write(np.array2string(repeat_results['best_conf_matrix_one_dominance'][fold][repeat_idx], separator=",") + "\n")
                    else:
                        file.write("  No one-dominance confusion matrix available.\n")

            
            # Write mean metrics for the current fold
            file.write(f"\n----------------- Mean Metrics for FOLD {fold + 1} -----------------\n")
            file.write(f'(thresholds={thr_desc_txt}\n')
            file.write(f'op_threshold_v: {op_threshold_v}\n')
            file.write(f'val threshold:{fold_result[17]} \n')
            file.write("Mean Best Confusion Matrix (Training):\n")
            file.write(np.array2string(fold_results['mean_best_cm'][fold], separator=",") + "\n")
            file.write(f"Mean Best Validation Accuracy: {fold_results['mean_best_val_acc'][fold]:.8f} ± {fold_results['std_best_val_acc'][fold]:.8f}\n")
            file.write(f"Mean recall: {fold_results['mean_best_recall'][fold]:.8f} ± {fold_results['std_best_recall'][fold]:.8f}\n")
            file.write(f"Mean specificity: {fold_results['mean_best_specificity'][fold]:.8f} ± {fold_results['std_best_specificity'][fold]:.8f}\n")
            file.write("Mean Best Confusion Matrix (Validation):\n")
            file.write(np.array2string(fold_results['mean_best_cm_v'][fold], separator=",") + "\n")
            file.write("\n######### Major Voting Strategy #########\n")
            file.write(f"Mean Balanced Accuracy for Major Voting: {fold_results['mean_best_bacc_major_voting'][fold]:.8f} ± {fold_results['std_best_bacc_major_voting'][fold]:.8f}\n")
            file.write(f"Mean recall for Major Voting: {fold_results['mean_best_recall_mj'][fold]:.8f} ± {fold_results['std_best_recall_mj'][fold]:.8f}\n")
            file.write(f"Mean specificity for Major Voting: {fold_results['mean_best_specificity_mj'][fold]:.8f} ± {fold_results['std_best_specificity_mj'][fold]:.8f}\n")
            file.write("Mean Confusion Matrix (Major Voting):\n")
            file.write(np.array2string(fold_results['mean_best_conf_matrix_major_voting'][fold], separator=",") + "\n")
            file.write("\n######### 1-Dominance #########\n")
            file.write(f"Mean Balanced Accuracy for 1-Dominance: {fold_results['mean_best_bacc_one_dominance'][fold]:.8f} ± {fold_results['std_best_bacc_one_dominance'][fold]:.8f}\n")
            file.write(f"Mean recall for 1d: {fold_results['mean_best_recall_1d'][fold]:.8f} ± {fold_results['std_best_recall_1d'][fold]:.8f}\n")
            file.write(f"Mean specificity for 1d: {fold_results['mean_best_specificity_1d'][fold]:.8f} ± {fold_results['std_best_specificity_1d'][fold]:.8f}\n")
            file.write("Mean Confusion Matrix (1-Dominance):\n")
            file.write(np.array2string(fold_results['mean_best_conf_matrix_one_dominance'][fold], separator=",") + "\n")

            if num_folds == 0:
                print('\n\n##### End of evaluation since only fold tiag')
                break


    print("\n----------------- Overall Analysis Across All Folds -----------------")
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    
    all_labels = []
    all_scores = []

    for fold in range(num_folds):
        y_path = os.path.join(save_path, f"y_true_fold{fold}.npy")
        s_path = os.path.join(save_path, f"probs_fold{fold}.npy")
        if not (os.path.exists(y_path) and os.path.exists(s_path)):
            print(f"[Overall] Missing predictions for fold {fold} at {y_path} or {s_path}. Skipping this fold.")
            continue

        y = np.load(y_path)
        s = np.load(s_path)
        all_labels.append(y)
        all_scores.append(s)

    if not all_labels or not all_scores:
        print("[Overall] No prediction files found across folds. Skipping aggregate ROC computation.")
        continue

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("WSI-level ROC (12-month survival)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "roc_center.png"), dpi=300)
    plt.close()

    from sklearn.metrics import confusion_matrix
    
    TP, FP, TN, FN = [], [], [], []
    
    for tau in thresholds:
        y_pred = (all_scores >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_labels, y_pred).ravel()
        TN.append(tn); FP.append(fp); FN.append(fn); TP.append(tp)
    
    TP = np.array(TP)
    FP = np.array(FP)
    TN = np.array(TN)
    FN = np.array(FN)
    
    eps = 1e-6  # to avoid log10(0)

    plt.figure(figsize=(6, 6))
    plt.plot(np.log10(FP + eps), TP)
    plt.xlabel("log10(FP)")
    plt.ylabel("TP")
    plt.title("TP vs log10(FP)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "roc_left_TP_logFP.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(np.log10(FN + eps), TN)
    plt.xlabel("log10(FN)")
    plt.ylabel("TN")
    plt.title("TN vs log10(FN)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "roc_right_TN_logFN.png"), dpi=300)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # center ROC
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC")
    
    # left
    axes[0].plot(np.log10(FP + eps), TP)
    axes[0].set_xlabel("log10(FP)")
    axes[0].set_ylabel("TP")
    axes[0].set_title("TP vs log10(FP)")
    
    # right
    axes[2].plot(np.log10(FN + eps), TN)
    axes[2].set_xlabel("log10(FN)")
    axes[2].set_ylabel("TN")
    axes[2].set_title("TN vs log10(FN)")
    
    for ax in axes:
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "roc_three_panels.png"), dpi=300)
    plt.close()

    if all_t =='all_files':
        # all-files WSI-level
        plot_roc_three_panels(
            y_true=np.load("y_true_wsi_all_files.npy"),
            scores=np.load("probs_wsi_all_files.npy"),
            save_path="plots",
            prefix="all_files"
        )
        
        # majority voting (patient level)
        plot_roc_three_panels(
            y_true=np.load("y_true_patient_all_files.npy"),
            scores=np.load("scores_mv_patient_all_files.npy"),
            save_path="plots",
            prefix="mv"
        )
        
        # one-dominance (patient level)
        plot_roc_three_panels(
            y_true=np.load("y_true_patient_all_files.npy"),
            scores=np.load("scores_1d_patient_all_files.npy"),
            save_path="plots",
            prefix="1d"
        )

    if umap_layers:
        # =============== UMAPs aggregated across ALL folds (per layer) ===============
        try:
            aggregate_umap_across_folds(
                per_fold_root=save_path,
                split_tag="val",  # or "train"
                layers=("input", "layer1_bn", "layer2_bn", "layer3_bn", "pre_head"),
                search_subdir_prefix="umap_fold",
                max_points=500_000,
                umap_n_neighbors=15,
                umap_min_dist=0.5,
                umap_metric="cosine",
                renyi_alpha=1.5,
                renyi_k=5,
                renyi_repeats=30
            )
        except Exception as e:
            print("[UMAP-ALL] Aggregation failed:", e)


    

    print("\n----------------- Overall Analysis Across All Folds -----------------")

    # Flatten all validation accuracies across folds and repeats
    global_val_acc = [val for fold_acc in repeat_results['best_val_acc'] for val in fold_acc]

    # Compute global metrics for validation accuracy
    global_mean_val_acc = np.mean(global_val_acc) if global_val_acc else 0.0
    global_std_val_acc = np.std(global_val_acc) if global_val_acc else 0.0

    global_recall = [
        val for fold_recall in repeat_results['best_recall'] for val in fold_recall
    ]
    global_mean_recall = np.mean(global_recall) if global_recall else 0.0
    global_std_recall = np.std(global_recall) if global_recall else 0.0
        
    global_specificity = [
        val for fold_specificity in repeat_results['best_specificity'] for val in fold_specificity
    ]
    global_mean_specificity = np.mean(global_specificity) if global_specificity else 0.0
    global_std_specificity = np.std(global_specificity) if global_specificity else 0.0
    
    # Handle training confusion matrices
    global_train_cm = [
        cm for fold_cm in repeat_results['best_cm'] for cm in fold_cm if cm is not None
    ]
    global_mean_train_cm = np.mean(np.stack(global_train_cm), axis=0) if global_train_cm else np.zeros((2, 2))
    global_std_best_cm = np.std(np.stack(global_train_cm), axis=0) if global_train_cm else np.zeros((2, 2))

    # Handle validation confusion matrices
    global_val_cm = [
        cm for fold_cm_v in repeat_results['best_cm_v'] for cm in fold_cm_v if cm is not None
    ]
    global_mean_val_cm = np.mean(np.stack(global_val_cm), axis=0) if global_val_cm else np.zeros((2, 2))
    global_std_best_cm_v = np.std(np.stack(global_val_cm), axis=0) if global_val_cm else np.zeros((2, 2))

    # Handle major voting confusion matrices
    global_conf_matrix_maj = [
        cm for fold_cm_maj in repeat_results['best_conf_matrix_major_voting'] for cm in fold_cm_maj if cm is not None
    ]
    global_mean_conf_matrix_maj = np.mean(np.stack(global_conf_matrix_maj), axis=0) if global_conf_matrix_maj else np.zeros((2, 2))
    global_std_conf_matrix_maj = np.std(np.stack(global_conf_matrix_maj), axis=0) if global_conf_matrix_maj else np.zeros((2, 2))

    # Handle 1-Dominance confusion matrices
    global_conf_matrix_1d = [
        cm for fold_cm_1d in repeat_results['best_conf_matrix_one_dominance'] for cm in fold_cm_1d if cm is not None
    ]
    global_mean_conf_matrix_1d = np.mean(np.stack(global_conf_matrix_1d), axis=0) if global_conf_matrix_1d else np.zeros((2, 2))
    global_std_conf_matrix_1d = np.std(np.stack(global_conf_matrix_1d), axis=0) if global_conf_matrix_1d else np.zeros((2, 2))

    # Handle major voting balanced accuracy
    global_bacc_major_voting = [
        val for fold_bacc_maj in repeat_results['best_bacc_major_voting'] for val in fold_bacc_maj
    ]
    global_mean_bacc_major_voting = np.mean(global_bacc_major_voting) if global_bacc_major_voting else 0.0
    global_std_bacc_major_voting = np.std(global_bacc_major_voting) if global_bacc_major_voting else 0.0

    global_specificity_mj = [
        val for fold_specificity_mj in repeat_results['best_specificity_mj'] for val in fold_specificity_mj
    ]
    global_mean_specificity_mj = np.mean( global_specificity_mj) if global_specificity_mj else 0.0
    global_std_specificity_mj = np.std( global_specificity_mj) if  global_specificity_mj else 0.0

    global_recall_mj = [
        val for fold_recall_mj in repeat_results['best_recall_mj'] for val in fold_recall_mj
    ]
    global_mean_recall_mj = np.mean( global_recall_mj) if global_recall_mj else 0.0
    global_std_recall_mj = np.std( global_recall_mj) if  global_recall_mj else 0.0
    
    # Handle 1-Dominance balanced accuracy
    global_bacc_one_dominance = [
        val for fold_bacc_1d in repeat_results['best_bacc_one_dominance'] for val in fold_bacc_1d
    ]
    global_mean_bacc_1d = np.mean(global_bacc_one_dominance) if global_bacc_one_dominance else 0.0
    global_std_bacc_1d = np.std(global_bacc_one_dominance) if global_bacc_one_dominance else 0.0
    
    # Handle 1-Dominance balanced accuracy
    global_recall_1d = [
        val for fold_recall_1d in repeat_results['best_recall_1d'] for val in fold_recall_1d
    ]
    global_mean_recall_1d = np.mean(global_recall_1d) if global_recall_1d else 0.0
    global_std_recall_1d = np.std(global_recall_1d) if global_recall_1d else 0.0
    
        
    # Handle 1-Dominance balanced accuracy
    global_specificity_1d = [
        val for fold_specificity_1d in repeat_results['best_specificity_1d'] for val in fold_specificity_1d
    ]
    global_mean_specificity_1d = np.mean(global_specificity_1d) if global_specificity_1d else 0.0
    global_std_specificity_1d = np.std(global_specificity_1d) if global_specificity_1d else 0.0
    
    
    op_threshold_v_array = np.array(op_threshold_v)

    mean_threshold = np.mean(op_threshold_v_array)
    std_threshold = np.std(op_threshold_v_array)
    
    with open(f"{save_path}/overall_metrics_per_fold_and_repeats.txt", "w") as file:
        # Print global metrics
        print(f"Global Mean Validation Accuracy: {global_mean_val_acc:.8f} ± {global_std_val_acc:.8f}")
        print(f"Global Mean recall: {global_mean_recall:.8f} ± {global_std_recall:.8f}")
        print(f"Global Mean sp: {global_mean_specificity:.8f} ± {global_std_specificity:.8f}")
        print(f"Global Mean Training Confusion Matrix:\n{global_mean_train_cm}")
        print(f"Global Std Training Confusion Matrix:\n{global_std_best_cm}")
        print(f"Global Mean Validation Confusion Matrix:\n{global_mean_val_cm}")
        print(f"Global Std Validation Confusion Matrix:\n{global_std_best_cm_v}")
        print("\n######### Major Voting Strategy #########")
        print(f"Global Mean Balanced Accuracy (Major Voting): {global_mean_bacc_major_voting:.8f} ± {global_std_bacc_major_voting:.8f}")
        print(f"Global Mean recall (mj): {global_mean_recall_mj:.8f} ± {global_std_recall_mj:.8f}")
        print(f"Global Mean sp (mj): {global_mean_specificity_mj:.8f} ± {global_std_specificity_mj:.8f}")
        print(f"Global Mean Confusion Matrix (Major Voting):\n{global_mean_conf_matrix_maj}")
        print(f"Global Std Confusion Matrix (Major Voting):\n{global_std_conf_matrix_maj}")
        print("\n######### 1-Dominance Strategy #########")
        print(f"Global Mean Balanced Accuracy (1-Dominance): {global_mean_bacc_1d:.8f} ± {global_std_bacc_1d:.8f}")
        print(f"Global Mean recall (1-Dominance): {global_mean_recall_1d:.8f} ± {global_std_recall_1d:.8f}")
        print(f"Global Mean sp (1-Dominance): {global_mean_specificity_1d:.8f} ± {global_std_specificity_1d:.8f}")
        print(f"Global Mean Confusion Matrix (1-Dominance):\n{global_mean_conf_matrix_1d}")
        print(f"Global Std Confusion Matrix (1-Dominance):\n{global_std_conf_matrix_1d}")

    # Save metrics to file
    #save_path = f'/home/ritav/logs_fold_{all_t}/{actualtime}_Model_{survival}_epoch{num_epochs}_hidden{hidden_channels}_lr{lr}_fold{fold}_batch{batch_sizee}_heads{headss}{headss_2}{headss_3}_repeat_{repeat}'
    #os.makedirs(save_path, exist_ok=True)

   # with open(f"{save_path}/metrics_per_fold_and_repeats.txt", "w") as file:
        
        # Write overall results
        file.write("\n----------------- Overall Analysis Across All Folds -----------------\n")
        # Suppose op_threshold_v is already populated
        file.write(f'op_threshold_v: {op_threshold_v}\n')
        file.write(f'Mean threshold: {mean_threshold:.8f} +- Std threshold: {std_threshold:.8f}\n')
        file.write(f'val threshold:{fold_result[17]} \n')
        file.write(f"Global Mean Validation Accuracy: {global_mean_val_acc:.8f} ± {global_std_val_acc:.8f}\n")
        file.write(f"Global Mean recall: {global_mean_recall:.8f} ± {global_std_recall:.8f}\n")
        file.write(f"Global Mean sp: {global_mean_specificity:.8f} ± {global_std_specificity:.8f}\n")
        file.write("Global Mean Training Confusion Matrix:\n")
        file.write(np.array2string(global_mean_train_cm, separator=",") + "\n")
        file.write("Global Mean Validation Confusion Matrix:\n")
        file.write(np.array2string(global_mean_val_cm, separator=",") + "\n")
        file.write("\n######### Major Voting Strategy #########\n")
        file.write(f"Global Mean Balanced Accuracy (Major Voting): {global_mean_bacc_major_voting:.8f} ± {global_std_bacc_major_voting:.8f}\n")
        file.write(f"Global Mean recall (mj): {global_mean_recall_mj:.8f} ± {global_std_recall_mj:.8f}\n")
        file.write(f"Global Mean sp (mj): {global_mean_specificity_mj:.8f} ± {global_std_specificity_mj:.8f}\n")
        file.write("Global Mean Confusion Matrix (Major Voting):\n")
        file.write(np.array2string(global_mean_conf_matrix_maj, separator=",") + "\n")
        file.write("\n######### 1-Dominance Strategy #########\n")
        file.write(f"Global Mean Balanced Accuracy (1-Dominance): {global_mean_bacc_1d:.8f} ± {global_std_bacc_1d:.8f}\n")
        file.write(f"Global Mean recall (1-Dominance): {global_mean_recall_1d:.8f} ± {global_std_recall_1d:.8f}\n")
        file.write(f"Global Mean sp (1-Dominance): {global_mean_specificity_1d:.8f} ± {global_std_specificity_1d:.8f}\n")
        file.write("Global Mean Confusion Matrix (1-Dominance):\n")
        file.write(np.array2string(global_mean_conf_matrix_1d, separator=",") + "\n")


    if 'tcga' in  all_t:
        results_dir = save_path  # diretório onde estão os ficheiros test_results_fold*.txt
        output_file = os.path.join(results_dir, "mean_std_test_results.txt")
        
        # Inicializar listas para as métricas
        baccs = []
        recalls = []
        specificities = []
        aucs = []
        
        baccs_mj = []
        recalls_mj = []
        specificities_mj = []
        
        baccs_1d = []
        recalls_1d = []
        specificities_1d = []
        
        # Extrair os valores numéricos dos resultados por fold
        for fold in range(num_folds):
            for repeat in range(repeatt):
                filename = f"test_results_fold{fold}_repeat{repeat}.txt"
                filepath = os.path.join(results_dir, filename)
        
                if not os.path.exists(filepath):
                    continue
        
                with open(filepath, "r") as f:
                    content = f.read()
        
                    def extract(pattern, text):
                        match = re.search(pattern, text)
                        return float(match.group(1)) if match else None
        
                    baccs.append(extract(r"Balanced Accuracy:\s*([0-9.]+)", content))
                    recalls.append(extract(r"Recall:\s*([0-9.]+)", content))
                    specificities.append(extract(r"Specificity:\s*([0-9.]+)", content))
                    aucs.append(extract(r"AUC:\s*([0-9.]+)", content))
        
                    baccs_mj.append(extract(r"Balanced Accuracy:\s*([0-9.]+)", content.split("Majority Voting")[1]))
                    recalls_mj.append(extract(r"Recall:\s*([0-9.]+)", content.split("Majority Voting")[1]))
                    specificities_mj.append(extract(r"Specificity:\s*([0-9.]+)", content.split("Majority Voting")[1]))
        
                    baccs_1d.append(extract(r"Balanced Accuracy:\s*([0-9.]+)", content.split("1-Dominance")[1]))
                    recalls_1d.append(extract(r"Recall:\s*([0-9.]+)", content.split("1-Dominance")[1]))
                    specificities_1d.append(extract(r"Specificity:\s*([0-9.]+)", content.split("1-Dominance")[1]))
        
        # Guardar resultados
        with open(output_file, "w") as f:
        
            def write_stats(name, values):
                values = [v for v in values if v is not None]
                mean = np.mean(values) if values else 0.0
                std = np.std(values) if values else 0.0
                f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")
        
            f.write("========== MÉDIAS E STD DOS TESTES (5 folds) ==========\n")
            write_stats("Balanced Accuracy (Test)", baccs)
            write_stats("Recall (Test)", recalls)
            write_stats("Specificity (Test)", specificities)
            write_stats("AUC (Test)", aucs)
        
            f.write("\n--- Majority Voting ---\n")
            write_stats("Balanced Accuracy (MJ)", baccs_mj)
            write_stats("Recall (MJ)", recalls_mj)
            write_stats("Specificity (MJ)", specificities_mj)
        
            f.write("\n--- 1-Dominance ---\n")
            write_stats("Balanced Accuracy (1D)", baccs_1d)
            write_stats("Recall (1D)", recalls_1d)
            write_stats("Specificity (1D)", specificities_1d)



    print("\n Metrics per fold and mean of all folds saved successfully.")
    print('\n######## End #########\n')

    import subprocess
    import subprocess

    message = "The Python script has finished successfully!" 
    url = "https://discord.com/api/webhooks/1293883063998091304/vli-GFnDE5o8uV2QnXkLtA-aKgf4EbL1x8Wxc81BGM5z0JH2gcptIXnaDxIYETaqD7d-"

subprocess.run([
    "curl", "-H", "Content-Type: application/json", "-X", "POST", 
    "-d", f'{{"content": "{message}"}}', url
])
                   

print("Message has been sent.")

