import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data, Batch
import torch.nn as nn
from torch_geometric.loader import DataLoader
#from torch.optim import Adam
#from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv,SAGEConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer, global_mean_pool, global_max_pool
from torch.utils.data import random_split
#from torch.nn import Linear
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
#from torch.nn import Sequential as Seq
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#import warnings
from sklearn.metrics import confusion_matrixgatconv
from torch.nn import Sequential, Linear, ReLU
#from torch_geometric.data import Batch
#from torch.utils.data import WeightedRandomSampler
#from sklearn.model_selection import KFold
#from torch_geometric.data import Batch
#from datetime import datetime
import numpy as np
#from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import csv
import random
# Dataset and Dataloader
from torch.utils.data import WeightedRandomSampler
from torch_geometric import seed_everything
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import NeighborLoader
import re
from torch_geometric.nn import HypergraphConv
from torch_geometric.utils import subgraph
from sklearn.metrics import roc_curve

seed_everything(47)
seed=47

# Additional seeds for other sources of randomness
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# Optional for CUDA determinism (may slightly affect performance)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

#torch.manual_seed(12345)
'''seed = 40 #antes 40  ###also in model_gnn
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
 torch.use_deterministic_algorithms(True)'''

import umap
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import HeteroData


from torch_geometric.data import Data
from torch_geometric.data import HeteroData
from torch_geometric.nn import global_max_pool
import numpy as np
import pandas as pd

from collections import OrderedDict
import umap
import seaborn as sns
sns.set(style="white")


################# UMAP ####################
# ================= UMAP & METRICS HELPERS =================
from collections import OrderedDict
import os, json
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

import umap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import silhouette_score


# --- UMAP layer metrics & plots: Silhouette + Rényi (orig & UMAP) ---
# Dependencies: numpy, pandas, matplotlib, seaborn, umap-learn, scikit-learn, torch, torch-geometric

import os, json
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from math import gamma

import torch
from torch import nn
from torch_geometric.data import HeteroData
# -*- coding: utf-8 -*-
"""
UMAP layer metrics & plots (Survival/Domain):
- Collect layer activations (incl. pre-head)
- Compute Silhouette and Rényi (Póczos k-NN; symmetric + balanced bootstrap)
- Do it in original space and UMAP space
- Save PNG plots and CSV/JSON metrics
- Aggregate across folds utilities

Dependencies:
  numpy, pandas, matplotlib, seaborn, umap-learn, scikit-learn, torch, torch-geometric
"""

# --- UMAP layer metrics & plots: Silhouette + Rényi (orig & UMAP) with PATIENT-LEVEL metrics ---
# Deps: numpy, pandas, matplotlib, seaborn, umap-learn, scikit-learn, torch, torch-geometric

import numpy as np
import torch
from collections import OrderedDict
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.metrics import pairwise_distances
from torch_geometric.data import Data as HeteroData
from scipy.special import gamma

# --- UMAP layer metrics & plots: Silhouette + Rényi (orig & UMAP) ---
# Dependencies: numpy, pandas, matplotlib, seaborn, umap-learn, scikit-learn, torch, torch-geometric

import os, json
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns

from collections import OrderedDict
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from math import gamma

import torch
from torch import nn
from torch_geometric.data import HeteroData

import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import OrderedDict
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from math import gamma
import torch
from torch import nn
from torch_geometric.data import HeteroData

# =========================================================
# Domain helpers (TCGA/CPTAC by first letter)
# =========================================================
def _first_letter_domain(cid) -> int:
    s = str(cid).strip()
    if not s:
        return -1
    c0 = s[0].upper()
    if c0 == "C":  # CPTAC
        return 1
    if c0 == "T":  # TCGA
        return 0
    return -1

def _extract_case_ids_from_batch(data, n_graphs: int):
    if hasattr(data, "case_id"):
        vals = list(data.case_id)
    elif hasattr(data, "case_ids"):
        vals = list(data.case_ids)
    elif hasattr(data, "case_id_strs"):
        vals = [str(s) for s in list(data.case_id_strs)]
    else:
        vals = [""] * n_graphs
    if len(vals) != n_graphs:
        vals = (vals * n_graphs)[:n_graphs]
    return [str(v) for v in vals]

# =========================================================
# Final head locator (for pre-head embedding)
# =========================================================
def get_classifier_module(model):
    if hasattr(model, "lin"):
        return model.lin
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "head"):
        return model.head
    raise AttributeError(f"No classifier/head found in {type(model)}")

# =========================================================
# Layers to hook (BNs + pre_head)
# =========================================================
def get_layer_modules(model):
    layers = OrderedDict()

    if isinstance(model, HyperGCN_survival):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
    elif isinstance(model, HyperGAT_survival):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
    elif isinstance(model, HeteroGCN_survival):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
    elif isinstance(model, HeteroGAT_survival):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
    elif isinstance(model, (GCN_survival, GNN)):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
    elif isinstance(model, AgglomerativeGCN_Survival):
        layers["enc_layer1_bn"] = model.enc.bn1
        layers["enc_layer2_bn"] = model.enc.bn2
        layers["enc_layer3_bn"] = model.enc.bn3
    elif isinstance(model, AgglomerativeGAT_Survival):
        layers["enc_layer1_bn"] = model.enc.bn1
        layers["enc_layer2_bn"] = model.enc.bn2
        layers["enc_layer3_bn"] = model.enc.bn3

    try:
        layers["pre_head"] = get_classifier_module(model)
    except AttributeError:
        pass

    return layers
    
def plot_attention_histograms(attentions, save_dir='attn_histograms', prefix=''):
    """
    Plot histograms of attention weights per layer, separately for intra and inter edges.
    """
    os.makedirs(save_dir, exist_ok=True)

    for layer_idx, attn in enumerate(attentions):
        if isinstance(attn, dict):  # HeteroGAT case
            for rel_type, (edge_index, alpha) in attn.items():
                alpha = alpha.cpu().detach().numpy()
                if isinstance(alpha, list) or len(alpha.shape) > 1:
                    alpha = np.concatenate(alpha, axis=0)

                plt.figure(figsize=(6, 4))
                plt.hist(alpha, bins=50, alpha=0.75, color='steelblue')
                plt.title(f"{prefix} Layer {layer_idx+1} - {rel_type.capitalize()} Attention")
                plt.xlabel("Attention Weight")
                plt.ylabel("Edge Count")
                plt.grid(True)

                filename = f"{prefix}attn_hist_layer{layer_idx+1}_{rel_type}.png"
                plt.savefig(os.path.join(save_dir, filename), dpi=300)
                plt.close()

        else:  # GAT case
            edge_index, alpha = attn
            alpha = alpha.cpu().detach().numpy()
            if isinstance(alpha, list) or len(alpha.shape) > 1:
                alpha = np.concatenate(alpha, axis=0)

            plt.figure(figsize=(6, 4))
            plt.hist(alpha, bins=50, alpha=0.75, color='steelblue')
            plt.title(f"{prefix} Layer {layer_idx+1} Attention")
            plt.xlabel("Attention Weight")
            plt.ylabel("Edge Count")
            plt.grid(True)

            filename = f"{prefix}attn_hist_layer{layer_idx+1}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close()
# =========================================================
# Collect activations + labels
# =========================================================
from torch_geometric.data import HeteroData

def collect_activations_for_umap(model, loader, device, max_nodes_per_layer=50000):
    """
    Runs the (already trained) model on `loader` and collects:
      - 'input' node features
      - features after each BN layer returned by get_layer_modules(model)
      - 'pre_head' graph embeddings (input to final classifier), if available.

    Returns:
      layer_acts:        {layer_name: (X, y_survival)}
      layer_acts_domain: {layer_name: (X, y_domain)}, domain in {0=TCGA,1=CPTAC,-1}
    """
    model.eval()
    model.to(device)

    layer_modules = get_layer_modules(model)

    # buffers for activations
    feats_buffers = {"input": []}
    for name in layer_modules.keys():
        feats_buffers[name] = []

    # buffers for pre_head graph labels
    pre_head_surv_labels_batches = []
    pre_head_dom_labels_batches  = []

    # forward hooks
    handles = []
    for lname, module in layer_modules.items():
        if lname == "pre_head":
            # we want the INPUT to the classifier: use forward_pre_hook
            def _pre_hook_fn(m, inp, lname=lname):
                x = inp[0].detach().cpu().numpy()  # [B_graphs, D]
                feats_buffers[lname].append(x)
            h = module.register_forward_pre_hook(_pre_hook_fn)
        else:
            def _hook_fn(m, inp, out, lname=lname):
                feats_buffers[lname].append(out.detach().cpu().numpy())
            h = module.register_forward_hook(_hook_fn)
        handles.append(h)

    node_surv_labels = []
    node_dom_labels  = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # ---------------- node features & batch vector ----------------
            if isinstance(data, HeteroData):
                x_nodes  = data.x_dict['tile']          # [N_nodes, F]
                batchvec = data['tile'].batch           # [N_nodes]
            else:
                x_nodes  = data.x                       # [N_nodes, F]
                batchvec = getattr(data, "batch", None)
                if batchvec is None:
                    batchvec = torch.zeros(
                        x_nodes.size(0),
                        dtype=torch.long,
                        device=x_nodes.device,
                    )

            # >>> this is exactly the bit you showed <<<
            feats_buffers["input"].append(x_nodes.detach().cpu().numpy())

            # ---------------- graph-level labels ----------------
            y_graph   = data.y.view(-1).detach().cpu().numpy()   # [B_graphs]
            batch_cpu = batchvec.detach().cpu().numpy()

            case_ids  = _extract_case_ids_from_batch(data, n_graphs=len(y_graph))
            dom_graph = np.array([_first_letter_domain(cid) for cid in case_ids],
                                 dtype=int)  # [B_graphs]

            # replicate graph labels to nodes
            if isinstance(model, BaselinePatientGAP):
                node_surv_labels.extend([int(y_graph[0])] * batch_cpu.shape[0])
                node_dom_labels.extend([int(dom_graph[0])] * batch_cpu.shape[0])
            else:
                for g_idx, y_g in enumerate(y_graph):
                    mask = (batch_cpu == g_idx)
                    n = int(mask.sum())
                    if n > 0:
                        node_surv_labels.extend([int(y_g)] * n)
                        node_dom_labels.extend([int(dom_graph[g_idx])] * n)

            # stash graph-level labels for pre_head (if present)
            if "pre_head" in layer_modules:
                pre_head_surv_labels_batches.append(y_graph.astype(int))
                pre_head_dom_labels_batches.append(dom_graph.astype(int))

            # ---------------- forward pass (to fire hooks) ----------------
            if isinstance(model, HeteroGCN_survival):
                _ = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
            elif isinstance(model, HeteroGAT_survival):
                _, _ = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
            elif isinstance(model, HyperGCN_survival):
                _ = model(data.x, data.edge_index, data.hyperedge_index, data.batch)
            elif isinstance(model, HyperGAT_survival):
                _ = model(data.x, data.edge_index, data.hyperedge_index, data.batch)
            elif isinstance(model, AgglomerativeGCN_Survival):
                _ = model(data.x, data.edge_index, data.batch)
            elif isinstance(model, AgglomerativeGAT_Survival):
                _, _ = model(data.x, data.edge_index, data.batch)
            elif isinstance(model, (GCN_survival, GNN)):
                _ = model(data.x, data.edge_index, data.batch)
            elif isinstance(model, GAT_survival):
                _, _ = model(data.x, data.edge_index, data.batch)
            elif isinstance(model, BaselinePatientGAP):
                _ = model(data)
            else:
                _ = model(data.x, data.edge_index, data.batch)

    # remove hooks
    for h in handles:
        h.remove()

    node_surv = np.asarray(node_surv_labels, dtype=int)
    node_dom  = np.asarray(node_dom_labels,  dtype=int)
    pre_head_surv = (
        np.concatenate(pre_head_surv_labels_batches, axis=0).astype(int)
        if pre_head_surv_labels_batches else None
    )
    pre_head_dom = (
        np.concatenate(pre_head_dom_labels_batches,  axis=0).astype(int)
        if pre_head_dom_labels_batches else None
    )

    # ---------------- build final dicts ----------------
    layer_acts        = {}
    layer_acts_domain = {}

    for lname, chunks in feats_buffers.items():
        if len(chunks) == 0:
            continue
        X = np.concatenate(chunks, axis=0)

        if lname == "pre_head":
            if pre_head_surv is None or pre_head_dom is None:
                continue
            y_s, y_d = pre_head_surv, pre_head_dom   # graph-level
        else:
            y_s, y_d = node_surv, node_dom          # node-level

        m = min(X.shape[0], y_s.shape[0], y_d.shape[0])
        X, y_s, y_d = X[:m], y_s[:m], y_d[:m]

        if X.shape[0] > max_nodes_per_layer:
            idx = np.random.choice(X.shape[0], max_nodes_per_layer, replace=False)
            X, y_s, y_d = X[idx], y_s[idx], y_d[idx]

        layer_acts[lname]        = (X, y_s)
        layer_acts_domain[lname] = (X, y_d)

    return layer_acts, layer_acts_domain

def plot_attention_histograms(attentions, save_dir='attn_histograms', prefix=''):
    """
    Plot histograms of attention weights per layer, separately for intra and inter edges.
    """
    os.makedirs(save_dir, exist_ok=True)

    for layer_idx, attn in enumerate(attentions):
        if isinstance(attn, dict):  # HeteroGAT case
            for rel_type, (edge_index, alpha) in attn.items():
                alpha = alpha.cpu().detach().numpy()
                if isinstance(alpha, list) or len(alpha.shape) > 1:
                    alpha = np.concatenate(alpha, axis=0)

                plt.figure(figsize=(6, 4))
                plt.hist(alpha, bins=50, alpha=0.75, color='steelblue')
                plt.title(f"{prefix} Layer {layer_idx+1} - {rel_type.capitalize()} Attention")
                plt.xlabel("Attention Weight")
                plt.ylabel("Edge Count")
                plt.grid(True)

                filename = f"{prefix}attn_hist_layer{layer_idx+1}_{rel_type}.png"
                plt.savefig(os.path.join(save_dir, filename), dpi=300)
                plt.close()

        else:  # GAT case
            edge_index, alpha = attn
            alpha = alpha.cpu().detach().numpy()
            if isinstance(alpha, list) or len(alpha.shape) > 1:
                alpha = np.concatenate(alpha, axis=0)

            plt.figure(figsize=(6, 4))
            plt.hist(alpha, bins=50, alpha=0.75, color='steelblue')
            plt.title(f"{prefix} Layer {layer_idx+1} Attention")
            plt.xlabel("Attention Weight")
            plt.ylabel("Edge Count")
            plt.grid(True)

            filename = f"{prefix}attn_hist_layer{layer_idx+1}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close()



# =========================================================
# Rényi α-divergence (Póczos et al.)
# =========================================================
def renyi_knn(P, Q, k=5, alpha=0.85, eps=1e-12, metric="euclidean"):
    assert alpha != 1.0, "alpha must differ from 1."
    n_p, n_q = P.shape[0], Q.shape[0]
    nn_P_in_P = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(P)
    nn_P_in_Q = NearestNeighbors(n_neighbors=k,   metric=metric).fit(Q)
    rho_k = nn_P_in_P.kneighbors(P, return_distance=True)[0][:, k]   + eps
    nu_k  = nn_P_in_Q.kneighbors(P, return_distance=True)[0][:, k-1] + eps
    B_k_a = (gamma(k)**2) / (gamma(k - alpha + 1) * gamma(k + alpha - 1))
    ratio = (((n_p - 1) * rho_k) / (n_q * nu_k)) ** (1.0 - alpha)
    estimation = B_k_a * np.mean(ratio)
    return (1.0 / (alpha - 1.0)) * np.log(estimation + eps)

def renyi_knn_symmetric(P, Q, k=5, alpha=0.85, eps=1e-12, metric="euclidean"):
    return 0.5 * (renyi_knn(P, Q, k=k, alpha=alpha, eps=eps, metric=metric) +
                  renyi_knn(Q, P, k=k, alpha=alpha, eps=eps, metric=metric))

def renyi_balanced_bootstrap(X, y, alpha=1.5, k=5, metric="euclidean", repeats=30, seed=0):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) != 2:
        return np.nan, np.nan
    X0, X1 = X[y==0], X[y==1]
    n = min(len(X0), len(X1))
    if n < max(k+1, 5):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(repeats):
        i0 = rng.choice(len(X0), n, replace=False)
        i1 = rng.choice(len(X1), n, replace=False)
        vals.append(renyi_knn_symmetric(X0[i0], X1[i1], k=k, alpha=alpha, metric=metric))
    vals = np.array(vals, dtype=float)
    return float(np.mean(vals)), float(np.std(vals))


def build_renyi_patch_selector(
    umap_df,
    train_patient_ids,
    apply_patient_ids,
    keep_fraction=None,
    threshold=None,
    percentile=None,
    k=4,
    alpha=0.85,
    eps=1e-7,
    apply_umap_df=None,
):
    """
    Compute per-fold Rényi-based patch selection without label leakage.

    Parameters
    ----------
    umap_df: pd.DataFrame
        DataFrame containing UMAP coordinates and labels for all patches. Must
        include columns ``UMAP1``, ``UMAP2``, ``patch_index``, ``patient_id``
        (or ``case_id``) and either ``filename_key`` or ``image_filename`` as
        the per-WSI identifier.
    train_patient_ids: Iterable
        Patient IDs used to *fit* the patch-level densities for the current fold.
    apply_patient_ids: Iterable
        Patients to which the trained densities should be applied (e.g., train +
        val/test of the fold). Validation labels are **not** used when fitting
        the densities.
    apply_umap_df: pd.DataFrame or None
        Optional dataframe containing UMAP coordinates for the patients in
        ``apply_patient_ids`` (e.g., a validation/test UMAP computed separately
        from training). When ``None``, ``umap_df`` is used for both fitting and
        scoring.
    keep_fraction: float
        Fraction of the best (lowest) Rényi scores from the *training* patches
        to keep. For example, ``0.25`` keeps the best 25% patches. Used only
        when neither ``threshold`` nor ``percentile`` is provided.
    threshold: float or None
        Absolute Rényi score threshold. When provided, ``keep_fraction`` is
        ignored and all patches with ``renyi_score <= threshold`` are kept,
        using scores computed from training-only densities.
    percentile: int or float or None
        Percentile of the computed Rényi scores to use as a dynamic threshold
        (e.g., 25/50/75). When set, ``keep_fraction`` is ignored and the
        threshold is computed from the scored patches.
    k: int
        k-NN parameter for the local Rényi estimator.
    alpha: float
        Rényi alpha parameter (must differ from 1).
    eps: float
        Numerical stability constant.

    Returns
    -------
    tuple
        ``(selector, threshold, kept_fraction)`` where ``selector`` maps
        ``{filename_key -> [patch_indices...]}`` with the selected patches for
        every patient in ``apply_patient_ids``.
    """

    if umap_df is None or len(umap_df) == 0:
        return {}, None, np.nan

    if np.isclose(alpha, 1.0):
        raise ValueError("alpha = 1 is not supported for the local Rényi estimator.")

    train_df = umap_df.copy()
    apply_df = apply_umap_df.copy() if apply_umap_df is not None else train_df.copy()

    key_col = "filename_key" if "filename_key" in train_df.columns else "image_filename"
    pid_col = "patient_id" if "patient_id" in train_df.columns else "case_id"
    lbl_col = "label" if "label" in train_df.columns else "vital_status_12"

    required_base = {"UMAP1", "UMAP2", pid_col, key_col, "patch_index"}
    train_required = required_base | {lbl_col}

    def _validate_cols(df_local, name, required):
        missing = [c for c in required if c not in df_local.columns]
        if missing:
            raise ValueError(
                f"UMAP dataframe for {name} missing required columns: {missing}"
            )

    _validate_cols(train_df, "training", train_required)
    _validate_cols(apply_df, "apply", required_base)

    train_df[pid_col] = train_df[pid_col].astype(str)
    apply_df[pid_col] = apply_df[pid_col].astype(str)

    train_patients = set(str(p) for p in train_patient_ids)
    apply_patients = set(str(p) for p in apply_patient_ids)

    train_df = train_df[train_df[pid_col].isin(train_patients)].copy()
    apply_df = apply_df[apply_df[pid_col].isin(apply_patients)].copy()

    if train_df.empty or apply_df.empty:
        return {}, None, np.nan

    lbl = train_df[lbl_col].astype(int)
    coords = train_df[["UMAP1", "UMAP2"]].values.astype(np.float32)
    train_dead = coords[lbl == 0]
    train_alive = coords[lbl == 1]

    if len(train_dead) <= k or len(train_alive) <= k:
        return {}, None, np.nan

    nn_dead = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(train_dead)
    nn_alive = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(train_alive)

    def _score_subset(sub_df, use_labels=True):
        sub_df = sub_df.copy()
        coords_local = sub_df[["UMAP1", "UMAP2"]].values.astype(np.float32)
        scores = np.full(len(sub_df), np.nan, dtype=np.float64)

        if use_labels:
            labels = sub_df[lbl_col].astype(int).values

            dead_mask = labels == 0
            alive_mask = labels == 1

            if dead_mask.any():
                rho = nn_dead.kneighbors(coords_local[dead_mask], return_distance=True)[0][:, k] + eps
                nu = nn_alive.kneighbors(coords_local[dead_mask], return_distance=True)[0][:, k - 1] + eps
                scores[dead_mask] = (((len(train_dead) - 1) * rho) / (len(train_alive) * nu)) ** (1.0 - alpha)

            if alive_mask.any():
                rho = nn_alive.kneighbors(coords_local[alive_mask], return_distance=True)[0][:, k] + eps
                nu = nn_dead.kneighbors(coords_local[alive_mask], return_distance=True)[0][:, k - 1] + eps
                scores[alive_mask] = (((len(train_alive) - 1) * rho) / (len(train_dead) * nu)) ** (1.0 - alpha)
        else:
            rho_dead = nn_dead.kneighbors(coords_local, return_distance=True)[0][:, k] + eps
            nu_dead = nn_alive.kneighbors(coords_local, return_distance=True)[0][:, k - 1] + eps
            score_dead = (((len(train_dead) - 1) * rho_dead) / (len(train_alive) * nu_dead)) ** (1.0 - alpha)

            rho_alive = nn_alive.kneighbors(coords_local, return_distance=True)[0][:, k] + eps
            nu_alive = nn_dead.kneighbors(coords_local, return_distance=True)[0][:, k - 1] + eps
            score_alive = (((len(train_alive) - 1) * rho_alive) / (len(train_dead) * nu_alive)) ** (1.0 - alpha)

            scores = np.minimum(score_dead, score_alive)

        sub_df["renyi_score"] = scores
        return sub_df

    scored_train = _score_subset(train_df, use_labels=True)
    scored_apply = _score_subset(apply_df, use_labels=False)

    selection_mode = None
    train_thr = None

    if threshold is not None:
        train_thr = float(threshold)
        selection_mode = "explicit_threshold"
    elif percentile is not None:
        # Percentile-based selection: use the requested percentile of all
        # scored patches (train + apply) so the kept fraction aligns with the
        # percentile specification. Lower scores are considered more informative
        # and are therefore kept when <= threshold.
        valid_scores = scored_apply["renyi_score"].values
        if len(valid_scores):
            train_thr = float(np.nanpercentile(valid_scores, percentile))
            selection_mode = "percentile"
    elif keep_fraction is not None:
        train_thr = np.nanquantile(scored_train["renyi_score"].values, keep_fraction)
        selection_mode = "keep_fraction"

    if train_thr is None or np.isnan(train_thr):
        print(
            f"[Rényi] Skipping selector build: unable to compute threshold (mode={selection_mode})."
        )
        return {}, None, np.nan

    selected = scored_apply[scored_apply["renyi_score"] <= train_thr].copy()

    selector = {}
    for fname, grp in selected.groupby(key_col):
        selector[fname] = grp["patch_index"].astype(int).tolist()

    kept_fraction = len(selected) / len(scored_apply) if len(scored_apply) else float("nan")
    pct_txt = f"p{percentile}" if percentile is not None else selection_mode or "manual"
    print(
        f"[Rényi] {pct_txt} → threshold={train_thr:.4f}, kept {len(selected)} / {len(scored_apply)} ({kept_fraction:.3f})."
    )

    return selector, float(train_thr), float(kept_fraction)


def summarize_patch_selector_usage(umap_df, patch_selector, patient_ids=None):
    """
    Compute how many patches are kept by a selector for a patient subset.

    Parameters
    ----------
    umap_df: pd.DataFrame
        Same frame used to build the selector.
    patch_selector: dict
        Mapping {filename_key -> [patch_indices...]}
    patient_ids: Iterable or None
        If provided, statistics are restricted to these patients.

    Returns
    -------
    dict
        ``{"total_patches": int, "selected_patches": int, "kept_fraction": float}``
    """
    if patch_selector is None:
        return {"total_patches": 0, "selected_patches": 0, "kept_fraction": np.nan}

    if umap_df is None or len(umap_df) == 0:
        return {"total_patches": 0, "selected_patches": 0, "kept_fraction": np.nan}

    df = umap_df.copy()
    key_col = "filename_key" if "filename_key" in df.columns else "image_filename"
    pid_col = "patient_id" if "patient_id" in df.columns else "case_id"

    if patient_ids is not None:
        wanted = set(str(p) for p in patient_ids)
        df = df[df[pid_col].astype(str).isin(wanted)]

    if df.empty:
        return {"total_patches": 0, "selected_patches": 0, "kept_fraction": np.nan}

    total_patches = len(df)
    selected_patches = 0

    for fname in df[key_col].astype(str).unique():
        selected_patches += len(patch_selector.get(fname, []))

    kept_fraction = selected_patches / total_patches if total_patches else np.nan

    return {
        "total_patches": int(total_patches),
        "selected_patches": int(selected_patches),
        "kept_fraction": float(kept_fraction),
    }

# =========================================================
# Silhouette (guarded)
# =========================================================
def _safe_silhouette(X, y):
    try:
        y = np.asarray(y)
        m = (y >= 0)
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            return np.nan
        for cls in np.unique(y[m]):
            if (y[m] == cls).sum() < 2:
                return np.nan
        return float(silhouette_score(X[m], y[m]))
    except Exception:
        return np.nan

# =========================================================
# Per-layer plots + metrics (UMAP once; metrics in original & UMAP)
# =========================================================
def plot_umap_per_layer_dual_and_metrics(
    layer_acts, layer_acts_domain, save_dir, prefix="val",
    umap_n_neighbors=15, umap_min_dist=0.5, umap_metric="cosine", umap_random_state=42,
    renyi_alpha=1.5, renyi_k=5, renyi_repeats=30
):
    os.makedirs(save_dir, exist_ok=True)
    metrics_all = {}

    for lname, (X, y_surv) in layer_acts.items():
        if lname not in layer_acts_domain:
            continue
        Xd, y_dom = layer_acts_domain[lname]
        m = min(X.shape[0], Xd.shape[0], y_surv.shape[0], y_dom.shape[0])
        X, y_surv, y_dom = X[:m], y_surv[:m], y_dom[:m]
        if X.shape[0] < 10:
            print(f"[UMAP] Skipping layer {lname}: too few points ({X.shape[0]})")
            continue

        # --- Silhouette (original) ---
        sil_surv_orig = _safe_silhouette(X, y_surv)
        sil_dom_orig  = _safe_silhouette(X, y_dom)

        # --- Rényi (original, balanced bootstrap) ---
        renyi_surv_orig_mean, renyi_surv_orig_std = renyi_balanced_bootstrap(
            X, y_surv, alpha=renyi_alpha, k=renyi_k, metric="euclidean", repeats=renyi_repeats
        )
        dom_mask = (y_dom >= 0)
        renyi_dom_orig_mean, renyi_dom_orig_std = (np.nan, np.nan)
        if dom_mask.sum() > 20 and len(np.unique(y_dom[dom_mask])) == 2:
            renyi_dom_orig_mean, renyi_dom_orig_std = renyi_balanced_bootstrap(
                X[dom_mask], y_dom[dom_mask], alpha=renyi_alpha, k=renyi_k,
                metric="euclidean", repeats=renyi_repeats
            )

        # --- UMAP once ---
        reducer = umap.UMAP(
            n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric,
            random_state=umap_random_state
        )
        emb = reducer.fit_transform(X)

        # --- Silhouette (UMAP) ---
        sil_surv_umap = _safe_silhouette(emb, y_surv)
        sil_dom_umap  = _safe_silhouette(emb, y_dom)

        # --- Rényi (UMAP, balanced bootstrap) ---
        renyi_surv_umap_mean, renyi_surv_umap_std = renyi_balanced_bootstrap(
            emb, y_surv, alpha=renyi_alpha, k=renyi_k, metric="euclidean", repeats=renyi_repeats
        )
        renyi_dom_umap_mean, renyi_dom_umap_std = (np.nan, np.nan)
        if dom_mask.sum() > 20 and len(np.unique(y_dom[dom_mask])) == 2:
            renyi_dom_umap_mean, renyi_dom_umap_std = renyi_balanced_bootstrap(
                emb[dom_mask], y_dom[dom_mask], alpha=renyi_alpha, k=renyi_k,
                metric="euclidean", repeats=renyi_repeats
            )

        # --- Plots (same emb for both colorings) ---
        ttl = "Patient embedding (pre-head)" if lname == "pre_head" else lname

        # Survival
        df_s = pd.DataFrame({"UMAP1": emb[:,0], "UMAP2": emb[:,1], "label": y_surv})
        plt.figure(figsize=(7,6))
        sns.scatterplot(data=df_s, x="UMAP1", y="UMAP2",
                        hue=df_s["label"].map({0:"Alive",1:"Dead"}).fillna("Unknown"),
                        s=3, alpha=0.6)
        cap_s = (f"Sil orig={sil_surv_orig:.3f}, Sil umap={sil_surv_umap:.3f} | "
                 f"Rényiα={renyi_alpha} orig={renyi_surv_orig_mean:.3f}, "
                 f"umap={renyi_surv_umap_mean:.3f}")
        plt.title(f"UMAP — {prefix} — {ttl} — Survival\n{cap_s}")
        plt.legend(title="Survival", loc="best")
        plt.tight_layout()
        out_a = os.path.join(save_dir, f"umap_{prefix}_{lname}_survival.png")
        plt.savefig(out_a, dpi=300); plt.close()
        df_s.to_csv(os.path.join(save_dir, f"umap_{prefix}_{lname}_survival.csv"), index=False)

        # Domain
        dom_map = {0:"TCGA", 1:"CPTAC", -1:"Unknown"}
        df_d = pd.DataFrame({"UMAP1": emb[:,0], "UMAP2": emb[:,1], "label": y_dom})
        plt.figure(figsize=(7,6))
        sns.scatterplot(
            data=df_d, x="UMAP1", y="UMAP2",
            hue=df_d["label"].map(dom_map).fillna("Unknown"),
            s=3, alpha=0.6
        )
        cap_d = (f"Sil orig={sil_dom_orig:.3f}, Sil umap={sil_dom_umap:.3f} | "
                 f"Rényiα={renyi_alpha} orig={renyi_dom_orig_mean:.3f}, "
                 f"umap={renyi_dom_umap_mean:.3f}")
        plt.title(f"UMAP — {prefix} — {ttl} — Domain\n{cap_d}")
        plt.legend(title="Dataset", loc="best")
        plt.tight_layout()
        out_b = os.path.join(save_dir, f"umap_{prefix}_{lname}_domain.png")
        plt.savefig(out_b, dpi=300); plt.close()
        df_d.to_csv(os.path.join(save_dir, f"umap_{prefix}_{lname}_domain.csv"), index=False)

        # --- persist metrics ---
        metrics_all[lname] = {
            "silhouette_survival_original": sil_surv_orig,
            "silhouette_survival_umap":     sil_surv_umap,
            "silhouette_domain_original":   sil_dom_orig,
            "silhouette_domain_umap":       sil_dom_umap,
            "renyi_alpha": renyi_alpha, "renyi_k": renyi_k, "renyi_repeats": renyi_repeats,
            "renyi_survival_original": renyi_surv_orig_mean,
            "renyi_survival_original_std": renyi_surv_orig_std,
            "renyi_survival_umap": renyi_surv_umap_mean,
            "renyi_survival_umap_std": renyi_surv_umap_std,
            "renyi_domain_original": renyi_dom_orig_mean,
            "renyi_domain_original_std": renyi_dom_orig_std,
            "renyi_domain_umap": renyi_dom_umap_mean,
            "renyi_domain_umap_std": renyi_dom_umap_std,
            "n_points": int(X.shape[0]),
        }

    with open(os.path.join(save_dir, f"umap_metrics_{prefix}.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"[UMAP] Saved metrics JSON -> {os.path.join(save_dir, f'umap_metrics_{prefix}.json')}")

# =========================================================
# Save per-fold arrays (for later “all folds” UMAP)
# =========================================================
def save_layer_acts_for_fold(layer_acts, layer_acts_domain, out_dir, fold, split_tag):
    os.makedirs(out_dir, exist_ok=True)
    for lname, (X, y_surv) in layer_acts.items():
        if lname not in layer_acts_domain:
            continue
        _, y_dom = layer_acts_domain[lname]
        X      = np.asarray(X)
        y_surv = np.asarray(y_surv, dtype=int).ravel()
        y_dom  = np.asarray(y_dom,  dtype=int).ravel()
        m = min(X.shape[0], y_surv.shape[0], y_dom.shape[0])
        X, y_surv, y_dom = X[:m], y_surv[:m], y_dom[:m]
        np.savez_compressed(
            os.path.join(out_dir, f"{split_tag}_fold{fold}_{lname}.npz"),
            X=X, y_surv=y_surv, y_dom=y_dom
        )

# =========================================================
# Aggregate across folds (concatenate -> single UMAP + metrics)
# =========================================================
def _concat_npzs(npz_paths, max_points=None, rng_seed=42):
    Xs, ys_surv, ys_dom = [], [], []
    for p in npz_paths:
        z = np.load(p)
        Xs.append(z["X"]); ys_surv.append(z["y_surv"]); ys_dom.append(z["y_dom"])
    X = np.concatenate(Xs, axis=0) if Xs else np.empty((0, 2))
    y_surv = np.concatenate(ys_surv, axis=0) if ys_surv else np.empty((0,))
    y_dom  = np.concatenate(ys_dom,  axis=0) if ys_dom  else np.empty((0,))
    if max_points is not None and X.shape[0] > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(X.shape[0], max_points, replace=False)
        X, y_surv, y_dom = X[idx], y_surv[idx], y_dom[idx]
    return X, y_surv, y_dom

def plot_umap_dual_and_metrics_from_arrays(
    X, y_surv, y_dom, save_dir, fname_prefix,
    umap_n_neighbors=15, umap_min_dist=0.5, umap_metric="cosine",
    renyi_alpha=1.5, renyi_k=5, renyi_repeats=30, point_size=3, alpha=0.6
):
    os.makedirs(save_dir, exist_ok=True)

    reducer = umap.UMAP(
        n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric,
        random_state=42,
    )
    emb = reducer.fit_transform(X)

    def _sil(x, y):
        try: return float(silhouette_score(x, y))
        except Exception: return float("nan")

    # Silhouette
    sil_surv_orig = _sil(X, y_surv) if X.shape[0] > 10 else float("nan")
    dom_mask = (y_dom >= 0)
    sil_dom_orig  = _sil(X[dom_mask], y_dom[dom_mask]) if dom_mask.sum()>10 else float("nan")
    sil_surv_umap = _sil(emb, y_surv) if emb.shape[0] > 10 else float("nan")
    sil_dom_umap  = _sil(emb[dom_mask], y_dom[dom_mask]) if dom_mask.sum()>10 else float("nan")

    # Rényi (balanced bootstrap) — professor’s estimator only
    jr_surv_orig_mean, jr_surv_orig_std = renyi_balanced_bootstrap(
        X, y_surv, alpha=renyi_alpha, k=renyi_k, repeats=renyi_repeats
    )
    jr_surv_umap_mean, jr_surv_umap_std = renyi_balanced_bootstrap(
        emb, y_surv, alpha=renyi_alpha, k=renyi_k, repeats=renyi_repeats
    )
    if dom_mask.sum()>20 and len(np.unique(y_dom[dom_mask]))==2:
        Xd, yd = X[dom_mask], y_dom[dom_mask]
        Ed, yd2 = emb[dom_mask], y_dom[dom_mask]
        jr_dom_orig_mean, jr_dom_orig_std = renyi_balanced_bootstrap(Xd, yd, alpha=renyi_alpha, k=renyi_k, repeats=renyi_repeats)
        jr_dom_umap_mean, jr_dom_umap_std = renyi_balanced_bootstrap(Ed, yd2, alpha=renyi_alpha, k=renyi_k, repeats=renyi_repeats)
    else:
        jr_dom_orig_mean = jr_dom_umap_mean = np.nan
        jr_dom_orig_std  = jr_dom_umap_std  = np.nan

    # Plots
    df = pd.DataFrame({"UMAP1": emb[:,0], "UMAP2": emb[:,1], "y_surv": y_surv.astype(int), "y_dom": y_dom.astype(int)})
    surv_map = {0:"Alive", 1:"Dead"}; dom_map = {0:"TCGA", 1:"CPTAC"}

    plt.figure(figsize=(7,6))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue=df["y_surv"].map(surv_map), s=point_size, alpha=alpha)
    plt.title(f"{fname_prefix} — SURV\nsil(orig)={sil_surv_orig:.3f}  sil(umap)={sil_surv_umap:.3f}  "
              f"Rényiα={renyi_alpha} orig={jr_surv_orig_mean:.3f}  umap={jr_surv_umap_mean:.3f}")
    plt.legend(title="Survival", loc="best"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname_prefix}_UMAP_survival.png"), dpi=300); plt.close()

    plt.figure(figsize=(7,6))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue=df["y_dom"].map(dom_map), s=point_size, alpha=alpha)
    plt.title(f"{fname_prefix} — DOMAIN\nsil(orig)={sil_dom_orig:.3f}  sil(umap)={sil_dom_umap:.3f}  "
              f"Rényiα={renyi_alpha} orig={jr_dom_orig_mean:.3f}  umap={jr_dom_umap_mean:.3f}")
    plt.legend(title="Dataset", loc="best"); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{fname_prefix}_UMAP_domain.png"), dpi=300); plt.close()

    met = pd.DataFrame([{
        "sil_surv_orig": sil_surv_orig, "sil_surv_umap": sil_surv_umap,
        "sil_dom_orig":  sil_dom_orig,  "sil_dom_umap":  sil_dom_umap,
        "renyi_alpha": renyi_alpha, "renyi_k": renyi_k, "renyi_repeats": renyi_repeats,
        "renyi_surv_orig":  jr_surv_orig_mean, "renyi_surv_orig_std":  jr_surv_orig_std,
        "renyi_surv_umap":  jr_surv_umap_mean, "renyi_surv_umap_std":  jr_surv_umap_std,
        "renyi_dom_orig":   jr_dom_orig_mean,  "renyi_dom_orig_std":   jr_dom_orig_std,
        "renyi_dom_umap":   jr_dom_umap_mean,  "renyi_dom_umap_std":   jr_dom_umap_std,
        "n_points": X.shape[0]
    }])
    met.to_csv(os.path.join(save_dir, f"{fname_prefix}_metrics.csv"), index=False)

# Aggregate UMAP Across Folds (Final UMAP Per Layer)
def aggregate_umap_across_folds(
    per_fold_root, split_tag="val",
    layers=("input","layer1_bn","layer2_bn","layer3_bn","pre_head"),
    search_subdir_prefix="umap_fold", max_points=120_000,
    umap_n_neighbors=15, umap_min_dist=0.5, umap_metric="cosine",
    renyi_alpha=1.5, renyi_k=5, renyi_repeats=30
):
    import glob
    umap_dirs = sorted(glob.glob(os.path.join(per_fold_root, f"{search_subdir_prefix}*")))
    out_agg = os.path.join(per_fold_root, f"UMAP_{split_tag}_ALL_FOLDS")
    os.makedirs(out_agg, exist_ok=True)

    for lname in layers:
        npz_paths = []
        for d in umap_dirs:
            npz_paths.extend(glob.glob(os.path.join(d, f"{split_tag}_fold*_{lname}.npz")))
        if not npz_paths:
            print(f"[UMAP-ALL] No data for layer '{lname}' (split={split_tag}). Skipping.")
            continue

        X, y_surv, y_dom = _concat_npzs(npz_paths, max_points=max_points)
        if X.shape[0] < 10:
            print(f"[UMAP-ALL] Too few points for layer '{lname}'. Skipping.")
            continue

        plot_umap_dual_and_metrics_from_arrays(
            X, y_surv, y_dom, save_dir=out_agg,
            fname_prefix=f"{split_tag}_ALLFOLDS_{lname}",
            umap_n_neighbors=umap_n_neighbors, umap_min_dist=umap_min_dist, umap_metric=umap_metric,
            renyi_alpha=renyi_alpha, renyi_k=renyi_k, renyi_repeats=renyi_repeats
        )
        print(f"[UMAP-ALL] Saved aggregated UMAPs for layer '{lname}' → {out_agg}")


'''def get_layer_modules(model):
    """
    Return an OrderedDict {name: module} with the node-level layers
    where we want to hook activations for UMAP.

    Supports:
    - HyperGCN_survival, HyperGAT_survival
    - HeteroGCN_survival, HeteroGAT_survival
    - GCN_survival, GAT_survival, GNN
    - AgglomerativeGCN_Survival, AgglomerativeGAT_Survival
    - BaselinePatientGAP (no internal layers -> only input)
    """
    layers = OrderedDict()

    # ------------------- HYPER MODELS -------------------
    if isinstance(model, HyperGCN_survival):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
        return layers

    if isinstance(model, HyperGAT_survival):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
        return layers

    # ------------------- HETERO MODELS -------------------
    if isinstance(model, HeteroGCN_survival):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
        return layers

    if isinstance(model, HeteroGAT_survival):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
        return layers

    # ------------------- HOMO GNNs -------------------
    if isinstance(model, (GCN_survival, GAT_survival, GNN)):
        layers["layer1_bn"] = model.bn1
        layers["layer2_bn"] = model.bn2
        layers["layer3_bn"] = model.bn3
        return layers

    # ------------------- AGGLOMERATIVE -------------------
    if isinstance(model, AgglomerativeGCN_Survival):
        layers["enc_layer1_bn"] = model.enc.bn1
        layers["enc_layer2_bn"] = model.enc.bn2
        layers["enc_layer3_bn"] = model.enc.bn3
        return layers

    if isinstance(model, AgglomerativeGAT_Survival):
        layers["enc_layer1_bn"] = model.enc.bn1
        layers["enc_layer2_bn"] = model.enc.bn2
        layers["enc_layer3_bn"] = model.enc.bn3
        return layers

    # ------------------- BASELINE -------------------
    if isinstance(model, BaselinePatientGAP):
        # No GNN layers: we'll only use the input embedding for UMAP.
        return layers

    print(f"[WARN] get_layer_modules: unknown model type {type(model)} — no hooks.")
    return layers

def collect_activations_for_umap(
    model,
    loader,
    device,
    max_nodes_per_layer=50000,
):
    """
    Runs the (already trained) model on `loader` and collects:
      - 'input' node features
      - features after each BN layer returned by get_layer_modules(model)

    Returns:
      final_layer_acts: dict[layer_name -> (X, labels_layer)]
          X: np.array [N_nodes_layer, D_layer]
          labels_layer: np.array [N_nodes_layer], survival labels (0/1)
    """
    model.eval()
    model.to(device)

    layer_modules = get_layer_modules(model)

    # We'll store: 'input' + each BN layer
    layer_acts = {"input": []}
    for name in layer_modules.keys():
        layer_acts[name] = []

    # forward hooks
    handles = []
    for lname, module in layer_modules.items():
        def hook_fn(m, inp, out, lname=lname):
            # out: [N_nodes_batch, D]
            layer_acts[lname].append(out.detach().cpu().numpy())
        h = module.register_forward_hook(hook_fn)
        handles.append(h)

    node_labels = []  # survival label for each node (global across loader)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Get node features + batch vector for this batch
            if isinstance(data, HeteroData):
                x_nodes = data.x_dict['tile']      # [N_nodes, F]
                batch_vec = data['tile'].batch     # [N_nodes]
            else:
                x_nodes = data.x                   # [N_nodes, F]
                batch_vec = data.batch             # [N_nodes]

            # Save "input" features for this batch
            layer_acts["input"].append(x_nodes.detach().cpu().numpy())

            # Build node-level labels from graph- or patient-level y
            y_graph = data.y.view(-1).detach().cpu().numpy()  # [B_graphs or 1]
            batch_cpu = batch_vec.detach().cpu().numpy()      # [N_nodes]

            if isinstance(model, BaselinePatientGAP):
                # 1 patient per batch, one label for all WSIs/nodes
                assert y_graph.shape[0] == 1, "Expected 1 label per patient batch for BaselinePatientGAP"
                node_labels.extend([int(y_graph[0])] * batch_cpu.shape[0])
            else:
                # Standard case: batch index = graph index
                for g_idx, y_g in enumerate(y_graph):
                    node_mask = (batch_cpu == g_idx)
                    n_nodes_g = node_mask.sum()
                    if n_nodes_g > 0:
                        node_labels.extend([int(y_g)] * n_nodes_g)

            # ---- Forward pass (so hooks fire) ----
            if isinstance(model, HeteroGCN_survival):
                _ = model(data.x_dict, data.edge_index_dict, data['tile'].batch)

            elif isinstance(model, HeteroGAT_survival):
                _logits, _attn = model(data.x_dict, data.edge_index_dict, data['tile'].batch)

            elif isinstance(model, HyperGCN_survival):
                _ = model(data.x, data.edge_index, data.hyperedge_index, data.batch)

            elif isinstance(model, HyperGAT_survival):
                _ = model(data.x, data.edge_index, data.hyperedge_index, data.batch)

            elif isinstance(model, AgglomerativeGCN_Survival):
                _ = model(data.x, data.edge_index, data.batch)

            elif isinstance(model, AgglomerativeGAT_Survival):
                _logits, _attn = model(data.x, data.edge_index, data.batch)

            elif isinstance(model, GCN_survival) or isinstance(model, GNN):
                _ = model(data.x, data.edge_index, data.batch)

            elif isinstance(model, GAT_survival):
                _logits, _attn = model(data.x, data.edge_index, data.batch)

            elif isinstance(model, BaselinePatientGAP):
                _ = model(data)  # Baseline ignores edge_index, only pools

            else:
                # Fallback: assume standard (x, edge_index, batch) signature
                _ = model(data.x, data.edge_index, data.batch)

    # Remove hooks
    for h in handles:
        h.remove()
    
    # Concatenate per-layer lists and (optionally) subsample
    node_labels_all = np.array(node_labels, dtype=int)
    
    final_layer_acts = {}
    for lname, chunks in layer_acts.items():
        if len(chunks) == 0:
            continue
    
        X = np.concatenate(chunks, axis=0)  # [N_nodes_total, D_l]
        labels_this = node_labels_all
    
        if X.shape[0] != labels_this.shape[0]:
            print(f"[WARN] layer {lname}: #nodes {X.shape[0]} != #labels {labels_this.shape[0]}")
            min_n = min(X.shape[0], labels_this.shape[0])
            X = X[:min_n]
            labels_this = labels_this[:min_n]
    
        # Optional subsample for speed
        if X.shape[0] > max_nodes_per_layer:
            idx = np.random.choice(X.shape[0], max_nodes_per_layer, replace=False)
            X_sub = X[idx]
            labels_layer = labels_this[idx]
        else:
            X_sub = X
            labels_layer = labels_this
    
        final_layer_acts[lname] = (X_sub, labels_layer)

    return final_layer_acts



def plot_umap_per_layer(layer_acts, save_dir, prefix="val", label_map={0: "Alive", 1: "Dead"}):
    """
    layer_acts: dict[layer_name -> (X, labels)]
      - X: [N, D] features
      - labels: [N] 0/1 survival
    """
    os.makedirs(save_dir, exist_ok=True)

    for lname, (X, labels) in layer_acts.items():
        if X.shape[0] < 10:
            print(f"[UMAP] Skipping layer {lname}: too few points ({X.shape[0]})")
            continue

        labels = np.array(labels)
        label_str = np.vectorize(lambda z: label_map.get(z, str(z)))(labels)

        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.5,
            metric="cosine",
            random_state=42,
        )
        emb = reducer.fit_transform(X)  # [N, 2]

        df = pd.DataFrame({
            "UMAP1": emb[:, 0],
            "UMAP2": emb[:, 1],
            "label": labels,
            "label_str": label_str,
        })

        plt.figure(figsize=(7, 6))
        sns.scatterplot(
            data=df,
            x="UMAP1", y="UMAP2",
            hue="label_str",
            s=3,
            alpha=0.6
        )
        plt.title(f"UMAP — {prefix} — {lname}")
        plt.legend(title="Survival", loc="best")
        plt.tight_layout()
        out_png = os.path.join(save_dir, f"umap_{prefix}_{lname}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()

        # Optional: save CSV of embedding
        df.to_csv(os.path.join(save_dir, f"umap_{prefix}_{lname}.csv"), index=False)
        print(f"[UMAP] Saved {out_png}")
'''

########################################

def plot_roc_three_panels(y_true, scores, save_path, prefix):
    """
    y_true : 1D array of 0/1 labels
    scores : 1D array of probabilities or scores
    save_path : folder to save figures
    prefix : e.g. 'all_files', 'mv', '1d' (used in filenames)
    """
    os.makedirs(save_path, exist_ok=True)

    # ROC
    fpr, tpr, thresholds = roc_curve(y_true, scores)

    # Recover TP, FP, TN, FN along the curve
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    TP = tpr * P
    FP = fpr * N
    FN = P - TP
    TN = N - FP

    eps = 1e-6  # avoid log10(0)

    # --- Single left panel ---
    plt.figure(figsize=(6, 6))
    plt.plot(np.log10(FP + eps), TP)
    plt.xlabel("log10(FP)")
    plt.ylabel("TP")
    plt.title("TP vs log10(FP)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{prefix}_roc_left_TP_logFP.png"), dpi=300)
    plt.close()

    # --- Single right panel ---
    plt.figure(figsize=(6, 6))
    plt.plot(np.log10(FN + eps), TN)
    plt.xlabel("log10(FN)")
    plt.ylabel("TN")
    plt.title("TN vs log10(FN)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{prefix}_roc_right_TN_logFN.png"), dpi=300)
    plt.close()

    # --- 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # center ROC
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_xlabel("FPR")
    axes[1].set_ylabel("TPR")
    axes[1].set_title("ROC")

    # left: TP vs log10(FP)
    axes[0].plot(np.log10(FP + eps), TP)
    axes[0].set_xlabel("log10(FP)")
    axes[0].set_ylabel("TP")
    axes[0].set_title("TP vs log10(FP)")


    # right: TN vs log10(FN)
    axes[2].plot(np.log10(FN + eps), TN)
    axes[2].set_xlabel("log10(FN)")
    axes[2].set_ylabel("TN")
    axes[2].set_title("TN vs log10(FN)")

    for ax in axes:
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{prefix}_roc_three_panels.png"), dpi=300)
    plt.close()
        
def verify_post_normalization(dataset):
    all_feats = []
    for data in dataset:
        all_feats.append(data.x.cpu())

    all_feats_tensor = torch.cat(all_feats, dim=0)
    global_mean = all_feats_tensor.mean(dim=0)
    global_std = all_feats_tensor.std(dim=0)

    print(f"\n\nMean over all features (after normalization): {global_mean.mean():.6f}")
    print(f"Std over all features (after normalization): {global_std.mean():.6f}")
    print(f"Min feature value: {all_feats_tensor.min():.4f}")
    print(f"Max feature value: {all_feats_tensor.max():.4f}")


def initialize_repeat_results(num_folds):
    """
    Initialize repeat results for storing metrics across folds.
    """
    return {
        'best_val_acc': [[] for _ in range(num_folds)],  # Accumulate validation accuracies
        'best_score_matrix_v': [[] for _ in range(num_folds)],  # Accumulate validation score matrices
        'best_cm_v': [[] for _ in range(num_folds)],  # Accumulate validation confusion matrices
        'best_score_matrix_t': [[] for _ in range(num_folds)],  # Accumulate training score matrices
        'best_cm': [[] for _ in range(num_folds)],  # Accumulate training confusion matrices
        'best_bacc_major_voting': [[] for _ in range(num_folds)],  # Accumulate major voting accuracies
        'best_bacc_one_dominance': [[] for _ in range(num_folds)],  # Accumulate 1-dominance accuracies
        'best_conf_matrix_major_voting': [[] for _ in range(num_folds)],  # Accumulate major voting confusion matrices
        'best_conf_matrix_one_dominance': [[] for _ in range(num_folds)],  # Accumulate 1-dominance confusion matrices
        'best_classification_reports_major': [[] for _ in range(num_folds)],  # Accumulate major voting reports
        'best_classification_reports_one_dominance': [[] for _ in range(num_folds)],  # Accumulate 1-dominance reports
        'best_specificity':[[] for _ in range(num_folds)],  # Accumulate validation accuracies
        'best_recall':[[] for _ in range(num_folds)],  # Accumulate validation accuracies
        'best_specificity_1d':[[] for _ in range(num_folds)],  # Accumulate validation accuracies
        'best_recall_1d':[[] for _ in range(num_folds)],  # Accumulate validation accuracies
        'best_specificity_mj':[[] for _ in range(num_folds)],  # Accumulate validation accuracies
        'best_recall_mj':[[] for _ in range(num_folds)],  # Accumulate validation accuracies
        
    }



def initialize_fold_results(num_folds):
    """
    Initialize fold results for storing aggregated metrics.
    """
    return {
        'mean_best_val_acc': [0.0 for _ in range(num_folds)],  # Mean validation accuracy per fold
        'std_best_val_acc': [0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'mean_best_score_matrix_v': [np.zeros((2, 2)) for _ in range(num_folds)],  # Mean validation score matrix
        'mean_best_cm_v': [np.zeros((2, 2)) for _ in range(num_folds)],  # Mean validation confusion matrix
        'mean_best_score_matrix_t': [np.zeros((2, 2)) for _ in range(num_folds)],  # Mean training score matrix
        'mean_best_cm': [np.zeros((2, 2)) for _ in range(num_folds)],  # Mean training confusion matrix
        'mean_best_bacc_major_voting': [0.0 for _ in range(num_folds)],  # Mean major voting balanced accuracy
        'mean_best_bacc_one_dominance': [0.0 for _ in range(num_folds)],  # Mean 1-dominance balanced accuracy
        'std_best_bacc_major_voting': [0.0 for _ in range(num_folds)],  # Std dev of major voting balanced accuracy
        'std_best_bacc_one_dominance': [0.0 for _ in range(num_folds)],  # Std dev of 1-dominance balanced accuracy
        'mean_best_conf_matrix_major_voting': [np.zeros((2, 2)) for _ in range(num_folds)],  # Major voting confusion matrix
        'mean_best_conf_matrix_one_dominance': [np.zeros((2, 2)) for _ in range(num_folds)],  # 1-dominance confusion matrix
        'mean_best_specificity':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'mean_best_recall':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'mean_best_specificity_1d':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'mean_best_recall_1d':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'mean_best_specificity_mj':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'mean_best_recall_mj':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold  
        'std_best_specificity':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'std_best_recall':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'std_best_specificity_1d':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'std_best_recall_1d':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'std_best_specificity_mj':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        'std_best_recall_mj':[0.0 for _ in range(num_folds)],  # Std dev of validation accuracy per fold
        
    }

def plot_attention_histograms(attentions, save_dir='attn_histograms', prefix=''):
    """
    Plot histograms of attention weights per layer, separately for intra and inter edges.
    """
    os.makedirs(save_dir, exist_ok=True)

    for layer_idx, attn in enumerate(attentions):
        if isinstance(attn, dict):  # HeteroGAT case
            for rel_type, (edge_index, alpha) in attn.items():
                alpha = alpha.cpu().detach().numpy()
                if isinstance(alpha, list) or len(alpha.shape) > 1:
                    alpha = np.concatenate(alpha, axis=0)

                plt.figure(figsize=(6, 4))
                plt.hist(alpha, bins=50, alpha=0.75, color='steelblue')
                plt.title(f"{prefix} Layer {layer_idx+1} - {rel_type.capitalize()} Attention")
                plt.xlabel("Attention Weight")
                plt.ylabel("Edge Count")
                plt.grid(True)

                filename = f"{prefix}attn_hist_layer{layer_idx+1}_{rel_type}.png"
                plt.savefig(os.path.join(save_dir, filename), dpi=300)
                plt.close()

        else:  # GAT case
            edge_index, alpha = attn
            alpha = alpha.cpu().detach().numpy()
            if isinstance(alpha, list) or len(alpha.shape) > 1:
                alpha = np.concatenate(alpha, axis=0)

            plt.figure(figsize=(6, 4))
            plt.hist(alpha, bins=50, alpha=0.75, color='steelblue')
            plt.title(f"{prefix} Layer {layer_idx+1} Attention")
            plt.xlabel("Attention Weight")
            plt.ylabel("Edge Count")
            plt.grid(True)

            filename = f"{prefix}attn_hist_layer{layer_idx+1}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close()



def store_fold_data(fold_results, train_dataset, val_dataset, train_df, val_df, task='12months'):
    """
    Store dataset sizes, patient IDs, and class distributions for each fold in fold_results.

    Parameters:
    - fold_results: Dictionary to store the results for each fold.
    - train_dataset: Training dataset for the current fold.
    - val_dataset: Validation dataset for the current fold.
    - train_df: DataFrame containing training data for the current fold.
    - val_df: DataFrame containing validation data for the current fold.
    """
    # Store dataset sizes and unique patient IDs
    fold_results['train_dataset_sizes'].append(len(train_dataset))
    fold_results['val_dataset_sizes'].append(len(val_dataset))
    fold_results['train_patient_ids'].append(train_df['case_id'].unique())
    fold_results['val_patient_ids'].append(val_df['case_id'].unique())

    label_col = 'vital_status_12' if task == '12months' else 'event'
    # Calculate and store class distributions for training and validation sets
    train_0s = train_df[train_df[label_col] == 0].groupby('case_id').size()
    train_1s = train_df[train_df[label_col] == 1].groupby('case_id').size()
    val_0s = val_df[val_df[label_col] == 0].groupby('case_id').size()
    val_1s = val_df[val_df[label_col] == 1].groupby('case_id').size()

    fold_results['train_0s_per_patient'].append(train_0s)
    fold_results['train_1s_per_patient'].append(train_1s)
    fold_results['val_0s_per_patient'].append(val_0s)
    fold_results['val_1s_per_patient'].append(val_1s)


def prepare_fold_data(df_train, df_val, df, df_test,all_t,unique_patients, task, train_index=None, val_index=None):
    """
    Prepare training and validation data based on the configuration of `all_t`.

    Parameters:
    - df_train (pd.DataFrame): The full training DataFrame for predefined splits.
    - df_val (pd.DataFrame): The full validation DataFrame for predefined splits.
    - df (pd.DataFrame): The main DataFrame with patient and feature information.
    - all_t (str): Indicator for dataset configuration.
    - train_index (array-like, optional): Indices for training patients in cross-validation (only used if `all_t` is 'all_files').
    - val_index (array-like, optional): Indices for validation patients in cross-validation (only used if `all_t` is 'all_files').

    Returns:
    - train_df (pd.DataFrame): DataFrame for training.
    - val_df (pd.DataFrame): DataFrame for validation.
    - num_folds (int): Number of folds, `0` for predefined split, otherwise the fold count.
    """
    if 'all_files' in all_t :
        # Use cross-validation with patient-based splits
        #unique_patients = df.groupby('case_id')['image_filename'].first().reset_index()
        print('Activated 5-fold cross-validation')

        # Select train and validation patients based on indices
        #train_patients = unique_patients.iloc[train_index]['case_id']
        #val_patients = unique_patients.iloc[val_index]['case_id']
        train_patients = unique_patients.iloc[train_index]['case_id']
        #print('\n Number of train patients',train_patients)
        print('\n Number of  train patients',len(train_patients))
        val_patients = unique_patients.iloc[val_index]['case_id']
        #print('\n Number of val patients',val_patients)
        print('\n Number of val patients',len(val_patients))
        
        # Filter the main DataFrame to create train and validation sets
        train_df = df[df['case_id'].isin(train_patients)]
        val_df = df[df['case_id'].isin(val_patients)]
        
        
        label_cols = ['image_filename', 'case_id', 'vital_status_12'] if task == '12months' else ['image_filename', 'case_id', 'event', 'time']
        train_df = train_df[label_cols]
        val_df = val_df[label_cols]

        train_patients = len(train_patients)
        val_patients = len(val_patients)
        # Set number of folds to 5 for cross-validation
        num_folds = 5
        test_df =[]
        test_patients = []
        if 'tcga' in all_t:
            test_df = df_test[label_cols]
            test_patients = len(test_patients)
            
    else:
        # Use predefined train/val split when `all_t` is not 'all_files'
        print('Using predefined train/validation split')
        '''if all_t=='all_tiago':
            # Directly assign predefined train and val DataFrames
            train_df = df_train['case_id'].unique()                ####################### ver isto          
            val_df = df_val['case_id'].unique()
            
        else:'''
        train_df = df_train
        val_df = df_val
        test_df = []

        # Set number of folds to 0 for predefined split
        num_folds = 0

        train_patients = len(train_df)
        val_patients = len(val_df)
        test_patients = []
    return train_df, val_df, test_df, num_folds,train_patients,val_patients, test_patients
 

# Function to calculate max neighbors dynamically
def get_max_neighbors(data, edge_type=('tile', 'intra', 'tile')):
    """
    Compute max degree (number of neighbors) of any node for a given edge type in a HeteroData object.
    """
    if isinstance(data, HeteroData):
        edge_index = data[edge_type].edge_index
    else:
        edge_index = data.edge_index

    degrees = torch.bincount(edge_index[0])
    return degrees.max().item()



def setup_data_loaders(all_t,train_df, val_df, test_df, mean_features, std_dev, batch_size, device,sampller,survival,node_batch_size,virtual_percent,individual,dataset, patch_selector=None, task='12months'):

                       
    # =========================
    # NEW: agglomerative mode
    # =========================
    if 'aglomerative' in all_t:
        dropout_rate = 0.3
        root = f"{dataset}_univ2_patchgraphadj_selfloop"  # per-WSI graphs
    
        # one row per patient (label once), but keep full WSI rows to locate files
        train_pat_df = train_df[['image_filename', label_col, 'case_id']].drop_duplicates('case_id')
        val_pat_df   = val_df  [['image_filename', label_col, 'case_id']].drop_duplicates('case_id')
    
        test_present = isinstance(test_df, pd.DataFrame) and (not test_df.empty) and ('tcga' in all_t)
        test_pat_df  = (test_df[['image_filename', label_col, 'case_id']].drop_duplicates('case_id')
                        if test_present else None)
    
        # datasets: normalization identical to your GraphDataset_featsnorml
        train_dataset = PatientWSIPackDataset_featsnorml(root, train_pat_df, train_df, device, mean_features, std_dev, virtual_percent, patch_selector=patch_selector)
        val_dataset   = PatientWSIPackDataset_featsnorml(root, val_pat_df,   val_df,   device, mean_features, std_dev, virtual_percent, patch_selector=patch_selector)
        test_dataset  = (PatientWSIPackDataset_featsnorml(root, test_pat_df, test_df, device, mean_features, std_dev, virtual_percent, patch_selector=patch_selector)
                         if test_present else [])
    
        # Early sanity check to avoid cryptic sampler errors later
        if len(train_dataset) == 0:
            raise RuntimeError(f"[aglomerative] No train patients left after filtering. "
                               f"Check root path '{root}' and that *.pt files exist for the listed case_ids.")
    
        # Helper to fetch label list
        def _labels_from(df):
            return df[label_col].astype(int).tolist() if (df is not None and not df.empty and label_col in df) else []
    
        # Prefer labels from the dataset (patient-level df); fallback to the patient DF you just built
        labels = _labels_from(getattr(train_dataset, 'df', None))
        if len(labels) == 0:
            print("[aglomerative] train_dataset.df empty or missing label column; falling back to train_pat_df.")
            labels = _labels_from(train_pat_df)
    
        # --- Build sampler only if explicitly requested AND we actually have labels ---
        sampler = None
        if sampller == 'True' and len(labels) > 0:
            from collections import Counter
            cnt = Counter(labels)
            n_pos = cnt.get(1, 0)
            n_neg = cnt.get(0, 0)
            total_samples = n_pos + n_neg
    
            # keep your original weighting convention
            weight_pos = total_samples / max(1, n_pos)
            weight_neg = total_samples / max(1, n_neg)
            sample_weights = [weight_pos if y == 0 else weight_neg for y in labels]
    
            # WeightedRandomSampler expects a non-empty floating tensor
            sampler = WeightedRandomSampler(
                torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True
            )
        elif sampller == 'True' and len(labels) == 0:
            print("[aglomerative] No train labels found; falling back to shuffle=True (no sampler).")
    
        # DataLoaders: one PATIENT per batch; avoid Batch-of-Batch with a passthrough collate
        collate_passthrough = (lambda xs: xs[0])
        generator = torch.Generator().manual_seed(47)
    
        if sampler is not None:
            train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler,drop_last=False, generator=generator,collate_fn=collate_passthrough)
        else:
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=False, generator=generator, collate_fn=collate_passthrough)
    
        val_loader = DataLoader(val_dataset,batch_size=1, shuffle=False, drop_last=False,collate_fn=collate_passthrough)
        if test_present:
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False,collate_fn=collate_passthrough)
        else:
            test_loader = []
    
        # neighbor stats (works on the packed patient batch too)
        first_graph = train_dataset[0]
        edge_index = first_graph.edge_index
        if edge_index.numel() > 0:
            degrees = torch.bincount(edge_index[0])
            max_neighbors = degrees.max().item()
            mean_neighbors = degrees.float().mean().item()
            min_neighbors = degrees.min().item()
        else:
            max_neighbors = 0; mean_neighbors = 0.0; min_neighbors = 0
        print(f"\n\n🔹 Maximum neighbors per node: {max_neighbors}")
        print(f"🔹 Average neighbors per node: {mean_neighbors:.2f}")
        print(f"🔹 Minimum neighbors per node: {min_neighbors}")
    
        max_neighbors = get_max_neighbors(first_graph)
        print(f"\n\n🔹 Maximum neighbors per node: {max_neighbors}")
    
        # keep interface compatibility (neighbor sampling not used in agglomerative)
        num_neighbors = [9, 3, 3]
        return (train_loader, val_loader, test_loader,
                train_dataset, val_dataset, test_dataset,
            num_neighbors, node_batch_size)
        
    ## hetero gnn:
    if 'hetero' in all_t:
         if  'combined_' in all_t:
             if individual ==  'True':
                print('\nNormalization of train')
                train_dataset = GraphDataset_featsnorml_hetero(f"{dataset}_univ2_combinedknn_hetero_selfloop_names_individualTrue",train_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
                print('len train:', len(train_dataset))
                print('\nNormalization of val')
                val_dataset = GraphDataset_featsnorml_hetero(f"{dataset}_univ2_combinedknn_hetero_selfloop_names_individualTrue",val_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
                print('len val:', len(val_dataset))
                test_dataset = []
             else:
                print('\nNormalization of train')
                train_dataset = GraphDataset_featsnorml_hetero(f"{dataset}_univ2_combinedknn_hetero_selfloop_names",train_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
                print('len train:', len(train_dataset))
                print('\nNormalization of val')
                val_dataset = GraphDataset_featsnorml_hetero(f"{dataset}_univ2_combinedknn_hetero_selfloop_names",val_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
                print('len val:', len(val_dataset))
                test_dataset = []
            
         if 'tcga' in all_t:
            if 'combined_' in all_t:
                print('\nNormalization of test knn')
                test_dataset = GraphDataset_featsnorml_hetero(f"{dataset}_univ2_combinedknn_hetero_selfloop_names", test_df, device, mean_features, std_dev, virtual_percent, patch_selector=patch_selector, task=task)
            
            print('len test:', len(test_dataset))
    
         else:
            test_dataset =[]
    elif 'hyper' in all_t:
        # infer percent from caller (already parsed as virtual_percent)
        hyper_percent = int(virtual_percent)
        root = f"{dataset}_univ2_combinedknn_hypergraph_gpu_selfloop_names"
        if individual == 'True':
            root += "_individualTrue"
    
        train_dataset = GraphDataset_featsnorml_hyperinc(root, train_df, device, mean_features, std_dev, hyper_percent, patch_selector=patch_selector, task=task)
        val_dataset   = GraphDataset_featsnorml_hyperinc(root, val_df,   device, mean_features, std_dev, hyper_percent, patch_selector=patch_selector, task=task)
    
        if 'tcga' in all_t:
            test_dataset  = GraphDataset_featsnorml_hyperinc(root, test_df,  device, mean_features, std_dev, hyper_percent, patch_selector=patch_selector, task=task)
        else:
            test_dataset = []

    ## homo gnn
    else:
        if  'combined_' in all_t:
            if individual ==  'True':
                print('\nNormalization of train')
                train_dataset = GraphDataset_featsnorml(f"{dataset}_univ2_combinedknn_homo_selfloop_names_individualTrue",train_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
                print('len train:', len(train_dataset))
                print('\nNormalization of val')
                val_dataset = GraphDataset_featsnorml(f"{dataset}_univ2_combinedknn_homo_selfloop_names_individualTrue",val_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
                print('len val:', len(val_dataset))
                test_dataset = []
            else:
                print('\nNormalization of train')
                train_dataset = GraphDataset_featsnorml(f"{dataset}_univ2_combinedknn_selfloop_names",train_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
                print('len train:', len(train_dataset))
                print('\nNormalization of val')
                val_dataset = GraphDataset_featsnorml(f"{dataset}_univ2_combinedknn_selfloop_names",val_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
                print('len val:', len(val_dataset))
                test_dataset = []
        else:
            train_dataset = GraphDataset_featsnorml(f"{dataset}_univ2_patchgraphadj_selfloop", train_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
            print('len train:', len(train_dataset))
            val_dataset = GraphDataset_featsnorml(f"{dataset}_univ2_patchgraphadj_selfloop", val_df, device,mean_features, std_dev,virtual_percent, patch_selector=patch_selector, task=task)
            print('len val:', len(val_dataset))
            
        if 'tcga' in all_t:
            if 'combined_' in all_t:
                print('\nNormalization of test knn')
                test_dataset = GraphDataset_featsnorml(f"{dataset}_univ2_combinedknn_selfloop_names", test_df, device, mean_features, std_dev, virtual_percent, patch_selector=patch_selector, task=task)
            else:
                print('\nNormalization of test all files')
                test_dataset = GraphDataset_featsnorml(f"{dataset}_univ2_patchgraphadj_selfloop", test_df, device, mean_features, std_dev, virtual_percent, patch_selector=patch_selector, task=task)
            print('len test:', len(test_dataset))
    
        else:
            test_dataset =[]
            
    total_samples = train_dataset.n_pos + train_dataset.n_neg
    # Calculate weights
    weight_pos = total_samples / train_dataset.n_pos  # Larger weight for the minority class
    weight_neg = total_samples / train_dataset.n_neg  # Smaller weight for the majority class

    # Assign weights inversely proportional to class frequencies
    sample_weights = [weight_pos if label == 0 else weight_neg for label in train_dataset.df[label_col]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    generator = torch.Generator().manual_seed(47)
    #print('\n\n ####survival[-4:]',survival[-4:])
    
    #first_graph = train_dataset[0]  # Access the first graph in the dataset
    #edge_index = first_graph.edge_index  # Get edge_index from a sample graph
    if 'hetero' in all_t:
        first_graph = train_dataset[0]
        # Use both edge types for neighbor analysis (optional)
        intra = first_graph['tile', 'intra', 'tile'].edge_index

        if ('tile', 'inter', 'tile') in first_graph.edge_types:
            inter = first_graph['tile', 'inter', 'tile'].edge_index
            print(f"Inter edges: {inter.shape[1]}")
        else:
            inter = torch.empty((2, 0), dtype=torch.long).to(intra.device)
            print("No inter-tile edges found (likely single WSI).")
        
        edge_index = torch.cat([intra, inter], dim=1)

    else:
        first_graph = train_dataset[0]  # Access the first graph in the dataset
        edge_index = first_graph.edge_index


    # Compute node degrees (number of neighbors per node)
    degrees = torch.bincount(edge_index[0])  # Count occurrences of each node

    # Compute statistics
    max_neighbors = degrees.max().item()
    mean_neighbors = degrees.float().mean().item()
    min_neighbors = degrees.min().item()

    # Print results
    print(f"\n\n🔹 Maximum neighbors per node: {max_neighbors}")
    print(f"🔹 Average neighbors per node: {mean_neighbors:.2f}")
    print(f"🔹 Minimum neighbors per node: {min_neighbors}")

    # Compute dynamic neighbor limit
   # first_graph = train_dataset[0]  # Get the first graph in the dataset
    max_neighbors = get_max_neighbors(first_graph)

    #max_neighbors = get_max_neighbors(train_dataset.data)
    print(f"\n\n🔹 Maximum neighbors per node: {max_neighbors}")
    
    if sampller =='True':
        # Compute `num_neighbors`
        adaptive = True if "Adaptive" in survival else False
        if adaptive:
            num_neighbors = torch.clamp(degrees, max=max_neighbors).tolist()  # Adaptive neighbor selection
        else:
            num_neighbors = [9,3,3] #[max_neighbors, max_neighbors, max_neighbors]  # Fixed neighbor sampling
    
        #input_nodes = torch.arange(train_dataset.data.num_nodes)  # Use all nodes
        #first_graph = train_dataset[0]  # Extract a single WSI graph
        input_nodes = torch.arange(first_graph.num_nodes)  # Get number of nodes
        
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,drop_last=False
, generator=generator) #shuffle=True#sampler=sampler
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)
        
    elif sampller == 'False':
       # Compute `num_neighbors`
        adaptive = True if "Adaptive" in survival else False
        if adaptive:
            num_neighbors = torch.clamp(degrees, max=max_neighbors).tolist()  # Adaptive neighbor selection
        else:
            num_neighbors = [9,3,3] #[max_neighbors, max_neighbors, max_neighbors]  # Fixed neighbor 
    
        input_nodes = torch.arange(first_graph.num_nodes)  # Use all nodes
     
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=False
, generator=generator) #shuffle=True#sampler=sampler
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)
    if 'tcga' in all_t:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)
    else:
        test_loader = []
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, num_neighbors, node_batch_size


def get_class_distribution(df, label_col='vital_status_12'):
    return {
        0: df[df[label_col] == 0].groupby('case_id')[label_col].apply(list),
        1: df[df[label_col] == 1].groupby('case_id')[label_col].apply(list)
    }
'''
For the all_files it is requested - but used in all 
'''
def preprocess_data(df):
    df.reset_index(drop=True, inplace=True)
    # Group by 'case_id' and get the first filename for each group
    unique_patients = df.groupby('case_id')['image_filename'].first().reset_index()
    # Get labels for each unique patient
    labels = df.groupby('case_id')['vital_status_12'].first()
    return unique_patients, labels

'''
# Function to track and save the order of every batch
'''
def track_batch_order(train_loader, epoch, fold, actualtime,all_t):
    batch_order = []  # List to store the order of batches

    # Iterate through the batches and track the order of case IDs
    for batch_idx, batch in enumerate(train_loader):
        batch_case_ids = batch.case_id  # Assuming case_id is part of the batch
        batch_order.append((batch_idx, [case_id for case_id in batch_case_ids]))  # Store batch index and case IDs

    # Save the batch order to CSV0
    save_dir = f'check-tensorboard-univ2/logs_fold_{all_t}_{dataset}/{actualtime}'
    save_path = f"check-tensorboard-univ2/logs_fold_{all_t}_{dataset}/{actualtime}/batch_order_fold{fold}_epoch{epoch}.csv"
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Write the batch order to CSV
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Batch_Index', 'Case_IDs'])
        for batch in batch_order:
            writer.writerow([batch[0], ','.join(map(str, batch[1]))])

    print(f"Batch order for fold {fold}, epoch {epoch} saved to {save_path}")

    return batch_order

def load_checkpoint(model, optimizer, device, checkpoint_path, filename):
    """
    Load the model and optimizer state from a checkpoint file, along with the epoch number and best validation accuracy.

    Parameters:
    - model (torch.nn.Module): The model to load state into.
    - optimizer (torch.optim.Optimizer): The optimizer to load state into.
    - device (torch.device): The device to map the checkpoint to (e.g., 'cpu' or 'cuda').
    - checkpoint_path (str): Path to the directory where the checkpoint file is stored.
    - filename (str): Name of the checkpoint file. Default is 'last_checkpoint.pth.tar'.

    Returns:
    - start_epoch (int): The epoch number to resume training from.
    - best_val_acc (float): The best validation accuracy achieved so far.
    """
    # Construct full path to the checkpoint file
    checkpoint_file = os.path.join(checkpoint_path, filename)

    # Check if the checkpoint file exists
    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_file}'")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # Load state dictionaries into model and optimizer
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load additional information (epoch number, best validation accuracy)
    start_epoch = checkpoint.get('epoch', 0) + 1  # Start from the next epoch
    best_val_acc = checkpoint.get('best_val_acc', 0.0)

    print(f"Checkpoint loaded from '{checkpoint_file}' (Epoch {start_epoch - 1})")

    return start_epoch, best_val_acc


def compute_mean_std(train_path, graphs_folder,device):
    """Compute mean and standard deviation of node features in one pass."""
    total_features_sum = None
    sum_squared_diffs = None
    total_num_samples = 0

    print(train_path)
    for filename in train_path['image_filename']:
        graph_path = os.path.join(graphs_folder, filename[:-3] + '.pt')
        graph_data = torch.load(graph_path, map_location=device)
        x = graph_data.x.to(device)

        if total_features_sum is None:
            total_features_sum = torch.sum(x, dim=0)
            sum_squared_diffs = torch.zeros_like(total_features_sum)
        else:
            total_features_sum += torch.sum(x, dim=0)

        total_num_samples += x.shape[0]

    # Compute mean
    mean_features = total_features_sum / total_num_samples

    for filename in train_path['image_filename']:
        graph_path = os.path.join(graphs_folder, filename[:-3] + '.pt')
        graph_data = torch.load(graph_path, map_location=device)
        x = graph_data.x.to(device)

        squared_diffs = (x - mean_features) ** 2
        sum_squared_diffs += torch.sum(squared_diffs, dim=0)

    # Compute std deviation
    std_dev = torch.sqrt(sum_squared_diffs / total_num_samples)
    mean_features = mean_features.to('cpu')
    std_dev = std_dev.to('cpu')
    
    #print('mean',mean_features)
    #print('std',std_dev)
    return mean_features, std_dev

def compute_mean_std_hetero(train_path, graphs_folder, device):
    """Compute mean and standard deviation of node features in hetero graphs."""
    total_features_sum = None
    sum_squared_diffs = None
    total_num_samples = 0

    for filename in train_path['image_filename']:
        graph_path = os.path.join(graphs_folder, filename[:-3] + '_hetero.pt')
        graph_data = torch.load(graph_path, map_location=device)
        x = graph_data['tile'].x.to(device)

        if total_features_sum is None:
            total_features_sum = torch.sum(x, dim=0)
            sum_squared_diffs = torch.zeros_like(total_features_sum)
        else:
            total_features_sum += torch.sum(x, dim=0)

        total_num_samples += x.shape[0]

    mean_features = total_features_sum / total_num_samples

    for filename in train_path['image_filename']:
        graph_path = os.path.join(graphs_folder, filename[:-3] + '_hetero.pt')
        graph_data = torch.load(graph_path, map_location=device)
        x = graph_data['tile'].x.to(device)

        squared_diffs = (x - mean_features) ** 2
        sum_squared_diffs += torch.sum(squared_diffs, dim=0)

    std_dev = torch.sqrt(sum_squared_diffs / total_num_samples)
    return mean_features.cpu(), std_dev.cpu()


def compute_mean_std_hyperinc(train_df, graphs_root, device):
    """
    Compute feature mean/std using {case_id}_combined_hyper.pt (patient-level).
    """
    sums = None
    sums_sq = None
    count = 0

    # one file per *patient*
    seen = set()
    for _, row in train_df.iterrows():
        cid = row['case_id']
        if cid in seen:
            continue
        seen.add(cid)

        p = os.path.join(graphs_root, f"{cid}_combined_hyper.pt")
        data = torch.load(p, map_location='cpu')
        x = data.x.float()

        if sums is None:
            sums = x.sum(dim=0)
            sums_sq = (x ** 2).sum(dim=0)
            count = x.size(0)
        else:
            sums += x.sum(dim=0)
            sums_sq += (x ** 2).sum(dim=0)
            count += x.size(0)

    mean = (sums / count).cpu().numpy()
    var = (sums_sq / count - (sums / count) ** 2).clamp(min=1e-12)
    std = var.sqrt().cpu().numpy()
    return mean, std


def _selector_key_candidates(filename):
    base = os.path.basename(filename)
    root, _ = os.path.splitext(base)
    return [filename, base, root]


def _get_selector_indices(patch_selector, filename):
    if patch_selector is None:
        return None
    for key in _selector_key_candidates(filename):
        if key in patch_selector:
            return patch_selector[key]
    return None


def _normalize_keep_indices(indices, num_nodes, device=None):
    keep = torch.as_tensor(
        sorted(set(int(i) for i in indices)), dtype=torch.long, device=device
    )
    keep = keep[keep >= 0]
    keep = keep[keep < num_nodes]
    return keep


def _filter_edge_index_with_mapping(edge_index, src_map, dst_map):
    src_idx = edge_index[0]
    dst_idx = edge_index[1]
    src_new = src_map[src_idx] if src_map is not None else src_idx
    dst_new = dst_map[dst_idx] if dst_map is not None else dst_idx
    mask = (src_new >= 0) & (dst_new >= 0)
    return torch.stack([src_new[mask], dst_new[mask]], dim=0)


def apply_patch_selector_to_data(data, filename, patch_selector):
    indices = _get_selector_indices(patch_selector, filename)
    if indices is None:
        return data

    keep = _normalize_keep_indices(indices, data.x.size(0), device=data.x.device)
    if keep.numel() == 0:
        return data

    data.x = data.x[keep]
    if data.edge_index is not None and data.edge_index.numel() > 0:
        edge_index, _ = subgraph(keep, data.edge_index, relabel_nodes=True)
        data.edge_index = edge_index
        data.edge_num = edge_index.size(1)
    data.features_num = data.x.size(0)
    return data


def apply_patch_selector_to_hypergraph(data, filename, patch_selector):
    indices = _get_selector_indices(patch_selector, filename)
    if indices is None:
        return data

    original_nodes = data.x.size(0)
    keep = _normalize_keep_indices(indices, original_nodes, device=data.x.device)
    if keep.numel() == 0:
        return data

    data.x = data.x[keep]

    if data.edge_index is not None and data.edge_index.numel() > 0:
        edge_index, _ = subgraph(keep, data.edge_index, relabel_nodes=True)
        data.edge_index = edge_index
        data.edge_num = edge_index.size(1)

    if hasattr(data, "hyperedge_index") and data.hyperedge_index is not None:
        mapping = -torch.ones(int(original_nodes), dtype=torch.long, device=data.x.device)
        mapping[keep] = torch.arange(keep.numel())
        src = data.hyperedge_index[0]
        dst = data.hyperedge_index[1]
        mapped_src = mapping[src]
        mask = mapped_src >= 0
        data.hyperedge_index = torch.stack([mapped_src[mask], dst[mask]], dim=0)

    data.features_num = data.x.size(0)
    return data


def apply_patch_selector_to_heterodata(data, filename, patch_selector, node_type="tile"):
    indices = _get_selector_indices(patch_selector, filename)
    if indices is None or node_type not in data.node_types:
        return data

    keep = _normalize_keep_indices(indices, data[node_type].x.size(0), device=data[node_type].x.device)
    if keep.numel() == 0:
        return data

    tile_map = -torch.ones(
        data[node_type].x.size(0), dtype=torch.long, device=data[node_type].x.device
    )
    tile_map[keep] = torch.arange(keep.numel())

    data[node_type].x = data[node_type].x[keep]
    data[node_type].num_nodes = keep.numel()

    for et in data.edge_types:
        src_type, _, dst_type = et
        eidx = data[et].edge_index
        src_map = tile_map if src_type == node_type else None
        dst_map = tile_map if dst_type == node_type else None
        data[et].edge_index = _filter_edge_index_with_mapping(eidx, src_map, dst_map)

    return data



######################################################
# Graph Dataaset definition
######################################################    
class GraphDataset_featsnorml(torch.utils.data.Dataset):
    def __init__(self, root, df, device, mean_features=None, std_dev=None, virtual_percent=0, patch_selector=None, task='12months'):
        self.root = root
        self.df = df
        self.device = device
        self.mean_features = mean_features.clone().detach().to(device) if mean_features is not None else None
        self.std_dev = std_dev.clone().detach().to(device) if std_dev is not None else None
        self.virtual_percent = virtual_percent  # 0 = original, 5/10/20 = virtual edge percent
        self.patch_selector = patch_selector
        self.task = task
        label_col = 'vital_status_12' if task == '12months' else 'event'
        self.n_pos = (self.df[label_col] == 1).sum() if label_col in self.df else 0
        self.n_neg = (self.df[label_col] == 0).sum() if label_col in self.df else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['image_filename'].iloc[idx]
        case_id = self.df['case_id'].iloc[idx]
        if self.task == '12months':
            label = torch.tensor(self.df['vital_status_12'].iloc[idx], dtype=torch.long)
        else:
            label = torch.tensor(self.df['event'].iloc[idx], dtype=torch.float32)
            time = torch.tensor(self.df['time'].iloc[idx], dtype=torch.float32)

        # Load graph data
        graph_path = os.path.join(self.root, filename[:-3] + '.pt')
        graph_data = torch.load(graph_path, map_location=self.device)

        # Node features
        x = graph_data.x

        # Select the correct edge_index
        if self.virtual_percent == 0:
            edge_index = graph_data.edge_index  # original graph without virtual KNN edges
        else:
            attr_name = f'edge_index_{self.virtual_percent}'
            if hasattr(graph_data, attr_name):
                edge_index = getattr(graph_data, attr_name)
            else:
                raise ValueError(f"Graph does not contain edge index for {self.virtual_percent}% virtual edges.")

        # Normalize features
        if self.mean_features is not None and self.std_dev is not None:
            x_new = (x - self.mean_features) / (self.std_dev + 1e-8)
            # Mean and std across all features (vector-wise)
            #mean_all_features = x_new.mean().item()
            #std_all_features = x_new.std().item()
        
            # Mean of means across features (feature-wise mean and std)
            #mean_per_feature = x_new.mean(dim=0)
            #std_per_feature = x_new.std(dim=0)
        
            #print(f"\n[{filename}] - Mean of all x: {mean_all_features:.4f}, Std of all x: {std_all_features:.4f}")
            #print(f"[{filename}] - Mean of feature-wise means: {mean_per_feature.mean().item():.4f}, Std of feature-wise means: {mean_per_feature.std().item():.4f}")
            #print(f"[{filename}] - Mean of feature-wise stds: {std_per_feature.mean().item():.4f}, Std of feature-wise stds: {std_per_feature.std().item():.4f}")

        else:
            x_new = x  
            print('!!!!!!!!!!!!! Warining!!!!!!!!!! No normalization')

        num_edges = edge_index.shape[1] if edge_index is not None else 0
        features = x_new.shape[0] if x_new is not None else 0

        # Create Data object
        data = Data(
            x=x_new,
            edge_index=edge_index,
            y=label,
            edge_num=num_edges,
            features_num=features,
            case_id=case_id,
            image_filename=filename
        )
        if self.task == 'risk':
            data.event = label
            data.time = time
        data = apply_patch_selector_to_data(data, filename, self.patch_selector)
        return data

 ## hetero gnn 
from torch_geometric.data import HeteroData

class GraphDataset_featsnorml_hetero(torch.utils.data.Dataset):
    def __init__(self, root, df, device, mean_features=None, std_dev=None, virtual_percent=0, patch_selector=None, task='12months'):
        self.root = root
        self.df = df
        self.device = device
        self.mean_features = mean_features.to(device) if mean_features is not None else None
        self.std_dev = std_dev.to(device) if std_dev is not None else None
        self.virtual_percent = virtual_percent
        self.patch_selector = patch_selector
        self.task = task
        label_col = 'vital_status_12' if task == '12months' else 'event'
        self.n_pos = (self.df[label_col] == 1).sum() if label_col in self.df else 0
        self.n_neg = (self.df[label_col] == 0).sum() if label_col in self.df else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['image_filename'].iloc[idx]
        case_id = self.df['case_id'].iloc[idx]
        if self.task == '12months':
            label = torch.tensor(self.df['vital_status_12'].iloc[idx], dtype=torch.long)
        else:
            label = torch.tensor(self.df['event'].iloc[idx], dtype=torch.float32)
            time = torch.tensor(self.df['time'].iloc[idx], dtype=torch.float32)

        # Load HeteroData graph
        graph_path = os.path.join(self.root, filename[:-3] + '_hetero.pt')
        data = torch.load(graph_path, map_location=self.device)

        # Normalize node features
        if self.mean_features is not None and self.std_dev is not None:
            data['tile'].x = (data['tile'].x - self.mean_features) / (self.std_dev + 1e-8)
        else:
            print("⚠️ Warning: No normalization applied to", filename)

        data.y = label
        if self.task == 'risk':
            data.event = label
            data.time = time
        data.case_id = case_id
        data.image_filename = filename

        return apply_patch_selector_to_heterodata(data, filename, self.patch_selector)

from torch_geometric.data import Data
class GraphDataset_featsnorml_hyperinc(torch.utils.data.Dataset):
    """
    Loader for {case_id}_combined_hyper.pt that exposes BOTH:
      - data.edge_index        → intra adjacency (prefers edge_index, else edge_index_base)
      - data.hyperedge_index   → true hypergraph incidence (node, hyperedge_id) for the chosen percent
    and keeps your familiar metadata fields.
    """
    def __init__(self, root, df, device, mean_features, std_dev,
                 hyper_percent: int = 100, label_col: str = 'vital_status_12', patch_selector=None, task='12months'):
        import os
        self.root = root
        self.df = df.reset_index(drop=True)
        self.device = device
        self.mean = torch.tensor(mean_features, dtype=torch.float32)
        self.std = torch.tensor(std_dev, dtype=torch.float32)
        self.hyper_key = f"hyperedge_index_{int(hyper_percent)}"
        self.task = task
        self.label_col = 'vital_status_12' if task == '12months' else 'event'
        self.patch_selector = patch_selector

        self.n_pos = int((self.df[self.label_col] == 1).sum()) if self.label_col in self.df else 0
        self.n_neg = int((self.df[self.label_col] == 0).sum()) if self.label_col in self.df else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from torch_geometric.data import Data
        row = self.df.iloc[idx]
        case_id = row['case_id']
        filename = row.get('image_filename', f"{case_id}_combined_hyper.pt")

        # label as float [1]
        label_val = float(row[self.label_col]) if self.label_col in row else 0.0
        label = torch.tensor([label_val], dtype=torch.float32)
        if self.task == 'risk':
            time_val = float(row['time']) if 'time' in row else 0.0
            time = torch.tensor([time_val], dtype=torch.float32)

        graph_path = os.path.join(self.root, f"{case_id}_combined_hyper.pt")
        G = torch.load(graph_path, map_location='cpu')

        # --- features (normalize like before)
        x = G.x.float()
        x_new = (x - self.mean) / (self.std + 1e-8)
        features = x_new.size(0)  # keep same convention you used earlier (num nodes)

        # --- INTRA adjacency for standard GNNs (prefer edge_index; fallback to edge_index_base)
        if hasattr(G, 'edge_index') and G.edge_index is not None:
            edge_index_intra = G.edge_index.long().contiguous()
        elif hasattr(G, 'edge_index_base') and G.edge_index_base is not None:
            edge_index_intra = G.edge_index_base.long().contiguous()
        else:
            edge_index_intra = torch.empty((2, 0), dtype=torch.long)

        # --- HYPER incidence for HypergraphConv
        if not hasattr(G, self.hyper_key):
            # try a robust fallback (e.g., use 100% if exact percent missing)
            if hasattr(G, 'hyperedge_index_100'):
                hyperedge_index = getattr(G, 'hyperedge_index_100')
            else:
                hyperedge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            hyperedge_index = getattr(G, self.hyper_key)

        hyperedge_index = hyperedge_index.long().contiguous() if hyperedge_index.numel() > 0 \
                          else torch.empty((2, 0), dtype=torch.long)

        # counts (keep your old keys, plus add a hyper count)
        num_edges_intra = edge_index_intra.size(1)
        num_edges_hyper = hyperedge_index.size(1)

        data = Data(
            x=x_new,
            edge_index=edge_index_intra,      # intra adjacency here
            y=label,
            edge_num=num_edges_intra,         # keep your old name for compatibility
            features_num=features,
            case_id=case_id,
            image_filename=filename
        )
        # attach hyper incidence too
        data.hyperedge_index = hyperedge_index
        if self.task == 'risk':
            data.event = label
            data.time = time

        if hasattr(G, 'pos'):
            data.pos = G.pos

        # (optional) also expose clique-expanded edges if you want them later:
        # key_clique = f"edge_index_hyper_clique_{self.hyper_key.split('_')[-1]}"
        # if hasattr(G, key_clique): data.edge_index_hyper_clique = getattr(G, key_clique)

        return apply_patch_selector_to_hypergraph(data, filename, self.patch_selector)

'''class GraphDataset_featsnorml_hyperinc(torch.utils.data.Dataset):
    """
    Patient-level loader for {case_id}_combined_hyper.pt using INCIDENCE (node, hyperedge_id).
    Returns Data with the same fields you had before:
        x=x_new, edge_index=edge_index (INCIDENCE), y=label,
        edge_num=num_edges, features_num=features, case_id=case_id, image_filename=filename
    """
    def __init__(self, root, df, device, mean_features, std_dev,
                 hyper_percent: int = 100, label_col: str = 'vital_status_12'):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.device = device
        self.mean = torch.tensor(mean_features, dtype=torch.float32)
        self.std = torch.tensor(std_dev, dtype=torch.float32)
        self.hyper_key = f"hyperedge_index_{int(hyper_percent)}"
        self.label_col = label_col

        # optional: simple counters you used before
        self.n_pos = int((self.df[self.label_col] == 1).sum()) if self.label_col in self.df else 0
        self.n_neg = int((self.df[self.label_col] == 0).sum()) if self.label_col in self.df else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = row['case_id']
        filename = row.get('image_filename', f"{case_id}_combined_hyper.pt")

        # label (float [1]) to match your existing loops
        if self.label_col in row:
            label_val = float(row[self.label_col])
        else:
            # fallback if you're in a multi-label setting; adapt as needed
            label_val = 0.0
        label = torch.tensor([label_val], dtype=torch.float32)

        graph_path = os.path.join(self.root, f"{case_id}_combined_hyper.pt")
        data_disk = torch.load(graph_path, map_location='cpu')

        if not hasattr(data_disk, self.hyper_key):
            raise KeyError(f"{graph_path} missing attribute {self.hyper_key}")

        # === features: normalize like before ===
        x = data_disk.x.float()
        x_new = (x - self.mean) / (self.std + 1e-8)
        features = x_new.size(1)

        # === edge_index: use INCIDENCE as requested ===
        edge_index = getattr(data_disk, self.hyper_key)
        if edge_index.numel() == 0:
            # keep shape (2,0) and dtype long
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = edge_index.long().contiguous()
        num_edges = edge_index.size(1)  # number of (node, hyperedge) memberships

        # Build Data exactly with your keys
        data = Data(
            x=x_new,
            edge_index=edge_index,     # INCIDENCE, not clique
            y=label,
            edge_num=num_edges,
            features_num=features,
            case_id=case_id,
            image_filename=filename
        )

        # keep pos if you sometimes need it (won't break anything)
        if hasattr(data_disk, 'pos'):
            data.pos = data_disk.pos

        return data'''



########################################
#Patientwsisummary 
########################################
class PatientWSIPackDataset_featsnorml(torch.utils.data.Dataset):
    """
    One item = one patient (case_id).
    Packs all that patient's WSIs into a single torch_geometric Batch.
    Uses ONLY adjacency edges stored as `edge_index` in each per-WSI .pt file.

    Returns a Batch whose stacked item graphs each have:
        x, edge_index, y, edge_num, features_num, case_id, image_filename
    And the Batch has two convenience attributes:
        .case_id (str), .num_wsis (int)
    """

    def __init__(self, root, df_patients, df_wsis, device,
                 mean_features=None, std_dev=None, virtual_percent=0, patch_selector=None):
        """
        root            : folder with per-WSI .pt graphs (adjacency only)
        df_patients     : DataFrame with at least ['case_id','vital_status_12'] (one row per patient)
        df_wsis         : DataFrame listing WSIs per patient (has ['case_id','image_filename'])
        device          : torch device to place tensors
        mean_features   : tensor/ndarray/list of shape [F] (feature means)
        std_dev         : tensor/ndarray/list of shape [F] (feature stds)
        virtual_percent : ignored here (kept for API compatibility)
        """
        self.root = root
        self.device = device

        # store normalization tensors (if provided)
        def _to_tensor(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.to(device=device, dtype=torch.float32)
            return torch.tensor(x, dtype=torch.float32, device=device)

        self.mean_features = _to_tensor(mean_features)
        self.std_dev       = _to_tensor(std_dev)
        self.patch_selector = patch_selector

        # one row per patient with label
        need_cols = ['case_id', 'vital_status_12']
        for c in need_cols:
            if c not in df_patients.columns:
                raise ValueError(f"df_patients must contain column '{c}'")
        self.df = df_patients[need_cols].drop_duplicates('case_id').reset_index(drop=True)

        # build case_id -> list of per-WSI .pt paths (robust mapping)
        self.case_to_files = {}
        for _, r in df_wsis.iterrows():
            cid = str(r['case_id'])
            fn  = str(r['image_filename'])
            cand = []

            # 1) direct .pt path under root
            if fn.endswith('.pt'):
                p = os.path.join(self.root, fn)
                if os.path.exists(p): cand.append(p)

            # 2) replace extension by .pt under root
            base = os.path.splitext(fn)[0]
            p2 = os.path.join(self.root, base + '.pt')
            if os.path.exists(p2) and p2 not in cand: cand.append(p2)

            # 3) fallback: any file that starts with case_id
            if not cand:
                for g in glob.glob(os.path.join(self.root, f"{cid}*.pt")):
                    cand.append(g)

            if cand:
                self.case_to_files.setdefault(cid, []).extend(cand)

        # deduplicate paths per patient
        for cid, paths in list(self.case_to_files.items()):
            uniq = []
            seen = set()
            for p in paths:
                if p not in seen:
                    uniq.append(p); seen.add(p)
            self.case_to_files[cid] = uniq

        # keep only patients that actually have files
        self.df = self.df[self.df['case_id'].isin(self.case_to_files.keys())].reset_index(drop=True)

        # class counts for your samplers/metrics
        self.n_pos = int((self.df['vital_status_12'] == 1).sum())
        self.n_neg = int((self.df['vital_status_12'] == 0).sum())

        if len(self.df) == 0:
            raise RuntimeError(
                f"[PatientWSIPackDataset_featsnorml] No patients resolved to .pt files under '{self.root}'. "
                f"Check filename→.pt mapping or the root path."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cid = str(row['case_id'])
        label = torch.tensor(int(row['vital_status_12']), dtype=torch.long, device=self.device)

        data_list = []
        for fpath in self.case_to_files[cid]:
            G = torch.load(fpath, map_location=self.device)

            # features (+ optional normalization)
            x = G.x.float()
            if (self.mean_features is not None) and (self.std_dev is not None):
                x = (x - self.mean_features) / (self.std_dev + 1e-8)

            # adjacency only
            edge_index = getattr(G, 'edge_index',
                                 torch.empty((2, 0), dtype=torch.long, device=self.device))

            num_edges    = int(edge_index.size(1))
            features_num = int(x.size(0))   # number of nodes

            d = Data(
                x=x,
                edge_index=edge_index,
                y=label,                              # one label for all WSIs of this patient
                edge_num=num_edges,
                features_num=features_num,
                case_id=cid,
                image_filename=os.path.basename(fpath)
            )
            # keep pos if present (useful for diagnostics/plots)
            if hasattr(G, 'pos'):
                d.pos = G.pos
            d = apply_patch_selector_to_data(d, os.path.basename(fpath), self.patch_selector)
            data_list.append(d)

        B = Batch.from_data_list(data_list)  # stacks WSIs; node indices get auto-offset
        B.case_id = cid
        B.num_wsis = len(data_list)
        B.y = label.view(1, 1).float()
        return B
# -------- dataset (patient → packed WSIs) --------
'''class PatientWSIPackDataset_featsnorml(torch.utils.data.Dataset):
    """
    One item = one patient (case_id).
    Packs ALL that patient's WSIs into a single `Batch`.
    This variant ALWAYS uses the per-WSI adjacency `edge_index` (no inter/virtual edges).
    Matches your original Data keys for compatibility.
    """
    def __init__(self, root, df_patients, df_wsis, device,
                 mean_features=None, std_dev=None):
        self.root   = root
        self.device = device

        # normalization tensors (keep behavior consistent with your previous classes)
        if mean_features is not None:
            self.mean_features = mean_features.clone().detach().to(device)
        else:
            self.mean_features = None
        if std_dev is not None:
            self.std_dev = std_dev.clone().detach().to(device)
        else:
            self.std_dev = None

        # one row per patient with label
        self.df = (df_patients[['image_filename', 'vital_status_12', 'case_id']]
                   .drop_duplicates('case_id')
                   .reset_index(drop=True))

        # robust mapping case_id → list of .pt files under `root`
        self.case_to_files = {}
        for _, r in df_wsis.iterrows():
            cid = str(r['case_id'])
            fn  = str(r['image_filename'])

            candidates = []

            # 1) CSV already points to .pt within root
            if fn.endswith('.pt'):
                p = os.path.join(self.root, fn)
                if os.path.exists(p):
                    candidates.append(p)

            # 2) swap extension → .pt
            base_no_ext, _ = os.path.splitext(fn)
            p2 = os.path.join(self.root, base_no_ext + '.pt')
            if os.path.exists(p2) and p2 not in candidates:
                candidates.append(p2)

            # 3) fallback: any {case_id}*.pt
            if not candidates:
                for g in glob.glob(os.path.join(self.root, f"{cid}*.pt")):
                    candidates.append(g)

            if candidates:
                self.case_to_files.setdefault(cid, []).extend(candidates)

        # de-duplicate paths per patient
        for cid, paths in list(self.case_to_files.items()):
            uniq = []
            seen = set()
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    uniq.append(p)
            self.case_to_files[cid] = uniq

        # keep only patients that actually resolved to files
        self.df = self.df[self.df['case_id'].isin(self.case_to_files.keys())].reset_index(drop=True)

        # class counts (patient-level) for optional sampler
        self.n_pos = int((self.df['vital_status_12'] == 1).sum())
        self.n_neg = int((self.df['vital_status_12'] == 0).sum())

        if len(self.df) == 0:
            raise RuntimeError(
                f"[PatientWSIPackDataset_featsnorml] No patients resolved to .pt under '{self.root}'. "
                f"Check the root path and filename→.pt mapping."
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        cid   = str(row['case_id'])
        label = torch.tensor(int(row['vital_status_12']), dtype=torch.long, device=self.device)

        data_list = []
        for fpath in self.case_to_files[cid]:
            G = torch.load(fpath, map_location=self.device)

            # features
            x = G.x
            if (self.mean_features is not None) and (self.std_dev is not None):
                x_new = (x - self.mean_features) / (self.std_dev + 1e-8)
            else:
                x_new = x  # allowed

            # EDGES: ADJACENCY ONLY (always use G.edge_index)
            edge_index = getattr(G, 'edge_index',
                                 torch.empty((2, 0), dtype=torch.long, device=self.device))

            num_edges    = int(edge_index.size(1)) if edge_index is not None else 0
            features_num = int(x_new.size(0)) if x_new is not None else 0  # count of nodes

            d = Data(
                x=x_new,
                edge_index=edge_index,
                y=label,
                edge_num=num_edges,
                features_num=features_num,
                case_id=cid,
                image_filename=os.path.basename(fpath)
            )
            # keep pos if present (optional)
            if hasattr(G, 'pos'):
                d.pos = G.pos
            data_list.append(d)

        # Pack all WSIs of that patient in a single Batch
        B = Batch.from_data_list(data_list)
        B.case_id  = cid
        B.num_wsis = len(data_list)
        return B'''
'''class PatientWSIPackDataset_featsnorml(torch.utils.data.Dataset):
        from torch_geometric.data import Data, Batch
        row = self.df.iloc[idx]
        cid = str(row['case_id'])
        label = torch.tensor(int(row['vital_status_12']), dtype=torch.long, device=self.device)

        data_list = []
        for fpath in self.case_to_files[cid]:
            G = torch.load(fpath, map_location=self.device)

            # Features
            x = G.x
            if (self.mean_features is not None) and (self.std_dev is not None):
                x_new = (x - self.mean_features) / (self.std_dev + 1e-8)
            else:
                x_new = x  # fine; you already print a warning elsewhere if desired

            # Edges: mirror your virtual_percent behavior
            if self.virtual_percent == 0:
                edge_index = getattr(G, 'edge_index',
                                     torch.empty((2, 0), dtype=torch.long, device=self.device))
            else:
                attr_name = f'edge_index_{self.virtual_percent}'
                if hasattr(G, attr_name):
                    edge_index = getattr(G, attr_name)
                else:
                    raise ValueError(f"{os.path.basename(fpath)} missing {attr_name}")

            num_edges = int(edge_index.size(1)) if edge_index is not None else 0
            # Keep this consistent with your other dataset: count of nodes
            features_num = int(x_new.size(0)) if x_new is not None else 0

            d = Data(
                x=x_new,
                edge_index=edge_index,
                y=label,
                edge_num=num_edges,
                features_num=features_num,
                case_id=cid,
                image_filename=os.path.basename(fpath)
            )
            data_list.append(d)

        B = Batch.from_data_list(data_list)
        B.case_id = cid
        B.num_wsis = len(data_list)
        return B'''
        




######################################################
# Graph Architectures
######################################################
'''https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing#scrollTo=ecJCNRmT2RsF
 As multiple papers pointed out (Xu et al. (2018), Morris et al. (2018)), applying neighborhood normalization decreases the expressivity of GNNs in distinguishing certain graph structures. An alternative formulation (Morris et al. (2018)) omits neighborhood normalization completely and adds a simple skip-connection to the GNN layer in order to preserve central node information:'''
# Updated GNN
from torch_geometric.nn import HeteroConv, GCNConv, global_max_pool
import torch.nn.functional as F
from torch.nn import Linear, Dropout, GroupNorm

######################
# Hyper
#######################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, HypergraphConv, global_max_pool

def _safe_gn(num_channels: int, max_groups: int = 16) -> nn.GroupNorm:
    # Use up to 16 groups, but make sure it divides num_channels; fallback to 1 group.
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    return nn.GroupNorm(1, num_channels)  # ultra-safe fallback


class HyperGCN_survival(nn.Module):
    """
    Hybrid: GCN over intra adjacency + HypergraphConv over incidence, summed per layer.
    Forward: (x, edge_index, hyperedge_index, batch) → logits
    """
    def __init__(self, in_channels, h1, h2, h3, out_channels, dropout_rate=0.3):
        super().__init__()
        # Intra (GCN)
        self.gcn1 = GCNConv(in_channels, h1)
        self.gcn2 = GCNConv(h1, h2)
        self.gcn3 = GCNConv(h2, h3)

        # Hyper (no attention → no hyperedge_attr required)
        self.hyp1 = HypergraphConv(in_channels, h1, use_attention=False)
        self.hyp2 = HypergraphConv(h1,         h2, use_attention=False)
        self.hyp3 = HypergraphConv(h2,         h3, use_attention=False)

        self.bn1  = _safe_gn(h1)
        self.bn2  = _safe_gn(h2)
        self.bn3  = _safe_gn(h3)
        self.lin  = nn.Linear(h3, out_channels)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, hyperedge_index, batch):
        # L1
        x_intra = self.gcn1(x, edge_index)
        x_hyper = self.hyp1(x, hyperedge_index)
        x = self.bn1(x_intra + x_hyper).relu(); x = self.dropout(x)

        # L2
        x_intra = self.gcn2(x, edge_index)
        x_hyper = self.hyp2(x, hyperedge_index)
        x = self.bn2(x_intra + x_hyper).relu(); x = self.dropout(x)

        # L3
        x_intra = self.gcn3(x, edge_index)
        x_hyper = self.hyp3(x, hyperedge_index)
        x = self.bn3(x_intra + x_hyper).relu(); x = self.dropout(x)

        x = global_max_pool(x, batch)
        return self.lin(x)


class HyperGAT_survival(nn.Module):
    """
    Hybrid: GAT over intra adjacency + HypergraphConv(attention) over incidence, summed per layer.
    Returns logits only (PyG does not return hyper attention weights).
    Forward: (x, edge_index, hyperedge_index, batch) → logits
    """
    def __init__(self, in_channels, hidden1, hidden2, hidden3,
                 out_channels, heads1, heads2, heads3, dropout_rate=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Intra (GAT)
        self.gat1 = GATConv(in_channels,              hidden1, heads=heads1, concat=True)
        self.gat2 = GATConv(hidden1 * heads1,         hidden2, heads=heads2, concat=True)
        self.gat3 = GATConv(hidden2 * heads2,         hidden3, heads=heads3, concat=True)

        # Hyper (with attention) — dims match GAT branch for sum-fusion
        self.hyp1 = HypergraphConv(in_channels,              hidden1, use_attention=True, heads=heads1, concat=True)
        self.hyp2 = HypergraphConv(hidden1 * heads1,         hidden2, use_attention=True, heads=heads2, concat=True)
        self.hyp3 = HypergraphConv(hidden2 * heads2,         hidden3, use_attention=True, heads=heads3, concat=True)

        self.bn1 = _safe_gn(hidden1 * heads1)
        self.bn2 = _safe_gn(hidden2 * heads2)
        self.bn3 = _safe_gn(hidden3 * heads3)
        self.lin = nn.Linear(hidden3 * heads3, out_channels)

    @staticmethod
    def _mean_hyperedge_attr(x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Build per-hyperedge features by mean pooling incident node features.
        Shape matches the node-channel at this layer, satisfying HypergraphConv(attention) needs.
        """
        if hyperedge_index.numel() == 0:
            return x.new_zeros((0, x.size(1)))
        v, e = hyperedge_index[0], hyperedge_index[1]
        H = int(e.max().item()) + 1
        he_sum = x.new_zeros((H, x.size(1)))
        he_cnt = x.new_zeros((H, 1))
        he_sum.index_add_(0, e, x[v])
        he_cnt.index_add_(0, e, torch.ones((e.numel(), 1), device=x.device, dtype=x.dtype))
        return he_sum / he_cnt.clamp_min(1.0)

    def forward(self, x, edge_index, hyperedge_index, batch):
        # L1
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x_intra = self.gat1(x, edge_index)
        x_hyper = self.hyp1(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn1(x_intra + x_hyper).relu(); x = self.dropout(x)

        # L2
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x_intra = self.gat2(x, edge_index)
        x_hyper = self.hyp2(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn2(x_intra + x_hyper).relu(); x = self.dropout(x)

        # L3
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x_intra = self.gat3(x, edge_index)
        x_hyper = self.hyp3(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn3(x_intra + x_hyper).relu(); x = self.dropout(x)

        x = global_max_pool(x, batch)
        return self.lin(x)


'''class HyperGCN_survival(torch.nn.Module):
    """
    3× HypergraphConv on incidence matrix (node, hyperedge_id).
    Forward signature stays (x, edge_index, batch) so train loop is unchanged.
    """
    def __init__(self, in_channels, h1, h2, h3, out_channels, dropout_rate=0.3):
        super().__init__()
        self.conv1 = HypergraphConv(in_channels, h1)
        self.bn1 = torch.nn.GroupNorm(16, h1)
        self.conv2 = HypergraphConv(h1, h2)
        self.bn2 = torch.nn.GroupNorm(16, h2)
        self.conv3 = HypergraphConv(h2, h3)
        self.bn3 = torch.nn.GroupNorm(16, h3)
        self.lin = torch.nn.Linear(h3, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, hyperedge_index, batch):
        x = self.conv1(x, hyperedge_index); x = self.bn1(x).relu(); x = self.dropout(x)
        x = self.conv2(x, hyperedge_index); x = self.bn2(x).relu(); x = self.dropout(x)
        x = self.conv3(x, hyperedge_index); x = self.bn3(x).relu(); x = self.dropout(x)
        x = global_max_pool(x, batch)
        return self.lin(x)'''
'''class HyperGCN_survival(torch.nn.Module):
    """
    Hybrid: GCN over intra adjacency + HypergraphConv over incidence, summed per layer.
    Forward: (x, edge_index, hyperedge_index, batch) → logits
    """
    def __init__(self, in_channels, h1, h2, h3, out_channels, dropout_rate=0.3):
        super().__init__()
        self.gcn1 = GCNConv(in_channels, h1)
        self.hyp1 = HypergraphConv(in_channels, h1)
        self.bn1  = torch.nn.GroupNorm(16, h1)

        self.gcn2 = GCNConv(h1, h2)
        self.hyp2 = HypergraphConv(h1, h2)
        self.bn2  = torch.nn.GroupNorm(16, h2)

        self.gcn3 = GCNConv(h2, h3)
        self.hyp3 = HypergraphConv(h2, h3)
        self.bn3  = torch.nn.GroupNorm(16, h3)

        self.lin  = torch.nn.Linear(h3, out_channels)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, hyperedge_index, batch):
        x1 = self.gcn1(x, edge_index)
        x1 = x1 + self.hyp1(x, hyperedge_index)
        x  = self.bn1(x1).relu(); x = self.dropout(x)

        x2 = self.gcn2(x, edge_index)
        x2 = x2 + self.hyp2(x, hyperedge_index)
        x  = self.bn2(x2).relu(); x = self.dropout(x)

        x3 = self.gcn3(x, edge_index)
        x3 = x3 + self.hyp3(x, hyperedge_index)
        x  = self.bn3(x3).relu(); x = self.dropout(x)

        x  = global_max_pool(x, batch)
        return self.lin(x)'''


## PGY DOES NOT SUPPORT YEY VISUALLIZATION OF WEIGHT TENSORS
'''class HyperGAT_survival(nn.Module):
    """
    3× HypergraphConv with attention over the INCIDENCE matrix (node, hyperedge_id).
    Returns (logits, attn_weights) when return_attn=True, else just logits.
    attn_weights is a list of layer-wise dicts: [{'hyper': (edge_index, alpha)}, ...]
    where 'edge_index' is the incidence used that layer and 'alpha' are attention scores.
    """
    def __init__(self, in_channels, hidden1, hidden2, hidden3,
                 out_channels, heads1, heads2, heads3,
                 dropout_rate=0.3, return_attn=False):
        super().__init__()
        torch.manual_seed(47)
        self.dropout = nn.Dropout(p=dropout_rate)

        # Attention-enabled HypergraphConv (concat=True → output dim = hidden*heads)
        self.h1 = HypergraphConv(in_channels,            hidden1, use_attention=True, heads=heads1, concat=True)
        self.bn1 = nn.GroupNorm(16, hidden1 * heads1)

        self.h2 = HypergraphConv(hidden1 * heads1,       hidden2, use_attention=True, heads=heads2, concat=True)
        self.bn2 = nn.GroupNorm(16, hidden2 * heads2)

        self.h3 = HypergraphConv(hidden2 * heads2,       hidden3, use_attention=True, heads=heads3, concat=True)
        self.bn3 = nn.GroupNorm(16, hidden3 * heads3)

        self.lin = nn.Linear(hidden3 * heads3, out_channels)

    @staticmethod
    def _mean_hyperedge_attr(x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        """
        Build per-hyperedge features by mean pooling incident node features.
        x: [N, C], hyperedge_index: [2, M] with rows (node_idx, hyperedge_id).
        Returns: he_attr [H, C], where H = #hyperedges.
        """
        device, dtype = x.device, x.dtype
        C = x.size(1)
        if hyperedge_index.numel() == 0:
            return x.new_zeros((0, C))  # no hyperedges → empty attr, still valid

        v = hyperedge_index[0].to(device)  # node indices
        e = hyperedge_index[1].to(device)  # hyperedge ids
        H = int(e.max().item()) + 1

        he_sum = x.new_zeros((H, C))
        he_cnt = x.new_zeros((H, 1))
        he_sum.index_add_(0, e, x[v])
        he_cnt.index_add_(0, e, torch.ones((e.numel(), 1), device=device, dtype=dtype))
        he_attr = he_sum / he_cnt.clamp_min(1.0)
        return he_attr

    def forward(self, x, hyperedge_index, batch):
        # L1 (hyperedge features must match current node-channel C)
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x = self.h1(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn1(x).relu(); x = self.dropout(x)

        # L2
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x = self.h2(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn2(x).relu(); x = self.dropout(x)

        # L3
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x = self.h3(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn3(x).relu(); x = self.dropout(x)

        x = global_max_pool(x, batch)
        return self.lin(x)'''

'''class HyperGAT_survival(nn.Module):
    """
    Hybrid: GAT over intra adjacency + HypergraphConv(attn) over incidence, summed per layer.
    Returns logits only (PyG doesn't expose hyper attention weights).
    """
    def __init__(self, in_channels, hidden1, hidden2, hidden3,
                 out_channels, heads1, heads2, heads3, dropout_rate=0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Intra (GAT)
        self.gat1 = GATConv(in_channels, hidden1, heads=heads1, concat=True)
        self.gat2 = GATConv(hidden1*heads1, hidden2, heads=heads2, concat=True)
        self.gat3 = GATConv(hidden2*heads2, hidden3, heads=heads3, concat=True)

        # Hyper (HypergraphConv with attention)
        self.hyp1 = HypergraphConv(in_channels,            hidden1, use_attention=True, heads=heads1, concat=True)
        self.hyp2 = HypergraphConv(hidden1*heads1,         hidden2, use_attention=True, heads=heads2, concat=True)
        self.hyp3 = HypergraphConv(hidden2*heads2,         hidden3, use_attention=True, heads=heads3, concat=True)

        self.bn1 = nn.GroupNorm(16, hidden1*heads1)
        self.bn2 = nn.GroupNorm(16, hidden2*heads2)
        self.bn3 = nn.GroupNorm(16, hidden3*heads3)
        self.lin = nn.Linear(hidden3*heads3, out_channels)

    @staticmethod
    def _mean_hyperedge_attr(x: torch.Tensor, hyperedge_index: torch.Tensor) -> torch.Tensor:
        # builds per-hyperedge features by mean pooling incident node features
        if hyperedge_index.numel() == 0:
            return x.new_zeros((0, x.size(1)))
        v, e = hyperedge_index[0], hyperedge_index[1]
        H = int(e.max().item()) + 1
        he_sum = x.new_zeros((H, x.size(1)))
        he_cnt = x.new_zeros((H, 1))
        he_sum.index_add_(0, e, x[v])
        he_cnt.index_add_(0, e, torch.ones((e.numel(), 1), device=x.device, dtype=x.dtype))
        return he_sum / he_cnt.clamp_min(1.0)

    def forward(self, x, edge_index, hyperedge_index, batch):
        # L1
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x_intra = self.gat1(x, edge_index)
        x_hyper = self.hyp1(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn1(x_intra + x_hyper).relu(); x = self.dropout(x)

        # L2
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x_intra = self.gat2(x, edge_index)
        x_hyper = self.hyp2(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn2(x_intra + x_hyper).relu(); x = self.dropout(x)

        # L3
        he_attr = self._mean_hyperedge_attr(x, hyperedge_index)
        x_intra = self.gat3(x, edge_index)
        x_hyper = self.hyp3(x, hyperedge_index, hyperedge_attr=he_attr)
        x = self.bn3(x_intra + x_hyper).relu(); x = self.dropout(x)

        x = global_max_pool(x, batch)
        return self.lin(x)'''
            
###############################
# Hetero
###############################
class HeteroGCN_survival(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, hidden3, out_channels,selected_inter,dropout_rate=0.3):
        super().__init__()

        self.dropout = Dropout(p=dropout_rate)
        self.selected_inter = selected_inter  # e.g., 'inter_5', 'inter_10', ...
        
        self.conv1 = HeteroConv({
            ('tile', 'intra', 'tile'): GCNConv(in_channels, hidden1),
            ('tile', selected_inter, 'tile'): GCNConv(in_channels, hidden1),
        }, aggr='sum')
        self.bn1 = GroupNorm(16, hidden1)

        self.conv2 = HeteroConv({
            ('tile', 'intra', 'tile'): GCNConv(hidden1, hidden2),
            ('tile', selected_inter, 'tile'): GCNConv(hidden1, hidden2),
        }, aggr='sum')
        self.bn2 = GroupNorm(16, hidden2)

        self.conv3 = HeteroConv({
            ('tile', 'intra', 'tile'): GCNConv(hidden2, hidden3),
            ('tile', selected_inter, 'tile'): GCNConv(hidden2, hidden3),
        }, aggr='sum')
        self.bn3 = GroupNorm(16, hidden3)

        self.lin = Linear(hidden3, out_channels)

    def forward(self, x_dict, edge_index_dict, batch):
        x = x_dict['tile']
        inter = self.selected_inter

        x = self.conv1({'tile': x}, {
            ('tile', 'intra', 'tile'): edge_index_dict[('tile', 'intra', 'tile')],
            ('tile', inter, 'tile'): edge_index_dict[('tile', inter, 'tile')],
        })['tile']
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2({'tile': x}, {
            ('tile', 'intra', 'tile'): edge_index_dict[('tile', 'intra', 'tile')],
            ('tile', inter, 'tile'): edge_index_dict[('tile', inter, 'tile')],
        })['tile']
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3({'tile': x}, {
            ('tile', 'intra', 'tile'): edge_index_dict[('tile', 'intra', 'tile')],
            ('tile', inter, 'tile'): edge_index_dict[('tile', inter, 'tile')],
        })['tile']
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_max_pool(x, batch)
        return self.lin(x)


class HeteroGAT_survival(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, hidden3,
                 out_channels,selected_inter, heads1=1, heads2=1, heads3=1, dropout_rate=0.3, return_attn=True):
        super().__init__()
        torch.manual_seed(47)
        self.return_attn = return_attn
        self.dropout = Dropout(p=dropout_rate)
        self.selected_inter = selected_inter
        # Layer 1
        self.gat_intra_1 = GATConv(in_channels, hidden1, heads=heads1, concat=True)
        self.gat_inter_1 = GATConv(in_channels, hidden1, heads=heads1, concat=True)
        self.bn1 = GroupNorm(16, hidden1 * heads1)

        # Layer 2
        self.gat_intra_2 = GATConv(hidden1 * heads1, hidden2, heads=heads2, concat=True)
        self.gat_inter_2 = GATConv(hidden1 * heads1, hidden2, heads=heads2, concat=True)
        self.bn2 = GroupNorm(16, hidden2 * heads2)

        # Layer 3
        self.gat_intra_3 = GATConv(hidden2 * heads2, hidden3, heads=heads3, concat=True)
        self.gat_inter_3 = GATConv(hidden2 * heads2, hidden3, heads=heads3, concat=True)
        self.bn3 = GroupNorm(16, hidden3 * heads3)

        # Classifier
        self.lin = Linear(hidden3 * heads3, out_channels)

    def forward(self, x_dict, edge_index_dict, batch):
        x = x_dict['tile']
        inter = self.selected_inter
        attn_weights = [] if self.return_attn else None

        # Layer 1
        if self.return_attn:
            out_intra1, attn_intra1 = self.gat_intra_1(x, edge_index_dict[('tile', 'intra', 'tile')], return_attention_weights=True)
            out_inter1, attn_inter1 = self.gat_inter_1(x, edge_index_dict[('tile', inter, 'tile')], return_attention_weights=True)
            attn_weights.append({'intra': attn_intra1, 'inter': attn_inter1})
        else:
            out_intra1 = self.gat_intra_1(x, edge_index_dict[('tile', 'intra', 'tile')])
            out_inter1 = self.gat_inter_1(x, edge_index_dict[('tile', inter, 'tile')])
        x = out_intra1 + out_inter1
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        if self.return_attn:
            out_intra2, attn_intra2 = self.gat_intra_2(x, edge_index_dict[('tile', 'intra', 'tile')], return_attention_weights=True)
            out_inter2, attn_inter2 = self.gat_inter_2(x, edge_index_dict[('tile', inter, 'tile')], return_attention_weights=True)
            attn_weights.append({'intra': attn_intra2, 'inter': attn_inter2})
        else:
            out_intra2 = self.gat_intra_2(x, edge_index_dict[('tile', 'intra', 'tile')])
            out_inter2 = self.gat_inter_2(x, edge_index_dict[('tile', inter, 'tile')])
        x = out_intra2 + out_inter2
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3
        if self.return_attn:
            out_intra3, attn_intra3 = self.gat_intra_3(x, edge_index_dict[('tile', 'intra', 'tile')], return_attention_weights=True)
            out_inter3, attn_inter3 = self.gat_inter_3(x, edge_index_dict[('tile', inter, 'tile')], return_attention_weights=True)
            attn_weights.append({'intra': attn_intra3, 'inter': attn_inter3})
        else:
            out_intra3 = self.gat_intra_3(x, edge_index_dict[('tile', 'intra', 'tile')])
            out_inter3 = self.gat_inter_3(x, edge_index_dict[('tile', inter, 'tile')])
        x = out_intra3 + out_inter3
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_max_pool(x, batch)
        logits = self.lin(x)

        if self.return_attn:
            return logits, attn_weights
        else:
            return logits

       

##############################
# Homo gnn
##############################
            
class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes,dropout_rate=0.3):
        super(GNN, self).__init__()
        torch.manual_seed(47)
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.GroupNorm(16,hidden_channels)#, track_running_stats=True)
        self.conv2 = GraphConv(hidden_channels, hidden_2)
        self.bn2 = torch.nn.GroupNorm(16,hidden_2)#, track_running_stats=True)
        self.conv3 = GraphConv(hidden_2, hidden_3)
        self.bn3 =torch.nn.GroupNorm(16,hidden_3)#, track_running_stats=True)
        self.lin = Linear(hidden_3, num_classes)
        self.dropout =torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x


class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes, dropout_rate=0.3):
        super(GIN, self).__init__()
        torch.manual_seed(47)
        self.conv1 = GINConv(nn=Sequential(Linear(num_node_features, hidden_channels), ReLU()))
        self.bn1 =torch.nn.GroupNorm(16,hidden_channels)#, track_running_stats=True)
        self.conv2 = GINConv(nn=Sequential(Linear(hidden_channels, hidden_2), ReLU()))
        self.bn2 =torch.nn.GroupNorm(16,hidden_2)#, track_running_stats=True)
        self.conv3 = GINConv(nn=Sequential(Linear(hidden_2, hidden_3)))
        self.bn3 =torch.nn.GroupNorm(16,hidden_3)#, track_running_stats=True)
        self.lin = Linear(hidden_3, num_classes)
        self.dropout =torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x

class GCN_survival(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes, dropout_rate=0.3):
        super(GCN_survival, self).__init__()
        torch.manual_seed(47)

        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.GroupNorm(16,hidden_channels)#, track_running_stats=True)  # BatchNorm after GCN layer
        self.conv2 = GCNConv(hidden_channels, hidden_2)
        self.bn2 = torch.nn.GroupNorm(16,hidden_2)#, track_running_stats=True)
        self.conv3 = GCNConv(hidden_2, hidden_3)
        self.bn3 = torch.nn.GroupNorm(16,hidden_3)#, track_running_stats=True)

        # Fully Connected Linear Layer
        self.lin = Linear(hidden_3, num_classes)

        # Dropout
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x, edge_index, batch):
        # First GCN Layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)  # Apply Batch Normalization
        x = x.relu()
        x = self.dropout(x)  # Apply Dropout

        # Second GCN Layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)

        # Third GCN Layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)

        # Pooling and Final Linear Layer
        x = global_max_pool(x, batch)  # Pooling across nodes in the graph
        x = self.lin(x)

        return x

class GAT_survival(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3,
                 num_classes, heads, heads_2, heads_3, dropout_rate=0.3, return_attn=True):
        super(GAT_survival, self).__init__()
        torch.manual_seed(47)
        self.return_attn = return_attn

        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads)
        self.bn1 = GroupNorm(16, hidden_channels * heads)

        self.conv2 = GATConv(hidden_channels * heads, hidden_2, heads=heads_2)
        self.bn2 = GroupNorm(16, hidden_2 * heads_2)

        self.conv3 = GATConv(hidden_2 * heads_2, hidden_3, heads=heads_3)
        self.bn3 = GroupNorm(16, hidden_3 * heads_3)

        self.lin = Linear(hidden_3 * heads_3, num_classes)
        self.dropout = torch.nn.Dropout(p= dropout_rate)

    def forward(self, x, edge_index, batch, return_attn=True):
        #self. collect attention weights if needed
        attns = [] if self.return_attn or return_attn else None

        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x).relu()
        x = self.dropout(x)
        if attns is not None:
            attns.append(attn1)

        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x).relu()
        x = self.dropout(x)
        if attns is not None:
            attns.append(attn2)

        x, attn3 = self.conv3(x, edge_index, return_attention_weights=True)
        x = self.bn3(x).relu()
        x = self.dropout(x)
        if attns is not None:
            attns.append(attn3)

        x = global_max_pool(x, batch)
        out = self.lin(x)

        if attns is not None:
            return out, attns
        else:
            return out  

########################################
# Patient Homo Aglomerative approach 
########################################
class HomoGCN_Encoder(torch.nn.Module):
    """Same layers as your GCN_survival, but returns node features (no pooling/classifier)."""
    def __init__(self, in_ch, h1, h2, h3, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_ch, h1)
        self.bn1 = torch.nn.GroupNorm(16, h1)
        self.conv2 = GCNConv(h1, h2)
        self.bn2 = torch.nn.GroupNorm(16, h2)
        self.conv3 = GCNConv(h2, h3)
        self.bn3 = torch.nn.GroupNorm(16, h3)
        self.drop = torch.nn.Dropout(p=dropout)
        self.out_dim = h3

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index); x = self.bn1(x).relu(); x = self.drop(x)
        x = self.conv2(x, edge_index); x = self.bn2(x).relu(); x = self.drop(x)
        x = self.conv3(x, edge_index); x = self.bn3(x).relu(); x = self.drop(x)
        return x

class HomoGAT_Encoder(torch.nn.Module):
    """
    Same structure as your GAT_survival (3×GATConv + GroupNorm + Dropout),
    but returns node features and attention weights per layer.
    """
    def __init__(self, in_ch, h1, h2, h3, heads1, heads2, heads3, dropout=0.3, return_attn=True):
        super().__init__()
        self.return_attn = return_attn
        self.conv1 = GATConv(in_ch, h1, heads=heads1)
        self.bn1 = torch.nn.GroupNorm(16, h1*heads1)
        self.conv2 = GATConv(h1*heads1, h2, heads=heads2)
        self.bn2 = torch.nn.GroupNorm(16, h2*heads2)
        self.conv3 = GATConv(h2*heads2, h3, heads=heads3)
        self.bn3 = torch.nn.GroupNorm(16, h3*heads3)
        self.drop = torch.nn.Dropout(p=dropout)
        self.out_dim = h3*heads3

    def forward(self, x, edge_index):
        attns = [] if self.return_attn else None

        x, a1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = self.bn1(x).relu(); x = self.drop(x)
        if attns is not None: attns.append(a1)

        x, a2 = self.conv2(x, edge_index, return_attention_weights=True)
        x = self.bn2(x).relu(); x = self.drop(x)
        if attns is not None: attns.append(a2)

        x, a3 = self.conv3(x, edge_index, return_attention_weights=True)
        x = self.bn3(x).relu(); x = self.drop(x)
        if attns is not None: attns.append(a3)

        if attns is not None:
            return x, attns
        return x

class AgglomerativeGCN_Survival(torch.nn.Module):
    def __init__(self, in_ch, h1, h2, h3, out_ch, reduce='mean', dropout=0.3):
        super().__init__()
        assert reduce in ('mean','max')
        self.reduce = reduce
        self.enc = HomoGCN_Encoder(in_ch, h1, h2, h3, dropout=dropout)
        self.lin = torch.nn.Linear(self.enc.out_dim, out_ch)

    def forward(self, x, edge_index, batch):
        z = self.enc(x, edge_index)                 # [nodes, h3]
        wsi_emb = global_max_pool(z, batch)         # [num_wsis, h3]
        if self.reduce == 'mean':
            patient_emb = wsi_emb.mean(dim=0, keepdim=True)
        else:
            patient_emb, _ = wsi_emb.max(dim=0, keepdim=True)
        return self.lin(patient_emb)                # [1, out_ch]

class AgglomerativeGAT_Survival(torch.nn.Module):
    def __init__(self, in_ch, h1, h2, h3, out_ch,
                 heads1=1, heads2=1, heads3=1, reduce='mean',
                 dropout=0.3, return_attn=True):
        super().__init__()
        assert reduce in ('mean','max')
        self.reduce = reduce
        self.enc = HomoGAT_Encoder(in_ch, h1, h2, h3, heads1, heads2, heads3,
                                   dropout=dropout, return_attn=return_attn)
        self.lin = torch.nn.Linear(self.enc.out_dim, out_ch)
        self.return_attn = return_attn

    def forward(self, x, edge_index, batch):
        out = self.enc(x, edge_index)
        if isinstance(out, tuple):
            z, attns = out
        else:
            z, attns = out, None

        wsi_emb = global_max_pool(z, batch)        # [num_wsis, dim]
        if self.reduce == 'mean':
            patient_emb = wsi_emb.mean(dim=0, keepdim=True)
        else:
            patient_emb, _ = wsi_emb.max(dim=0, keepdim=True)

        logits = self.lin(patient_emb)             # [1, out_ch]
        if self.return_attn:
            return logits, attns                   # attns: list of (edge_index, alpha) per layer
        return logits


## Baseline Aglomerative
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch import nn
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool

class BaselinePatientGAP(nn.Module):
    """
    Compatible with BOTH:
      - model(data)                          # Data must have .x and .batch
      - model(x, edge_index, batch=None)     # edge_index is ignored
    Pools: nodes -> WSI (by batch), then WSIs -> patient, then Linear -> logits.
    """
    def __init__(self, in_dim: int, num_classes: int = 1,
                 wsi_pool: str = "mean", patient_pool: str = "mean"):
        super().__init__()
        assert wsi_pool in ("mean", "max")
        assert patient_pool in ("mean", "max")
        self.wsi_pool = wsi_pool
        self.patient_pool = patient_pool
        self.lin = nn.Linear(in_dim, num_classes)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # nodes -> WSI vectors
        x_wsi = (global_max_pool(x, batch) if self.wsi_pool == "max"
                 else global_mean_pool(x, batch))              # [num_wsis, d]
        # WSIs -> patient vector (you pass 1 patient per step)
        x_patient = (x_wsi.max(dim=0, keepdim=True).values if self.patient_pool == "max"
                     else x_wsi.mean(dim=0, keepdim=True))     # [1, d]
        return x_patient

    def forward(self, *args, **kwargs):
        # Accept model(data) OR model(x, edge_index, batch)
        if len(args) == 1 and hasattr(args[0], "x"):
            data = args[0]
            x = data.x
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            x = args[0]                                   # x
            batch = None
            if len(args) >= 3:                            # (x, edge_index, batch)
                batch = args[2]
            if batch is None:
                batch = kwargs.get("batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x_patient = self._pool(x, batch)                  # [1, d]
        return self.lin(x_patient)                        # [1, num_classes] logits


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def aggregate_patient_features(loader, wsi_pool: str = "mean", patient_pool: str = "mean"):
    """
    Convert a patient-level loader (1 patient per batch) into a feature matrix.

    Pools patches -> WSI -> patient using the same GAP strategy as BaselinePatientGAP.
    Supports optional "meanmax" pooling at the patient level by concatenating mean and max.
    Returns (X, y, case_ids).
    """
    if loader is None or (hasattr(loader, "__len__") and len(loader) == 0):
        return np.empty((0, 0)), np.empty((0,)), []

    feats = []
    labels = []
    case_ids = []

    for data in loader:
        if hasattr(data, "to"):
            data = data.cpu()

        x = data.x
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)

        # nodes -> WSI
        if wsi_pool == "max":
            x_wsi = global_max_pool(x, batch)
        else:
            x_wsi = global_mean_pool(x, batch)

        # WSI -> patient
        if patient_pool == "max":
            x_patient = x_wsi.max(dim=0, keepdim=True).values
        elif patient_pool == "meanmax":
            x_patient = torch.cat(
                [x_wsi.mean(dim=0, keepdim=True), x_wsi.max(dim=0, keepdim=True).values],
                dim=1,
            )
        else:
            x_patient = x_wsi.mean(dim=0, keepdim=True)

        feats.append(x_patient.squeeze(0).numpy())
        labels.append(int(data.y.view(-1)[0].item()))
        case_ids.append(getattr(data, "case_id", None))

    if len(feats) == 0:
        return np.empty((0, 0)), np.empty((0,)), []
    return np.vstack(feats), np.array(labels), case_ids


def _balanced_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Return the probability threshold that maximizes balanced accuracy."""
    if probs.size == 0:
        return 0.5
    thresholds = np.unique(probs)
    best_thr, best_bacc = 0.5, -1.0
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        bacc = balanced_accuracy_score(labels, preds)
        if bacc > best_bacc:
            best_bacc = bacc
            best_thr = float(thr)
    return best_thr


def run_baseline_lda(train_loader, val_loader, test_loader=None, all_t: str = "", calculate_threshold: str = "False"):
    """
    Patient-level baseline that applies GAP over patches/WSIs and fits an LDA head.
    Returns a tuple aligned with train_accum_graddient_new_sigmoid_threshold outputs.
    """

    patient_pool = "meanmax" if "meanmax" in all_t.lower() else ("max" if "max" in all_t.lower() else "mean")
    wsi_pool = "max" if "max" in all_t.lower() else "mean"

    X_train, y_train, _ = aggregate_patient_features(train_loader, wsi_pool=wsi_pool, patient_pool=patient_pool)
    X_val, y_val, _ = aggregate_patient_features(val_loader, wsi_pool=wsi_pool, patient_pool=patient_pool)
    X_test, y_test, test_case_ids = aggregate_patient_features(test_loader, wsi_pool=wsi_pool, patient_pool=patient_pool)

    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(X_train, y_train)

    def _infer_probs(X):
        if X.size == 0:
            return np.array([])
        if hasattr(lda, "decision_function"):
            scores = lda.decision_function(X)
            if scores.ndim > 1:
                scores = scores[:, 0]
            return _sigmoid_array(scores)
        return lda.predict_proba(X)[:, 1]

    val_probs = _infer_probs(X_val)
    test_probs = _infer_probs(X_test)

    if calculate_threshold == "True":
        threshold = _balanced_threshold(val_probs, y_val)
    else:
        threshold = 0.5

    def _compute_metrics(probs, labels):
        preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(labels, preds) if labels.size else np.zeros((2, 2), dtype=int)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        bacc = balanced_accuracy_score(labels, preds) if labels.size else 0.0
        auc = roc_auc_score(labels, probs) if labels.size and len(np.unique(labels)) > 1 else 0.0
        report = classification_report(labels, preds, zero_division=0) if labels.size else ""
        return bacc, cm, recall, specificity, auc, report

    val_bacc, val_cm, val_recall, val_specificity, val_auc, val_report = _compute_metrics(val_probs, y_val)
    train_bacc, train_cm, train_recall, train_specificity, train_auc, _ = _compute_metrics(_infer_probs(X_train), y_train)
    test_bacc, test_cm, test_recall, test_specificity, test_auc, test_report = _compute_metrics(test_probs, y_test)

    # Align with expected return structure
    score_matrix_v = val_report
    score_matrix_t = f"AUC: {train_auc:.4f} | BACC: {train_bacc:.4f}"

    patient_mv_scores = val_probs
    patient_1d_scores = val_probs

    return (
        val_bacc,
        score_matrix_v,
        val_cm,
        score_matrix_t,
        train_cm,
        val_bacc,
        val_report,
        val_cm,
        val_bacc,
        val_report,
        val_cm,
        val_recall,
        val_specificity,
        val_recall,
        val_specificity,
        val_recall,
        val_specificity,
        threshold,
        y_val,
        val_probs,
        y_val,
        patient_mv_scores,
        patient_1d_scores,
    )

'''class BaselinePatientGAP(nn.Module):
    """
    Use with PatientWSIPackDataset_featsnorml and collate_passthrough (1 patient per batch).
    Pools nodes to WSI vectors, then WSIs to a single patient vector, then a linear head.
    """
    def __init__(self, in_dim: int, num_classes: int = 1,
                 wsi_pool: str = "mean", patient_pool: str = "mean"):
        super().__init__()
        assert wsi_pool in ("mean", "max")
        assert patient_pool in ("mean", "max")
        self.wsi_pool = wsi_pool
        self.patient_pool = patient_pool
        self.lin = nn.Linear(in_dim, num_classes)

    def forward(self, data):
        # 1) Nodes -> WSI vectors
        if self.wsi_pool == "max":
            x_wsi = global_max_pool(data.x, data.batch)   # [num_wsis, d]
        else:
            x_wsi = global_mean_pool(data.x, data.batch)  # [num_wsis, d]

        # 2) WSIs -> Patient vector (single patient per batch thanks to passthrough collate)
        if self.patient_pool == "max":
            x_patient = x_wsi.max(dim=0, keepdim=True).values  # [1, d]
        else:
            x_patient = x_wsi.mean(dim=0, keepdim=True)        # [1, d]

        # 3) Linear head -> logits (use BCEWithLogitsLoss outside)
        return self.lin(x_patient)  # shape [1, num_classes]'''

'''
class HeteroGAT_survival(torch.nn.Module):
    def __init__(self, in_channels, hidden1, hidden2, hidden3,
                 out_channels, heads1=1, heads2=1, heads3=1, dropout_rate=0.3):
        super().__init__()
        torch.manual_seed(47)
        self.dropout = Dropout(p=dropout_rate)

        # Conv Layer 1
        self.conv1 = HeteroConv({
            ('tile', 'intra', 'tile'): GATConv(in_channels, hidden1, heads=heads1, concat=True),
            ('tile', 'inter', 'tile'): GATConv(in_channels, hidden1, heads=heads1, concat=True),
        }, aggr='sum')
        self.bn1 = GroupNorm(16, hidden1 * heads1)

        # Conv Layer 2
        self.conv2 = HeteroConv({
            ('tile', 'intra', 'tile'): GATConv(hidden1 * heads1, hidden2, heads=heads2, concat=True),
            ('tile', 'inter', 'tile'): GATConv(hidden1 * heads1, hidden2, heads=heads2, concat=True),
        }, aggr='sum')
        self.bn2 = GroupNorm(16, hidden2 * heads2)

        # Conv Layer 3
        self.conv3 = HeteroConv({
            ('tile', 'intra', 'tile'): GATConv(hidden2 * heads2, hidden3, heads=heads3, concat=True),
            ('tile', 'inter', 'tile'): GATConv(hidden2 * heads2, hidden3, heads=heads3, concat=True),
        }, aggr='sum')
        self.bn3 = GroupNorm(16, hidden3 * heads3)

        # Linear Classifier
        self.lin = Linear(hidden3 * heads3, out_channels)

    def forward(self, x_dict, edge_index_dict, batch):
        x = x_dict['tile']
    
        x = self.conv1({'tile': x}, edge_index_dict,, return_attention_weights=True)['tile']
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
    
        x = self.conv2({'tile': x}, edge_index_dict)['tile']
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
    
        x = self.conv3({'tile': x}, edge_index_dict)['tile']
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
    
        x = global_max_pool(x, batch)
        return self.lin(x)
        '''
'''class GAT_survival(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes, heads, heads_2, heads_3, dropout_rate=0.3):
        super(GAT_survival, self).__init__()
        torch.manual_seed(47)
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads)
        self.bn1 =torch.nn.GroupNorm(16,hidden_channels * heads)#, track_running_stats=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_2, heads=heads_2)
        self.bn2 =torch.nn.GroupNorm(16,hidden_2 * heads_2)#, track_running_stats=True)
        self.conv3 = GATConv(hidden_2 * heads_2, hidden_3, heads=heads_3)
        self.bn3 =torch.nn.GroupNorm(16,hidden_3 * heads_3)#, track_running_stats=True)
        self.lin = Linear(hidden_3 * heads_3, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x'''
    
###################################### testar com relu e elu !!!!!!!!!!!!!!!!!!!!!!!!!
# Updated GraphSAGE
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes,dropout_rate=0.3):
        super(GraphSAGE, self).__init__()
        torch.manual_seed(47)
        self.conv1 = SAGEConv(num_node_features, hidden_channels,aggr='mean')
        self.bn1 =torch.nn.GroupNorm(16,hidden_channels)#, track_running_stats=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_2,aggr='mean')
        self.bn2 =torch.nn.GroupNorm(16,hidden_2)#, track_running_stats=True)
        self.conv3 = SAGEConv(hidden_2, hidden_3,aggr='mean')
        self.bn3 =torch.nn.GroupNorm(16,hidden_3)#, track_running_stats=True)
        self.lin = Linear(hidden_3, num_classes)
        self.dropout =torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x

# Define Adaptive GraphSAGE model
class AdaptiveGraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes, dropout_rate=0.3):
        super(AdaptiveGraphSAGE, self).__init__()
        torch.manual_seed(47)
        self.conv1 = SAGEConv(num_node_features, hidden_channels,aggr='mean')
        self.bn1 =torch.nn.GroupNorm(16,hidden_channels)#, track_running_stats=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_2,aggr='mean')
        self.bn2 =torch.nn.GroupNorm(16,hidden_2)#, track_running_stats=True)
        self.conv3 = SAGEConv(hidden_2, hidden_3,aggr='mean')
        self.bn3 =torch.nn.GroupNorm(16,hidden_3)#, track_running_stats=True)
        self.lin = Linear(hidden_3, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x

##max
class GraphSAGE_max(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes,dropout_rate=0.3):
        super(GraphSAGE_max, self).__init__()
        torch.manual_seed(47)
        self.conv1 = SAGEConv(num_node_features, hidden_channels,aggr='max')
        self.bn1 =torch.nn.GroupNorm(16,hidden_channels)#, track_running_stats=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_2,aggr='max')
        self.bn2 =torch.nn.GroupNorm(16,hidden_2)#, track_running_stats=True)
        self.conv3 = SAGEConv(hidden_2, hidden_3,aggr='max')
        self.bn3 =torch.nn.GroupNorm(16,hidden_3)#, track_running_stats=True)
        self.lin = Linear(hidden_3, num_classes)
        self.dropout =torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x

# Define Adaptive GraphSAGE model
class AdaptiveGraphSAGE_max(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes, dropout_rate=0.3):
        super(AdaptiveGraphSAGE_max, self).__init__()
        torch.manual_seed(47)
        self.conv1 = SAGEConv(num_node_features, hidden_channels,aggr='max')
        self.bn1 =torch.nn.GroupNorm(16,hidden_channels)#, track_running_stats=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_2,aggr='max')
        self.bn2 =torch.nn.GroupNorm(16,hidden_2)#, track_running_stats=True)
        self.conv3 = SAGEConv(hidden_2, hidden_3,aggr='max')
        self.bn3 =torch.nn.GroupNorm(16,hidden_3)#, track_running_stats=True)
        self.lin = Linear(hidden_3, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x
    
    
#######################################################
# Reset weights and model initialization
#######################################################
def reset_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.manual_seed(seed)  # Ensure consistent weight initialization
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
     
def model_init(survival,num_node_features,num_epochs,hidden_channels,hidden_2,hidden_3,lr_r,batch_sizee,more_data,num_classes, headss,headss_2,headss_3,paralel,device,fold,all_t,best_model_path_init,sigmoid,repeat,individual,threshold,percent,dataset):
    
    interperct=f'inter_{percent}'
    dropout_rate=0.3
    if 'hetero' in all_t:
        initial_weights_path = f"{best_model_path_init}/initial_weights_fold_sigmoid{sigmoid}_{survival}{num_node_features}{hidden_channels}{hidden_2}{hidden_3}{headss}{headss_2}{headss_3}_{interperct}-other.pth"  # Define the path for initializing weights
        ### HETERO GNNS
        if survival == 'gcn':
           print('\n\n Selected the HETERO GCN model')
           best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hetero/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3_{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
           model = HeteroGCN_survival(num_node_features,hidden_channels,hidden_2,hidden_3, num_classes,interperct).to(device)
        elif survival == 'gat':
          print('\n\n Selected the HETERO GAT model')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hetero/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3:{hidden_3}_heads:{headss}_{headss_2}_{headss_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          model= HeteroGAT_survival(num_node_features,hidden_channels,hidden_2,hidden_3, num_classes,interperct, headss,headss_2,headss_3,return_attn=True).to(device)
        print('\n',model)

    elif 'hyper' in all_t:
        print('\n\n ########## Selected the HypergraphConv (incidence) model ################')
        initial_weights_path = f"{best_model_path_init}/initial_weights_fold_sigmoid{sigmoid}_{survival}{num_node_features}{hidden_channels}{hidden_2}{hidden_3}{headss}{headss_2}{interperct}-other.pth"  # Define the path for initializing weights
        if survival == 'gcn':
           print('\n\n Selected the HYPER GCN model')
           best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hyper/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3_{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
           model = HyperGCN_survival(num_node_features, hidden_channels, hidden_2, hidden_3, num_classes).to(device)
        elif survival == 'gat':
          print('\n\n Selected the Hyper GAT model')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hyper/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3:{hidden_3}_heads:{headss}_{headss_2}_{headss_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          dropout_rate=0.3
          model = HyperGAT_survival(in_channels=num_node_features,hidden1=hidden_channels,hidden2=hidden_2,hidden3=hidden_3,out_channels=num_classes,heads1=headss, heads2=headss_2, heads3=headss_3,dropout_rate=dropout_rate).to(device)
          print('\n',model)

    elif 'aglomerative' in all_t:
        ### HETERO GNNS
        if 'baseline' in all_t:
           initial_weights_path = f"{best_model_path_init}/initial_weights_fold_sigmoid{sigmoid}_{num_node_features}{hidden_channels}{hidden_2}{hidden_3}{headss}{headss_2}{headss_3}_{interperct}-other.pth"  # Define the path for initializing weights
           print('\n\n Selected the  BASELINE aglomerative NON-GCN model')
           best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_aglomerative/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3_{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
           if "baseline_aglomerative_mean" in all_t:
                model = BaselinePatientGAP(
                    in_dim=num_node_features,
                    num_classes=num_classes,
                    wsi_pool="mean",
                    patient_pool="mean",
                ).to(device)
           elif "baseline_aglomerative_max" in all_t:
                model = BaselinePatientGAP(
                    in_dim=num_node_features,
                    num_classes=num_classes,
                    wsi_pool="max",
                    patient_pool="max",
                ).to(device)
           else:
                raise ValueError(f"Unsupported baseline agglomerative model type: {all_t}")
           print('\n',model)
        else:
            # reducer from all_t
            agg_reduce = 'mean' if 'mean' in all_t else 'max'
            initial_weights_path = f"{best_model_path_init}/initial_weights_fold_sigmoid{sigmoid}_{survival}{num_node_features}{hidden_channels}{hidden_2}{hidden_3}{headss}{headss_2}{headss_3}_{interperct}-other.pth"  # Define the path for initializing weights
            if survival == 'gcn':
               print('\n\n Selected the  Homo aglomerative GCN model')
               best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_aglomerative/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3_{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
               model = AgglomerativeGCN_Survival(in_ch=num_node_features,h1=hidden_channels, h2=hidden_2, h3=hidden_3,out_ch=num_classes,reduce=agg_reduce, dropout=dropout_rate).to(device)
            elif survival == 'gat':
              print('\n\n Selected the Homo aglomerative GAT model')
              best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_aglomerative/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3:{hidden_3}_heads:{headss}_{headss_2}_{headss_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
              model= AgglomerativeGAT_Survival(in_ch=num_node_features,h1=hidden_channels, h2=hidden_2, h3=hidden_3,out_ch=num_classes,heads1=headss, heads2=headss_2, heads3=headss_3,reduce=agg_reduce,dropout=dropout_rate,return_attn=True).to(device)
            print('\n',model)
        
    else:
        initial_weights_path = f"{best_model_path_init}/initial_weights_fold_sigmoid{sigmoid}_{survival}{num_node_features}{hidden_channels}{hidden_2}{hidden_3}{headss}{headss_2}{headss_3}-other.pth"  # Define the path for initializing weights
        if survival == 'gcn':
          print('\n\n ########## Selected the GCN survival model ################')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3_{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          model = GCN_survival(num_node_features, hidden_channels,hidden_2,hidden_3,num_classes).to(device)
        elif survival == 'survival_iter':
          print('\n\n ########## Selected the GCN survival model ################')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_num_layers{num_layers}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}.pt"
          model = GCN_survival_iter(num_node_features, hidden_channels,num_layers,num_classes).to(device)
        elif survival == 'gcn_other':
          print('\n\n Selected the GCN model')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3_{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          model = GCN(num_node_features, hidden_channels,hidden_2,hidden_3,num_classes).to(device)
        elif survival == 'gnn':
          print('\n\n Selected the GNN model (without neighborhood normalization)')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3:{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          model = GNN(num_node_features, hidden_channels,hidden_2,hidden_3,num_classes).to(device)
        elif survival == 'gin':
          print('\n\n Selected the GIN model')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2{hidden_2}_hidden3:{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          model = GIN(num_node_features, hidden_channels,hidden_2,hidden_3,num_classes).to(device)
        elif survival == 'GraphSAGE'or survival == 'AdaptiveGraphSAGE':
          print('\n\n Selected the  GraphSAGE model')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage/{all_t}_best_model_epoch{num_epochs}{survival}_hidden{hidden_channels}_hidden2{hidden_2}_hidden3:{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          model =  GraphSAGE(num_node_features, hidden_channels,hidden_2,hidden_3,num_classes).to(device)
        elif survival == 'GraphSAGE_max' or survival == 'AdaptiveGraphSAGE_max':
          print('\n\n Selected the  GraphSAGE model')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage/{all_t}_best_model_epoch{num_epochs}_{survival}_hidden{hidden_channels}_hidden2{hidden_2}_hidden3:{hidden_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          model =  GraphSAGE_max(num_node_features, hidden_channels,hidden_2,hidden_3,num_classes).to(device)        
        # Create a GAT_survival model instance
        elif survival == 'gat':
          print('\n\n Selected the GAT model')
          best_model_path = f"Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}/{all_t}_best_model_epoch{num_epochs}_hidden{hidden_channels}_hidden2:{hidden_2}_hidden3:{hidden_3}_heads:{headss}_{headss_2}_{headss_3}_lr{lr_r}_batch_{batch_sizee}_fold{fold}_repeat{repeat}_threshold{threshold}_individual{individual}.pt"
          model = GAT_survival(num_node_features, hidden_channels,hidden_2,hidden_3, num_classes, headss,headss_2,headss_3, return_attn=True).to(device)
            
          # Wrap the model with DataParallel and move it to the GPU
          if paralel == 'True':
            model = nn.DataParallel(model).to(device)
              
    if 'hetero' in all_t:
        if not os.path.exists(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hetero'):
           os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hetero')
           os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hetero/init')
        if 'SAGE' in survival:
            if not os.path.exists(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage_hetero'):
               os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage_hetero')
               os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage_hetero/init')
    elif 'hyper' in all_t:
        if not os.path.exists(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hyper'):
           os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hyper')
           os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_hyper/init')
            
    elif 'aglomerative' in all_t:
        if 'baseline' in all_t:
            if not os.path.exists(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_aglomerative_baseline'):
               os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_aglomerative_baseline')
               os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_aglomerative_baseline/init')
        if not os.path.exists(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_aglomerative'):
           os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_aglomerative')
           os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}_aglomerative/init')
    else:
        if not os.path.exists(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}'):
           os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}')
           os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}/init')
        if 'SAGE' in survival:
            if not os.path.exists(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage'):
               os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage')
               os.makedirs(f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage/init')
    
    print('initial_weights_path',initial_weights_path)
     # Check if we need to load the pretrained/initial weights for initialization
    if os.path.exists(initial_weights_path):
        print(f"Loading initial weights from {initial_weights_path}")
        model.load_state_dict(torch.load(initial_weights_path, map_location=device))
  # Load model's initial weights
    else:
        print("No saved initial weights found. Starting with a fresh initialization.")
        model.apply(reset_weights)  # Reset weights if no saved initial weights are found
        torch.save(model.state_dict(), initial_weights_path)  # Save these initialized weights for future runs

    return model,best_model_path,initial_weights_path




#######################################################
# Training and validation
#######################################################

def save_checkpoint(model, optimizer, epoch, val_acc, path, filename):
    """Save the model and optimizer state."""
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_acc': val_acc,
    }
    filepath = os.path.join(path, filename)
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")

def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    denom = event.sum().clamp_min(1.0)
    return (-(risk - log_cumsum) * event).sum() / denom

def concordance_index_torch(time, risk, event):
    time = time.detach().view(-1).float().cpu()
    risk = risk.detach().view(-1).float().cpu()
    event = event.detach().view(-1).float().cpu()
    n = len(time)
    concordant = 0.0
    ties = 0.0
    comparable = 0.0
    for i in range(n):
        if event[i] <= 0:
            continue
        for j in range(n):
            if time[i] < time[j]:
                comparable += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    ties += 1.0
    if comparable == 0:
        return 0.5
    return float((concordant + 0.5 * ties) / comparable)

def validate_risk(loader, model, device, survival, all_t):
    model.eval()
    all_time, all_event, all_risk = [], [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if isinstance(data, HeteroData):
                if survival == 'gat':
                    out, _ = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                else:
                    out = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
            elif 'hyper' in all_t:
                out = model(data.x, data.edge_index, data.hyperedge_index, data.batch)
            else:
                if survival == 'gat' and 'hyper' not in all_t:
                    out, _ = model(data.x, data.edge_index, data.batch)
                else:
                    out = model(data.x, data.edge_index, data.batch)
            risk = out.view(-1)
            time = data.time.view(-1).float()
            event = data.event.view(-1).float()
            total_loss += cox_ph_loss(risk, time, event).item()
            n_batches += 1
            all_time.append(time)
            all_event.append(event)
            all_risk.append(risk)
    if n_batches == 0:
        return 0.0, 0.5
    times = torch.cat(all_time)
    events = torch.cat(all_event)
    risks = torch.cat(all_risk)
    c_index = concordance_index_torch(times, risks, events)
    return total_loss / n_batches, c_index


    
def train_accum_graddient_new_sigmoid_threshold(train_loader, val_loader, model, criterion, optimizer, device, early_stopping_rounds,
                                                lr_scheduler_patience, lr_scheduler_factor, writer, best_model_path, weight_tensor, num_epochs,
                                                hidden_channels, lr, num_node_features, class_weights, best_val_acc, best_score_matrix_v, best_cm_v,
                                                best_score_matrix_t, best_cm, fold, lrrr, grad_accum, load, survival,
                                                all_t, actualtime, effective_batch, calculate_threshold, track, track_lr, repeat, num_neighbors,
                                                node_batch_size, batch_sizee,dataset, task='12months'):
    """
    Train the model with gradient accumulation and validation, implementing early stopping and learning rate scheduling.

    This function is specifically designed for a criterion where `reduction='sum'`.

    Parameters:
    - train_loader, val_loader: Dataloaders for training and validation.
    - model: Model to be trained.
    - criterion: Loss function (expected to have `reduction='sum'`).
    - optimizer: Optimizer.
    - device: Device to run computations on.
    - early_stopping_rounds: Rounds for early stopping.
    - lr_scheduler_patience, lr_scheduler_factor: Learning rate scheduler parameters.
    - writer: TensorBoard writer.
    - best_model_path: Path to save the best model.
    - weight_tensor: Weights for the loss function.
    - num_epochs, hidden_channels, lr, num_node_features: Hyperparameters.
    - class_weights, fold, grad_accum: Additional training parameters.
    - load: Boolean to load previous model checkpoint.
    - survival, all_t, actualtime: Additional identifiers.
    - effective_batch: Effective batch size for gradient accumulation.
    - lrrr: Learning rate reduction mode.
    - best_val_acc, best_score_matrix_v, best_cm_v, best_score_matrix_t, best_cm: Initial best metrics (can be None or initial values).
    - calculate_threshold: Flag for threshold calculation in validation.
    - track: Metric to track for best model saving ('loss' or 'bacc').
    - track_lr: Metric to track for LR scheduling ('loss' or 'bacc').
    - repeat: Repetition number for logging.
    - num_neighbors, node_batch_size, batch_sizee: Parameters for NeighborLoader if using SAGE.
    """
    
    # Initialize best metrics if not provided or set to default (e.g., 0.0 for accuracy, high for loss)
    # Note: best_avg_loss_val should be initialized to infinity for 'min' tracking
    best_avg_loss_val = 1000
    # Initialize if not provided by caller
    best_val_acc = 0.0

    early_stopping_counter = 0
    
    # Initialize variables to hold the best metrics for return
    best_score_matrix_v_epoch = best_score_matrix_v
    best_cm_v_epoch = best_cm_v
    best_score_matrix_t_epoch = best_score_matrix_t
    best_cm_epoch = best_cm
    
    # Initialize variables that are returned but might not be updated if no improvement
    best_bacc_majority_voting = None
    best_bacc_one_dominance = None
    best_majority_voting_predictions = None
    best_class_report_majority_voting = None
    best_conf_matrix_majority_voting = None
    best_one_dominance_predictions = None
    best_class_report_one_dominance = None
    best_conf_matrix_one_dominance = None
    best_recall_v = None
    best_specificity_v = None
    best_recall_mj_v = None
    best_specificity_mj_v = None
    best_recall_1d_v = None
    best_specificity_1d_v = None
    best_optimal_threshold_v = None
    best_y_true_wsi_level = None
    best_probs_wsi_level = None
    best_y_true_patient_level = None
    best_patient_mv_scores = None
    best_patient_1d_scores = None
    

    
    if lrrr == 'ritap':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0.000005) # Corrected args.num_epochs to num_epochs
    elif lrrr == 'reduceplateau':
        if track_lr == 'loss':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_scheduler_patience)
        elif track_lr == 'bacc':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=lr_scheduler_patience, factor=0.5)
    
    print('\n\nStarting training with gradient accumulation. Effective batch size:', effective_batch)

    # Load checkpoint if specified
    if load == 'True':
        if 'SAGE' in survival:
            start_epoch, best_val_acc = load_checkpoint(
                model, optimizer, device, f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage',
                filename=f'{all_t}_last_checkpoint_{survival}.pth.tar'
            )
        else:
            start_epoch, best_val_acc = load_checkpoint(
                model, optimizer, device, f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}',
                filename=f'{all_t}_last_checkpoint_{survival}.pth.tar'
            )

    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        total_loss = 0.0  # To accumulate total loss for the epoch
        num_samples = 0
        warmup_epochs = 5  # Number of epochs before allowing early stopping or model saving
        
        model.train()
        optimizer.zero_grad()  # Initialize gradients
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs} - WSI Level', leave=True) as wsi_pbar:
            # Training loop with gradient accumulation
            for i, data in enumerate(train_loader, 1):
                if "SAGE" in survival:
                    # In this block, 'data' is expected to be a single WSI graph
                    # The NeighborLoader then generates subgraphs from this WSI.
                    # The label for the entire WSI (data.y) should be applied to all subgraphs.
                    if data.y.size(0) > 1:
                        # If data.y has multiple elements, assume it's a batch of WSIs
                        # and you need to determine which label corresponds to the current WSI.
                        # For now, let's assume each 'data' from train_loader corresponds to one WSI.
                        # If it's a batch, you'd need to iterate through the batch or adapt.
                        # For simplicity, assuming data.y is [1] for a single WSI.
                        full_wsi_label = data.y.to(device).float()
                    else:
                        full_wsi_label = data.y.to(device).float() # Assuming data.y is already a tensor [1] or similar
                    
                    wsi_loader = NeighborLoader(
                        data,
                        num_neighbors=num_neighbors,
                        batch_size=node_batch_size,
                        input_nodes=torch.arange(data.x.size(0)),
                        shuffle=True,
                        drop_last=False,
                        generator=torch.Generator().manual_seed(47),
                        pin_memory=True
                    )
                    
                    # Subgraph-Level Progress Bar (uncommented for actual use)
                    with tqdm(total=len(wsi_loader), desc=f'WSI {i} - Subgraphs', leave=False) as subgraph_pbar:
                        for subgraph in wsi_loader:
                            subgraph = subgraph.to(device)
                            output = model(subgraph.x, subgraph.edge_index, subgraph.batch)
                            
                            # Assign full WSI label to all subgraphs
                            # Ensure target has the same batch size as output
                            target = full_wsi_label.expand_as(output) # Broadcast the WSI label to all subgraphs in the batch
                            
                            # Calculate loss (criterion returns sum, so divide by effective_batch)
                            loss = criterion(output, target) / effective_batch
                            loss.backward()
                            
                            # Accumulate raw (unscaled) loss for logging (criterion already summed)
                            total_loss += criterion(output, target).item()
                            num_samples += output.size(0) # Accumulate based on number of predictions (subgraphs)
                            
                            subgraph_pbar.update(1)

                    # Optimizer step after processing subgraphs for the current WSI
                    if i % grad_accum == 0 or i == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()
                else: # For non-SAGE models (assuming typical batch processing)
                    data = data.to(device)  # Move data to GPU
                    data.y = data.y.view(-1, 1).float()  # Reshape target to [batch_size, 1]

                    if isinstance(data, HeteroData):
                        if survival == 'gat':
                            out,_ = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                        else:
                            out = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                    elif 'hyper' in all_t:
                        out = model(data.x, data.edge_index, data.hyperedge_index, data.batch)
                    else:
                        if survival == 'gat'  and 'hyper' not in all_t:
                            out,_ = model(data.x, data.edge_index, data.batch)
                        else:
                            out = model(data.x, data.edge_index, data.batch)
  # Output logits of shape [batch_size, 1]

                    if task == 'risk':
                        risk = out.view(-1)
                        time = data.time.view(-1).float()
                        event = data.event.view(-1).float()
                        loss_raw = cox_ph_loss(risk, time, event)
                    else:
                        loss_raw = criterion(out, data.y)
                    loss = loss_raw / effective_batch
                    loss.backward()

                    # Step optimizer if gradient accumulation steps are met
                    if i % grad_accum == 0 or i == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()

                    # Accumulate raw (unscaled) loss for logging (criterion already summed)
                    total_loss += loss_raw.item()
                    num_samples += data.num_graphs # Assuming data.num_graphs is appropriate for batch size

                # Update WSI progress
                wsi_pbar.update(1)  # Update tqdm progress bar

        # Calculate epoch training metrics (overall mean loss per sample for the epoch)
        train_loss = total_loss / num_samples

        # Validation loop
        if task == 'risk':
            avg_loss_val, c_index_val = validate_risk(val_loader, model, device, survival, all_t)
            avg_loss_train, c_index_train = validate_risk(train_loader, model, device, survival, all_t)
            val_acc = val_bacc = c_index_val
            train_accuracy = train_bacc = c_index_train
            score_matrix_v = cm_v = score_matrix_t = cm_t = None
            majority_voting_predictions = class_report_majority_voting = conf_matrix_majority_voting = None
            one_dominance_predictions = class_report_one_dominance = conf_matrix_one_dominance = None
            bacc_majority_voting = bacc_one_dominance = 0.0
            auc_v = c_index_val
            auc_t = c_index_train
            recall_v = specificity_v = recall_mj_v = specificity_mj_v = recall_1d_v = specificity_1d_v = 0.0
            optimal_threshold_v = 0.5
            y_true_wsi_level = probs_wsi_level = y_true_patient_level = patient_mv_scores = patient_1d_scores = []
        else:
            mode = 'val'
            avg_loss_val, val_acc, val_bacc, score_matrix_v, cm_v, majority_voting_predictions, bacc_majority_voting, class_report_majority_voting, conf_matrix_majority_voting, one_dominance_predictions, bacc_one_dominance, class_report_one_dominance, conf_matrix_one_dominance, auc_v, recall_v, specificity_v, recall_mj_v, specificity_mj_v, recall_1d_v, specificity_1d_v, optimal_threshold_v,y_true_wsi_level, probs_wsi_level,y_true_patient_level, patient_mv_scores, patient_1d_scores = validate_sigmoid_threshold(
                val_loader, model, weight_tensor, device, mode, criterion, calculate_threshold,survival,all_t)
            mode = 'train'
            avg_loss_train, train_accuracy, train_bacc, score_matrix_t, cm_t, _, _, _, _, _, _, _, _, auc_t, _, _, _, _, _, _, _, _, _,_,_,_ = validate_sigmoid_threshold(train_loader, model, weight_tensor, device, mode, criterion, calculate_threshold,survival,all_t)

        if 'SAGE' in survival:
            # Save checkpoint at the end of each epoch
            save_checkpoint(model, optimizer, epoch, val_acc, f'Checkpoints-univ2/Checkpoint_{dataset}_adj_Sage',
                            filename=f'{all_t}_last_checkpoint_{survival}.pth.tar')
        else:
            # Save checkpoint at the end of each epoch
            save_checkpoint(model, optimizer, epoch, val_acc, f'Checkpoints-univ2/Checkpoint_{dataset}_adj_{survival}',
                            filename=f'{all_t}_last_checkpoint_{survival}.pth.tar')

        metric_name = "C-index" if task == 'risk' else "Bacc"
        # Print epoch summary
        print(f'\n\nFOLD {fold + 1} repeat {repeat+1}- Epoch: {epoch:03d}, Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.6f}, Train {metric_name}: {train_bacc:.6f} Train auc: {auc_t}, Val Loss: {avg_loss_val:.6f}, Val Acc: {val_acc:.6f}, Val {metric_name}: {val_bacc:.6f} Val auc: {auc_v}\n -------------------------------------------------------------------------\n')
        print('Train classification report:')
        print(cm_t)
        print(score_matrix_t)
        print('\n')
        print('Val classification report:')
        print(cm_v)
        print(score_matrix_v)
        print('\n')

        # Save the model if validation metric improves
        if track == 'loss':
            if avg_loss_val < best_avg_loss_val:
                best_avg_loss_val = avg_loss_val
                best_val_acc = val_bacc # Keep tracking balanced accuracy for consistency with 'bacc' track
                
                # Update best metrics for saving and returning
                best_score_matrix_v_epoch = score_matrix_v
                best_cm_v = cm_v
                best_score_matrix_t_epoch = score_matrix_t
                best_cm = cm_t
                best_bacc_majority_voting = bacc_majority_voting
                best_bacc_one_dominance = bacc_one_dominance
                best_majority_voting_predictions = majority_voting_predictions
                best_class_report_majority_voting = class_report_majority_voting
                best_conf_matrix_majority_voting = conf_matrix_majority_voting
                best_one_dominance_predictions = one_dominance_predictions
                best_class_report_one_dominance = class_report_one_dominance
                best_conf_matrix_one_dominance = conf_matrix_one_dominance
                best_recall_v = recall_v
                best_specificity_v = specificity_v
                best_recall_mj_v = recall_mj_v
                best_specificity_mj_v = specificity_mj_v
                best_recall_1d_v = recall_1d_v
                best_specificity_1d_v = specificity_1d_v
                best_optimal_threshold_v = optimal_threshold_v
                best_y_true_wsi_level = y_true_wsi_level
                best_probs_wsi_level = probs_wsi_level
                best_y_true_patient_level = y_true_patient_level
                best_patient_mv_scores = patient_mv_scores
                best_patient_1d_scores = patient_1d_scores

                torch.save(model.state_dict(), best_model_path)
                print(f'\nSaved the lowest loss model with Val loss: {avg_loss_val:.6f} (Bacc: {val_bacc:.6f}) at Epoch: {epoch:03d}\n')
                early_stopping_counter = 0  # Reset early stopping counter if accuracy improves
            else:
                early_stopping_counter += 1  # Increment early stopping counter
            
            # Learning rate scheduling for 'loss' tracking
            if lrrr == 'counter':
                if epoch > 1 and avg_loss_val >= best_avg_loss_val: # Only reduce LR if loss is not improving
                    lr_scheduler_counter += 1
                    if lr_scheduler_counter >= lr_scheduler_patience:
                        lr_scheduler_counter = 0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_scheduler_factor
                            print(f'\nReduced learning rate to: {param_group["lr"]}\n')
                else: # Loss improved or first epoch, reset counter
                    lr_scheduler_counter = 0
            elif lrrr == 'reduceplateau':
                if epoch > 1:
                    scheduler.step(avg_loss_val)
            elif lrrr == 'ritap':
                scheduler.step()
                
        elif track == 'bacc':
            # Skip model saving and early stopping during the warm-up period
            if epoch > warmup_epochs:
                # Save the model if test accuracy improves
                if val_bacc > best_val_acc:
                    best_val_acc = val_bacc
                    # Save best performance metrics and model state
                    best_score_matrix_v = score_matrix_v
                    best_cm_v = cm_v
                    best_score_matrix_t = score_matrix_t
                    best_cm = cm_t
                    best_bacc_majority_voting = bacc_majority_voting
                    best_bacc_one_dominance = bacc_one_dominance
                    best_majority_voting_predictions = majority_voting_predictions
                    best_class_report_majority_voting = class_report_majority_voting
                    best_conf_matrix_majority_voting = conf_matrix_majority_voting
                    best_one_dominance_predictions = one_dominance_predictions
                    best_class_report_one_dominance = class_report_one_dominance
                    best_conf_matrix_one_dominance = conf_matrix_one_dominance
                    best_recall_v = recall_v
                    best_specificity_v = specificity_v
                    best_recall_mj_v = recall_mj_v
                    best_specificity_mj_v = specificity_mj_v
                    best_recall_1d_v = recall_1d_v
                    best_specificity_1d_v = specificity_1d_v
                    best_optimal_threshold_v = optimal_threshold_v
                    best_y_true_wsi_level = y_true_wsi_level
                    best_probs_wsi_level = probs_wsi_level
                    best_y_true_patient_level = y_true_patient_level
                    best_patient_mv_scores = patient_mv_scores
                    best_patient_1d_scores = patient_1d_scores

                    torch.save(model.state_dict(), best_model_path)
                    print(f'\nSaved the best model with Val {metric_name}: {val_bacc:.6f} at Epoch: {epoch:03d}\n')
                    early_stopping_counter = 0  # Reset early stopping counter if accuracy improves
                else:
                    early_stopping_counter += 1  # Increment early stopping counter
            
            # Learning rate scheduling for 'bacc' tracking
            if lrrr == 'counter':
                if epoch > 1 and val_bacc <= best_val_acc: # In risk mode val_bacc stores C-index
                    lr_scheduler_counter += 1
                    if lr_scheduler_counter >= lr_scheduler_patience:
                        lr_scheduler_counter = 0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_scheduler_factor
                            print(f'\nReduced learning rate to: {param_group["lr"]}\n')
                else: # Bacc improved or first epoch, reset counter
                    lr_scheduler_counter = 0
            elif lrrr == 'reduceplateau':
                if epoch > 1:
                    scheduler.step(val_bacc)
            elif lrrr == 'ritap':
                scheduler.step()

        # Early stopping check
        if early_stopping_counter >= early_stopping_rounds:
            print(f'\nEarly stopping triggered after {early_stopping_rounds} epochs of no improvement.\n')
            break

        # Add scalars to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Evaluation train', avg_loss_train, epoch)
        writer.add_scalar('Loss/Evaluation validation', avg_loss_val, epoch)
        if task == 'risk':
            writer.add_scalar('CIndex/Evaluation validation', val_bacc, epoch)
            writer.add_scalar('CIndex/Evaluation train', train_bacc, epoch)
        else:
            writer.add_scalar('Bacc/Evaluation validation', val_bacc, epoch)
            writer.add_scalar('Bacc/Evaluation train', train_bacc, epoch)
        writer.add_scalar('AUC/Evaluation train', auc_t, epoch)
        writer.add_scalar('AUC/Evaluation validation', auc_v, epoch)

    # Load the best model
    print('\nTraining complete: loading the best model')
    model.load_state_dict(torch.load(best_model_path))
    
    # Log hyperparameters and close the writer
    writer.add_scalar('Variables/num_epochs', num_epochs)
    writer.add_scalar('Variables/hidden_channels', hidden_channels)
    writer.add_scalar('Variables/lr', lr)
    writer.add_scalar('Variables/num_node_features', num_node_features)
    writer.add_scalar('Variables/weights0', class_weights[0])
    writer.add_scalar('Variables/weights1', class_weights[1])
    writer.add_scalar('Variables/Early-stopping-rounds', early_stopping_rounds)
    writer.add_scalar('Variables/LR-scheduler-patience', lr_scheduler_patience)
    writer.add_scalar('Variables/LR-scheduler-factor', lr_scheduler_factor)
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
    writer.add_scalar('Best_model/val_acc', best_val_acc) # For risk mode this stores best C-index
    writer.close()

    return (best_val_acc,  # 0: This is now the best patient-level BAcc (or loss if tracking loss)
            best_score_matrix_v,  # 1: Sample-level score matrix for best model
            best_cm_v,  # 2: Sample-level CM for best model
            best_score_matrix_t,  # 3: Sample-level score matrix for best model (training)
            best_cm,  # 4: Sample-level CM for best model (training)
            # Removed redundant last-epoch patient-level predictions/BAcc/reports
            best_bacc_majority_voting,  # 5: Best patient-level Majority Voting BAcc
            best_class_report_majority_voting,  # 6: Best patient-level Majority Voting report
            best_conf_matrix_majority_voting,  # 7: Best patient-level Majority Voting CM
            best_bacc_one_dominance,  # 8: Best patient-level 1-Dominance BAcc
            best_class_report_one_dominance,  # 9: Best patient-level 1-Dominance report
            best_conf_matrix_one_dominance, # 10: Best patient-level 1-Dominance CM
            best_recall_v, # 11: Sample-level recall for best model
            best_specificity_v, # 12: Sample-level specificity for best model
            best_recall_mj_v, # 13: Patient-level Majority Voting Recall for best model
            best_specificity_mj_v, # 14: Patient-level Majority Voting Specificity for best model
            best_recall_1d_v, # 15: Patient-level 1-Dominance Recall for best model
            best_specificity_1d_v, # 16: Patient-level 1-Dominance Specificity for best model
            best_optimal_threshold_v, # 17: Optimal threshold for best model
            best_y_true_wsi_level, 
            best_probs_wsi_level,
            best_y_true_patient_level,
            best_patient_mv_scores, 
            best_patient_1d_scores) 
    


###################################
# Validation Function
###################################
def validate_sigmoid_threshold(loader, model, weight_tensor, device, mode, criterion, calculate_threshold,survival,all_t):
    model.eval()
    total_loss = 0.0
    
    y_true_wsi_level = []      # For WSI-level true labels
    probs_wsi_level = []       # For WSI-level probabilities
    
    # Dictionaries for patient-level aggregation
    patient_wsi_probs = defaultdict(list)    # Stores probabilities for each WSI of a patient
    patient_true_labels = {}                 # Stores the single true label for each patient (WSI ID is patient ID here)

    with torch.no_grad():
        with tqdm(total=len(loader), desc=f'Validation - WSI Level', leave=False) as pbar:
            for data in loader:
                data = data.to(device)

                # Ensure target shape matches output shape
                data.y = data.y.view(-1, 1).float()

                if isinstance(data, HeteroData):
                    if survival == 'gat':
                        out,_ = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                    else:
                        out = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                elif 'hyper' in all_t:
                    out = model(data.x, data.edge_index, data.hyperedge_index, data.batch)
                else:
                    if survival == 'gat' and 'hyper' not in all_t:
                        out,_ = model(data.x, data.edge_index, data.batch)
                    else:
                        out = model(data.x, data.edge_index, data.batch)

                probs = torch.sigmoid(out).view(-1).cpu().numpy()  # probs now contains one probability per WSI in the batch

                probs_wsi_level.extend(probs)
                y_true_wsi_level.extend(data.y.cpu().numpy().flatten())

                # Calculate loss
                loss = criterion(out, data.y)
                total_loss += loss.item() # Assuming criterion is reduction='sum' or already averaged per batch item

                # Store WSI-wise probabilities and true labels for patient-level aggregation
                # Iterating over the number of WSIs in the current batch
                for i in range(len(probs)):
                    current_wsi_id = data.case_id[i]
                    current_wsi_true_label = data.y[i].item() # Correctly extracts WSI label for i-th WSI
                    current_wsi_prob = probs[i]

                    # Store the true label for this WSI ID (patient ID).
                    # This will set the true label once per unique WSI ID encountered.
                    if current_wsi_id not in patient_true_labels:
                        patient_true_labels[current_wsi_id] = current_wsi_true_label
                    
                    patient_wsi_probs[current_wsi_id].append(current_wsi_prob)
                pbar.update(1)

    # Calculate optimal threshold
    if calculate_threshold == 'True':
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true_wsi_level, probs_wsi_level)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    else: # calculate_threshold == 'False'
        optimal_threshold = 0.5  # Default threshold if not calculating

    # Convert WSI-level probabilities to binary predictions using the optimal threshold
    # This prepares the `patient_wsi_probs` for the majority_voting/one_dominance helpers
    patient_wsi_binary_preds = defaultdict(list)
    for wsi_id, probs_list_for_wsi in patient_wsi_probs.items():
        patient_wsi_binary_preds[wsi_id] = [1 if prob >= optimal_threshold else 0 for prob in probs_list_for_wsi]
    
    # Generate WSI-level binary predictions for overall metrics
    y_pred_wsi_level = (np.array(probs_wsi_level) >= optimal_threshold).astype(int)
    
    # Calculate majority voting and one-dominance predictions per patient
    patient_pred_majority_voting = {}
    patient_pred_one_dominance = {}

    for wsi_id in patient_true_labels.keys(): # Iterate through unique WSI IDs
        # Pass the binarized WSI predictions to the helper functions
        patient_pred_majority_voting[wsi_id] = majority_voting(patient_wsi_binary_preds[wsi_id])
        patient_pred_one_dominance[wsi_id] = one_dominance(patient_wsi_binary_preds[wsi_id])
            
    # Convert patient_true_labels and patient_pred_X to lists for sklearn metrics
    sorted_wsi_ids = sorted(patient_true_labels.keys()) # Ensure consistent order
    y_true_patient_level = [patient_true_labels[wsi_id] for wsi_id in sorted_wsi_ids]
    y_pred_patient_majority_voting = [patient_pred_majority_voting[wsi_id] for wsi_id in sorted_wsi_ids]
    y_pred_patient_one_dominance = [patient_pred_one_dominance[wsi_id] for wsi_id in sorted_wsi_ids]

    # --- NEW: continuous patient-level scores for ROC ---
    # MV score = mean of WSI probabilities for that patient
    patient_mv_scores = np.array([
        np.mean(patient_wsi_probs[wsi_id]) for wsi_id in sorted_wsi_ids
    ])

    # 1D score = max WSI probability for that patient
    patient_1d_scores = np.array([
        np.max(patient_wsi_probs[wsi_id]) for wsi_id in sorted_wsi_ids
    ])

    y_true_patient_level = np.array(y_true_patient_level)


    # ----------------------------------------
    # Calculate WSI-level (Sample-level) metrics
    # ----------------------------------------
    correct_wsi_level = sum(1 for true, pred in zip(y_true_wsi_level, y_pred_wsi_level) if true == pred)
    # The 'len(loader.dataset)' is the total number of WSIs in the validation set
    accuracy_wsi_level = correct_wsi_level / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset) # Average loss per WSI
    
    balanced_acc_wsi_level = balanced_accuracy_score(y_true_wsi_level, y_pred_wsi_level)
    auc_wsi_level = roc_auc_score(y_true_wsi_level, probs_wsi_level) # AUC uses probabilities
    class_report_wsi_level = classification_report(y_true_wsi_level, y_pred_wsi_level, zero_division=1)
    cm_wsi_level = confusion_matrix(y_true_wsi_level, y_pred_wsi_level)
    tn_s, fp_s, fn_s, tp_s = cm_wsi_level.ravel()
    recall_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
    specificity_s = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0


    # ----------------------------------------
    # Calculate Patient-level (Majority Voting) metrics
    # ----------------------------------------
    bacc_majority_voting = balanced_accuracy_score(y_true_patient_level, y_pred_patient_majority_voting)
    class_report_majority_voting = classification_report(y_true_patient_level, y_pred_patient_majority_voting, zero_division=1)
    conf_matrix_majority_voting = confusion_matrix(y_true_patient_level, y_pred_patient_majority_voting)
    tnmj, fpmj, fnmj, tpmj = conf_matrix_majority_voting.ravel()
    recall_mj = tpmj / (tpmj + fnmj) if (tpmj + fnmj) > 0 else 0
    specificity_mj = tnmj / (tnmj + fpmj) if (tnmj + fpmj) > 0 else 0
    
    if mode == 'val':
        print(f'\nTotal majority voting patients correct: {sum(t == p for t, p in zip(y_true_patient_level, y_pred_patient_majority_voting))}/{len(y_true_patient_level)}')


    # ----------------------------------------
    # Calculate Patient-level (One-Dominance) metrics
    # ----------------------------------------
    bacc_one_dominance = balanced_accuracy_score(y_true_patient_level, y_pred_patient_one_dominance)
    class_report_one_dominance = classification_report(y_true_patient_level, y_pred_patient_one_dominance, zero_division=1)
    conf_matrix_one_dominance = confusion_matrix(y_true_patient_level, y_pred_patient_one_dominance)
    tn1d, fp1d, fn1d, tp1d = conf_matrix_one_dominance.ravel()
    recall_1d = tp1d / (tp1d + fn1d) if (tp1d + fn1d) > 0 else 0
    specificity_1d = tn1d / (tn1d + fp1d) if (tn1d + fp1d) > 0 else 0

    if mode == 'val':
        print(f'Total 1-D patients correct: {sum(t == p for t, p in zip(y_true_patient_level, y_pred_patient_one_dominance))}/{len(y_true_patient_level)}')

        # Convert lists to numpy arrays for later ROC computation
    y_true_wsi_level = np.array(y_true_wsi_level)
    probs_wsi_level = np.array(probs_wsi_level)

    return (avg_loss, accuracy_wsi_level, balanced_acc_wsi_level, class_report_wsi_level, cm_wsi_level,
            y_pred_patient_majority_voting, bacc_majority_voting, class_report_majority_voting, conf_matrix_majority_voting,
            y_pred_patient_one_dominance, bacc_one_dominance, class_report_one_dominance, conf_matrix_one_dominance,
            auc_wsi_level, recall_s, specificity_s, recall_mj, specificity_mj, recall_1d, specificity_1d, optimal_threshold,y_true_wsi_level, probs_wsi_level, y_true_patient_level, patient_mv_scores, patient_1d_scores)


###################################
# Test Function
###################################
def test_model_on_loader(loader, model, device, threshold=0.5, criterion=None):
    model.eval()
    
    y_true_wsi_level = []
    probs_wsi_level = []
    total_loss = 0.0
    
    patient_wsi_probs = defaultdict(list)
    patient_true_labels = {}

    with torch.no_grad():
        with tqdm(total=len(loader), desc=f'Testing - WSI Level', leave=False) as pbar:
            for data in loader:
                data = data.to(device)
                data.y = data.y.view(-1, 1).float()

                if isinstance(data, HeteroData):
                    if survival == 'gat':
                        out,_ = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                    else:
                        out = model(data.x_dict, data.edge_index_dict, data['tile'].batch)
                else:
                    if survival == 'gat':
                        out,_ = model(data.x, data.edge_index, data.batch)
                    else:
                        out = model(data.x, data.edge_index, data.batch)

                probs = torch.sigmoid(out).view(-1).cpu().numpy()

                probs_wsi_level.extend(probs)
                y_true_wsi_level.extend(data.y.cpu().numpy().flatten())

                # Store WSI-wise probabilities and true labels for patient-level aggregation
                # Iterating over the number of WSIs in the current batch
                for i in range(len(probs)):
                    current_wsi_id = data.case_id[i]
                    current_wsi_true_label = data.y[i].item()
                    current_wsi_prob = probs[i]

                    # Store the true label for this WSI ID (patient ID).
                    # This will set the true label once per unique WSI ID encountered.
                    if current_wsi_id not in patient_true_labels:
                        patient_true_labels[current_wsi_id] = current_wsi_true_label
                    
                    patient_wsi_probs[current_wsi_id].append(current_wsi_prob)

                if criterion:
                    total_loss += criterion(out, data.y).item()
                pbar.update(1)

    # Convert WSI-level probabilities to binary predictions using the provided threshold
    y_pred_wsi_level = (np.array(probs_wsi_level) >= threshold).astype(int)
    
    # ----------------------------------------
    # Consistency for patient-level aggregation:
    # First, binarize probabilities for each WSI within a patient
    patient_wsi_binary_preds = defaultdict(list)
    for wsi_id, probs_list_for_wsi in patient_wsi_probs.items():
        patient_wsi_binary_preds[wsi_id] = [1 if prob >= threshold else 0 for prob in probs_list_for_wsi]
    
    # Then, apply the consistent helper functions
    patient_pred_majority_voting = {}
    patient_pred_one_dominance = {}
    for wsi_id in patient_true_labels.keys():
        patient_pred_majority_voting[wsi_id] = majority_voting(patient_wsi_binary_preds[wsi_id])
        patient_pred_one_dominance[wsi_id] = one_dominance(patient_wsi_binary_preds[wsi_id])
    
    # Convert patient_true_labels and aggregated predictions to lists for sklearn metrics
    sorted_wsi_ids = sorted(patient_true_labels.keys())
    y_true_patient_level = [patient_true_labels[wsi_id] for wsi_id in sorted_wsi_ids]
    y_pred_patient_majority_voting = [patient_pred_majority_voting[wsi_id] for wsi_id in sorted_wsi_ids]
    y_pred_patient_one_dominance = [patient_pred_one_dominance[wsi_id] for wsi_id in sorted_wsi_ids]
    # ----------------------------------------


    # ----------------------------------------
    # Calculate WSI-level (Sample-level) metrics
    # ----------------------------------------
    correct_wsi_level = sum(int(t == p) for t, p in zip(y_true_wsi_level, y_pred_wsi_level))
    total_wsis = len(y_true_wsi_level)
    accuracy_wsi_level = correct_wsi_level / total_wsis
    avg_loss = total_loss / total_wsis if criterion else None
    
    bacc_wsi_level = balanced_accuracy_score(y_true_wsi_level, y_pred_wsi_level)
    auc_wsi_level = roc_auc_score(y_true_wsi_level, probs_wsi_level)
    cm_wsi_level = confusion_matrix(y_true_wsi_level, y_pred_wsi_level)
    report_wsi_level = classification_report(y_true_wsi_level, y_pred_wsi_level, zero_division=1)
    tn_s, fp_s, fn_s, tp_s = cm_wsi_level.ravel()
    recall_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
    specificity_s = tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0

    # ----------------------------------------
    # Calculate Patient-level (Majority Voting) metrics
    # ----------------------------------------
    bacc_mj = balanced_accuracy_score(y_true_patient_level, y_pred_patient_majority_voting)
    cm_mj = confusion_matrix(y_true_patient_level, y_pred_patient_majority_voting)
    report_mj = classification_report(y_true_patient_level, y_pred_patient_majority_voting, zero_division=1)
    tnmj, fpmj, fnmj, tpmj = cm_mj.ravel()
    recall_mj = tpmj / (tpmj + fnmj) if (tpmj + fnmj) > 0 else 0
    specificity_mj = tnmj / (tnmj + fpmj) if (tnmj + fpmj) > 0 else 0

    # ----------------------------------------
    # Calculate Patient-level (One-Dominance) metrics
    # ----------------------------------------
    bacc_1d = balanced_accuracy_score(y_true_patient_level, y_pred_patient_one_dominance)
    cm_1d = confusion_matrix(y_true_patient_level, y_pred_patient_one_dominance)
    report_1d = classification_report(y_true_patient_level, y_pred_patient_one_dominance, zero_division=1)
    tn1d, fp1d, fn1d, tp1d = cm_1d.ravel()
    recall_1d = tp1d / (tp1d + fn1d) if (tp1d + fn1d) > 0 else 0
    specificity_1d = tn1d / (tn1d + fp1d) if (tn1d + fp1d) > 0 else 0

    return [
        bacc_wsi_level,           # [0] WSI-level Balanced Accuracy
        report_wsi_level,         # [1] WSI-level Classification Report
        cm_wsi_level,             # [2] WSI-level Confusion Matrix
        auc_wsi_level,            # [3] WSI-level AUC-ROC
        recall_s,                 # [4] WSI-level Recall
        specificity_s,            # [5] WSI-level Specificity
        bacc_mj,                  # [6] Patient-level Majority Voting Balanced Accuracy
        recall_mj,                # [7] Patient-level Majority Voting Recall
        specificity_mj,           # [8] Patient-level Majority Voting Specificity
        report_mj,                # [9] Patient-level Majority Voting Classification Report
        cm_mj,                    # [10] Patient-level Majority Voting Confusion Matrix
        bacc_1d,                  # [11] Patient-level One-Dominance Balanced Accuracy
        recall_1d,                # [12] Patient-level One-Dominance Recall
        specificity_1d,           # [13] Patient-level One-Dominance Specificity
        report_1d,                # [14] Patient-level One-Dominance Classification Report
        cm_1d                     # [15] Patient-level One-Dominance Confusion Matrix
    ]

############
######################
# Majority voting and one dominance
###################################
def majority_voting(predictions):
    if len(predictions) == 1:
        return predictions[0]  # Direct prediction
    return max(set(predictions), key=predictions.count)


def one_dominance(predictions):
    if 0 in predictions:
        return 0
    else:
        return 1



def collect_feature_stats(dataloader):
    all_features = []

    for data in dataloader:
        x = data.x.cpu()
        all_features.append(x)

    all_features = torch.cat(all_features, dim=0)  # [total_nodes, num_features]
    feature_means = all_features.mean(dim=0)
    feature_stds = all_features.std(dim=0)

    return feature_means, feature_stds, all_features







########################################################
#No use
#########################################################
    
'''
def validate_sigmoid_threshold(loader, model, weight_tensor, device, mode,criterion,calculate_threshold): 
    model.eval()
    total_loss = 0.0
    correct = 0
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
    y_true = []
    y_pred = []
    probs_list = []  # Store probabilities for threshold calculation
    y_predd = defaultdict(list)  # Dictionary to store predictions per patient
    y_truee = defaultdict(list)  # Dictionary to store true labels per patient
    majority_voting_predictions = defaultdict(list)
    one_dominance_predictions = defaultdict(list)
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Ensure target shape matches output shape
            data.y = data.y.view(-1, 1).float()

            out = model(data.x, data.edge_index, data.batch)
            #pred = out.argmax(dim=1)
            
            ##probs = torch.sigmoid(out).squeeze()  # Convert logits to probabilities
             #probs = torch.sigmoid(out).squeeze()  # Convert logits to probabilities
            ##probs_list.extend(probs.cpu().numpy())  # Store probabilities for global threshold calculation
            
            probs = torch.sigmoid(out).view(-1).cpu().numpy()  # Ensure 1D array
            probs_list.extend(probs)


            # Store true labels for metric calculation
            y_true.extend(data.y.cpu().numpy().flatten())

            # Calculate loss
            loss = criterion(out, data.y)
            total_loss += criterion(out, data.y).item() #* data.num_graphs

            # Store patient-wise predictions
            for id in range(len(probs)):
                patient_id_str = data.case_id[id]
                y_predd[patient_id_str].append(probs[id].item())
                y_truee[patient_id_str] = data.y[id].item()

    # Calculate optimal threshold
    if calculate_threshold == 'True':
        #print('\nCalculating Optimal Threshold...')
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, probs_list)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
    elif calculate_threshold == 'False':
        optimal_threshold = 0.5  # Default threshold if not calculating

    #print(f"Optimal Threshold: {optimal_threshold:.4f}")
    # Convert probabilities to predictions using the optimal threshold
    for id in y_predd.keys():
        y_predd[id] = [1 if prob >= optimal_threshold else 0 for prob in y_predd[id]]
    
    # Apply threshold to generate binary predictions
    y_pred = (np.array(probs_list) >= optimal_threshold).astype(int)
    
    # Count correct predictions for `correct` metric
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    
    # Calculate majority voting and one-dominance predictions per patient
    for id in y_predd.keys():
        majority_voting_predictions[id] = majority_voting(y_predd[id])
        one_dominance_predictions[id] = one_dominance(y_predd[id]) 

        # Print direct prediction, majority voting, and one-dominance for each patient
        #print(f"Patient ID: {id}, Direct Prediction: {y_predd[id]}, Majority Voting: {majority_voting_predictions[id]}, One-Dominance: {one_dominance_predictions[id]}")
          
    match_count_maj = sum(1 for true, pred in zip(y_truee.values(), majority_voting_predictions.values()) if true == pred)
    match_count_onedominanc = sum(1 for true, pred in zip(y_truee.values(), one_dominance_predictions.values()) if true == pred)
    
    if mode == 'val':
        print(f'\nTotal majority voting patients correct: {match_count_maj}/{len(y_predd)}')
        print(f'Total 1-D patients correct: {match_count_onedominanc}/{len(y_predd)}')

    print('correct',correct)
    # Calculate metrics
    accuracy = correct / len(loader.dataset)  # Derive ratio of correct predictions
    avg_loss = total_loss / len(loader.dataset)  # Calculate average loss

    # Compute balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(y_true, y_pred, zero_division=1)

    # Calculate confusion matrix for validation set
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Calculate Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Calculate Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    ### Calculate metrics for majority voting
    bacc_majority_voting = balanced_accuracy_score(list(y_truee.values()), list(majority_voting_predictions.values()))
    class_report_majority_voting = classification_report(list(y_truee.values()), list(majority_voting_predictions.values()), zero_division=1)
    conf_matrix_majority_voting = confusion_matrix(list(y_truee.values()), list(majority_voting_predictions.values()))
    tnmj, fpmj, fnmj, tpmj = confusion_matrix(list(y_truee.values()), list(majority_voting_predictions.values())).ravel()
    # Calculate Recall (Sensitivity)
    recall_mj = tpmj / (tpmj + fnmj) if (tpmj + fnmj) > 0 else 0
    # Calculate Specificity
    specificity_mj = tnmj / (tnmj + fpmj) if (tnmj + fpmj) > 0 else 0
    
    ### Calculate metrics for one-dominance
    bacc_one_dominance = balanced_accuracy_score(list(y_truee.values()), list(one_dominance_predictions.values()))
    class_report_one_dominance = classification_report(list(y_truee.values()), list(one_dominance_predictions.values()), zero_division=1)
    conf_matrix_one_dominance = confusion_matrix(list(y_truee.values()), list(one_dominance_predictions.values()))
    tn1d, fp1d, fn1d, tp1d = confusion_matrix(list(y_truee.values()), list(one_dominance_predictions.values())).ravel()
    # Calculate Recall (Sensitivity)
    recall_1d = tp1d / (tp1d + fn1d) if (tp1d + fn1d) > 0 else 0
    # Calculate Specificity
    specificity_1d = tn1d / (tn1d + fp1d) if (tn1d + fp1d) > 0 else 0

    return avg_loss, accuracy, balanced_acc, class_report, cm, \
           majority_voting_predictions, bacc_majority_voting, class_report_majority_voting, conf_matrix_majority_voting, \
           one_dominance_predictions, bacc_one_dominance, class_report_one_dominance, conf_matrix_one_dominance,auc,recall,specificity,recall_mj,specificity_mj,recall_1d,specificity_1d,optimal_threshold

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, roc_auc_score
from collections import defaultdict
import numpy as np
import torch

def test_model_on_loader(loader, model, device, threshold=0.5, criterion=None):
    model.eval()
    y_true = []
    probs_list = []
    total_loss = 0.0
    y_predd = defaultdict(list)
    y_truee = defaultdict(list)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.y = data.y.view(-1, 1).float()

            out = model(data.x, data.edge_index, data.batch)
            probs = torch.sigmoid(out).view(-1).cpu().numpy()

            probs_list.extend(probs)
            y_true.extend(data.y.cpu().numpy().flatten())

            for id in range(len(probs)):
                pid = data.case_id[id]
                y_predd[pid].append(probs[id])
                y_truee[pid] = data.y[id].item()

            if criterion:
                total_loss += criterion(out, data.y).item()

    # Binary predictions
    y_pred = (np.array(probs_list) >= threshold).astype(int)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))

    total_samples = len(y_true)
    acc = correct / total_samples
    avg_loss = total_loss / total_samples if criterion else None
    bacc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs_list)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=1)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Majority voting using same threshold
    y_pred_major = [int(np.mean(v) >= threshold) for v in y_predd.values()]
    y_true_major = list(y_truee.values())
    bacc_mj = balanced_accuracy_score(y_true_major, y_pred_major)
    cm_mj = confusion_matrix(y_true_major, y_pred_major)
    report_mj = classification_report(y_true_major, y_pred_major, zero_division=1)
    tnmj, fpmj, fnmj, tpmj = cm_mj.ravel()
    recall_mj = tpmj / (tpmj + fnmj) if (tpmj + fnmj) > 0 else 0
    specificity_mj = tnmj / (tnmj + fpmj) if (tnmj + fpmj) > 0 else 0

    # One-dominance
    y_pred_1d = [int(sum(v) >= 1) for v in y_predd.values()]
    bacc_1d = balanced_accuracy_score(y_true_major, y_pred_1d)
    cm_1d = confusion_matrix(y_true_major, y_pred_1d)
    report_1d = classification_report(y_true_major, y_pred_1d, zero_division=1)
    tn1d, fp1d, fn1d, tp1d = cm_1d.ravel()
    recall_1d = tp1d / (tp1d + fn1d) if (tp1d + fn1d) > 0 else 0
    specificity_1d = tn1d / (tn1d + fp1d) if (tn1d + fp1d) > 0 else 0

    return [
        bacc,              # [0]
        report,            # [1]
        cm,                # [2]
        auc,               # [3]
        recall,            # [4]
        specificity,       # [5]
        bacc_mj,           # [6]
        recall_mj,         # [7]
        specificity_mj,    # [8]
        report_mj,         # [9]
        cm_mj,             # [10]
        bacc_1d,           # [11]
        recall_1d,         # [12]
        specificity_1d,    # [13]
        report_1d,         # [14]
        cm_1d              # [15]
    ]


'''
'''def get_csv_paths():
    """Return file paths for different dataset configurations."""
    return {
        'all_tiago': {
            'train': "/home/ritav/Graphs/cptac_233_adj_selfloop/encoded_train_allfiles.csv",
            'val': "/home/ritav/Graphs/cptac_233_adj_selfloop/encoded_test_allfiles.csv"
        },
        'patient_tiago': {
            'train': "//home/ritav/Graphs/cptac_233_adj_selfloop/filtered_combined_case_ids_train.csv",
            'val': "/home/ritav/Graphs/cptac_233_adj_selfloop/filtered_combined_case_ids_test.csv"
        }
    }'''
'''def train_accum_graddient_new_sigmoid_threshold(train_loader, val_loader, model, criterion, optimizer, device, early_stopping_rounds,
        lr_scheduler_patience, lr_scheduler_factor, writer, best_model_path, weight_tensor, num_epochs,
        hidden_channels, lr, num_node_features, class_weights,best_val_acc,best_score_matrix_v,best_cm_v,best_score_matrix_t, best_cm, fold, lrrr, grad_accum, load, survival,
        all_t, actualtime, effective_batch,calculate_threshold,track,track_lr ,repeat,num_neighbors,node_batch_size,batch_sizee
):
    """
    Train the model with gradient accumulation and validation, implementing early stopping and learning rate scheduling.

    Parameters:
    - train_loader, val_loader: Dataloaders for training and validation.
    - model: Model to be trained.
    - criterion: Loss function.
    - optimizer: Optimizer.
    - device: Device to run computations on.
    - early_stopping_rounds: Rounds for early stopping.
    - lr_scheduler_patience, lr_scheduler_factor: Learning rate scheduler parameters.
    - writer: TensorBoard writer.
    - best_model_path: Path to save the best model.
    - weight_tensor: Weights for the loss function.
    - num_epochs, hidden_channels, lr, num_node_features: Hyperparameters.
    - class_weights, fold, grad_accum: Additional training parameters.
    - load: Boolean to load previous model checkpoint.
    - survival, all_t, actualtime: Additional identifiers.
    - effective_batch: Effective batch size for gradient accumulation.
    - lrrr: Learning rate reduction mode.
    """
    best_val_acc = 0.0
    best_avg_loss_val=1000
    early_stopping_counter = 0
    lr_scheduler_counter = 0
    if lrrr == 'ritap':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    elif lrrr == 'reduceplateau':
        if track_lr =='loss':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_scheduler_patience)
        elif track_lr =='bacc':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=lr_scheduler_patience, factor=0.5)
            
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=lr_scheduler_patience,factor=lr_scheduler_factor, verbose=True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 0.000005)

    print('\n\nStarting training with gradient accumulation. Effective batch size:', effective_batch)

    # Load checkpoint if specified
    if load == 'True': 
            if 'SAGE' in survival:
                start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, device, f'Checkpoints-univ2/Checkpoint_cptac_adj_Sage',
            filename=f'{all_t}_last_checkpoint_{survival}.pth.tar'
        )
            else:
                start_epoch, best_val_acc = load_checkpoint(
            model, optimizer, device, f'Checkpoints-univ2/Checkpoint_cptac_adj_{survival}',
            filename=f'{all_t}_last_checkpoint_{survival}.pth.tar'
        )

    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_true_labels = []
        total_pred_labels = []
        
        total_loss = 0.0  # To accumulate total loss for the epoch
        num_samples = 0 
        # Set a warm-up period
        warmup_epochs = 5  # Number of epochs before allowing early stopping or model saving
        model.train()
        optimizer.zero_grad()  # Initialize gradients
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs} - WSI Level',leave=True) as wsi_pbar:

            # Identify last GCN layer (if needed for debugging)
                #last_gcn_layer = next(
                #(name for name, param in model.named_parameters() if
               #  'conv' in name or 'gcn' in name or 'graphconv' in name),
               # None
            )           

            # Training loop with gradient accumulation
            for i, data in enumerate(train_loader, 1):
                if "SAGE" in survival:
                    wsi_loader = NeighborLoader(
                        data,
                        num_neighbors=num_neighbors,
                        batch_size=node_batch_size,
                        input_nodes=torch.arange(data.x.size(0)),
                        shuffle=True,
                        drop_last=False,
                        generator=torch.Generator().manual_seed(47),
                        pin_memory=True
                    )
                    print('wsi_loader \n')
                     # Subgraph-Level Progress Bar
                    co=0
                    #with tqdm(total=len(wsi_loader), desc=f'WSI {i*batch_sizee} - Subgraphs', leave=False) as subgraph_pbar:
                    for subgraph in wsi_loader:
                        #print('\n## New subgraph\n',subgraph)
                        #print(f" - image_filename: {subgraph.image_filename}")
                        #subgraph = subgraph.to(device)
                        # Forward pass with global max pooling (graph-level output)
                        #output = model(subgraph.x, subgraph.edge_index, subgraph.batch)  # Output: [num_graphs, 1]

                        # Ensure subgraph.y matches the number of graphs (not nodes)
                        #subgraph.y = subgraph.y[:output.size(0)].view(-1, 1).float()

                        # Check dimensions before computing loss
                        #assert output.size() == subgraph.y.size(), f"Output: {output.size()}, Target: {subgraph.y.size()}"

                        # Compute loss
                        #loss = criterion(output, subgraph.y) / effective_batch
                        #loss.backward()

                        # Update total loss
                        #total_loss += criterion(output, subgraph.y).item() #* subgraph.num_graphs
                        #num_samples += subgraph.num_graphs
                        subgraph = subgraph.to(device)
                        output = model(subgraph.x, subgraph.edge_index, subgraph.batch)
    
                        # Assign full WSI label to all subgraphs
                        target = full_wsi_label[:output.size(0)]
    
                        loss = criterion(output, target) / effective_batch
                        loss.backward()
    
                        # Accumulate raw (unscaled) loss for logging
                        total_loss += criterion(output, target).item()
                        total_samples += output.size(0)
                    
                        co=co+1
                        print(co)

                        # Update WSI progress
                        #subgraph_pbar.update(1)

                    # Optimizer step after processing subgraphs
                    if i % grad_accum == 0 or i == len(train_loader):
                        #print('\n\n\n ##### \n\n\naccum')
                        optimizer.step()
                        optimizer.zero_grad()
                        
                else:
                
                    data = data.to(device)  # Move data to GPU
                # Ensure target shape matches the output shape
                    data.y = data.y.view(-1, 1).float()  # Reshape target to [batch_size, 1]

                    out = model(data.x, data.edge_index, data.batch)  # Output logits of shape [batch_size, 1]

                    # Calculate loss
                    loss = criterion(out, data.y) / effective_batch  # Scale the loss for gradient accumulation
                    loss.backward()

                    # Step optimizer if gradient accumulation steps are met
                    if i % grad_accum == 0 or i == len(train_loader):
                        optimizer.step()
                        optimizer.zero_grad()

                    #train_loss += loss.item() #* data.num_graphs

                    total_loss += criterion(out, data.y).item() #* data.num_graphs  # Unscale to match the dataset size
                    num_samples += data.num_graphs 

                # Update WSI progress
                wsi_pbar.update(1)  # Update tqdm progress bar

            # Calculate epoch training metrics
            train_loss = total_loss / num_samples
            
        #track_batch_order(train_loader, epoch, fold, actualtime, all_t)

        # Validation loop
        mode = 'val'  # just to print maj and 1-d once for validation
        avg_loss_val, val_acc, val_bacc, score_matrix_v, cm_v, majority_voting_predictions, bacc_majority_voting, class_report_majority_voting, conf_matrix_majority_voting, one_dominance_predictions, bacc_one_dominance, class_report_one_dominance, conf_matrix_one_dominance,auc_v,recall_v,specificity_v,recall_mj_v,specificity_mj_v,recall_1d_v,specificity_1d_v,optimal_threshold_v = validate_sigmoid_threshold(
            val_loader, model, weight_tensor, device, mode,criterion,calculate_threshold)
        mode = 'train'
        avg_loss_train, train_accuracy, train_bacc, score_matrix_t, cm_t, _, _, _, _, _, _, _, _,auc_t,_,_,_,_,_,_,_ = validate_sigmoid_threshold(train_loader,model,weight_tensor,device, mode,criterion,calculate_threshold)
        
        
        if 'SAGE' in survival:
            # Save checkpoint at the end of each epoch
            save_checkpoint(model, optimizer, epoch, val_acc,f'Checkpoints-univ2/Checkpoint_cptac_adj_Sage',
                        filename=f'{all_t}_last_checkpoint_{survival}.pth.tar')

        else:
            # Save checkpoint at the end of each epoch
            save_checkpoint(model, optimizer, epoch, val_acc,f'Checkpoints-univ2/Checkpoint_cptac_adj_{survival}',
                        filename=f'{all_t}_last_checkpoint_{survival}.pth.tar')

        # Print epoch summary
        print(f'\n\nFOLD {fold + 1} repeat {repeat+1}- Epoch: {epoch:03d}, Train Loss: {train_loss:.6f}, Train Acc: {train_accuracy:.6f}, Train Bacc: {train_bacc:.6f} Train auc: {auc_t}, Val Loss: {avg_loss_val:.6f}, Val Acc: {val_acc:.6f}, Val Bacc: {val_bacc:.6f} Val auc: {auc_v}\n -------------------------------------------------------------------------\n')
        print('Train classification report:')
        print(cm_t)
        print(score_matrix_t)
        print('\n')
        print('Val classification report:')
        print(cm_v)
        print(score_matrix_v)
        print('\n')

        # Save the model if validation accuracy improves
        if track =='loss':
            if avg_loss_val < best_avg_loss_val:
                best_avg_loss_val = avg_loss_val
                best_val_acc = val_bacc
                best_score_matrix_v = score_matrix_v
                best_cm_v = cm_v
                best_score_matrix_t = score_matrix_t
                best_cm = cm_t
                best_bacc_majority_voting = bacc_majority_voting
                best_bacc_one_dominance = bacc_one_dominance
                best_majority_voting_predictions = majority_voting_predictions
                best_class_report_majority_voting = class_report_majority_voting
                best_conf_matrix_majority_voting = conf_matrix_majority_voting
                best_one_dominance_predictions = one_dominance_predictions
                best_class_report_one_dominance = class_report_one_dominance
                best_conf_matrix_one_dominance = conf_matrix_one_dominance
                best_recall_v = recall_v 
                best_specificity_v = specificity_v
                best_recall_mj_v = recall_mj_v
                best_specificity_mj_v = specificity_mj_v
                best_recall_1d_v = recall_1d_v 
                best_specificity_1d_v = specificity_1d_v
                best_optimal_threshold_v = optimal_threshold_v

                torch.save(model.state_dict(), best_model_path)
                print(f'\nSaved the lowest model with Val loss: {val_bacc:.6f} at Epoch: {epoch:03d}\n')
                early_stopping_counter = 0  # Reset early stopping counter if accuracy improves
            else:
                early_stopping_counter += 1  # Increment early stopping counter
                
            if lrrr == 'counter':
                if epoch > 1 and avg_loss_val <= best_avg_loss_val:
                    lr_scheduler_counter += 1
                    if lr_scheduler_counter >= lr_scheduler_patience:
                        lr_scheduler_counter = 0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_scheduler_factor
                            print(f'\nReduced learning rate to: {param_group["lr"]}\n')
            elif lrrr == 'reduceplateau':
                if epoch > 1:
                    if track_lr  == 'bacc':
                        scheduler.step(val_bacc)  # Update learning rate based on validation accurac
                    elif track_lr  =='loss':
                        scheduler.step(avg_loss_val)
            elif lrrr=='ritap':
                scheduler.step()

                
        # Save the model if test accuracy improves
        elif track == 'bacc':
            # Skip model saving and early stopping during the warm-up period
            if epoch > warmup_epochs:
                # Save the model if test accuracy improves
                if val_bacc > best_val_acc:
                    best_val_acc = val_bacc
                    # Save best performance metrics and model state
                    best_score_matrix_v = score_matrix_v
                    best_cm_v = cm_v
                    best_score_matrix_t = score_matrix_t
                    best_cm = cm_t
                    best_bacc_majority_voting = bacc_majority_voting
                    best_bacc_one_dominance = bacc_one_dominance
                    best_majority_voting_predictions = majority_voting_predictions
                    best_class_report_majority_voting = class_report_majority_voting
                    best_conf_matrix_majority_voting = conf_matrix_majority_voting
                    best_one_dominance_predictions = one_dominance_predictions
                    best_class_report_one_dominance = class_report_one_dominance
                    best_conf_matrix_one_dominance = conf_matrix_one_dominance
                    best_recall_v = recall_v 
                    best_specificity_v = specificity_v
                    best_recall_mj_v = recall_mj_v
                    best_specificity_mj_v = specificity_mj_v
                    best_recall_1d_v = recall_1d_v 
                    best_specificity_1d_v = specificity_1d_v
                    best_optimal_threshold_v = optimal_threshold_v

                    torch.save(model.state_dict(), best_model_path)
                    print(f'\nSaved the best model with Val bAcc: {val_bacc:.6f} at Epoch: {epoch:03d}\n')
                    early_stopping_counter = 0  # Reset early stopping counter if accuracy improves
                else:
                    early_stopping_counter += 1  # Increment early stopping counter
            
            # Learning rate scheduling
            if lrrr == 'counter':
                if epoch > 1 and val_bacc <= best_val_acc:
                    lr_scheduler_counter += 1
                    if lr_scheduler_counter >= lr_scheduler_patience:
                        lr_scheduler_counter = 0
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= lr_scheduler_factor
                            print(f'\nReduced learning rate to: {param_group["lr"]}\n')
            elif lrrr == 'reduceplateau':
                if epoch > 1:
                    if track_lr  == 'bacc':
                        scheduler.step(val_bacc)  # Update learning rate based on validation accurac
                    elif track_lr  =='loss':
                        scheduler.step(avg_loss_val) # Update learning rate based on validation accuracy
            elif lrrr=='ritap':
                scheduler.step()


        # Early stopping check
        if early_stopping_counter >= early_stopping_rounds:
            print(f'\nEarly stopping triggered after {early_stopping_rounds} epochs of no improvement.\n')
            break


        # Add scalars to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Evaluation train', avg_loss_train, epoch)
        #writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        #writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Loss/Evaluation validation', avg_loss_val, epoch)
        
        writer.add_scalar('Bacc/Evaluation validation', val_bacc, epoch)
        writer.add_scalar('Bacc/Evaluation train', train_bacc, epoch)
        writer.add_scalar('AUC/Evaluation train', auc_t, epoch)
        writer.add_scalar('AUC/Evaluation validation', auc_v, epoch)

    # Load the best model
    print('\nTraining complete: loading the best model')
    model.load_state_dict(torch.load(best_model_path))

    # Log hyperparameters and close the writer
    writer.add_scalar('Variables/num_epochs', num_epochs)
    writer.add_scalar('Variables/hidden_channels', hidden_channels)
    writer.add_scalar('Variables/lr', lr)
    writer.add_scalar('Variables/num_node_features', num_node_features)
    writer.add_scalar('Variables/weights0', class_weights[0])
    writer.add_scalar('Variables/weights1', class_weights[1])
    writer.add_scalar('Variables/Early-stopping-rounds', early_stopping_rounds)
    writer.add_scalar('Variables/LR-scheduler-patience', lr_scheduler_patience)
    writer.add_scalar('Variables/LR-scheduler-factor', lr_scheduler_factor)
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
    writer.add_scalar('Best_model/val_acc', val_acc)
    writer.close()

    return  (best_val_acc,  # 0
            best_score_matrix_v,  # 1
            best_cm_v,  # 2
            best_score_matrix_t,  # 3
            best_cm,  # 4
            majority_voting_predictions,  # 5
            bacc_majority_voting,  # 6
            class_report_majority_voting,  # 7
            conf_matrix_majority_voting,  # 8
            one_dominance_predictions,  # 9
            bacc_one_dominance,  # 10
            class_report_one_dominance,  # 11
            conf_matrix_one_dominance,  # 12
            best_bacc_majority_voting,  # 13
            best_bacc_one_dominance,  # 14
            best_majority_voting_predictions,  # 15
            best_class_report_majority_voting,  # 16
            best_conf_matrix_majority_voting,  # 17
            best_one_dominance_predictions,  # 18
            best_class_report_one_dominance,  # 19
            best_conf_matrix_one_dominance, #20
            best_recall_v, #21
            best_specificity_v, #22
            best_recall_mj_v, #23
            best_specificity_mj_v, #24
            best_recall_1d_v, #25
            best_specificity_1d_v, #26
            best_optimal_threshold_v) # 27

def compute_mean_features(fold_train_path, graphs_folder):
    total_features_sum = None
    sum_squared_diffs = None
    total_num_samples = 0
    for file in fold_train_path['0']:
        filename = os.path.basename(file)
        if filename == 'patient_035_node_2.h5':
            continue
        graph_path = os.path.join(graphs_folder, filename[:-3] + '.pt')
        graph_data = torch.load(graph_path)
        x = graph_data.x.numpy()
        #train_region_feats.append(region_feats_train)
        if total_features_sum is None:
            total_features_sum = np.sum(x, axis=0)
        else:
            total_features_sum += np.sum(x, axis=0)
                
        total_num_samples += x.shape[0]

    mean_features = total_features_sum / total_num_samples
    print('Mean computed...')
    
    for file in fold_train_path['0']:
        filename = os.path.basename(file)
        if filename == 'patient_035_node_2.h5':
            continue
        graph_path = os.path.join(graphs_folder, filename[:-3] + '.pt')
        
        graph_data = torch.load(graph_path)
        x = graph_data.x.numpy()
        
        squared_diffs = (x - mean_features)**2
        if sum_squared_diffs is None:
            sum_squared_diffs = np.sum(squared_diffs, axis=0)
        else:
            sum_squared_diffs += np.sum(squared_diffs, axis=0)

    std_dev = np.sqrt(sum_squared_diffs/total_num_samples)
    print('Std deviation computed...')
    
    mean_features = mean_features.to('cpu')
    std_dev = std_dev.to('cpu')
    
    return mean_features, std_dev
    
class GraphDataset_combined(torch.utils.data.Dataset):
    def __init__(self, root, df,device):
        self.root = root
        self.df = df
        self.device = device
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['image_filename'].iloc[idx]
        case_id = self.df['case_id'].iloc[idx]   # Extract case ID from filename
        label = self.df['vital_status_12'].iloc[idx]  
        case_id_combined = self.df['combined_file'].iloc[idx]  
        label = torch.tensor(label, dtype=torch.long)  # Assuming label is integer

        # Load graph data
        graph_path = os.path.join(self.root,case_id_combined)
        graph_data = torch.load(graph_path,map_location=self.device)

        # Convert the graph data into a PyTorch Geometric Data object
        x = graph_data['x']  # Node features
        edge_index = graph_data['edge_index']  # Edge indices
        num_edges = edge_index.shape[1] if edge_index is not None else 0
        features = x.shape[0] if x is not None else 0

        # Create a PyTorch Geometric Data object with additional case_id attribute
        data = Data(x=x, edge_index=edge_index, y=label, edge_num=num_edges, features_num=features, case_id=case_id, image_filename=filename)  
        return data

    
# Define the WSINeighborDataset
class WSINeighborDataset(torch.utils.data.Dataset):
    def __init__(self, wsi_graphs, num_neighbors, node_batch_size, shuffle=True, generator=None):
        self.wsi_graphs = wsi_graphs
        self.num_neighbors = num_neighbors
        self.node_batch_size = node_batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __len__(self):
        return len(self.wsi_graphs)

    def __getitem__(self, idx):
        wsi_graph = self.wsi_graphs[idx]
        wsi_loader = NeighborLoader(
            wsi_graph,
            num_neighbors=self.num_neighbors,
            batch_size=self.node_batch_size,
            input_nodes=torch.arange(wsi_graph.x.size(0)),
            shuffle=self.shuffle,
            drop_last=False,
            generator=self.generator,
            pin_memory=False
        )
        return wsi_loader

       
class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root, df, device, mean_features=None, std_dev=None, patch_selector=None):
        self.root = root
        self.df = df
        self.device = device
        self.mean_features = mean_features.clone().detach().to(device) if mean_features is not None else None
        self.std_dev = std_dev.clone().detach().to(device) if std_dev is not None else None
        self.patch_selector = patch_selector
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['image_filename'].iloc[idx]
        case_id = self.df['case_id'].iloc[idx]   # Extract case ID from filename
        label = self.df['vital_status_12'].iloc[idx]  
        label = torch.tensor(label, dtype=torch.long)  # Assuming label is integer

        # Load graph data
        graph_path = os.path.join(self.root, filename[:-3] + '.pt')
        graph_data = torch.load(graph_path,map_location=self.device)

        # Convert the graph data into a PyTorch Geometric Data object
        x = graph_data['x']  # Node features
        edge_index = graph_data['edge_index']  # Edge indices
        num_edges = edge_index.shape[1] if edge_index is not None else 0
        features = x.shape[0] if x is not None else 0

        # Create a PyTorch Geometric Data object with additional case_id attribute
        data = Data(x=x, edge_index=edge_index, y=label, edge_num=num_edges, features_num=features, case_id=case_id, image_filename=filename)
        return apply_patch_selector_to_data(data, filename, self.patch_selector)


def setup_data_loaders_nonorm(all_t,train_df, val_df, mean_features, std_dev, batch_size, device,sampller,node_batch_size, patch_selector=None):
    if all_t == 'combined':
        train_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop_knn",train_df, device,mean_features, std_dev, patch_selector=patch_selector)
        print('len train:', len(train_dataset))
        val_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop_knn",val_df, device, mean_features, std_dev, patch_selector=patch_selector)
        print('len val:', len(val_dataset))
    elif all_t == 'combined_0.1':
        train_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop_knn_new_faiss_0.1",train_df, device,mean_features, std_dev, patch_selector=patch_selector)
        print('len train:', len(train_dataset))
        val_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop_knn_new_faiss_0.1",val_df, device,mean_features, std_dev, patch_selector=patch_selector)
        print('len val:', len(val_dataset))
    elif all_t == 'combined_0.1_allfiles':
        train_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop_knn_new_faiss_0.1",train_df, device,mean_features, std_dev, patch_selector=patch_selector)
        print('len train:', len(train_dataset))
        val_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop_knn_new_faiss_0.1",val_df, device,mean_features, std_dev, patch_selector=patch_selector)
        print('len val:', len(val_dataset))
    else:
        train_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop", train_df, device,mean_features, std_dev, patch_selector=patch_selector)
        print('len train:', len(train_dataset))
        val_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop", val_df, device,mean_features, std_dev, patch_selector=patch_selector)
        print('len val:', len(val_dataset))

    generator = torch.Generator().manual_seed(47)
    # Get edge index from your dataset
    first_graph = train_dataset[0]  # Access the first graph in the dataset
    
    edge_index = first_graph.edge_index  # Get edge_index from a sample graph
  # Assuming train_dataset is your dataset object

    # Compute node degrees (number of neighbors per node)
    degrees = torch.bincount(edge_index[0])  # Count occurrences of each node

    # Compute statistics
    max_neighbors = degrees.max().item()
    mean_neighbors = degrees.float().mean().item()
    min_neighbors = degrees.min().item()

    # Print results
    print(f"\n\n🔹 Maximum neighbors per node: {max_neighbors}")
    print(f"🔹 Average neighbors per node: {mean_neighbors:.2f}")
    print(f"🔹 Minimum neighbors per node: {min_neighbors}")

    # Compute dynamic neighbor limit
    max_neighbors = get_max_neighbors(train_dataset.data)
    print(f"\n\n🔹 Maximum neighbors per node: {max_neighbors}")
    
    if sampller =='True':
        # Compute `num_neighbors`
        adaptive = True if "Adaptive" in survival else False
        if adaptive:
            num_neighbors = torch.clamp(degrees, max=max_neighbors).tolist()  # Adaptive neighbor selection
        else:
            num_neighbors = [5,5,3]#[max_neighbors, max_neighbors, max_neighbors]  # Fixed neighbor sampling
    
        input_nodes = torch.arange(train_dataset.data.num_nodes)  # Use all nodes
    
'''
'''if "SAGE" in survival:

            train_neighbor_dataset = WSINeighborDataset(train_dataset, num_neighbors=num_neighbors, node_batch_size=node_batch_size)
            #val_neighbor_dataset = WSINeighborDataset(val_dataset, num_neighbors=num_neighbors, node_batch_size=node_batch_size, shuffle=False)

        # Batch WSIs together
            train_loader = DataLoader(train_neighbor_dataset, batch_size=batch_size, sampler=sampler,drop_last=False
, generator=generator) #shuffle=True#sampler=sampler
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)

        else:'''
'''
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,drop_last=False
, generator=generator) #shuffle=True#sampler=sampler
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)
    elif sampller == 'False':
       # Compute `num_neighbors`
        adaptive = True if "Adaptive" in survival else False
        if adaptive:
            num_neighbors = torch.clamp(degrees, max=max_neighbors).tolist()  # Adaptive neighbor selection
        else:
            num_neighbors = [5,5,3] #[max_neighbors, max_neighbors, max_neighbors]  # Fixed neighbor 
    
        input_nodes = torch.arange(first_graph.num_nodes)  # Use all nodes
'''
''' if "SAGE" in survival:

            train_neighbor_dataset = WSINeighborDataset(train_dataset, num_neighbors=num_neighbors, node_batch_size=node_batch_size)
            #val_neighbor_dataset = WSINeighborDataset(val_graphs, num_neighbors=num_neighbors, node_batch_size=node_batch_size, shuffle=False)

            # Batch WSIs together
            train_loader = DataLoader(train_neighbor_dataset, batch_size=batch_size, sampler=sampler,drop_last=False
, generator=generator)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)

        else:'''
'''train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=False
, generator=generator) #shuffle=True#sampler=sampler
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
)
    return train_loader, val_loader, train_dataset, val_dataset,num_neighbors,node_batch_size'''

#######################################################
# Functions to initialize metrics
#######################################################
'''def initialize_metrics():
    early_stopping_counter = 0
    lr_scheduler_counter = 0
    best_val_acc = 0.0
    
    # Metrics initialization
    best_val_acc=[]
    best_score_matrix_v=[]
    best_cm_v=[]
    best_score_matrix_t=[]
    best_cm=[]
    best_bacc_majority_voting=[]
    best_bacc_one_dominance=[]
    # Initialize lists to store predictions
    classification_reports_major_all=[]
    classification_reports_one_dominance_all = []
    bacc_major_voting_all=[]
    bacc_one_dominance_all=[]
    conf_matrix_major_voting_all=[]
    conf_matrix_one_dominance_all = []
    train_dataset_all=[]
    train_patients_all=[]
    val_dataset_all=[]
    val_patients_all=[]
    train_0s_per_patient_all=[]
    train_1s_per_patient_all=[]
    val_0s_per_patient_all=[]
    val_1s_per_patient_all=[]    
    best_val_acc=[]
    best_cm_v_all=[]
    best_cm_all=[]  
    bacc_major_voting_all=[]
    bacc_one_dominance_all=[]
    conf_matrix_one_dominance_all=[]
    conf_matrix_major_voting_all=[]
   
    return locals()'''


# Updated GAT_survival
'''class GAT_survival(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, hidden_2, hidden_3, num_classes, heads, heads_2, heads_3, dropout_rate=0.3):
        super(GAT_survival, self).__init__()
        torch.manual_seed(47)
        self.conv1 = GATConv(num_node_features, hidden_channels, heads=heads)
        #self.bn1 =torch.nn.BatchNorm1d(hidden_channels * heads, track_running_stats=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_2, heads=heads_2)
        #self.bn2 =torch.nn.BatchNorm1d(hidden_2 * heads_2, track_running_stats=True)
        self.conv3 = GATConv(hidden_2 * heads_2, hidden_3, heads=heads_3)
        #self.bn3 =torch.nn.BatchNorm1d(hidden_3 * heads_3, track_running_stats=True)
        self.lin = Linear(hidden_3 * heads_3, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        #x = self.bn1(x)
        x = x.relu()
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
       # x = self.bn2(x)
        x = x.relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        #x = self.bn3(x)
        x = x.relu()
        x = self.dropout(x)
        x = global_max_pool(x, batch)
        x = self.lin(x)
        return x'''

'''def get_class_distribution(df, label_col='vital_status_12'):
    return {
        0: df[df[label_col] == 0].groupby('case_id').size(),
        1: df[df[label_col] == 1].groupby('case_id').size()
    }
'''


''' 
According to the type of data you wish to analyse it will select the correct folder and csv
'''
'''def load_data(all_t,cross_validation,num_folds,fold,device):
    if all_t== 'all_tiago':
        csv_path_train = "/home/ritav/Graphs/cptac_233_adj_selfloop/encoded_train_allfiles.csv"
        df_train = pd.read_csv(csv_path_train)
        csv_path_val = "/home/ritav/Graphs/cptac_233_adj_selfloop/encoded_test_allfiles.csv"
        df_val = pd.read_csv(csv_path_val)
        df = pd.concat([df_train, df_val], ignore_index=True)
    elif all_t == 'patient_tiago':
        csv_path_train = "/home/ritav/Graphs/cptac_233_adj_selfloop/filtered_combined_case_ids_train.csv"
        df_train = pd.read_csv(csv_path_train)
        csv_path_val = "/home/ritav/Graphs/cptac_233_adj_selfloop/filtered_combined_case_ids_test.csv"
        df_val = pd.read_csv(csv_path_val)
        df = pd.concat([df_train, df_val], ignore_index=True)
    elif all_t == 'combined':
        #complete data
        csv_path_train = "/home/ritav/Graphs/cptac_233_adj_selfloop/allpatients_train_test_combinedgnn.csv"
        df_train = pd.read_csv(csv_path_train)
        csv_path_val = "/home/ritav/Graphs/cptac_233_adj_selfloop/allpatient_test_split_combinedgnn.csv"
        df_val = pd.read_csv(csv_path_val)   
        df = pd.concat([df_train, df_val], ignore_index=True)
    elif all_t == 'combined_0.1':
        csv_path_train = "/home/ritav/Graphs/cptac_233_adj_selfloop/allpatients_train_test.csv"
        df_train = pd.read_csv(csv_path_train)
        csv_path_val = "/home/ritav/Graphs/cptac_233_adj_selfloop/allpatient_test_split_combinedgnn.csv"
        df_val = pd.read_csv(csv_path_val)
        df = pd.concat([df_train, df_val], ignore_index=True)
    elif all_t =='all_files':
        csv_path = "/home/ritav/Graphs/cptac_233_adj_selfloop/all_filesencoded_three_updated_with_filenames_cptac_ts_dx_full.csv"
        df = pd.read_csv(csv_path)
        
    if all_t== 'all_files': 
        cross_validation = 'True'
        print('\nCross-valitation  activated')
    else:
        num_folds = 0
        fold = 0
        print('\nCross-valitation  deactivated')

    if all_t == 'combined':
        train_dataset = GraphDataset_combined("/home/ritav/Graphs/cptac_233_adj_selfloop_knn", df_train,device)
        print('len train:',len(train_dataset))
        val_dataset = GraphDataset_combined("/home/ritav/Graphs/cptac_233_adj_selfloop_knn", df_val,device)
        print('len val:',len(val_dataset))
    elif all_t == 'combined_0.1':
        train_dataset = GraphDataset_combined("/home/ritav/Graphs/cptac_233_adj_selfloop_knn_new_faiss_0.1", df_train,device)
        print('len train:',len(train_dataset))
        val_dataset = GraphDataset_combined("/home/ritav/Graphs/cptac_233_adj_selfloop_knn_new_faiss_0.1", df_val,device)
        print('len val:',len(val_dataset))
    elif all_t == 'patient_tiago':
        train_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop", df_train,device)
        print('len train:',len(train_dataset))
        val_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop", df_val,device)
        print('len val:',len(val_dataset))
    elif all_t == 'all_tiago':
        train_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop", df_train,device)
        print('len train:',len(train_dataset))
        val_dataset = GraphDataset("/home/ritav/Graphs/cptac_233_adj_selfloop", df_val,device)
    print('len val:',len(val_dataset))
    
    print(df_train)
        
    return df, df_train, df_val,train_dataset,fold,num_folds,cross_validation'''

    
'''def save_report(report, fold, output_dir="reports"):
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the filename based on the current timestamp and fold number
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"report_fold_{fold}_{timestamp}.txt"

# Save the report to a file
with open(os.path.join(output_dir, filename), "w") as f:
    f.write(report)''' 
   
'''def reset_weights(m):

    #Try resetting model weights to avoid
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()'''

'''class GCN_survival_iter(torch.nn.Module):
def __init__(self, num_node_features, hidden_channels, num_layers, num_classes):
    super(GCN_survival_iter, self).__init__()

    self.num_layers = num_layers
    self.convs = torch.nn.ModuleList()

    # Add the first GCN layer
    self.convs.append(GCNConv(num_node_features, hidden_channels))

    # Add intermediate GCN layers
    for _ in range(num_layers - 1):
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

   # Add the last GCN layer
    self.lin = Linear(hidden_channels, num_classes)

def forward(self, x, edge_index, batch):
    for layer in self.convs[:-1]:
        x = layer(x, edge_index)
        x = x.relu()

    # Apply the last convolutional layer without ReLU activation
    x = self.convs[-1](x, edge_index)

    # Perform global mean pooling
    x = global_mean_pool(x, batch)
    x = self.lin(x)
    return x'''

'''from scipy.sparse import csr_matrix

def normalize_adj(edge_list, num_nodes):
    # Convert edge list to CSR format
    src_nodes, dst_nodes = edge_list[:, 0], edge_list[:, 1]
    adj = csr_matrix((np.ones(len(src_nodes)), (src_nodes, dst_nodes)), shape=(num_nodes, num_nodes))
    
    # Calculate the degree matrix
    deg = np.array(adj.sum(1))
    deg_inv_sqrt = np.power(deg, -0.5).flatten()
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
    deg_inv_sqrt_mat = np.diag(deg_inv_sqrt)
    
    # Normalize adjacency matrix
    adj_normalized = adj.dot(deg_inv_sqrt_mat).transpose().dot(deg_inv_sqrt_mat)
    
    return adj_normalized'''


'''def validate_maj(loader, model, weight_tensor, device, num_folds):
    model.eval()
    correct = 0
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)  # Define CrossEntropyLoss for calculating loss

    # Initialize dictionaries to store predictions
    patient_predictions = {}

    with torch.no_grad():
        y_true = []  # True labels
        y_pred = []  # Predicted labels
        for data in loader:  # Iterate in batches over the training/test dataset.
            data = data.to(device)  # Move data to GPU
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)  # Use the class with highest probability.

            # Calculate loss
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

            # Collect true and predicted labels
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

            correct += int((pred == data.y).sum())  # Check against ground-truth labels.

            # Collect predictions for each patient
            for i in range(len(data.batch)):
                patient_id = data.batch[i].item()
                if patient_id not in patient_predictions:
                    patient_predictions[patient_id] = [pred[i].item()]
                else:
                    patient_predictions[patient_id].append(pred[i].item())

    # Apply majority voting and 1-dominance strategy for each patient
    majority_voting_predictions = {}
    one_dominance_predictions = {}

    for patient_id, predictions in patient_predictions.items():
        # Majority voting
        majority_vote = max(set(predictions), key=predictions.count)
        majority_voting_predictions[patient_id] = majority_vote

        # 1-dominance strategy
        unique_predictions = set(predictions)
        if len(unique_predictions) == 1:
            one_dominance_predictions[patient_id] = unique_predictions.pop()
        elif 0 in unique_predictions:
            one_dominance_predictions[patient_id] = 0
        else:
            # If all predictions are 1, then predict 1 for that patient
            one_dominance_predictions[patient_id] = 1

    accuracy = correct / len(loader.dataset)  # Derive ratio of correct predictions.
    avg_loss = total_loss / len(loader.dataset)  # Calculate average loss

    # Compute balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(y_true, y_pred, zero_division=1)

    # Calculate confusion matrix for validation set
    cm = confusion_matrix(y_true, y_pred)

    return avg_loss, accuracy, balanced_acc, class_report, cm, majority_voting_predictions, one_dominance_predictions'''


'''class GraphDataset_norm(torch.utils.data.Dataset):
    def __init__(self, root, df,device):
        self.root = root
        self.df = df
        self.device = device
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['image_filename'].iloc[idx]
        case_id = self.df['case_id'].iloc[idx]   # Extract case ID from filename
        label = self.df['vital_status_12'].iloc[idx]  
        label = torch.tensor(label, dtype=torch.long)  # Assuming label is integer

        # Load graph data
        graph_path = os.path.join(self.root, filename[:-3] + '.pt')
        graph_data = torch.load(graph_path,map_location=self.device)

        # Normalize the adjacency matrix (edge_index) if it exists
        if 'edge_index' in graph_data:
            edge_index = graph_data['edge_index']
            edge_index_normalized = normalize_adj(edge_index)
        else:
            edge_index_normalized = None

        # Convert the graph data into a PyTorch Geometric Data object
        x = graph_data['x']  # Node features
        num_edges = edge_index.shape[1] if edge_index is not None else 0
        features = x.shape[0] if x is not None else 0

        # Create a PyTorch Geometric Data object with additional case_id attribute
        data = Data(x=x, edge_index=edge_index_normalized, y=label, edge_num=num_edges, features_num=features, case_id=case_id, image_filename=filename)  
        return data'''

####### Class that loads the data and gives it in this format:
##  x- features, edge_index = adjency matrix, y -label survival, edge_num - edgesin that graph, features_num - features in that graph
##  Data(x=[3895, 1024], edge_index=[2, 30368], y=1, edge_num=30368, features_num=3895)
##  Data(x=[5572, 1024], edge_index=[2, 42874], y=1, edge_num=42874, features_num=5572)
'''class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, root, df):
        self.root = root
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df['image_filename'].iloc[idx]
        label = self.df['vital_status_12'].iloc[idx]  
        label = torch.tensor(label, dtype=torch.long)  # Assuming label is integer

        # Load graph data
        graph_path = os.path.join(self.root, filename[:-3] + '.pt')
        graph_data = torch.load(graph_path)
        
        # Convert the graph data into a PyTorch Geometric Data object
        x = graph_data['x']  # Node features
        edge_index = graph_data['edge_index']  # Edge indices
        num_edges = edge_index.shape[1] if edge_index is not None else 0
        features = x.shape[0] if x is not None else 0

        data = Data(x=x, edge_index=edge_index, y=label, edge_num = num_edges, features_num = features)  
        return data'''

'''def majority_voting(predictions):
    mean_prediction = np.mean(predictions)
    if mean_prediction >= 0.5:
        return 1
    else:
        return 0'''

'''def one_dominance(predictions):
    if len(predictions) == 1:
        return predictions[0]  # Direct prediction
    # Implement the actual one-dominance logic here for multiple predictions
    # Example: return 1 if there are more 1s than 0s, else return 0
    return 1 if predictions.count(1) > predictions.count(0) else 0'''

'''def one_dominance(predictions):
    if 0 in predictions:
        return 0
    else:
        return 1'''

'''def normalize_adj(adj):
    # Convert adjacency matrix to CSR format
    adj = csr_matrix(adj)
    
    # Calculate the degree matrix
    deg = np.array(adj.sum(1))
    deg_inv_sqrt = np.power(deg, -0.5).flatten()
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.
    deg_inv_sqrt_mat = np.diag(deg_inv_sqrt)
    
    # Normalize adjacency matrix
    adj_normalized = adj.dot(deg_inv_sqrt_mat).transpose().dot(deg_inv_sqrt_mat)
    
    return adj_normalized'''

'''def validate(loader,model,weight_tensor,device):
        model.eval()
        correct = 0
        total_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)  # Define CrossEntropyLoss for calculating loss

        #print('\n\n #######No weights')
        #criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            y_true = []  # True labels
            y_pred = []  # Predicted labels
            for data in loader:  # Iterate in batches over the training/test dataset.
                data = data.to(device)  # Move data to GPU
                out = model(data.x, data.edge_index, data.batch)  
                pred = out.argmax(dim=1)  # Use the class with highest probability.

                # Calculate loss
                loss = criterion(out, data.y)
                total_loss += loss.item() * data.num_graphs

                # Collect true and predicted labels
                y_true.extend(data.y.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())


                correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        #print('VAL:\ny_true', y_true)
        #print('y_pred',y_pred)
        #print('\n ############## VAL correct',correct)

        accuracy = correct / len(loader.dataset)  # Derive ratio of correct predictions.
        #print('acc',accuracy)
        avg_loss = total_loss / len(loader.dataset)  # Calculate average loss

        # Compute balanced accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Generate classification report
        class_report = classification_report(y_true, y_pred, zero_division=1)

        # Calculate confusion matrix for validation set
        cm = confusion_matrix(y_true, y_pred)
        
        #print('validate funtion pred',pred,data.y)
        #print('train pred',train_predictions,train_labels)

        return avg_loss, accuracy, balanced_acc, class_report,cm,y_pred,  y_true'''


'''def validate(loader, model, weight_tensor, device,mode):
    model.eval()
    total_loss = 0.0
    correct = 0
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    y_true = []
    y_pred = []
    y_predd = defaultdict(list)  # Dictionary to store predictions per patient
    y_truee = defaultdict(list)  # Dictionary to store predictions per patient
    majority_voting_predictions = defaultdict(list)
    one_dominance_predictions = defaultdict(list)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            
            for id in range(0,len(pred)):
                #print('data',data)
                patient_id_str = data.case_id[id]
                y_predd[patient_id_str].append(pred[id].cpu().numpy().item())           
                y_truee[patient_id_str] = data.y[id].cpu().numpy().item()
                
                #print('pred',len(pred))
                #print('data',data.case_id[id])
            
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs

                
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            
            correct += int((pred == data.y).sum())
            

    # Calculate majority voting and one-dominance predictions per patient
    for id in y_predd.keys():
        majority_voting_predictions[id] = majority_voting(y_predd[id])
        one_dominance_predictions[id] = one_dominance(y_predd[id]) 
        
          
    match_count_maj = sum(1 for true, pred in zip(y_truee.values(), majority_voting_predictions.values()) if true == pred)
    match_count_onedominanc = sum(1 for true, pred in zip(y_truee.values(), one_dominance_predictions.values()) if true == pred)
    if mode == 'val':
        print(f'\nTotal majority voting patients correct: {match_count_maj}/{len(y_predd)}')
        print(f'Total 1-D patients correct: {match_count_onedominanc}/{len(y_predd)}')

    # Calculate metrics
    accuracy = correct / len(loader.dataset)  # Derive ratio of correct predictions.
    #print('acc',accuracy)
    avg_loss = total_loss / len(loader.dataset)  # Calculate average loss

    # Compute balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(y_true, y_pred, zero_division=1)

    # Calculate confusion matrix for validation set
    cm = confusion_matrix(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    
    #accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, zero_division=1)
    conf_matrix = confusion_matrix(y_true, y_pred)

    ### Calculate metrics for majority voting
    bacc_majority_voting = balanced_accuracy_score(list(y_truee.values()), list(majority_voting_predictions.values()))
    class_report_majority_voting = classification_report(list(y_truee.values()), list(majority_voting_predictions.values()), zero_division=1)
    conf_matrix_majority_voting = confusion_matrix(list(y_truee.values()), list(majority_voting_predictions.values()))

    ### Calculate metrics for one-dominance
    bacc_one_dominance = balanced_accuracy_score(list(y_truee.values()), list(one_dominance_predictions.values()))
    class_report_one_dominance = classification_report(list(y_truee.values()), list(one_dominance_predictions.values()), zero_division=1)
    conf_matrix_one_dominance = confusion_matrix(list(y_truee.values()), list(one_dominance_predictions.values()))

    return avg_loss, accuracy, balanced_acc, class_report, conf_matrix, \
           majority_voting_predictions, bacc_majority_voting, class_report_majority_voting, conf_matrix_majority_voting, \
           one_dominance_predictions, bacc_one_dominance, class_report_one_dominance, conf_matrix_one_dominance
'''
