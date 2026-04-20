"""
compare_ig_gradients.py -- Does Integrated Gradients add anything over plain gradients?

Compares feature attributions from:
  - Plain gradients  (backprop of recommended node score w.r.t. input)
  - Integrated Gradients  (Captum, 50 steps, zero baseline)

For 10 SIRS+ / high-CRP test cases, reports:
  - Top-5 features for each method
  - Spearman rank correlation between methods
  - Mean absolute difference in normalised attributions

If correlation > 0.95 on average, IG adds no meaningful signal.
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from torch_geometric.data import Data
from torch_geometric.nn import GATConv

torch.manual_seed(42)

script_dir = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Load data + vocab
# ---------------------------------------------------------------------------
with open(script_dir / 'labeled_training_data_karibdis.pkl', 'rb') as f:
    labeled_data = pickle.load(f)
with open(script_dir / 'vocabulary_karibdis.pkl', 'rb') as f:
    vocab = pickle.load(f)

node_dim = vocab.get('node_dim', 33)
edge_dim  = vocab.get('edge_dim', 5)

case_to_examples = defaultdict(list)
for idx, ex in enumerate(labeled_data):
    case_to_examples[ex['decision_point']['case_id']].append(idx)

unique_cases = list(case_to_examples.keys())
train_cases, temp  = train_test_split(unique_cases, test_size=0.3, random_state=42)
val_cases,  test_c = train_test_split(temp,          test_size=0.5, random_state=42)
test_indices = [i for c in test_c for i in case_to_examples[c]]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class PrescriptiveGAT(nn.Module):
    def __init__(self, node_dim=33, edge_dim=5, hidden=64):
        super().__init__()
        self.conv1 = GATConv(node_dim, 32, heads=4, edge_dim=edge_dim)
        self.conv2 = GATConv(128, hidden, heads=1, edge_dim=edge_dim)
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_attr=None):
        x_out = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x_out[0] if isinstance(x_out, tuple) else x_out)
        x = F.dropout(x, p=0.1, training=self.training)
        x_out = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x_out[0] if isinstance(x_out, tuple) else x_out)
        return self.node_scorer(x).squeeze(-1)

device = torch.device('cpu')
model = PrescriptiveGAT(node_dim=node_dim, edge_dim=edge_dim).to(device)
model.load_state_dict(
    torch.load(script_dir / 'prescriptive_gat_karibdisv3.pt', map_location=device)
)
model.eval()
print(f"[ok] Model loaded")

# ---------------------------------------------------------------------------
# Captum / IG check
# ---------------------------------------------------------------------------
try:
    from captum.attr import IntegratedGradients
    USE_IG = True
    print("[ok] Captum available -- IG will run")
except ImportError:
    USE_IG = False
    print("[!] Captum not installed -- cannot compare, IG unavailable")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------
BOOL_PVS = [
    'SIRSCriteria2OrMore', 'SIRSCritHeartRate', 'SIRSCritTemperature',
    'SIRSCritTachypnea', 'SIRSCritLeucos',
    'InfectionSuspected', 'DisfuncOrg', 'Hypotensie', 'Hypoxie',
    'Oligurie', 'Infusion',
    'DiagnosticBlood', 'DiagnosticArtAstrup', 'DiagnosticIC',
    'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther',
    'DiagnosticXthorax', 'DiagnosticUrinaryCulture', 'DiagnosticLacticAcid',
    'DiagnosticUrinarySediment', 'DiagnosticECG',
]
FEATURE_NAMES = (
    ['is_Task', 'is_Activity', 'CRP', 'Leucocytes', 'LacticAcid', 'Age'] +
    BOOL_PVS +
    ['temporal_position', 'declare_involvement', 'execution_count',
     'is_executed_in_prefix', 'Diagnose']
)

# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def subgraph_to_pyg(example):
    sg = example['subgraph']
    nf = example['node_features']
    et_vocab = vocab.get('edge_types', [])
    node_list = sg['nodes']
    n2i = {n: i for i, n in enumerate(node_list)}

    x = torch.from_numpy(np.array([nf[n] for n in node_list])).float()

    edges, efeats = [], []
    for src, etype, dst in sg['edges']:
        if src in n2i and dst in n2i:
            edges.append([n2i[src], n2i[dst]])
            ef = torch.zeros(len(et_vocab))
            if etype in et_vocab:
                ef[et_vocab.index(etype)] = 1.0
            efeats.append(ef)

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        edge_attr  = torch.stack(efeats)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, len(et_vocab)), dtype=torch.float)

    cands = sg['candidate_activities']
    cand_idx = [n2i[c] for c in cands if c in n2i]
    return x, edge_index, edge_attr, node_list, cand_idx

# ---------------------------------------------------------------------------
# Attribution functions
# ---------------------------------------------------------------------------
def plain_gradients(x, ei, ea, target_idx):
    x_g = x.clone().detach().requires_grad_(True)
    scores = model(x_g, ei, ea)
    scores[target_idx].backward()
    return x_g.grad.detach()  # [num_nodes, node_dim]

def integrated_gradients(x, ei, ea, target_idx, n_steps=50):
    baseline = torch.zeros_like(x)

    def fwd(x_in):
        s = model(x_in, ei, ea)
        return s[target_idx].unsqueeze(0)

    ig = IntegratedGradients(fwd)
    attrs, delta = ig.attribute(x, baseline,
                                n_steps=n_steps,
                                return_convergence_delta=True)
    return attrs.detach(), float(delta.abs().mean())

# ---------------------------------------------------------------------------
# Collect SIRS+ or high-CRP test cases
# ---------------------------------------------------------------------------
def is_interesting(example):
    """Return True if patient has SIRS or elevated CRP."""
    sg = example['subgraph']
    nf = example['node_features']
    task_nodes = [n for n in sg['nodes']
                  if sg['node_types'].get(n) == 'Task' and n in nf]
    if not task_nodes:
        return False
    latest = max(task_nodes, key=lambda n: nf[n][28])
    f = nf[latest]
    has_sirs = f[6] > 0.5
    high_crp  = float(f[2]) * 300.0 > 100.0
    return has_sirs or high_crp

target_examples = []
for idx in test_indices:
    ex = labeled_data[idx]
    if is_interesting(ex):
        target_examples.append(ex)
    if len(target_examples) >= 10:
        break

print(f"[ok] Found {len(target_examples)} SIRS+/high-CRP test cases\n")

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
correlations = []
mean_diffs   = []

print("=" * 70)
print("IG vs Plain Gradients -- Feature Attribution Comparison")
print("=" * 70)

for case_num, ex in enumerate(target_examples, 1):
    sg = ex['subgraph']
    nf = ex['node_features']

    x, ei, ea, node_list, cand_idx = subgraph_to_pyg(ex)

    if not cand_idx:
        print(f"Case {case_num}: no candidates, skipping")
        continue

    # Pick recommended candidate
    with torch.no_grad():
        scores  = model(x, ei, ea)
        c_scores = scores[torch.tensor(cand_idx)]
        ranked  = torch.argsort(c_scores, descending=True).tolist()

    rec_node_idx = cand_idx[ranked[0]]

    # Find latest Task node index in node_list
    task_nodes = [n for n in sg['nodes']
                  if sg['node_types'].get(n) == 'Task' and n in nf]
    if not task_nodes:
        continue
    latest_task_uri = max(task_nodes, key=lambda n: nf[n][28])
    if latest_task_uri not in node_list:
        continue
    task_idx = node_list.index(latest_task_uri)

    # Run both methods
    grads = plain_gradients(x, ei, ea, rec_node_idx)  # [N, 33]
    ig_attrs, conv_delta = integrated_gradients(x, ei, ea, rec_node_idx, n_steps=300)

    # Focus on latest Task node features (where clinical values live)
    g_vec  = grads[task_idx].numpy()       # [33]
    ig_vec = ig_attrs[task_idx].numpy()    # [33]

    # Normalise to unit norm for comparison
    g_norm  = g_vec  / (np.linalg.norm(g_vec)  + 1e-12)
    ig_norm = ig_vec / (np.linalg.norm(ig_vec) + 1e-12)

    corr, _ = spearmanr(g_norm, ig_norm)
    mad     = float(np.mean(np.abs(g_norm - ig_norm)))
    correlations.append(corr)
    mean_diffs.append(mad)

    # Top features by IG
    top_ig  = np.argsort(np.abs(ig_norm))[::-1][:5]
    top_g   = np.argsort(np.abs(g_norm))[::-1][:5]

    rec_name = node_list[rec_node_idx].split('Activity_')[-1].replace('%20', ' ') \
               if 'Activity_' in node_list[rec_node_idx] else node_list[rec_node_idx]

    print(f"\nCase {case_num} | rec={rec_name} | conv_delta={conv_delta:.4f}")
    print(f"  Spearman rho = {corr:.4f}  |  Mean |IG - Grad| = {mad:.4f}")
    print(f"  Top-5 IG:   {[FEATURE_NAMES[i] for i in top_ig]}")
    print(f"  Top-5 Grad: {[FEATURE_NAMES[i] for i in top_g]}")

    # Highlight differences
    ig_set  = set(top_ig)
    g_set   = set(top_g)
    only_ig = [FEATURE_NAMES[i] for i in ig_set - g_set]
    only_g  = [FEATURE_NAMES[i] for i in g_set  - ig_set]
    if only_ig or only_g:
        print(f"  ** Unique to IG:   {only_ig}")
        print(f"  ** Unique to Grad: {only_g}")
    else:
        print(f"  [same top-5 features]")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
if correlations:
    mean_corr = float(np.mean(correlations))
    mean_mad  = float(np.mean(mean_diffs))
    print(f"  Cases analysed:          {len(correlations)}")
    print(f"  Mean Spearman rho:       {mean_corr:.4f}")
    print(f"  Mean |IG - Grad| (norm): {mean_mad:.4f}")
    print()
    if mean_corr > 0.95:
        print("  VERDICT: IG and plain gradients are nearly identical (rho > 0.95).")
        print("  For this model, IG does NOT add meaningful signal over plain gradients.")
        print("  Recommendation: simplify framework to 2 layers (GNNExplainer + Narrative).")
    elif mean_corr > 0.80:
        print("  VERDICT: IG and plain gradients are broadly similar (rho 0.80-0.95).")
        print("  IG shows some differences but may not justify the added complexity.")
        print("  Consider keeping IG if convergence delta is consistently low (<0.01).")
    else:
        print("  VERDICT: IG diverges meaningfully from plain gradients (rho < 0.80).")
        print("  IG is adding genuine signal -- justified in the framework.")
else:
    print("  No cases were processed -- check data loading.")
