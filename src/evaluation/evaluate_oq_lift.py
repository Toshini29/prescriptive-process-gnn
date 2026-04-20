"""
evaluate_oq_lift.py - Prescriptive quality evaluation: outcome quality lift.

Core question: does following the model's recommendation lead to better
outcomes than what actually happened?

Since we cannot run a prospective trial, we use the training data as a proxy:
- Cases where the model's top-1 recommendation MATCHES the actual chosen
  activity represent cases where the model would have "agreed" with the
  clinician. Their OQ reflects outcomes when the recommended action was taken.
- Cases where the model DISAGREES represent cases where the clinician did
  something the model would not have chosen. Their OQ reflects outcomes
  when a lower-ranked action was taken.

If OQ(agree) > OQ(disagree), the model has prescriptive value:
  actions it ranks highest tend to lead to better outcomes.

Additional analyses:
  - OQ lift by activity type (where does the model add most value?)
  - Rank of actual choice in model ranking (how wrong is the model when it disagrees?)
  - OQ by model confidence quartile (does higher confidence correlate with better OQ?)
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

print("=" * 70)
print("PRESCRIPTIVE QUALITY EVALUATION -- Outcome Quality Lift")
print("=" * 70)

torch.manual_seed(42)
random.seed(42)

from pathlib import Path
script_dir = Path(__file__).resolve().parent

with open(script_dir / 'labeled_training_data_karibdis.pkl', 'rb') as f:
    labeled_data = pickle.load(f)

with open(script_dir / 'vocabulary_karibdis.pkl', 'rb') as f:
    vocab = pickle.load(f)

node_dim = vocab.get('node_dim', 33)
edge_dim  = vocab.get('edge_dim', 5)
print(f"Loaded {len(labeled_data)} examples, node_dim={node_dim}")

# -- Same case-level split as training ------------------------------------------
case_to_examples = defaultdict(list)
for idx, example in enumerate(labeled_data):
    case_to_examples[example['decision_point']['case_id']].append(idx)

unique_cases = list(case_to_examples.keys())
train_cases, temp_cases = train_test_split(unique_cases, test_size=0.3, random_state=42)
val_cases, test_cases   = train_test_split(temp_cases, test_size=0.5, random_state=42)
test_indices = [i for c in test_cases for i in case_to_examples[c]]
print(f"Test set: {len(test_indices)} decision points")

# -- Model ----------------------------------------------------------------------
class PrescriptiveGAT(nn.Module):
    def __init__(self, node_dim=33, edge_dim=5, hidden=64):
        super().__init__()
        self.conv1 = GATConv(node_dim, 32, heads=4, edge_dim=edge_dim)
        self.conv2 = GATConv(128, hidden, heads=1, edge_dim=edge_dim)
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x_out = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x_out[0] if isinstance(x_out, tuple) else x_out)
        x = F.dropout(x, p=0.1, training=self.training)
        x_out = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x_out[0] if isinstance(x_out, tuple) else x_out)
        return self.node_scorer(x).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = PrescriptiveGAT(node_dim=node_dim, edge_dim=edge_dim).to(device)
model.load_state_dict(
    torch.load(script_dir / 'prescriptive_gat_karibdisv3.pt', map_location=device)
)
model.eval()
print(f"Loaded model (device: {device})")

# -- Graph builder --------------------------------------------------------------
def subgraph_to_pyg(example):
    sg  = example['subgraph']
    nf  = example['node_features']
    etv = vocab.get('edge_types', [])
    node_list   = sg['nodes']
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.from_numpy(np.array([nf[n] for n in node_list])).float()

    eil, efl = [], []
    for src, etype, dst in sg['edges']:
        if src in node_to_idx and dst in node_to_idx:
            eil.append([node_to_idx[src], node_to_idx[dst]])
            ef = torch.zeros(len(etv))
            if etype in etv:
                ef[etv.index(etype)] = 1.0
            efl.append(ef)

    if eil:
        edge_index = torch.tensor(eil, dtype=torch.long).t()
        edge_attr  = torch.stack(efl)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, len(etv)), dtype=torch.float)

    candidates = sg['candidate_activities']
    cand_indices = [node_to_idx[c] for c in candidates if c in node_to_idx]
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), candidates, cand_indices

def extract_name(uri):
    if 'Activity_' in uri:
        return uri.split('Activity_')[1].replace('%20', ' ')
    return uri.split('/')[-1].replace('_', ' ')

def activity_type(name):
    if name in ('Admission IC', 'Admission NC'):
        return 'Admission'
    if name in ('IV Antibiotics', 'IV Liquid'):
        return 'Treatment'
    if name in ('CRP', 'Leucocytes', 'LacticAcid'):
        return 'Lab Test'
    if name.startswith('Release'):
        return 'Discharge'
    if name == 'Return ER':
        return 'Return ER'
    return 'Other'

# -- Main evaluation loop -------------------------------------------------------
print("\nRunning evaluation on test set...")

records = []

with torch.no_grad():
    for idx in test_indices:
        example = labeled_data[idx]
        try:
            data, candidates, cand_indices = subgraph_to_pyg(example)
            if not cand_indices:
                continue

            data = data.to(device)
            cand_idx_t  = torch.tensor(cand_indices, dtype=torch.long, device=device)
            node_scores = model(data.x, data.edge_index, data.edge_attr)
            cand_scores = node_scores[cand_idx_t]
            probs       = F.softmax(cand_scores, dim=0).cpu().numpy()

            ranked_uris  = [candidates[i] for i in torch.argsort(cand_scores, descending=True).cpu().tolist()]
            ranked_names = [extract_name(u) for u in ranked_uris]

            rec_name    = ranked_names[0]
            rec_score   = float(probs[torch.argsort(cand_scores, descending=True)[0].item()])
            actual_name = example['decision_point']['chosen_decision']
            oq          = example['outcome_quality']

            # Rank of actual choice in model's ranking (0 = top)
            rank_of_actual = ranked_names.index(actual_name) if actual_name in ranked_names else -1

            agrees = (rec_name == actual_name)

            records.append({
                'rec_name':       rec_name,
                'rec_type':       activity_type(rec_name),
                'actual_name':    actual_name,
                'actual_type':    activity_type(actual_name),
                'oq':             oq,
                'rec_score':      rec_score,
                'agrees':         agrees,
                'rank_of_actual': rank_of_actual,
                'n_candidates':   len(cand_indices),
            })
        except Exception:
            continue

print(f"Evaluated {len(records)} decision points")

# ── Q1: OQ lift — agree vs disagree ───────────────────────────────────────────
print("\n" + "=" * 70)
print("Q1: OUTCOME QUALITY -- MODEL AGREEMENT vs DISAGREEMENT")
print("=" * 70)

agree    = [r for r in records if r['agrees']]
disagree = [r for r in records if not r['agrees']]

def oq_stats(group, label):
    if not group:
        print(f"  {label}: no cases")
        return
    oqs = [r['oq'] for r in group]
    print(f"  {label} (n={len(group)}):")
    print(f"    Mean OQ:   {np.mean(oqs):.3f}")
    print(f"    Std:       {np.std(oqs):.3f}")
    print(f"    Median:    {np.median(oqs):.3f}")

oq_stats(agree,    "Model agrees with actual choice")
print()
oq_stats(disagree, "Model disagrees with actual choice")

if agree and disagree:
    lift = np.mean([r['oq'] for r in agree]) - np.mean([r['oq'] for r in disagree])
    print(f"\n  OQ Lift (agree - disagree): {lift:+.3f}")
    print(f"  Agreement rate: {len(agree)/len(records)*100:.1f}%  "
          f"({len(agree)}/{len(records)})")
    if lift > 0.03:
        print("  >> Model has prescriptive value: agreed cases had meaningfully better outcomes")
    elif lift > 0:
        print("  >> Small positive lift -- model partially prescriptive")
    else:
        print("  >> No lift detected -- model predicts what happened, not what should happen")

# ── Q2: OQ lift by recommendation type ────────────────────────────────────────
print("\n" + "=" * 70)
print("Q2: OQ LIFT BY RECOMMENDED ACTIVITY TYPE")
print("    (Where does the model add most prescriptive value?)")
print("=" * 70)

print(f"\n  {'Rec type':<15s}  {'n_agree':>8s}  {'OQ(agree)':>10s}  "
      f"{'n_disag':>8s}  {'OQ(disag)':>10s}  {'Lift':>7s}")
print("  " + "-" * 68)

for atype in ('Admission', 'Treatment', 'Lab Test', 'Discharge', 'Return ER'):
    ag = [r for r in agree    if r['rec_type'] == atype]
    dg = [r for r in disagree if r['rec_type'] == atype]
    if not ag and not dg:
        continue
    oq_ag = np.mean([r['oq'] for r in ag]) if ag else float('nan')
    oq_dg = np.mean([r['oq'] for r in dg]) if dg else float('nan')
    lift_v = oq_ag - oq_dg if ag and dg else float('nan')
    print(f"  {atype:<15s}  {len(ag):>8d}  {oq_ag:>10.3f}  "
          f"{len(dg):>8d}  {oq_dg:>10.3f}  {lift_v:>+7.3f}")

# ── Q3: Rank of actual choice ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Q3: WHERE DOES THE ACTUAL CHOICE RANK IN MODEL'S ORDERING?")
print("    (Distribution of rank_of_actual across test set)")
print("=" * 70)

valid_ranks = [r['rank_of_actual'] for r in records if r['rank_of_actual'] >= 0]
from collections import Counter
rank_counts = Counter(valid_ranks)
total_valid = len(valid_ranks)

print(f"\n  {'Rank':>6s}  {'Count':>8s}  {'%':>7s}  {'Cumulative %':>14s}")
print("  " + "-" * 42)
cumulative = 0
for rank in sorted(rank_counts.keys())[:8]:
    count = rank_counts[rank]
    pct   = count / total_valid * 100
    cumulative += pct
    label = "  <- top-1" if rank == 0 else ""
    print(f"  {rank:>6d}  {count:>8d}  {pct:>6.1f}%  {cumulative:>13.1f}%{label}")

mean_rank = np.mean(valid_ranks)
print(f"\n  Mean rank of actual choice: {mean_rank:.2f}")
print(f"  (Lower = model ranking aligns with what clinicians actually chose)")

# ── Q4: OQ by model confidence quartile ───────────────────────────────────────
print("\n" + "=" * 70)
print("Q4: DOES HIGHER MODEL CONFIDENCE CORRELATE WITH BETTER OUTCOMES?")
print("    (OQ by confidence quartile)")
print("=" * 70)

scores = [r['rec_score'] for r in records]
q25, q50, q75 = np.percentile(scores, [25, 50, 75])

quartile_labels = [
    (f"Q1 (score <= {q25:.2f})",    lambda r: r['rec_score'] <= q25),
    (f"Q2 ({q25:.2f} < score <= {q50:.2f})", lambda r: q25 < r['rec_score'] <= q50),
    (f"Q3 ({q50:.2f} < score <= {q75:.2f})", lambda r: q50 < r['rec_score'] <= q75),
    (f"Q4 (score > {q75:.2f})",     lambda r: r['rec_score'] > q75),
]

print(f"\n  {'Quartile':<35s}  {'n':>5s}  {'Mean OQ':>8s}  {'Agree rate':>11s}")
print("  " + "-" * 65)
for label, fn in quartile_labels:
    group = [r for r in records if fn(r)]
    if not group:
        continue
    mean_oq  = np.mean([r['oq'] for r in group])
    agree_rt = 100 * sum(r['agrees'] for r in group) / len(group)
    print(f"  {label:<35s}  {len(group):>5d}  {mean_oq:>8.3f}  {agree_rt:>10.1f}%")

# ── Q5: Intervention recommendation — OQ comparison ───────────────────────────
print("\n" + "=" * 70)
print("Q5: WHEN MODEL RECOMMENDS INTERVENTION BUT CLINICIAN DID LAB TEST")
print("    (Key prescriptive scenario)")
print("=" * 70)

rec_interv_actual_lab = [
    r for r in records
    if r['rec_type'] == 'Admission' and r['actual_type'] == 'Lab Test'
]
rec_lab_actual_interv = [
    r for r in records
    if r['rec_type'] == 'Lab Test' and r['actual_type'] == 'Admission'
]

if rec_interv_actual_lab:
    oqs = [r['oq'] for r in rec_interv_actual_lab]
    print(f"\n  Model recommends Admission, clinician chose Lab Test (n={len(rec_interv_actual_lab)}):")
    print(f"    Mean OQ: {np.mean(oqs):.3f}  (these are cases where admission may have been better)")

if rec_lab_actual_interv:
    oqs = [r['oq'] for r in rec_lab_actual_interv]
    print(f"\n  Model recommends Lab Test, clinician admitted (n={len(rec_lab_actual_interv)}):")
    print(f"    Mean OQ: {np.mean(oqs):.3f}  (these are cases where lab test may have been better)")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
oq_agree  = np.mean([r['oq'] for r in agree])   if agree    else 0
oq_disag  = np.mean([r['oq'] for r in disagree]) if disagree else 0
top1_acc  = len(agree) / len(records) * 100
top3_acc  = sum(r['rank_of_actual'] <= 2 for r in records if r['rank_of_actual'] >= 0) / total_valid * 100
top5_acc  = sum(r['rank_of_actual'] <= 4 for r in records if r['rank_of_actual'] >= 0) / total_valid * 100

print(f"""
  Total test decision points evaluated: {len(records)}

  Top-1 agreement (model == actual):    {top1_acc:.1f}%
  Top-3 accuracy (actual in top 3):     {top3_acc:.1f}%
  Top-5 accuracy (actual in top 5):     {top5_acc:.1f}%

  Mean OQ when model agreed:            {oq_agree:.3f}
  Mean OQ when model disagreed:         {oq_disag:.3f}
  OQ Lift:                              {oq_agree - oq_disag:+.3f}

  Interpretation:
    A positive OQ lift means the model's preferred actions are associated
    with better outcomes in the data -- prescriptive value is present.
    A negative or zero lift means the model is primarily predictive
    (learns what was done, not what should be done).
""")
print("=" * 70)
