"""
train_gat.py - Node-level prescriptive GNN for next activity recommendation.

Architecture: single GAT that scores each candidate activity node using its
graph-contextualised representation. No separate scorer needed.

Loss: outcome-weighted cross-entropy over candidate nodes — the model learns
to rank the chosen activity highest, proportionally to how good the outcome was.
"""

import pickle
from pathlib import Path
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
print("NODE-LEVEL PRESCRIPTIVE GAT — OUTCOME WEIGHTED")
print("=" * 70)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

script_dir = Path(__file__).resolve().parent

with open(script_dir / 'labeled_training_data_karibdis.pkl', 'rb') as f:
    labeled_data = pickle.load(f)

with open(script_dir / 'vocabulary_karibdis.pkl', 'rb') as f:
    vocab = pickle.load(f)

node_dim = vocab.get('node_dim', 32)
edge_dim = vocab.get('edge_dim', len(vocab.get('edge_types', [])))
print(f"✓ {len(labeled_data)} examples, node_dim={node_dim}, edge_dim={edge_dim}")

# ── Case-level train/val/test split ─────────────────────────────────────────
case_to_examples = defaultdict(list)
for idx, example in enumerate(labeled_data):
    case_to_examples[example['decision_point']['case_id']].append(idx)

unique_cases = list(case_to_examples.keys())
train_cases, temp_cases = train_test_split(unique_cases, test_size=0.3, random_state=42)
val_cases, test_cases = train_test_split(temp_cases, test_size=0.5, random_state=42)

train_indices = [i for c in train_cases for i in case_to_examples[c]]
val_indices   = [i for c in val_cases   for i in case_to_examples[c]]
test_indices  = [i for c in test_cases  for i in case_to_examples[c]]

print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")


# ── Data preparation ─────────────────────────────────────────────────────────
def subgraph_to_pyg(example, vocab):
    """
    Build a PyG Data object from a decision-point subgraph.

    Returns
    -------
    data              : PyG Data
    candidate_indices : list[int]  node indices of all candidate activities
    label             : int        index within candidate_indices of the chosen activity
                                   (-1 if chosen activity not in subgraph)
    """
    sg = example['subgraph']
    node_features = example['node_features']
    edge_types_vocab = vocab.get('edge_types', [])

    node_list = sg['nodes']
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.from_numpy(
        np.array([node_features[n] for n in node_list])
    ).float()

    edge_index_list, edge_feat_list = [], []
    for src, etype, dst in sg['edges']:
        if src in node_to_idx and dst in node_to_idx:
            edge_index_list.append([node_to_idx[src], node_to_idx[dst]])
            ef = torch.zeros(len(edge_types_vocab))
            if etype in edge_types_vocab:
                ef[edge_types_vocab.index(etype)] = 1.0
            edge_feat_list.append(ef)

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t()
        edge_attr = torch.stack(edge_feat_list)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(edge_types_vocab)), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Candidate indices in the node list
    candidates = sg['candidate_activities']
    candidate_indices = [node_to_idx[c] for c in candidates if c in node_to_idx]

    # Label: position of chosen activity within candidate_indices
    chosen = example['decision_point']['chosen_decision']
    chosen_uri = f"http://example.org/Activity_{chosen.replace(' ', '%20')}"
    if chosen_uri in candidates and chosen_uri in node_to_idx:
        label = candidates.index(chosen_uri)
    else:
        label = -1

    return data, candidate_indices, label


# ── Model ────────────────────────────────────────────────────────────────────
class PrescriptiveGAT(nn.Module):
    """
    Node-level GAT for prescriptive process monitoring.

    Message passing aggregates context from the whole subgraph (DECLARE
    constraints, clinical ProcessValues, temporal edges) into each node's
    representation. Candidate activity nodes are then scored directly —
    no separate graph embedding or scorer module needed.
    """
    def __init__(self, node_dim=32, edge_dim=5, hidden=64):
        super().__init__()
        self.conv1 = GATConv(node_dim, 32, heads=4, edge_dim=edge_dim)
        self.conv2 = GATConv(128, hidden, heads=1, edge_dim=edge_dim)
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x_out = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x_out[0] if isinstance(x_out, tuple) else x_out)
        x = F.dropout(x, p=0.1, training=self.training)
        x_out = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x_out[0] if isinstance(x_out, tuple) else x_out)
        return self.node_scorer(x).squeeze(-1)  # [num_nodes]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model_file = script_dir / 'prescriptive_gat_karibdisv3.pt'

if model_file.exists():
    print("✓ Loading trained model...")
    model = PrescriptiveGAT(node_dim=node_dim, edge_dim=edge_dim).to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
else:
    print("\n[Training] PrescriptiveGAT — node-level, outcome-weighted...")
    model = PrescriptiveGAT(node_dim=node_dim, edge_dim=edge_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Split training indices into interventions vs. routine decisions.
    # Interventions (Admission IC/NC, IV Antibiotics, IV Liquid) are 22% of the
    # data but the model never recommends them without oversampling.
    # Strategy: 50% intervention examples, 50% routine per epoch.
    INTERVENTION_LABELS = {'Admission IC', 'Admission NC', 'IV Antibiotics', 'IV Liquid'}
    intervention_indices = [
        i for i in train_indices
        if labeled_data[i]['decision_point']['chosen_decision'] in INTERVENTION_LABELS
    ]
    routine_indices = [
        i for i in train_indices
        if labeled_data[i]['decision_point']['chosen_decision'] not in INTERVENTION_LABELS
    ]
    print(f"  Intervention examples: {len(intervention_indices)} "
          f"({len(intervention_indices)/len(train_indices)*100:.1f}%)")
    print(f"  Routine examples:      {len(routine_indices)}")

    for epoch in range(80):
        model.train()
        total_loss, count = 0, 0

        # 50/50 split per epoch: oversample interventions to counter class imbalance
        n_intervention = min(len(intervention_indices), 2500)
        n_routine      = min(5000 - n_intervention, len(routine_indices))
        epoch_indices  = (random.sample(intervention_indices, n_intervention) +
                          random.sample(routine_indices, n_routine))
        random.shuffle(epoch_indices)

        for idx in epoch_indices:
            example = labeled_data[idx]
            try:
                data, candidate_indices, label = subgraph_to_pyg(example, vocab)
                if label < 0 or len(candidate_indices) == 0:
                    continue

                data = data.to(device)
                cand_idx_t = torch.tensor(candidate_indices, dtype=torch.long, device=device)
                label_t = torch.tensor([label], dtype=torch.long, device=device)
                outcome = example['outcome_quality']

                optimizer.zero_grad()
                node_scores = model(data.x, data.edge_index, data.edge_attr)
                cand_scores = node_scores[cand_idx_t].unsqueeze(0)  # [1, num_candidates]

                # Outcome-weighted cross-entropy:
                # weight ∝ outcome quality → learn most from cases with good outcomes
                loss = F.cross_entropy(cand_scores, label_t) * outcome
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1
            except Exception:
                continue

        avg_loss = total_loss / max(count, 1)
        scheduler.step(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/80: loss={avg_loss:.4f} ({count} samples)")

    torch.save(model.state_dict(), model_file)
    print(f"✓ Saved {model_file.name}")

model.eval()

# ── Evaluation ───────────────────────────────────────────────────────────────
print("\n[Evaluation] Top-k accuracy on test set...")

c1, c3, c5, total = 0, 0, 0, 0

with torch.no_grad():
    for idx in test_indices:
        example = labeled_data[idx]
        try:
            data, candidate_indices, label = subgraph_to_pyg(example, vocab)
            if label < 0 or len(candidate_indices) == 0:
                continue

            data = data.to(device)
            cand_idx_t = torch.tensor(candidate_indices, dtype=torch.long, device=device)

            node_scores = model(data.x, data.edge_index, data.edge_attr)
            cand_scores = node_scores[cand_idx_t]
            ranked = torch.argsort(cand_scores, descending=True).cpu().tolist()

            if label == ranked[0]:
                c1 += 1
            if label in ranked[:3]:
                c3 += 1
            if label in ranked[:5]:
                c5 += 1
            total += 1
        except Exception:
            continue

print("\n" + "=" * 70)
print("RESULTS — Node-level PrescriptiveGAT")
print("=" * 70)
print(f"  Top-1 accuracy: {c1/total*100:.2f}%")
print(f"  Top-3 accuracy: {c3/total*100:.2f}%")
print(f"  Top-5 accuracy: {c5/total*100:.2f}%")
print(f"  (over {total} test decision points)")
print("=" * 70)
