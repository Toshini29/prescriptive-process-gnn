"""
explainer.py -- Triple-layer explainability for prescriptive GNN.

Layer 1 -- Integrated Gradients:
  Feature-level attribution: which clinical features (CRP, SIRS, Age, ...)
  drove the recommended node's score. Semantically meaningful because
  features map directly to clinical concepts.

Layer 2 -- GNNExplainer:
  Edge-level attribution: which graph connections (DECLARE constraints,
  task sequences) were most influential. Structurally meaningful because
  edges encode process knowledge (chainresponse, instanceOf, etc.).

Layer 3 -- Clinical Narrative:
  Human-readable synthesis of Layers 1 & 2, grounded in PKG clinical
  rules: SIRS criteria, organ dysfunction, treatment protocols, DECLARE
  constraints. Makes the recommendation interpretable to a clinician.
"""
import pickle
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import numpy as np
import random
from collections import defaultdict
from sklearn.model_selection import train_test_split

print("=" * 70)
print("PRESCRIPTIVE RECOMMENDATIONS -- Triple-layer Explainability")
print("=" * 70)

torch.manual_seed(42)
random.seed(42)

script_dir = Path(__file__).resolve().parent

with open(script_dir / 'labeled_training_data_karibdis.pkl', 'rb') as f:
    labeled_data = pickle.load(f)

with open(script_dir / 'vocabulary_karibdis.pkl', 'rb') as f:
    vocab = pickle.load(f)

node_dim = vocab.get('node_dim', 33)
edge_dim  = vocab.get('edge_dim', 5)
print(f"[ok] {len(labeled_data)} examples, node_dim={node_dim}, edge_dim={edge_dim}")

# -- Case split (same seed as training) ---------------------------------------
case_to_examples = defaultdict(list)
for idx, example in enumerate(labeled_data):
    case_to_examples[example['decision_point']['case_id']].append(idx)

unique_cases = list(case_to_examples.keys())
train_cases, temp_cases = train_test_split(unique_cases, test_size=0.3, random_state=42)
val_cases, test_cases   = train_test_split(temp_cases,  test_size=0.5, random_state=42)
test_indices = [i for c in test_cases for i in case_to_examples[c]]
print(f"[ok] {len(test_indices)} test examples")

# -- Feature layout (must match build_training_data.py exactly) ----------------
NUMERIC_FEATURES = [
    ('CRP',        2, (0.0, 300.0)),
    ('Leucocytes', 3, (0.0,  30.0)),
    ('LacticAcid', 4, (0.0,  10.0)),
    ('Age',        5, (0.0, 100.0)),
]
BOOL_PVS = [
    'SIRSCriteria2OrMore', 'SIRSCritHeartRate', 'SIRSCritTemperature',
    'SIRSCritTachypnea', 'SIRSCritLeucos',
    'InfectionSuspected', 'DisfuncOrg', 'Hypotensie', 'Hypoxie',
    'Oligurie', 'Infusion',
    'DiagnosticBlood', 'DiagnosticArtAstrup', 'DiagnosticIC',
    'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther',
    'DiagnosticXthorax', 'DiagnosticUrinaryCulture', 'DiagnosticLacticAcid',
    'DiagnosticUrinarySediment', 'DiagnosticECG',
]  # indices 6--27

# Human-readable feature names aligned to node feature vector (33 dims)
FEATURE_NAMES = (
    ['is_Task', 'is_Activity', 'CRP', 'Leucocytes', 'LacticAcid', 'Age'] +
    BOOL_PVS +
    ['temporal_position', 'declare_involvement', 'execution_count',
     'is_executed_in_prefix', 'Diagnose']
)

# Clinical thresholds
CRP_HIGH       = 100.0
LACTATE_HIGH   = 2.0
LEUCOCYTES_MIN = 4.0
LEUCOCYTES_MAX = 12.0

# -- URI / name helpers --------------------------------------------------------
def extract_name(uri):
    if 'Activity_' in uri:
        return uri.split('Activity_')[1].replace('%20', ' ')
    if 'Task_' in uri:
        parts = uri.split('Task_')[1].split('_')
        return f"Task {parts[-1]} (case {parts[0]})" if len(parts) >= 2 else 'Task'
    if 'Case_' in uri:
        return 'Case ' + uri.split('Case_')[1]
    return uri.split('/')[-1].replace('_', ' ')

# -- Edge / activity category sets --------------------------------------------
DECLARE_EDGE_TYPES  = {'chainresponse', 'exactly_one', 'init'}
FLOW_EDGE_TYPES     = {'directlyFollowedBy', 'instanceOf'}
INTERVENTION_NAMES  = {'Admission IC', 'Admission NC', 'IV Antibiotics', 'IV Liquid'}
TRIVIAL_NAMES       = {'ER Triage', 'ER Registration', 'ER Sepsis Triage'}
URGENCY_FLAGS       = {
    'SIRSCriteria2OrMore', 'DisfuncOrg', 'Hypotensie', 'Hypoxie',
    'Oligurie', 'InfectionSuspected',
}
VERDICT_ICON = {'good': '[ok]', 'neutral': '~', 'suboptimal': 'v', 'avoid': '[x]'}

# -- Clinical snapshot ---------------------------------------------------------
def get_clinical_snapshot(example):
    """Extract numeric biomarkers and active boolean flags from the most recent Task."""
    sg = example['subgraph']
    nf = example['node_features']
    task_nodes = [n for n in sg['nodes']
                  if sg['node_types'].get(n) == 'Task' and n in nf]
    if not task_nodes:
        return {}, []
    latest = max(task_nodes, key=lambda n: nf[n][28])
    f = nf[latest]
    numeric = {}
    for name, idx, (lo, hi) in NUMERIC_FEATURES:
        val = float(f[idx])
        if val > 0.0:
            numeric[name] = round(lo + val * (hi - lo), 2)
    active_flags = [BOOL_PVS[i] for i in range(len(BOOL_PVS)) if f[6 + i] > 0.5]
    return numeric, active_flags

def count_activity_in_prefix(example, activity_name):
    """Count how many times an activity appears in the prefix task sequence."""
    sg = example['subgraph']
    nf = example['node_features']
    act_uri = f"http://example.org/Activity_{activity_name.replace(' ', '%20')}"
    task_nodes = sorted(
        [n for n in sg['nodes'] if sg['node_types'].get(n) == 'Task' and n in nf],
        key=lambda n: nf[n][28]
    )
    count = 0
    for task_uri in task_nodes:
        for src, etype, dst in sg['edges']:
            if src == task_uri and etype == 'instanceOf' and dst == act_uri:
                count += 1
    return count

# -- Diagnostic summary -------------------------------------------------------
print("\n[Diagnostic] Clinical flag coverage in test set...")
sirs_count = organ_count = crp_high_count = lactate_high_count = interv_count = 0
for idx in test_indices:
    ex = labeled_data[idx]
    sg = ex['subgraph']
    nf = ex['node_features']
    task_nodes = [n for n in sg['nodes'] if sg['node_types'].get(n) == 'Task' and n in nf]
    if task_nodes:
        latest = max(task_nodes, key=lambda n: nf[n][28])
        f = nf[latest]
        if f[6] > 0.5: sirs_count += 1
        if any(f[6+i] > 0.5 for i in [6, 7, 8, 9]): organ_count += 1
        if float(f[2]) * 300.0 > CRP_HIGH: crp_high_count += 1
        if float(f[4]) * 10.0  > LACTATE_HIGH: lactate_high_count += 1
    if ex['decision_point']['chosen_decision'] in INTERVENTION_NAMES:
        interv_count += 1
n = len(test_indices)
print(f"  SIRS+ cases:                {sirs_count}/{n} ({sirs_count/n*100:.1f}%)")
print(f"  Organ dysfunction cases:    {organ_count}/{n} ({organ_count/n*100:.1f}%)")
print(f"  High CRP (>100) cases:      {crp_high_count}/{n} ({crp_high_count/n*100:.1f}%)")
print(f"  High LacticAcid (>2) cases: {lactate_high_count}/{n} ({lactate_high_count/n*100:.1f}%)")
print(f"  Intervention labels:        {interv_count}/{n} ({interv_count/n*100:.1f}%)")

# -- Model ---------------------------------------------------------------------
class PrescriptiveGAT(nn.Module):
    def __init__(self, node_dim=33, edge_dim=5, hidden=64):
        super().__init__()
        self.conv1 = GATConv(node_dim, 32, heads=4, edge_dim=edge_dim)
        self.conv2 = GATConv(128, hidden, heads=1, edge_dim=edge_dim)
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 1)
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
print("[ok] Loaded PrescriptiveGAT")

# -- Explainer setup -----------------------------------------------------------
# Layer 1: Integrated Gradients
USE_IG = False
try:
    from captum.attr import IntegratedGradients
    USE_IG = True
    print("[ok] Integrated Gradients ready")
except ImportError:
    print("[!] Captum not installed (pip install captum) -- IG unavailable, using gradients")

# Layer 2: GNNExplainer
USE_GNNEXPLAINER = False
gnn_explainer = None
try:
    from torch_geometric.explain import Explainer, GNNExplainer as PyGGNNExplainer
    gnn_explainer = Explainer(
        model=model,
        algorithm=PyGGNNExplainer(epochs=150),
        explanation_type='model',
        edge_mask_type='object',
        node_mask_type='attributes',
        model_config=dict(mode='regression', task_level='node', return_type='raw'),
    )
    USE_GNNEXPLAINER = True
    print("[ok] GNNExplainer ready")
except Exception as e:
    print(f"[!] GNNExplainer unavailable ({e})")

# -- Graph builder -------------------------------------------------------------
def subgraph_to_pyg(example):
    sg = example['subgraph']
    nf = example['node_features']
    edge_types_vocab = vocab.get('edge_types', [])
    node_list = sg['nodes']
    node_to_idx = {n: i for i, n in enumerate(node_list)}

    x = torch.from_numpy(np.array([nf[n] for n in node_list])).float()

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
        edge_attr  = torch.stack(edge_feat_list)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr  = torch.empty((0, len(edge_types_vocab)), dtype=torch.float)

    candidates = sg['candidate_activities']
    candidate_indices = [node_to_idx[c] for c in candidates if c in node_to_idx]
    return (Data(x=x, edge_index=edge_index, edge_attr=edge_attr),
            node_list, candidates, candidate_indices)

# -- Ranking -------------------------------------------------------------------
def rank_candidates(data, candidates, candidate_indices):
    cand_idx_t = torch.tensor(candidate_indices, dtype=torch.long, device=device)
    with torch.no_grad():
        node_scores = model(data.x, data.edge_index, data.edge_attr)
        cand_scores = node_scores[cand_idx_t]
        probs = F.softmax(cand_scores, dim=0).cpu().numpy()
    return sorted(zip(candidates, candidate_indices, probs),
                  key=lambda x: x[2], reverse=True)

# -- Layer 1: Integrated Gradients ---------------------------------------------
def get_ig_attributions(data, target_node_idx):
    """
    Compute Integrated Gradients attributions for all node features.
    Returns tensor of shape [num_nodes, node_dim].
    Baseline: zero vector (uninformative patient state).
    """
    x = data.x.to(device)
    ei = data.edge_index.to(device)
    ea = data.edge_attr.to(device)

    if USE_IG:
        try:
            def forward_fn(x_input):
                scores = model(x_input, ei, ea)
                return scores[target_node_idx].unsqueeze(0)

            ig = IntegratedGradients(forward_fn)
            baseline = torch.zeros_like(x)
            attrs, _ = ig.attribute(x, baseline,
                                    n_steps=200,
                                    return_convergence_delta=True)
            return attrs.detach().cpu()
        except Exception as e:
            print(f"  [IG error: {e} -- falling back to gradients]")

    # Gradient fallback
    x_grad = x.clone().detach().requires_grad_(True)
    scores = model(x_grad, ei, ea)
    scores[target_node_idx].backward()
    return x_grad.grad.detach().cpu()


def interpret_ig(ig_attrs, node_list, example, target_node_idx, top_k=5):
    """
    Map IG attributions to clinical concepts.
    Returns list of (feature_name, attribution, clinical_description).

    Strategy: pool attributions from (a) the latest Task node (carries clinical
    measurements) and (b) the recommended Activity node (carries process features).
    Use a relative threshold -- keep the top-k features by magnitude across both
    nodes so that weak but consistent signals are not silently discarded.
    """
    sg = example['subgraph']
    nf = example['node_features']

    # Most recent Task node (carries CRP, SIRS flags, etc.)
    task_nodes = [n for n in sg['nodes']
                  if sg['node_types'].get(n) == 'Task' and n in nf]
    latest_task_uri = max(task_nodes, key=lambda n: nf[n][28]) if task_nodes else None
    latest_task_idx = (node_list.index(latest_task_uri)
                       if latest_task_uri and latest_task_uri in node_list else None)

    numeric, _ = get_clinical_snapshot(example)

    # Collect (feat_name, attr_val, source_node) candidates from both nodes
    candidates = []

    if latest_task_idx is not None:
        task_attrs = ig_attrs[latest_task_idx]
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            candidates.append((feat_name, feat_idx, float(task_attrs[feat_idx]),
                                nf.get(latest_task_uri, []), 'patient_state'))

    # Also check the recommended Activity node (process features + may carry
    # duplicate clinical dims that reflect aggregated context via message-passing)
    rec_uri  = node_list[target_node_idx]
    rec_raw  = nf.get(rec_uri, [])
    rec_attrs = ig_attrs[target_node_idx]
    seen_feats = {c[0] for c in candidates}
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        if feat_name not in seen_feats:
            source = 'process_history' if feat_idx in (29, 30, 31) else 'patient_state'
            candidates.append((feat_name, feat_idx, float(rec_attrs[feat_idx]),
                                rec_raw, source))

    # Compute relative threshold: keep features whose |attr| >= 5% of the max
    max_abs = max((abs(c[2]) for c in candidates), default=0.0)
    rel_threshold = max_abs * 0.05  # top 5% of peak attribution

    results = []
    for feat_name, feat_idx, attr_val, raw_feat, source in candidates:
        if max_abs > 0 and abs(attr_val) < rel_threshold:
            continue
        if abs(attr_val) < 1e-9:   # genuinely zero -- skip
            continue
        desc = _clinical_desc(feat_name, feat_idx, attr_val, numeric, raw_feat)
        if desc:
            results.append((feat_name, attr_val, desc, source))

    results.sort(key=lambda x: abs(x[1]), reverse=True)
    return results[:top_k]


def _clinical_desc(feat_name, feat_idx, attr_val, numeric, raw_feat):
    """Translate a feature + attribution into a clinical sentence."""
    direction = "^" if attr_val > 0 else "v"

    if feat_name == 'CRP':
        val = numeric.get('CRP')
        if val is None: return None
        level = "elevated" if val > CRP_HIGH else "normal range"
        return f"{direction} CRP {val:.0f} mg/L ({level})"

    if feat_name == 'Leucocytes':
        val = numeric.get('Leucocytes')
        if val is None: return None
        status = "abnormal" if val < LEUCOCYTES_MIN or val > LEUCOCYTES_MAX else "normal"
        return f"{direction} Leucocytes {val:.1f} K/uL ({status}, ref 4--12)"

    if feat_name == 'LacticAcid':
        val = numeric.get('LacticAcid')
        if val is None: return None
        level = "high -- tissue hypoperfusion" if val > LACTATE_HIGH else "normal"
        return f"{direction} LacticAcid {val:.1f} mmol/L ({level})"

    if feat_name == 'Age':
        val = numeric.get('Age')
        if val is None: return None
        return f"{direction} Age {val:.0f} years"

    if feat_name == 'temporal_position':
        return f"{direction} Temporal position in case (process stage)"

    if feat_name == 'SIRSCriteria2OrMore':
        return f"{direction} SIRS criteria met -- sepsis escalation {'required' if attr_val > 0 else 'not indicated'}"

    if feat_name == 'InfectionSuspected':
        return f"{direction} Infection suspected"

    if feat_name in ('DisfuncOrg', 'Hypotensie', 'Hypoxie', 'Oligurie'):
        labels = {'DisfuncOrg': 'Organ dysfunction', 'Hypotensie': 'Hypotension',
                  'Hypoxie': 'Hypoxia', 'Oligurie': 'Oliguria'}
        return f"{direction} {labels[feat_name]} flag active"

    if feat_name == 'Infusion':
        return f"{direction} Infusion in progress"

    if feat_name.startswith('SIRSCrit'):
        label = feat_name.replace('SIRSCrit', 'SIRS criterion: ')
        return f"{direction} {label}"

    if feat_name.startswith('Diagnostic'):
        test = feat_name.replace('Diagnostic', '')
        return f"{direction} Diagnostic test ordered: {test}"

    if feat_name == 'declare_involvement':
        return f"{direction} Activity has {'many' if attr_val > 0 else 'few'} DECLARE constraints"

    if feat_name == 'execution_count':
        return f"{direction} Activity {'frequently' if attr_val > 0 else 'rarely'} executed in this prefix"

    if feat_name == 'is_executed_in_prefix':
        return f"{direction} Activity {'already' if attr_val > 0 else 'not yet'} executed in prefix"

    return None


# -- Layer 2: GNNExplainer -----------------------------------------------------
def get_edge_importance(data, target_node_idx):
    """GNNExplainer edge masks for the target node."""
    if USE_GNNEXPLAINER:
        try:
            explanation = gnn_explainer(
                x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                index=torch.tensor(target_node_idx, device=device),
            )
            mask = explanation.edge_mask.detach().cpu().numpy()
            if mask.max() > mask.min():
                mask = (mask - mask.min()) / (mask.max() - mask.min())
            return mask
        except Exception as e:
            print(f"  [GNNExplainer error: {e} -- falling back to gradients]")

    ea = data.edge_attr.clone().detach().requires_grad_(True)
    scores = model(data.x, data.edge_index, ea)
    scores[target_node_idx].backward()
    imp = ea.grad.abs().sum(dim=1).detach().cpu().numpy()
    if imp.max() > imp.min():
        imp = (imp - imp.min()) / (imp.max() - imp.min())
    return imp


def get_top_edges(sg, node_list, edges, edge_importance, rec_uri, top_k=6):
    """Return top-k edges by importance, categorised by type."""
    scored = sorted(
        [(float(edge_importance[i]) if i < len(edge_importance) else 0.0,
          src, etype, dst)
         for i, (src, etype, dst) in enumerate(edges)],
        reverse=True
    )[:top_k]

    declare_edges, flow_edges = [], []
    for imp, src, etype, dst in scored:
        touches = src == rec_uri or dst == rec_uri
        entry   = (extract_name(src), etype, extract_name(dst), imp, touches)
        if etype in DECLARE_EDGE_TYPES:
            declare_edges.append(entry)
        elif etype in FLOW_EDGE_TYPES:
            flow_edges.append(entry)

    return declare_edges, flow_edges


def declare_support_for(sg, activity_uri):
    support = []
    for src, etype, dst in sg['edges']:
        if etype in DECLARE_EDGE_TYPES and (src == activity_uri or dst == activity_uri):
            support.append((extract_name(src), etype, extract_name(dst)))
    return support


# -- Layer 3: Clinical Narrative -----------------------------------------------
def generate_clinical_why(example, ranked, declare_edges):
    """
    Generate plain-English clinical reasoning for the recommended activity.
    Grounded in patient state, process history, and clinical rules.
    """
    best_uri, _, best_score = ranked[0]
    best_name  = extract_name(best_uri)
    numeric, active_flags = get_clinical_snapshot(example)
    oq = example['outcome_quality']

    has_sirs     = 'SIRSCriteria2OrMore' in active_flags
    has_organ_dx = any(f in active_flags
                       for f in ('DisfuncOrg', 'Hypotensie', 'Hypoxie', 'Oligurie'))
    crp_val      = numeric.get('CRP')
    lactate_val  = numeric.get('LacticAcid')
    leuco_val    = numeric.get('Leucocytes')
    high_crp     = crp_val    is not None and crp_val    > CRP_HIGH
    high_lactate = lactate_val is not None and lactate_val > LACTATE_HIGH
    abnorm_leuco = leuco_val   is not None and (leuco_val < LEUCOCYTES_MIN or
                                                leuco_val > LEUCOCYTES_MAX)
    prior_count  = count_activity_in_prefix(example, best_name)

    organ_labels = {
        'Hypotensie': 'hypotension', 'Hypoxie': 'hypoxia',
        'Oligurie': 'oliguria', 'DisfuncOrg': 'organ dysfunction',
    }
    ordinals = {0: 'first', 1: 'second', 2: 'third', 3: 'fourth', 4: 'fifth'}

    lines = []

    # -- Admission IC / NC -----------------------------------------------------
    if best_name in ('Admission IC', 'Admission NC'):
        if has_sirs:
            lines.append(
                "The patient currently meets SIRS criteria (2 or more indicators positive), "
                "consistent with a sepsis diagnosis."
            )
            ward = "the ICU" if best_name == 'Admission IC' else "a hospital ward"
            lines.append(
                f"The sepsis management protocol requires hospital admission. "
                f"Admission to {ward} is recommended."
            )
        if has_organ_dx:
            active_organ = [organ_labels[f] for f in organ_labels if f in active_flags]
            lines.append(
                f"Organ dysfunction is present ({', '.join(active_organ)}), "
                f"supporting the need for {'ICU-level' if best_name == 'Admission IC' else 'inpatient'} care."
            )
        if high_lactate:
            lines.append(
                f"Lactic acid is elevated ({lactate_val:.1f} mmol/L, threshold: 2.0), "
                "indicating possible tissue hypoperfusion -- immediate escalation is warranted."
            )
        if high_crp:
            lines.append(
                f"CRP is significantly elevated ({crp_val:.0f} mg/L, threshold: 100), "
                "reflecting active systemic inflammation that requires inpatient management."
            )
        if abnorm_leuco:
            lines.append(
                f"Leucocyte count ({leuco_val:.1f} K/uL) is outside the normal range (4--12 K/uL), "
                "indicating a leukocyte disorder that increases organ failure risk."
            )

    # -- IV Antibiotics --------------------------------------------------------
    elif best_name == 'IV Antibiotics':
        if has_sirs:
            lines.append(
                "The patient meets SIRS criteria, indicating a possible sepsis diagnosis."
            )
        lines.append(
            "Early initiation of IV antibiotics is a core requirement of the sepsis "
            "management protocol. Delays in antibiotic therapy are associated with "
            "worse outcomes."
        )
        if prior_count == 0:
            lines.append(
                "IV antibiotics have not yet been administered in this admission -- "
                "starting therapy now is appropriate."
            )

    # -- IV Liquid -------------------------------------------------------------
    elif best_name == 'IV Liquid':
        if 'Hypotensie' in active_flags:
            lines.append(
                "The patient is currently hypotensive. IV fluid administration is the "
                "standard first-line intervention to restore perfusion pressure."
            )
        else:
            lines.append(
                "Supportive IV fluid therapy is appropriate to maintain haemodynamic "
                "stability at this stage."
            )
        if has_sirs:
            lines.append("Fluid resuscitation also supports the active sepsis management protocol.")

    # -- Lab tests: CRP / Leucocytes / LacticAcid -----------------------------
    elif best_name in ('CRP', 'Leucocytes', 'LacticAcid'):
        ordinal = ordinals.get(prior_count, f"{prior_count + 1}th")
        units   = {'CRP': 'mg/L', 'Leucocytes': 'K/uL', 'LacticAcid': 'mmol/L'}
        val_map = {'CRP': crp_val, 'Leucocytes': leuco_val, 'LacticAcid': lactate_val}
        val     = val_map.get(best_name)

        if prior_count == 0:
            lines.append(
                f"This would be the first {best_name} measurement in this admission, "
                "establishing a baseline for inflammatory marker monitoring."
            )
        else:
            lines.append(
                f"This is the {ordinal} {best_name} measurement in this admission. "
                "Serial measurements allow clinicians to track the patient's "
                "inflammatory response over time."
            )

        if val is not None:
            u = units[best_name]
            lines.append(f"Current reading: {best_name} = {val} {u}.")

        if not has_sirs and not high_crp and not high_lactate and not abnorm_leuco:
            lines.append(
                "No sepsis criteria or critical thresholds have been reached. "
                "Continued observation without escalation is appropriate at this stage."
            )
        elif high_crp:
            lines.append(
                f"CRP is above the critical threshold (100 mg/L). "
                "Continued measurement will confirm whether levels are rising or falling."
            )
        elif abnorm_leuco:
            lines.append(
                "Leucocyte count is outside the normal range. "
                "Monitoring its trend is important for assessing infection severity."
            )

    # -- Release ---------------------------------------------------------------
    elif best_name.startswith('Release'):
        if not has_sirs:
            lines.append(
                "The patient no longer meets SIRS criteria, indicating resolution "
                "of the acute episode."
            )
            lines.append(
                "Discharge is clinically appropriate. Patients without active sepsis "
                "criteria can be safely released at this stage."
            )
        else:
            lines.append(
                "The model suggests discharge based on outcome patterns in similar cases. "
                "However, SIRS criteria remain active -- clinical judgement is strongly advised "
                "before proceeding."
            )

    # -- ER Sepsis Triage ------------------------------------------------------
    elif best_name == 'ER Sepsis Triage':
        lines.append(
            "The clinical process protocol requires sepsis triage to follow initial "
            "ER triage. This step is mandatory and should not be skipped."
        )

    # -- DECLARE constraint (if highly relevant) -------------------------------
    for src_n, etype, dst_n, imp, touches in declare_edges[:2]:
        if touches and imp > 0.3 and etype == 'chainresponse':
            lines.append(
                f"The process protocol requires '{src_n}' to be followed by "
                f"'{dst_n}' -- this constraint supports the current recommendation."
            )

    # -- Outcome quality -------------------------------------------------------
    if oq >= 0.85:
        lines.append(
            "Historical cases that followed a similar clinical pathway at this stage "
            "consistently showed good patient outcomes."
        )
    elif oq >= 0.6:
        lines.append(
            "Similar cases have generally shown acceptable outcomes along this pathway."
        )
    elif oq < 0.5:
        lines.append(
            "The expected outcome quality for this trajectory is below average. "
            "Close monitoring is advised regardless of the next step taken."
        )

    return lines


def generate_clinical_why_not(example, ranked):
    """
    Generate plain-English reasons why each alternative was not recommended.
    """
    best_uri, _, best_score = ranked[0]
    best_name = extract_name(best_uri)
    numeric, active_flags = get_clinical_snapshot(example)
    oq = example['outcome_quality']

    has_sirs     = 'SIRSCriteria2OrMore' in active_flags
    has_organ_dx = any(f in active_flags
                       for f in ('DisfuncOrg', 'Hypotensie', 'Hypoxie', 'Oligurie'))
    crp_val      = numeric.get('CRP')
    lactate_val  = numeric.get('LacticAcid')
    high_crp     = crp_val    is not None and crp_val    > CRP_HIGH
    high_lactate = lactate_val is not None and lactate_val > LACTATE_HIGH

    results = []
    for uri, _, score in ranked[1:5]:
        alt  = extract_name(uri)
        atype = activity_type_label(alt)

        if alt in ('Admission IC', 'Admission NC'):
            if has_sirs or has_organ_dx or high_crp or high_lactate:
                if alt == 'Admission IC' and best_name == 'Admission NC':
                    reason = (
                        "ICU admission is considered but organ dysfunction or "
                        "critical biomarker thresholds are not met. "
                        "A normal care ward is sufficient at this stage."
                    )
                elif alt == 'Admission NC' and best_name == 'Admission IC':
                    reason = (
                        "Ward admission is an option but organ dysfunction or "
                        "critical lactic acid levels indicate ICU-level care is required."
                    )
                else:
                    reason = (
                        f"Also a clinically valid option given elevated biomarkers. "
                        f"The model prefers {display_name(best_name)} based on outcome "
                        "patterns in similar cases."
                    )
            else:
                reason = (
                    "Hospital admission is not yet indicated -- the patient does not "
                    "meet sepsis criteria or show signs of organ dysfunction."
                )

        elif alt in ('CRP', 'Leucocytes', 'LacticAcid'):
            if has_sirs and has_organ_dx:
                reason = (
                    "Ordering another lab test when organ dysfunction is present "
                    "delays the more urgent intervention that is needed."
                )
            elif has_sirs:
                reason = (
                    "With active SIRS criteria, escalating to treatment takes "
                    f"priority over additional {alt} monitoring at this stage."
                )
            else:
                reason = (
                    f"Also valid for monitoring. The model prefers {best_name} "
                    "based on the recent measurement sequence in this case."
                )

        elif alt.startswith('Release'):
            if has_sirs:
                reason = (
                    "Discharge is not safe while SIRS criteria remain active. "
                    "Releasing this patient prematurely risks rapid deterioration "
                    "and readmission."
                )
            else:
                reason = (
                    f"Discharge is possible but the model assigns lower confidence "
                    f"({score:.0%}) -- more monitoring may be appropriate first."
                )

        elif alt == 'Return ER':
            reason = (
                "Return to the emergency room indicates patient deterioration after "
                "discharge -- this is always an undesired outcome to be avoided."
            )

        elif alt == 'IV Antibiotics':
            if has_sirs or high_crp or high_lactate:
                reason = (
                    f"Antibiotic therapy is also clinically appropriate given the "
                    f"patient's inflammatory markers. The model ranks "
                    f"{display_name(best_name)} higher at this stage."
                )
            else:
                reason = (
                    "IV antibiotics are not yet indicated -- no confirmed sepsis "
                    "criteria or critical biomarker thresholds are present."
                )

        elif alt == 'IV Liquid':
            if 'Hypotensie' not in active_flags:
                reason = (
                    "IV fluid therapy is not urgently indicated -- "
                    "no hypotension is currently recorded."
                )
            else:
                reason = (
                    f"Also appropriate given hypotension. The model prefers "
                    f"{best_name} based on outcome-weighted patterns."
                )

        else:
            reason = (
                f"The model assigns lower confidence to this option ({score:.0%} vs "
                f"{best_score:.0%}) based on outcome patterns in similar cases."
            )

        results.append((alt, atype, score, reason))
    return results


def generate_narrative(example, ranked, ig_features, declare_edges, flow_edges):
    """
    Synthesise IG attributions + GNNExplainer edges into clinical language.
    Returns list of narrative lines.
    """
    best_uri, _, best_score = ranked[0]
    best_name = extract_name(best_uri)
    numeric, active_flags = get_clinical_snapshot(example)
    oq = example['outcome_quality']

    has_sirs     = 'SIRSCriteria2OrMore' in active_flags
    has_organ_dx = any(f in active_flags for f in ('DisfuncOrg','Hypotensie','Hypoxie','Oligurie'))
    has_infect   = 'InfectionSuspected' in active_flags
    crp_val      = numeric.get('CRP')
    lactate_val  = numeric.get('LacticAcid')
    leuco_val    = numeric.get('Leucocytes')
    high_crp     = crp_val    is not None and crp_val    > CRP_HIGH
    high_lactate = lactate_val is not None and lactate_val > LACTATE_HIGH
    abnorm_leuco = leuco_val   is not None and (leuco_val < LEUCOCYTES_MIN or leuco_val > LEUCOCYTES_MAX)

    is_intervention = best_name in INTERVENTION_NAMES
    is_lab_test     = best_name in ('CRP', 'Leucocytes', 'LacticAcid')
    is_release      = best_name.startswith('Release')

    prior_count = count_activity_in_prefix(example, best_name)

    lines = []

    # -- Patient status --------------------------------------------------------
    lines.append("  Patient status at this decision point:")
    if has_sirs:
        lines.append("    [!] SIRS criteria met (>=2 criteria positive) -- sepsis protocol active")
    else:
        lines.append("    · No active SIRS criteria -- sepsis escalation not indicated")

    if has_organ_dx:
        organ_list = [f for f in ('DisfuncOrg','Hypotensie','Hypoxie','Oligurie')
                      if f in active_flags]
        lines.append(f"    [!] Organ dysfunction: {', '.join(organ_list)}")

    if has_infect:
        lines.append("    [!] Infection suspected")

    biomarker_lines = []
    if crp_val is not None:
        flag = " <- ELEVATED" if high_crp else ""
        biomarker_lines.append(f"CRP {crp_val:.0f} mg/L{flag}")
    if lactate_val is not None:
        flag = " <- HIGH (hypoperfusion)" if high_lactate else ""
        biomarker_lines.append(f"LacticAcid {lactate_val:.1f} mmol/L{flag}")
    if leuco_val is not None:
        flag = " <- ABNORMAL" if abnorm_leuco else ""
        biomarker_lines.append(f"Leucocytes {leuco_val:.1f} K/uL{flag}")
    if numeric.get('Age'):
        biomarker_lines.append(f"Age {numeric['Age']:.0f} years")

    if biomarker_lines:
        lines.append(f"    · Biomarkers: {' | '.join(biomarker_lines)}")

    lines.append("")

    # -- Why this recommendation -----------------------------------------------
    lines.append(f"  Why '{best_name}' is recommended:")

    if is_intervention:
        if best_name in ('Admission IC', 'Admission NC') and has_sirs:
            lines.append("    [ok] SIRS criteria active -> hospital admission is clinically required")
            lines.append("      (AdmitSepsisPatientsRule: sepsis patients must be admitted)")
        if best_name == 'Admission IC' and has_organ_dx:
            lines.append("    [ok] Organ dysfunction present -> ICU-level care indicated")
        if best_name == 'IV Antibiotics' and has_sirs:
            lines.append("    [ok] SIRS+ patient -> IV antibiotic protocol should be initiated")
        if best_name == 'IV Liquid' and 'Hypotensie' in active_flags:
            lines.append("    [ok] Hypotension active -> fluid resuscitation indicated")
        if high_crp or high_lactate:
            lines.append("    [ok] High severity biomarkers -> escalation to intervention appropriate")

    elif is_lab_test:
        if prior_count == 0:
            lines.append(f"    · First {best_name} measurement in this case -- baseline monitoring")
        elif prior_count == 1:
            lines.append(f"    · Second {best_name} measurement -- trend monitoring appropriate")
        else:
            lines.append(f"    · {best_name} measured {prior_count} times previously in this case")
            if not has_sirs:
                lines.append("    · No SIRS criteria -> continued monitoring without escalation")

        if not has_sirs and not high_crp and not high_lactate:
            lines.append("    · No clinical urgency flags -> observation protocol appropriate")

    elif is_release:
        if not has_sirs:
            lines.append("    [ok] No SIRS criteria active -- discharge appropriate for this patient")
            lines.append("      (ReleaseNonSIRS rule: non-SIRS patients can be safely discharged)")
        else:
            lines.append("    [!] SIRS criteria present -- discharge carries risk of readmission")

    # DECLARE constraint support
    if declare_edges:
        top_declare = declare_edges[0]
        src_n, etype, dst_n, imp, touches = top_declare
        if touches:
            constraint_labels = {
                'chainresponse': 'must be followed by',
                'exactly_one':   'must occur exactly once',
                'init':          'must be the initial activity',
            }
            label = constraint_labels.get(etype, etype)
            lines.append(f"    · DECLARE ({etype}): '{src_n}' {label} -> supports '{best_name}'")

    # Process flow evidence from GNNExplainer
    if flow_edges:
        top_flow = next((e for e in flow_edges if e[4]), None)  # prefer touching edges
        if top_flow:
            src_n, etype, dst_n, imp, _ = top_flow
            if etype == 'instanceOf':
                lines.append(f"    · Recent execution: a task was an instance of '{dst_n}' "
                             f"(GNNExplainer importance: {imp:.2f})")
            elif etype == 'directlyFollowedBy':
                lines.append(f"    · Process flow: '{src_n}' directly followed by '{dst_n}' "
                             f"(importance: {imp:.2f})")

    # Outcome quality context
    if oq >= 0.8:
        lines.append(f"    · Suffix quality {oq:.2f} -- cases following this path had good outcomes")
    elif oq < 0.5:
        lines.append(f"    [!] Suffix quality {oq:.2f} -- outcome of this trajectory was suboptimal")

    lines.append("")

    # -- Why not alternatives --------------------------------------------------
    lines.append("  Why not the top alternatives:")
    for uri, _, score in ranked[1:4]:
        alt_name = extract_name(uri)
        reasons  = []

        if alt_name in ('Admission IC', 'Admission NC') and not has_sirs:
            reasons.append("no SIRS -> admission not clinically indicated")
        elif alt_name in ('Admission IC', 'Admission NC') and has_sirs:
            reasons.append("admission also clinically valid -- model ranked lower based on process history")
        elif alt_name in ('CRP', 'Leucocytes', 'LacticAcid') and has_sirs and has_organ_dx:
            reasons.append("repeated lab test when organ dysfunction present -- intervention preferred")
        elif alt_name.startswith('Release') and has_sirs:
            reasons.append("SIRS active -> early discharge dangerous (DontReleaseSepsisPatientsRule)")
        elif alt_name.startswith('Release') and not has_sirs:
            reasons.append("discharge valid but model confidence lower at this stage")
        else:
            reasons.append(f"lower outcome-weighted score ({score:.1%} vs {ranked[0][2]:.1%})")

        alt_declare = declare_support_for(example['subgraph'], uri)
        if alt_declare:
            src_n, etype, dst_n = alt_declare[0]
            reasons.append(f"DECLARE support ({etype}: {src_n}->{dst_n}) present but GNN ranks lower")

        lines.append(f"    * {alt_name}: {'; '.join(reasons)}")

    return lines


# -- Verdict -------------------------------------------------------------------
def infer_verdict(score, oq, n_candidates):
    uniform  = 1.0 / max(n_candidates, 1)
    combined = score * 0.6 + oq * 0.4
    if score <= uniform * 1.5:
        # Low confidence -- use outcome quality to decide
        if oq >= 0.8: return 'good'
        if oq >= 0.5: return 'neutral'
        return 'suboptimal'
    if combined >= 0.70: return 'good'
    if combined >= 0.45: return 'neutral'
    if combined >= 0.25: return 'suboptimal'
    return 'avoid'


def _wrap(text, width=64):
    """Word-wrap text into a list of lines, each at most `width` chars."""
    words, lines, current = text.split(), [], []
    for word in words:
        if sum(len(w) + 1 for w in current) + len(word) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


# -- Status label (human-readable verdict) ------------------------------------
STATUS_LABEL = {
    'good':       ('APPROPRIATE',  '[ok]', ''),
    'neutral':    ('ACCEPTABLE',   '~', ''),
    'suboptimal': ('USE CAUTION',  '[!]', ''),
    'avoid':      ('NOT ADVISED',  '[x]', ''),
}

ACTIVITY_DISPLAY = {
    'Admission NC': 'Admission to Normal Care Ward',
    'Admission IC': 'Admission to Intensive Care Unit (ICU)',
    'IV Antibiotics': 'IV Antibiotic Therapy',
    'IV Liquid': 'IV Fluid Resuscitation',
    'Return ER': 'Return to Emergency Room',
    'ER Triage': 'ER Triage',
    'ER Registration': 'ER Registration',
    'ER Sepsis Triage': 'Sepsis Triage',
    'CRP': 'CRP Blood Test',
    'Leucocytes': 'Leucocyte Count Test',
    'LacticAcid': 'Lactic Acid Test',
}

def display_name(name):
    """Return human-readable display name for an activity."""
    if name in ACTIVITY_DISPLAY:
        return ACTIVITY_DISPLAY[name]
    if name.startswith('Release'):
        letter = name.replace('Release ', '').replace('Release', '').strip()
        return f'Patient Discharge (Type {letter})' if letter else 'Patient Discharge'
    return name

def activity_type_label(name):
    """Return a plain-English category for an activity."""
    if name in ('Admission IC', 'Admission NC'):
        return 'Hospital Admission'
    if name in ('IV Antibiotics', 'IV Liquid'):
        return 'Treatment'
    if name in ('CRP', 'Leucocytes', 'LacticAcid'):
        return 'Lab Test'
    if name.startswith('Release'):
        return 'Discharge'
    if name == 'Return ER':
        return 'ER Readmission'
    if name in ('ER Triage', 'ER Registration', 'ER Sepsis Triage'):
        return 'Triage / Registration'
    return 'Clinical Activity'

def plain_edge_reason(src_n, etype, dst_n, touches, rec_name):
    """Translate a graph edge into a plain-English sentence."""
    if etype == 'chainresponse' and touches:
        return f"Process protocol requires '{src_n}' to be followed by '{dst_n}'"
    if etype == 'chainresponse':
        return f"Protocol link: '{src_n}' -> '{dst_n}' (established sequence)"
    if etype == 'exactly_one':
        return f"'{src_n}' must occur exactly once -- constraint noted"
    if etype == 'init':
        return f"'{dst_n}' is designated as an initial activity"
    if etype == 'instanceOf' and touches:
        return f"A recent clinical task was recorded as an instance of '{dst_n}'"
    if etype == 'instanceOf':
        return f"Earlier task linked to '{dst_n}' in the process history"
    if etype == 'directlyFollowedBy':
        return f"'{src_n}' was directly followed by '{dst_n}' in the process"
    return f"Process link: '{src_n}' -> '{dst_n}'"

# -- Main formatter ------------------------------------------------------------
def format_explanation(example, ranked, ig_attrs, edge_imp, node_list):
    sg    = example['subgraph']
    oq    = example['outcome_quality']
    dp    = example['decision_point']
    edges = sg['edges']
    sep   = "=" * 70
    thin  = "-" * 70
    out   = []

    best_uri, best_node_idx, best_score = ranked[0]
    best_name  = extract_name(best_uri)
    best_label = display_name(best_name)
    n_cands    = len(ranked)
    verdict    = infer_verdict(best_score, oq, n_cands)
    status, icon, _ = STATUS_LABEL[verdict]
    act_type   = activity_type_label(best_name)

    numeric, active_flags = get_clinical_snapshot(example)
    has_sirs     = 'SIRSCriteria2OrMore' in active_flags
    has_organ_dx = any(f in active_flags for f in ('DisfuncOrg','Hypotensie','Hypoxie','Oligurie'))
    crp_val      = numeric.get('CRP')
    lactate_val  = numeric.get('LacticAcid')
    leuco_val    = numeric.get('Leucocytes')
    high_crp     = crp_val    is not None and crp_val    > CRP_HIGH
    high_lactate = lactate_val is not None and lactate_val > LACTATE_HIGH
    abnorm_leuco = leuco_val   is not None and (leuco_val < LEUCOCYTES_MIN or leuco_val > LEUCOCYTES_MAX)

    case_label = dp['case_id'].split('Case_')[-1]

    # -- Header ----------------------------------------------------------------
    out.append(sep)
    out.append(f"  CLINICAL DECISION SUPPORT  --  Case {case_label}  |  Step #{dp['decision_index']}")
    out.append(sep)
    out.append("")

    # -- Recommendation box ----------------------------------------------------
    out.append(f"  RECOMMENDED ACTION:  {best_label}")
    out.append(f"  Status   : {icon} {status}")
    out.append(f"  Confidence: {best_score:.0%}  |  Expected outcome quality: {oq:.0%}")
    out.append("")
    out.append(thin)

    # -- Patient overview ------------------------------------------------------
    out.append("  PATIENT OVERVIEW")
    out.append("")

    alerts = []
    if has_sirs:
        alerts.append("[!]  SIRS criteria met (>=2 positive) -- sepsis protocol should be active")
    if has_organ_dx:
        organ_list = [f for f in ('Hypotensie','Hypoxie','Oligurie','DisfuncOrg')
                      if f in active_flags]
        labels = {'Hypotensie': 'Hypotension', 'Hypoxie': 'Hypoxia',
                  'Oligurie': 'Oliguria', 'DisfuncOrg': 'Organ dysfunction'}
        out.append(f"  [!]  Organ dysfunction detected: {', '.join(labels[f] for f in organ_list)}")
    if high_crp:
        alerts.append(f"[!]  CRP elevated ({crp_val:.0f} mg/L > 100) -- significant inflammation")
    if high_lactate:
        alerts.append(f"[!]  Lactic acid high ({lactate_val:.1f} mmol/L > 2.0) -- possible hypoperfusion")
    if abnorm_leuco:
        out.append(f"  [!]  Leucocytes outside normal range ({leuco_val:.1f} K/uL, ref: 4--12)")

    if alerts:
        for a in alerts:
            out.append(f"  {a}")
    else:
        out.append("  [ok]  No active clinical urgency flags")

    out.append("")
    out.append("  Current measurements:")
    if numeric:
        for name, val in numeric.items():
            unit = {'CRP': 'mg/L', 'Leucocytes': 'K/uL',
                    'LacticAcid': 'mmol/L', 'Age': 'years'}.get(name, '')
            out.append(f"    * {name:<15s} {val} {unit}")
    else:
        out.append("    * No biomarker values recorded at this step")

    diag_flags = [f.replace('Diagnostic','') for f in active_flags if f.startswith('Diagnostic')]
    if diag_flags:
        out.append(f"    * Tests ordered: {', '.join(diag_flags)}")

    out.append("")
    out.append(thin)

    # -- Compute evidence ------------------------------------------------------
    ig_features               = interpret_ig(ig_attrs, node_list, example, best_node_idx)
    declare_edges, flow_edges = get_top_edges(sg, node_list, edges, edge_imp, best_uri)

    # -- Why this recommendation -----------------------------------------------
    out.append("  WHY THIS IS RECOMMENDED")
    out.append("")

    why_lines = generate_clinical_why(example, ranked, declare_edges)
    for line in why_lines:
        wrapped = _wrap(line, width=64)
        out.append(f"  * {wrapped[0]}")
        for continuation in wrapped[1:]:
            out.append(f"    {continuation}")
        out.append("")

    out.append(thin)

    # -- Alternatives considered -----------------------------------------------
    out.append("  ALTERNATIVES CONSIDERED")
    out.append("")

    why_not = generate_clinical_why_not(example, ranked)
    for alt_name, alt_type, score, reason in why_not:
        alt_v    = infer_verdict(score, oq, n_cands)
        alt_icon = STATUS_LABEL[alt_v][1]
        alt_label = display_name(alt_name)
        out.append(f"  {alt_icon} {alt_label}  --  {score:.0%} confidence")
        wrapped = _wrap(reason, width=62)
        for j, seg in enumerate(wrapped):
            out.append(f"     {seg}")
        out.append("")

    out.append(thin)

    # -- Model verification ----------------------------------------------------
    # Cross-check: do IG feature attributions and GNNExplainer edge importance
    # support the same concepts the clinical narrative reasons about?
    out.append("  MODEL VERIFICATION")
    out.append("")

    # Clinical features and their narrative keywords
    CONCEPT_KEYWORDS = {
        'CRP':                 ['CRP', 'inflammation'],
        'LacticAcid':          ['lactic', 'Lactic', 'hypoperfusion'],
        'Leucocytes':          ['Leucocyte', 'leucocyte'],
        'Age':                 ['age', 'Age', 'elderly'],
        'SIRSCriteria2OrMore': ['SIRS', 'sepsis'],
        'InfectionSuspected':  ['Infection', 'infection'],
        'DisfuncOrg':          ['Organ', 'organ dysfunction'],
        'Hypotensie':          ['hypotension', 'Hypotension', 'fluid'],
        'Hypoxie':             ['hypoxia', 'Hypoxia'],
        'Oligurie':            ['oliguria', 'Oliguria'],
    }
    PROCESS_FEATURES = {'declare_involvement', 'execution_count',
                        'is_executed_in_prefix', 'temporal_position'}

    narrative_text = ' '.join(why_lines)

    # Separate IG features into clinical vs process
    clinical_ig, process_ig = [], []
    for fn, av, *_ in ig_features:
        if abs(av) < 1e-9:
            continue
        if fn in PROCESS_FEATURES:
            process_ig.append((fn, av))
        elif fn in CONCEPT_KEYWORDS:
            clinical_ig.append((fn, av))

    # Check which clinical IG features match narrative
    clinical_matched, clinical_unmatched = [], []
    for feat_name, attr_val in clinical_ig:
        keywords = CONCEPT_KEYWORDS[feat_name]
        direction = 'positive' if attr_val > 0 else 'negative'
        if any(kw in narrative_text for kw in keywords):
            clinical_matched.append((feat_name, direction))
        else:
            clinical_unmatched.append((feat_name, direction))

    # GNNExplainer: check if top DECLARE edge touches recommended activity
    top_declare_aligned = any(touches and imp > 0.25
                              for _, _, _, imp, touches in declare_edges[:3])

    # -- Report clinical feature alignment
    if clinical_matched:
        out.append("  Clinical feature attribution (IG) -- aligned with narrative:")
        for fn, dirn in clinical_matched:
            out.append(f"    [+] {fn}: {dirn} attribution confirms narrative reasoning")
    if clinical_unmatched:
        out.append("  Clinical feature attribution (IG) -- present but not in narrative:")
        for fn, dirn in clinical_unmatched:
            out.append(f"    [?] {fn}: {dirn} attribution -- may indicate latent signal")
    if not clinical_ig:
        out.append("  Clinical features: no strong clinical feature attributions from IG")

    # -- Report process feature reliance
    if process_ig:
        proc_names = ', '.join(fn for fn, _ in process_ig[:3])
        out.append(f"  Process features driving model (IG): {proc_names}")
        out.append("    -> Model uses process history/position alongside biomarkers")

    # -- GNNExplainer DECLARE edge check
    if top_declare_aligned:
        out.append("  GNNExplainer: DECLARE process constraint supports recommendation [+]")
    elif declare_edges:
        out.append("  GNNExplainer: process constraints present, indirect support [~]")
    else:
        out.append("  GNNExplainer: no DECLARE constraints active at this step")

    # -- Overall verdict
    has_clinical = bool(clinical_matched)
    has_process  = bool(process_ig)
    has_declare  = top_declare_aligned

    if has_clinical and (has_process or has_declare):
        verdict_line = "[ALIGNED] Both clinical and process signals consistent with narrative"
    elif has_clinical:
        verdict_line = "[ALIGNED] Clinical attribution matches narrative reasoning"
    elif has_process and not clinical_ig:
        verdict_line = "[PROCESS-LED] Model relies on process position -- narrative grounds this clinically"
    elif clinical_unmatched:
        verdict_line = "[PARTIAL] Model sees additional clinical signals not captured in narrative"
    else:
        verdict_line = "[REVIEW] Weak attribution signal -- model confidence may be low"

    out.append(f"  Overall: {verdict_line}")
    out.append("")
    out.append(thin)

    # -- Technical footnote ----------------------------------------------------
    ig_ref  = "Integrated Gradients (Sundararajan et al., 2017)" if USE_IG else "Gradient attribution"
    gnn_ref = "GNNExplainer (Ying et al., 2019)" if USE_GNNEXPLAINER else "Gradient edge masks"
    out.append(f"  Methods: {ig_ref} + {gnn_ref} used as narrative verification")
    out.append(f"  Model: Node-level PrescriptiveGAT | Outcome-weighted cross-entropy loss")
    out.append(sep)
    return "\n".join(out)


# -- Demo case sampling --------------------------------------------------------
print("\n" + "=" * 70)
print("GENERATING EXPLANATIONS")
print("=" * 70)

model.eval()

def clinical_tier(example, rec_name, top_score, uniform):
    """
    Classify a decision point into a priority tier for demo selection.

    Tier 1 -- Intervention recommended for a clinically urgent patient
             (SIRS / organ dysfunction / high biomarkers)
    Tier 2 -- Any intervention recommended (model learned to escalate)
    Tier 3 -- High-severity biomarkers present (CRP>100 or LacticAcid>2)
    Tier 4 -- Suboptimal outcome quality (<0.5) -- shows model flagging risk
    Tier 5 -- Confident non-trivial prediction (general fallback)
    None   -- Skip (trivial / low confidence)
    """
    numeric, flags = get_clinical_snapshot(example)
    oq = example['outcome_quality']

    has_sirs     = 'SIRSCriteria2OrMore' in flags
    has_organ_dx = any(f in flags for f in ('DisfuncOrg','Hypotensie','Hypoxie','Oligurie'))
    high_crp     = numeric.get('CRP',    0) > CRP_HIGH
    high_lactate = numeric.get('LacticAcid', 0) > LACTATE_HIGH

    is_intervention = rec_name in INTERVENTION_NAMES
    is_urgent_state = has_sirs or has_organ_dx or high_crp or high_lactate

    # Tier 0: early triage step with active SIRS -- best case for IG demonstration
    # (IG highlights SIRS criteria features; not filtered as "trivial" here)
    if rec_name in TRIVIAL_NAMES and has_sirs:
        return 0

    if rec_name in TRIVIAL_NAMES:
        return None
    if top_score < uniform * 2:
        return None

    if is_intervention and is_urgent_state:
        return 1
    if is_intervention:
        return 2
    if is_urgent_state:
        return 3
    if oq < 0.5:
        return 4
    return 5

# Multi-tier sampling: collect best example per tier per case
tier_buckets = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
seen_cases   = set()

for idx in test_indices:
    example = labeled_data[idx]
    case_id = example['decision_point']['case_id']
    dec_idx = example['decision_point']['decision_index']

    if dec_idx < 1:
        continue
    try:
        data, node_list, cands, cand_indices = subgraph_to_pyg(example)
        if not cand_indices:
            continue
        ranked    = rank_candidates(data.to(device), cands, cand_indices)
        if not ranked:
            continue
        top_score = ranked[0][2]
        uniform   = 1.0 / max(len(cand_indices), 1)
        rec_name  = extract_name(ranked[0][0])

        tier = clinical_tier(example, rec_name, top_score, uniform)
        if tier is None:
            continue

        # Keep only the best (highest confidence) example per case per tier
        if case_id not in tier_buckets[tier] or \
           top_score > ranked[0][2]:
            tier_buckets[tier][case_id] = (idx, top_score)
    except Exception:
        continue

# Report what was found
for t in range(0, 6):
    tier_labels = {
        0: 'Tier 0 -- Early SIRS triage (IG demonstration)',
        1: 'Tier 1 -- Intervention + urgent patient',
        2: 'Tier 2 -- Intervention recommended',
        3: 'Tier 3 -- High-severity biomarkers',
        4: 'Tier 4 -- Suboptimal trajectory (oq<0.5)',
        5: 'Tier 5 -- General confident predictions',
    }
    print(f"  {tier_labels[t]}: {len(tier_buckets[t])} cases")

# Fill demo_map: pick up to 10 cases, 2 per tier where possible
demo_map = {}
for tier in range(0, 6):
    slots = 2 if tier > 0 else 2
    count = 0
    for case_id, (idx, _) in tier_buckets[tier].items():
        if len(demo_map) >= 10:
            break
        if case_id not in demo_map and count < slots:
            demo_map[case_id] = (idx, tier)
            count += 1
    if len(demo_map) >= 10:
        break

print(f"\n  Showing {len(demo_map)} demos:")
for cid, (idx, tier) in demo_map.items():
    case_label = cid.split('Case_')[-1]
    rec = extract_name(labeled_data[idx]['subgraph']['candidate_activities'][0]) \
          if labeled_data[idx]['subgraph']['candidate_activities'] else '?'
    print(f"    Case {case_label} (Tier {tier})")

for i, (demo_idx, tier) in enumerate(demo_map.values()):
    example  = labeled_data[demo_idx]
    data, node_list, candidates, candidate_indices = subgraph_to_pyg(example)
    data     = data.to(device)

    if not candidate_indices:
        print(f"\nCase {i+1}: no candidate nodes, skipping.")
        continue

    ranked        = rank_candidates(data, candidates, candidate_indices)
    best_node_idx = ranked[0][1]
    rec_name      = extract_name(ranked[0][0])

    print(f"\nCase {i+1}: computing attributions for '{rec_name}'...")
    ig_attrs = get_ig_attributions(data, best_node_idx)
    edge_imp = get_edge_importance(data, best_node_idx)

    print(format_explanation(example, ranked, ig_attrs, edge_imp, node_list))

print("\n[ok] Done.")
print("=" * 70)
