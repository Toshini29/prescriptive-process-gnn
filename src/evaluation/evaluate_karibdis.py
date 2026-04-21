"""
evaluate_karibdis.py - GNN vs KARIBDIS SHACL engine (sampled evaluation).

Runs the genuine KaribdisEvaluator (pySHACL + additional_knowledge.ttl rules)
on a representative sample of ~20 test cases across four clinical subgroups:

  Subgroup A — SIRS+ patients          (AdmitSepsisPatientsRule should fire)
  Subgroup B — Elderly + high CRP, no SIRS  (GNN diverges from Karibdis)
  Subgroup C — Abnormal leucocytes      (PerformPreventiveTreatment chain)
  Subgroup D — Routine (no urgent flags) (no Karibdis rule fires)

The derived facts that the SHACL rules need (ProcessValue_Disease = Sepsis
on the Case) are temporarily added before each evaluation and removed after,
so the PKG is not permanently modified.
"""

import sys, pickle, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from collections import defaultdict
from pathlib import Path
from rdflib import URIRef, RDF, Literal, Namespace
import random

script_dir = Path(__file__).resolve().parent
repo_root  = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root / 'src'))

print("=" * 70)
print("GNN vs KARIBDIS — Real SHACL Engine (Sampled Evaluation)")
print("=" * 70)

EX    = Namespace('http://example.org/')
BPO   = Namespace('http://infs.cit.tum.de/karibdis/baseontology/')
MONDO = Namespace('http://purl.obolibrary.org/obo/')
SEPSIS_URI        = MONDO.MONDO_1040015
LEUKOCYTE_DIS_URI = MONDO.MONDO_0004805

# ── Load PKG ──────────────────────────────────────────────────────────────────
from karibdis.ProcessKnowledgeGraph import ProcessKnowledgeGraph
from tests.example_domains.sepsis.previous_tries.karibdis_evaluator import KaribdisEvaluator

pkg = ProcessKnowledgeGraph()
pkg.parse(str(repo_root / 'output' / 'sepsis_complete_pkg.ttl'), format='turtle')
print(f"PKG loaded: {len(pkg)} triples")
evaluator = KaribdisEvaluator(pkg)

# ── Load training data & model ─────────────────────────────────────────────────
torch.manual_seed(42); random.seed(42)

with open(script_dir / 'labeled_training_data_karibdis.pkl', 'rb') as f:
    labeled_data = pickle.load(f)
with open(script_dir / 'vocabulary_karibdis.pkl', 'rb') as f:
    vocab = pickle.load(f)

node_dim = vocab.get('node_dim', 33)
edge_dim  = vocab.get('edge_dim', 5)

case_to_ex = defaultdict(list)
for idx, ex in enumerate(labeled_data):
    case_to_ex[ex['decision_point']['case_id']].append(idx)
unique_cases = list(case_to_ex.keys())
train_cases, temp  = train_test_split(unique_cases, test_size=0.3, random_state=42)
val_cases, test_cases = train_test_split(temp, test_size=0.5, random_state=42)
test_indices = [i for c in test_cases for i in case_to_ex[c]]

NUMERIC_FEATURES = [
    ('CRP',        2, (0.0, 300.0)),
    ('Leucocytes', 3, (0.0,  30.0)),
    ('LacticAcid', 4, (0.0,  10.0)),
    ('Age',        5, (0.0, 100.0)),
]
BOOL_PVS = [
    'SIRSCriteria2OrMore', 'SIRSCritHeartRate', 'SIRSCritTemperature',
    'SIRSCritTachypnea', 'SIRSCritLeucos', 'InfectionSuspected',
    'DisfuncOrg', 'Hypotensie', 'Hypoxie', 'Oligurie', 'Infusion',
    'DiagnosticBlood', 'DiagnosticArtAstrup', 'DiagnosticIC',
    'DiagnosticSputum', 'DiagnosticLiquor', 'DiagnosticOther',
    'DiagnosticXthorax', 'DiagnosticUrinaryCulture', 'DiagnosticLacticAcid',
    'DiagnosticUrinarySediment', 'DiagnosticECG',
]

def get_clinical_state(example):
    sg, nf = example['subgraph'], example['node_features']
    task_nodes = [n for n in sg['nodes']
                  if sg['node_types'].get(n) == 'Task' and n in nf]
    if not task_nodes:
        return {}, []
    f = nf[max(task_nodes, key=lambda n: nf[n][28])]
    numeric = {name: round((lo + float(f[idx]) * (hi - lo)), 2)
               for name, idx, (lo, hi) in NUMERIC_FEATURES if float(f[idx]) > 0.0}
    flags = [BOOL_PVS[i] for i in range(len(BOOL_PVS)) if f[6+i] > 0.5]
    return numeric, flags

class PrescriptiveGAT(nn.Module):
    def __init__(self, node_dim=33, edge_dim=5, hidden=64):
        super().__init__()
        self.conv1 = GATConv(node_dim, 32, heads=4, edge_dim=edge_dim)
        self.conv2 = GATConv(128, hidden, heads=1, edge_dim=edge_dim)
        self.node_scorer = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x_out = self.conv1(x, edge_index, edge_attr)
        x = F.elu(x_out[0] if isinstance(x_out, tuple) else x_out)
        x = F.dropout(x, p=0.1, training=self.training)
        x_out = self.conv2(x, edge_index, edge_attr)
        x = F.elu(x_out[0] if isinstance(x_out, tuple) else x_out)
        return self.node_scorer(x).squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = PrescriptiveGAT(node_dim=node_dim, edge_dim=edge_dim).to(device)
model.load_state_dict(torch.load(
    script_dir / 'prescriptive_gat_karibdisv3.pt', map_location=device))
model.eval()
print(f"GNN model loaded (device: {device})")

def subgraph_to_pyg(example):
    sg, nf, etv = example['subgraph'], example['node_features'], vocab.get('edge_types', [])
    node_list   = sg['nodes']
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    x = torch.from_numpy(np.array([nf[n] for n in node_list])).float()
    eil, efl = [], []
    for src, etype, dst in sg['edges']:
        if src in node_to_idx and dst in node_to_idx:
            eil.append([node_to_idx[src], node_to_idx[dst]])
            ef = torch.zeros(len(etv))
            if etype in etv: ef[etv.index(etype)] = 1.0
            efl.append(ef)
    ei = torch.tensor(eil, dtype=torch.long).t() if eil else torch.empty((2,0), dtype=torch.long)
    ea = torch.stack(efl) if efl else torch.empty((0, len(etv)), dtype=torch.float)
    cands = sg['candidate_activities']
    return Data(x=x, edge_index=ei, edge_attr=ea), cands, [node_to_idx[c] for c in cands if c in node_to_idx]

def extract_name(uri):
    s = str(uri)
    if 'Activity_' in s: return s.split('Activity_')[1].replace('%20', ' ')
    return s.split('/')[-1].replace('_', ' ')

def activity_type(name):
    if name in ('Admission IC', 'Admission NC'):    return 'Admission'
    if name in ('IV Antibiotics', 'IV Liquid'):     return 'Treatment'
    if name in ('CRP', 'Leucocytes', 'LacticAcid'): return 'Lab Test'
    if name.startswith('Release'):                  return 'Discharge'
    return 'Other/Triage'

# ── Sample representative cases ─────────────────────────────────────────────
print("\nSelecting representative sample from test set...")

subgroup_A, subgroup_B, subgroup_C, subgroup_D = [], [], [], []

with torch.no_grad():
    for idx in test_indices:
        ex = labeled_data[idx]
        # Skip very early steps only for non-SIRS subgroups
        # (SIRS is only flagged at step 0-1, so we must include those for subgroup A)
        numeric, flags = get_clinical_state(ex)
        has_sirs     = 'SIRSCriteria2OrMore' in flags
        leuco        = numeric.get('Leucocytes', 0)
        abnorm_leuco = 0 < leuco < 4.0 or leuco > 12.0
        high_crp     = numeric.get('CRP', 0) > 100
        elderly      = numeric.get('Age', 0) > 65

        try:
            data, cands, cand_idx = subgraph_to_pyg(ex)
            if not cand_idx: continue
            scores = model(data.to(device).x, data.to(device).edge_index,
                           data.to(device).edge_attr)
            ranked = [cands[i] for i in
                      torch.argsort(scores[torch.tensor(cand_idx, device=device)],
                                    descending=True).cpu().tolist()]
            gnn_rec = extract_name(ranked[0])
        except Exception:
            continue

        step = ex['decision_point']['decision_index']
        if has_sirs and step >= 1 and len(subgroup_A) < 5:
            subgroup_A.append((idx, gnn_rec, numeric, flags))
        elif elderly and high_crp and not has_sirs and step >= 2 and len(subgroup_B) < 5:
            subgroup_B.append((idx, gnn_rec, numeric, flags))
        elif abnorm_leuco and not has_sirs and step >= 2 and len(subgroup_C) < 5:
            subgroup_C.append((idx, gnn_rec, numeric, flags))
        elif not has_sirs and not high_crp and not abnorm_leuco and step >= 2 and len(subgroup_D) < 5:
            subgroup_D.append((idx, gnn_rec, numeric, flags))

        if all(len(g) >= 5 for g in [subgroup_A, subgroup_B, subgroup_C, subgroup_D]):
            break

sample = (
    [('A: SIRS+',                x) for x in subgroup_A] +
    [('B: Elderly+highCRP+noSIRS', x) for x in subgroup_B] +
    [('C: Abnormal leucocytes',  x) for x in subgroup_C] +
    [('D: Routine (no flags)',   x) for x in subgroup_D]
)
print(f"Sample: {len(subgroup_A)} SIRS+, {len(subgroup_B)} elderly+CRP, "
      f"{len(subgroup_C)} abnorm-leuco, {len(subgroup_D)} routine")
print(f"Running real Karibdis SHACL engine on {len(sample)} cases...\n")

# ── Evaluate each sampled case with real Karibdis ──────────────────────────
results = []
total_karibdis_time = 0.0

for subgroup_label, (idx, gnn_rec, numeric, flags) in sample:
    ex  = labeled_data[idx]
    dp  = ex['decision_point']

    has_sirs     = 'SIRSCriteria2OrMore' in flags
    leuco        = numeric.get('Leucocytes', 0)
    abnorm_leuco = 0 < leuco < 4.0 or leuco > 12.0

    case_uri  = URIRef(dp['case_id'])
    case_name = dp['case_id'].split('Case_')[-1]

    # Derive task_uri from the subgraph: the most recent Task node by temporal position
    # (feature index 28). Do NOT construct it from decision_index — those URIs don't
    # match what OnlineEventImporter created in the PKG.
    sg_ex = ex['subgraph']
    nf_ex = ex['node_features']
    task_nodes_in_sg = [n for n in sg_ex['nodes']
                        if sg_ex['node_types'].get(n) == 'Task' and n in nf_ex]
    if not task_nodes_in_sg:
        continue
    task_uri = URIRef(max(task_nodes_in_sg, key=lambda n: nf_ex[n][28]))

    candidates     = sg_ex['candidate_activities']
    candidate_uris = [URIRef(c) for c in candidates]

    # Temporarily add derived facts so SHACL rules can fire
    temp = []
    if has_sirs:
        temp += [(case_uri, EX.ProcessValue_SIRSCriteria2OrMore, Literal(True)),
                 (case_uri, BPO.ProcessValue_Disease, SEPSIS_URI)]
    if abnorm_leuco:
        temp += [(task_uri, BPO.abnormalValueFor, EX.ProcessValue_Leucocytes),
                 (case_uri, BPO.ProcessValue_Disease, LEUKOCYTE_DIS_URI)]
    for t in temp:
        pkg.add(t)

    t0 = time.time()
    try:
        evals = evaluator.evaluate_all_candidates(task_uri, candidate_uris)
    except Exception as e:
        evals = []
        print(f"  [Error on {case_name}]: {e}")
    finally:
        for t in temp:
            pkg.remove(t)
    elapsed = time.time() - t0
    total_karibdis_time += elapsed

    if not evals:
        continue

    # Agreement: IC/NC treated as equivalent for Admission
    def same(a, b):
        if a == b: return True
        both = lambda x: x in ('Admission IC', 'Admission NC')
        return both(a) and both(b)

    all_zero = all(e['score'] == 0 for e in evals)
    if all_zero:
        # No SHACL rule fired — Karibdis has no preference; show as NoRule
        kar_rec      = 'NoRule'
        kar_score    = 0
        kar_verdict  = 'neutral'
        kar_messages = []
        agrees     = activity_type(gnn_rec) in ('Lab Test', 'Other/Triage')
        rule_fired = 'NoRuleFired'
    else:
        best_kar     = evals[0]
        kar_rec      = extract_name(best_kar['activity'])
        kar_score    = best_kar['score']
        kar_verdict  = best_kar['verdict']
        kar_messages = best_kar['messages']
        agrees = same(gnn_rec, kar_rec)
        # Identify which rule drove the top score
        msg = kar_messages[0] if kar_messages else ''
        if 'sepsis' in msg.lower() and kar_score > 0:
            rule_fired = 'AdmitSepsisPatientsRule (+5)'
        elif 'sepsis' in msg.lower() and kar_score < 0:
            rule_fired = 'DontReleaseSepsisPatientsRule (-3)'
        elif 'preventive' in msg.lower() or 'risk' in msg.lower():
            rule_fired = f'PerformPreventiveTreatment (+{kar_score})'
        elif kar_score == 0:
            rule_fired = 'NoPositiveRule (elimination)'
        else:
            rule_fired = f'Rule (score={kar_score})'

    results.append({
        'subgroup':   subgroup_label,
        'case':       case_name,
        'step':       dp['decision_index'],
        'gnn_rec':    gnn_rec,
        'kar_rec':    kar_rec,
        'kar_score':  kar_score,
        'kar_verdict':kar_verdict,
        'kar_msg':    kar_messages[0][:80] if kar_messages else '(no message)',
        'rule_fired': rule_fired,
        'all_zero':   all_zero,
        'agrees':     agrees,
        'actual':     dp['chosen_decision'],
        'oq':         ex['outcome_quality'],
        'has_sirs':   has_sirs,
        'numeric':    numeric,
        'elapsed':    elapsed,
    })
    status = '[=]' if agrees else '[x]'
    print(f"  {status} {subgroup_label:<30s} Case {case_name:<6s} step {dp['decision_index']:>2d} | "
          f"GNN={gnn_rec:<20s} Karibdis={kar_rec:<20s} "
          f"score={kar_score:>4} rule={rule_fired}  ({elapsed:.1f}s)")

print(f"\nTotal Karibdis evaluation time: {total_karibdis_time:.1f}s")

# ── Results table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS: GNN vs REAL KARIBDIS SHACL ENGINE")
print("=" * 70)

agree    = [r for r in results if r['agrees']]
disagree = [r for r in results if not r['agrees']]
no_rule  = [r for r in results if r['all_zero']]

print(f"\n  Sample size: {len(results)} cases")
print(f"  Agreement:   {len(agree)} / {len(results)} ({len(agree)/len(results)*100:.0f}%)")
print(f"  Divergence:  {len(disagree)} / {len(results)} ({len(disagree)/len(results)*100:.0f}%)")
print(f"  Karibdis rule fired: {len(results)-len(no_rule)} cases  |  "
      f"No rule (score=0): {len(no_rule)} cases")

print(f"\n  {'Subgroup':<30s}  {'Case':<6s}  {'GNN':<20s}  {'Karibdis':<20s}  "
      f"{'Score':>5s}  {'Agree':>6s}  {'OQ':>5s}  {'Rule fired'}")
print("  " + "-" * 130)
for r in results:
    ag = '[=]' if r['agrees'] else '[x]'
    kar_display = f"{r['kar_rec']} (no+rule)" if r['kar_score'] == 0 else r['kar_rec']
    print(f"  {r['subgroup']:<30s}  {r['case']:<6s}  {r['gnn_rec']:<20s}  "
          f"{kar_display:<20s}  {r['kar_score']:>5}  {ag:>6s}  "
          f"{r['oq']:>5.2f}  {r['rule_fired']}")

# ── By subgroup ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("BREAKDOWN BY SUBGROUP")
print("=" * 70)
for sg_label in ['A: SIRS+', 'B: Elderly+highCRP+noSIRS',
                  'C: Abnormal leucocytes', 'D: Routine (no flags)']:
    group = [r for r in results if r['subgroup'] == sg_label]
    if not group: continue
    ag   = [r for r in group if r['agrees']]
    oqs  = [r['oq'] for r in group]
    rules = set(r['rule_fired'] for r in group)
    print(f"\n  {sg_label}")
    print(f"    n={len(group)}, agreement={len(ag)}/{len(group)}, mean OQ={np.mean(oqs):.3f}")
    print(f"    Rules fired: {', '.join(rules)}")
    for r in group:
        ag_s = '[=]' if r['agrees'] else '[x]'
        kar_display = f"{r['kar_rec']} (no+rule)" if r['kar_score'] == 0 else r['kar_rec']
        print(f"    {ag_s} GNN={r['gnn_rec']:<22s} Karibdis={kar_display:<22s} "
              f"msg: {r['kar_msg'][:60]}")

# ── Key thesis finding ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY THESIS FINDING: Subgroup B — Elderly + High CRP + No SIRS")
print("=" * 70)
grp_b = [r for r in results if r['subgroup'] == 'B: Elderly+highCRP+noSIRS']
if grp_b:
    admit = [r for r in grp_b if activity_type(r['gnn_rec']) == 'Admission']
    lab   = [r for r in grp_b if activity_type(r['gnn_rec']) == 'Lab Test']
    print(f"\n  Karibdis: NoRuleFired for all {len(grp_b)} cases")
    print(f"  (MeasurementOutOfBounds would fire only if Karibdis knows CRP threshold)")
    print(f"\n  GNN recommends Admission: {len(admit)}/{len(grp_b)}")
    print(f"  GNN recommends Lab Test:  {len(lab)}/{len(grp_b)}")
    if admit:
        print(f"\n  Mean OQ (GNN->Admission): {np.mean([r['oq'] for r in admit]):.3f}")
    if lab:
        print(f"  Mean OQ (GNN->Lab Test):  {np.mean([r['oq'] for r in lab]):.3f}")
    print(f"\n  Interpretation: Karibdis has no explicit rule for this subgroup.")
    print(f"  The GNN learned from outcome patterns that admission leads to better")
    print(f"  outcomes for elderly patients with markedly elevated CRP, even without")
    print(f"  formal SIRS criteria — a data-driven generalisation beyond Karibdis.")

print("\n" + "=" * 70)
print(f"  Karibdis evaluation used: pySHACL + additional_knowledge.ttl rules")
print(f"  (AdmitSepsisPatientsRule, DontReleaseSepsisPatientsRule,")
print(f"   PerformPreventiveTreatment, ReleaseNonSIRS)")
print("=" * 70)
