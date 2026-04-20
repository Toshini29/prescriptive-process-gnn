"""
build_training_data.py — Training Data pipeline.

The PKG already contains EVERYTHING:
  - Task instances with all clinical values as literal properties
  - Case nodes with Age
  - Activity schema with DECLARE constraints between activities
  - Temporal sequence via directlyFollowedBy edges

Graph structure (faithful to PKG, no invented nodes):
  Task nodes    — carry clinical values directly as node features
  Activity nodes — carry DECLARE constraint info as node features
"""
import pickle
import numpy as np
from pathlib import Path
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD
from collections import defaultdict

print("=" * 70)
print("BUILD TRAINING DATA FROM PKG")
print("=" * 70)

script_dir = Path(__file__).resolve().parent
repo_root  = Path(__file__).resolve().parents[4]
ttl_path   = repo_root / "output" / "sepsis_complete_pkg.ttl"

# ── Namespaces ────────────────────────────────────────────────────────────────
BASE    = Namespace("http://infs.cit.tum.de/karibdis/baseontology/")
DECLARE = Namespace("http://infs.cit.tum.de/karibdis/declare/")
EX      = Namespace("http://example.org/")

DIRECTLY_FOLLOWED_BY = BASE.directlyFollowedBy
INSTANCE_OF          = BASE.instanceOf
PART_OF              = BASE.partOf
PERFORMED_BY         = BASE.performedBy
WRITES_VALUE         = BASE.writesValue
COMPLETED_AT         = BASE.completedAt

# DECLARE predicates we care about
DECLARE_PREDS = {
    str(DECLARE.chainresponse): 'chainresponse',
    str(DECLARE.exactly_one):   'exactly_one',
    str(DECLARE.init):          'init',
}

# Activities that determine outcome quality
RELEASE_URIS = {
    str(EX['Activity_Release%20A']), str(EX['Activity_Release%20B']),
    str(EX['Activity_Release%20C']), str(EX['Activity_Release%20D']),
    str(EX['Activity_Release%20E']),
}
RETURN_ER_URI        = str(EX['Activity_Return%20ER'])
ADMISSION_IC_URI     = str(EX['Activity_Admission%20IC'])
ADMISSION_NC_URI     = str(EX['Activity_Admission%20NC'])
IV_ANTIBIOTICS_URI   = str(EX['Activity_IV%20Antibiotics'])
IV_LIQUID_URI        = str(EX['Activity_IV%20Liquid'])
ER_TRIAGE_URI        = str(EX['Activity_ER%20Triage'])
ER_SEPSIS_TRIAGE_URI = str(EX['Activity_ER%20Sepsis%20Triage'])
ER_REGISTRATION_URI  = str(EX['Activity_ER%20Registration'])

# Activity categories
LAB_TEST_URIS = {
    str(EX['Activity_CRP']),
    str(EX['Activity_Leucocytes']),
    str(EX['Activity_LacticAcid']),
}
INTERVENTION_URIS = {
    ADMISSION_IC_URI,
    ADMISSION_NC_URI,
    IV_ANTIBIOTICS_URI,
    IV_LIQUID_URI,
}

# Normal ranges from PKG (:normal_min / :normal_max)
LEUCOCYTES_MIN, LEUCOCYTES_MAX = 4.0, 12.0

# Severity thresholds (clinical literature, standard sepsis criteria)
CRP_HIGH_THRESHOLD        = 100.0  # mg/L — elevated systemic inflammation
LACTICACID_HIGH_THRESHOLD = 2.0    # mmol/L — tissue hypoperfusion indicator

# Activities with exactly_one DECLARE constraint (from PKG)
EXACTLY_ONE_URIS = {
    ER_TRIAGE_URI,
    ER_SEPSIS_TRIAGE_URI,
}

# Feature normalization ranges
NUMERIC_RANGES = {
    'CRP':        (0.0, 300.0),
    'Leucocytes': (0.0,  30.0),
    'LacticAcid': (0.0,  10.0),
    'Age':        (0.0, 100.0),
}

# Ordered list of boolean ProcessValue predicates → feature indices [6..27]
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

NODE_DIM = 33
EDGE_TYPES = ['directlyFollowedBy', 'instanceOf', 'chainresponse', 'exactly_one', 'init']
EDGE_TYPE_TO_IDX = {et: i for i, et in enumerate(EDGE_TYPES)}
EDGE_DIM = len(EDGE_TYPES)

# ── Load PKG ──────────────────────────────────────────────────────────────────
print(f"\n[1/5] Loading PKG from {ttl_path.name}...")
pkg = Graph()
pkg.parse(str(ttl_path), format="turtle")
print(f"  ✓ {len(pkg):,} triples")

# ── Collect all Activity nodes and their DECLARE constraints ──────────────────
print("\n[2/5] Collecting Activity nodes and DECLARE constraints...")

all_activity_uris = sorted(
    str(s) for s, p, o in pkg.triples((None, RDF.type, BASE.Activity))
)
print(f"  ✓ {len(all_activity_uris)} activities")

# declare_edges: list of (src_uri, edge_type_str, dst_uri)
declare_edges = []
declare_involvement = defaultdict(int)  # uri → count of DECLARE edges touching it

for act_uri in all_activity_uris:
    act_ref = URIRef(act_uri)
    for s, p, o in pkg.triples((act_ref, None, None)):
        p_str = str(p)
        if p_str in DECLARE_PREDS:
            o_str = str(o)
            if o_str in all_activity_uris or o_str == act_uri:
                etype = DECLARE_PREDS[p_str]
                declare_edges.append((act_uri, etype, o_str))
                declare_involvement[act_uri] += 1
                declare_involvement[o_str]   += 1

max_declare = max(declare_involvement.values(), default=1)
print(f"  ✓ {len(declare_edges)} DECLARE constraint edges")

# ── Collect Case-level attributes (Age) ───────────────────────────────────────
print("\n[3/5] Collecting case-level attributes and task sequences...")

case_age = {}
for case_ref, p, o in pkg.triples((None, EX.ProcessValue_Age, None)):
    try:
        case_age[str(case_ref)] = float(o)
    except (ValueError, TypeError):
        pass

# Diagnose: stored as URI (log:Diagnose_X), get rdfs:label, ordinal-encode
case_diagnose_raw = {}
for case_ref, p, o in pkg.triples((None, EX.ProcessValue_Diagnose, None)):
    label = pkg.value(o, RDFS.label)
    if label is not None:
        case_diagnose_raw[str(case_ref)] = str(label)

all_diagnose_labels = sorted(set(case_diagnose_raw.values()))
diagnose_rank = {lbl: i / max(len(all_diagnose_labels) - 1, 1)
                 for i, lbl in enumerate(all_diagnose_labels)}
case_diagnose = {uri: diagnose_rank[lbl]
                 for uri, lbl in case_diagnose_raw.items()}
print(f"  ✓ {len(all_diagnose_labels)} distinct Diagnose values (ordinal encoded)")

# ── Build per-case task sequences ─────────────────────────────────────────────
def get_task_sequence(pkg, case_uri_str):
    """Return ordered list of task URIs for this case via directlyFollowedBy.

    Every case has a stub terminal task (only :partOf, no instanceOf/DFB).
    This stub and the real first task both have no incoming DFB edge, so we
    must pick the one that has an outgoing DFB edge (i.e. the real first).
    """
    case_ref = URIRef(case_uri_str)
    case_tasks = set(
        str(s) for s, p, o in pkg.triples((None, PART_OF, case_ref))
    )
    if not case_tasks:
        return []

    # Tasks with an outgoing directlyFollowedBy within this case
    has_outgoing = set(
        str(s) for s, p, o in pkg.triples((None, DIRECTLY_FOLLOWED_BY, None))
        if str(s) in case_tasks and str(o) in case_tasks
    )
    # Tasks that are targets of directlyFollowedBy within this case
    has_predecessor = set(
        str(o) for s, p, o in pkg.triples((None, DIRECTLY_FOLLOWED_BY, None))
        if str(s) in case_tasks and str(o) in case_tasks
    )

    # First task: no predecessor AND has an outgoing edge (excludes stub)
    first_candidates = (case_tasks - has_predecessor) & has_outgoing
    if not first_candidates:
        # fallback: any task without a predecessor
        first_candidates = case_tasks - has_predecessor
    if not first_candidates:
        return list(case_tasks)

    sequence = []
    current = next(iter(first_candidates))
    visited = set()
    while current and current not in visited:
        sequence.append(current)
        visited.add(current)
        nexts = [
            str(o) for s, p, o in pkg.triples((URIRef(current), DIRECTLY_FOLLOWED_BY, None))
            if str(o) in case_tasks
        ]
        current = nexts[0] if nexts else None

    return sequence


def get_task_clinical_values(pkg, task_uri_str):
    """
    Extract all ProcessValue literals from a Task node.
    Returns dict: pv_name → raw value (float or bool).
    """
    task_ref = URIRef(task_uri_str)
    values = {}
    for s, p, o in pkg.triples((task_ref, None, None)):
        p_str = str(p)
        # ProcessValue predicates look like http://example.org/ProcessValue_CRP
        if 'ProcessValue_' in p_str and isinstance(o, Literal):
            pv_name = p_str.split('ProcessValue_')[1]
            try:
                if o.datatype in (XSD.boolean,):
                    values[pv_name] = bool(o)
                else:
                    values[pv_name] = float(o)
            except (ValueError, TypeError):
                pass
    return values


def get_task_activity(pkg, task_uri_str):
    """Return the Activity URI this Task is an instanceOf, or None."""
    task_ref = URIRef(task_uri_str)
    for s, p, o in pkg.triples((task_ref, INSTANCE_OF, None)):
        return str(o)
    return None


def make_task_feature(clinical_values, position, total_tasks, age_norm, diagnose_norm):
    """Build the 33-dim feature vector for a Task node."""
    f = np.zeros(NODE_DIM, dtype=np.float32)
    f[0] = 1.0  # is_Task

    # Numeric biomarkers [2-5]
    for i, name in enumerate(['CRP', 'Leucocytes', 'LacticAcid']):
        val = clinical_values.get(name)
        if val is not None:
            lo, hi = NUMERIC_RANGES[name]
            f[2 + i] = float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))
    f[5] = age_norm  # Age from Case node

    # Boolean ProcessValues [6-27]
    for i, name in enumerate(BOOL_PVS):
        val = clinical_values.get(name)
        if val is not None:
            f[6 + i] = 1.0 if val else 0.0

    # Temporal position [28]
    f[28] = position / max(total_tasks - 1, 1)

    # Diagnose ordinal [32]
    f[32] = diagnose_norm

    # [29-31] are Activity-specific → stay 0 for Task nodes
    return f


def make_activity_feature(act_uri, declare_inv, max_decl, exec_count, max_exec, is_executed):
    """Build the 32-dim feature vector for an Activity node."""
    f = np.zeros(NODE_DIM, dtype=np.float32)
    f[1] = 1.0  # is_Activity
    # [2-28] clinical values → 0 (Activities have no clinical literals)
    f[29] = declare_inv / max(max_decl, 1)
    f[30] = exec_count / max(max_exec, 1)
    f[31] = 1.0 if is_executed else 0.0
    return f


def suffix_quality(activity_sequence, decision_index, clinical):
    """
    Karibdis-rule-informed suffix quality for decision at position i.

    clinical: dict of ProcessValue name → raw value for the current task
              (used to apply clinical decision rules from the PKG).

    Rules incorporated (from additional_knowledge.ttl / PKG SHACL rules):

      AdmitSepsisPatientsRule      (+0.35):
        SIRS-positive patients who get admitted (IC or NC) in the suffix
        → admission is the right clinical action for sepsis

      DontReleaseSepsisPatientsRule (-0.30):
        SIRS-positive patients released without prior admission
        → dangerous early discharge

      ReleaseNonSIRS               (+0.20):
        Non-SIRS patients released in suffix
        → correct discharge for mild cases

      PerformPreventiveTreatment   (+0.25):
        Abnormal Leucocytes (outside 4–12 K/μL) + ICU admission in suffix
        → Leukocyte disorder → organ failure risk → ICU is required preventive action

      MeasurementOutOfBounds penalty (-0.10):
        Abnormal Leucocytes present but NO admission in suffix
        → risk factor unaddressed

      OrganDysfunction + ICU       (+0.10 / -0.15):
        Organ dysfunction (Hypotensie/Hypoxie/Oligurie/DisfuncOrg) present:
          ICU in suffix → appropriate escalation
          No ICU in suffix → under-treatment penalty

      IVAntibiotics                (+0.10 / -0.05):
        SIRS-positive patients:
          IV Antibiotics in suffix → sepsis treatment protocol followed
          No IV Antibiotics in suffix → protocol gap

      SevereSepsis (high CRP/LacticAcid) (+0.10 / -0.10):
        High CRP (>100 mg/L) or high LacticAcid (>2 mmol/L) indicates
        severe sepsis / septic shock:
          Admission in suffix → correct escalation for severe patient
          Release before admission → dangerous under-treatment of severe case

      AtMostOnce violation         (-0.15):
        ER Triage / ER Sepsis Triage are exactly_one activities.
        Appearing in both prefix AND suffix = constraint violation.

    Plus base process signals:
      +0.15 any release in suffix
      -0.40 Return ER in suffix
      +0.10 short suffix (<10)
      -0.15 long suffix  (>20)
    """
    suffix = activity_sequence[decision_index + 1:]
    prefix = activity_sequence[:decision_index + 1]

    # ── Clinical state at this decision point ─────────────────────────────
    has_sirs = bool(clinical.get('SIRSCriteria2OrMore', False))
    leucocytes = clinical.get('Leucocytes', None)
    abnormal_leucocytes = (
        leucocytes is not None and
        (leucocytes < LEUCOCYTES_MIN or leucocytes > LEUCOCYTES_MAX)
    )
    has_organ_dysfunction = any(
        bool(clinical.get(f, False))
        for f in ('Hypotensie', 'Hypoxie', 'Oligurie', 'DisfuncOrg')
    )
    crp_val = clinical.get('CRP', None)
    lactate_val = clinical.get('LacticAcid', None)
    high_severity = (
        (crp_val is not None and crp_val > CRP_HIGH_THRESHOLD) or
        (lactate_val is not None and lactate_val > LACTICACID_HIGH_THRESHOLD)
    )

    # ── Suffix analysis ────────────────────────────────────────────────────
    if not suffix:
        # Last decision — assess whole-case quality
        has_release  = any(u in RELEASE_URIS for u in activity_sequence)
        has_return   = RETURN_ER_URI in activity_sequence
        has_admitted = (ADMISSION_IC_URI in activity_sequence or
                        ADMISSION_NC_URI in activity_sequence)
        has_admitted_ic = ADMISSION_IC_URI in activity_sequence
        has_iv_abx = IV_ANTIBIOTICS_URI in activity_sequence
        q = 0.5
        if has_sirs and has_admitted and has_release: q += 0.35
        elif has_sirs and has_release and not has_admitted: q -= 0.30
        if not has_sirs and has_release: q += 0.20
        if has_organ_dysfunction and has_admitted_ic: q += 0.10
        elif has_organ_dysfunction and not has_admitted: q -= 0.15
        if has_sirs and has_iv_abx: q += 0.10
        elif has_sirs and not has_iv_abx: q -= 0.05
        if high_severity and has_admitted: q += 0.10
        if has_return: q -= 0.40
        return float(np.clip(q, 0.0, 1.0))

    has_release      = any(u in RELEASE_URIS for u in suffix)
    has_return       = RETURN_ER_URI in suffix
    has_admission_ic = ADMISSION_IC_URI in suffix
    has_admission_nc = ADMISSION_NC_URI in suffix
    has_admission    = has_admission_ic or has_admission_nc
    has_iv_abx       = IV_ANTIBIOTICS_URI in suffix
    suffix_len       = len(suffix)

    # Check if release happens before any admission in suffix
    release_before_admission = False
    if has_release and not has_admission:
        release_before_admission = True
    elif has_release and has_admission:
        first_release = next(
            (j for j, u in enumerate(suffix) if u in RELEASE_URIS), suffix_len
        )
        first_admit = next(
            (j for j, u in enumerate(suffix)
             if u in (ADMISSION_IC_URI, ADMISSION_NC_URI)), suffix_len
        )
        release_before_admission = first_release < first_admit

    q = 0.5

    # ── Karibdis clinical rules ────────────────────────────────────────────
    # AdmitSepsisPatientsRule: SIRS → admission required
    if has_sirs and has_admission:
        q += 0.35

    # DontReleaseSepsisPatientsRule: SIRS → no early release
    if has_sirs and has_release and release_before_admission:
        q -= 0.30

    # ReleaseNonSIRS: non-SIRS patients should be discharged
    if not has_sirs and has_release:
        q += 0.20

    # PerformPreventiveTreatment:
    # Leukocyte disorder (abnormal WBC) → organ failure risk → ICU needed
    if abnormal_leucocytes and has_admission_ic:
        q += 0.25
    elif abnormal_leucocytes and not has_admission:
        q -= 0.10  # MeasurementOutOfBounds: risk unaddressed

    # OrganDysfunction: severe patients need ICU escalation
    if has_organ_dysfunction and has_admission_ic:
        q += 0.10
    elif has_organ_dysfunction and not has_admission:
        q -= 0.15  # organ failure without escalation → under-treatment

    # IVAntibiotics: sepsis treatment protocol signal
    if has_sirs and has_iv_abx:
        q += 0.10
    elif has_sirs and not has_iv_abx:
        q -= 0.05

    # SevereSepsis: high biomarkers → admission is critical
    if high_severity and has_admission:
        q += 0.10
    elif high_severity and release_before_admission:
        q -= 0.10  # severe patient released early

    # AtMostOnce violation: exactly_one activities must not repeat
    for act_uri in EXACTLY_ONE_URIS:
        if act_uri in prefix and act_uri in suffix:
            q -= 0.15
            break  # penalise once per decision point

    # ── Base process signals ───────────────────────────────────────────────
    if has_release:
        q += 0.15
    if has_return:
        q -= 0.40
    if suffix_len < 10:
        q += 0.10
    if suffix_len > 20:
        q -= 0.15

    return float(np.clip(q, 0.0, 1.0))


def action_quality_adjustment(chosen_uri, clinical, prefix_activity_uris, decision_index):
    """
    Action-specific quality adjustment — did the chosen activity make clinical sense
    given the patient's state at this decision point?

    This resolves the core ambiguity in suffix-only labeling: if a patient went
    CRP → Admission NC → Release, both CRP and Admission NC get the same high
    suffix quality, so the model can't distinguish them. This function rewards
    or penalises the CHOSEN activity based on whether it was the right call NOW.

    Cases covered
    ─────────────
    Admission IC
      +0.20  SIRS+ (sepsis admission required)
      +0.15  organ dysfunction (ICU appropriate for organ failure)
      +0.10  high severity (CRP>100 or LacticAcid>2)
      -0.10  no clinical urgency (over-treatment)

    Admission NC
      +0.20  SIRS+ without organ dysfunction (NC adequate)
      +0.05  SIRS+ with organ dysfunction (IC would be better, NC still ok)
      +0.05  high severity
      -0.10  no clinical urgency (over-treatment)

    IV Antibiotics
      +0.15  SIRS+ (sepsis treatment protocol)
      +0.05  high severity modifier
      +0.05  organ dysfunction not yet admitted (pre-escalation treatment)

    IV Liquid
      +0.15  hypotension present (fluid resuscitation standard)
      +0.05  SIRS+ supportive care
      +0.05  organ dysfunction

    CRP (lab test)
      +0.05  first measurement ever (appropriate baseline)
       0.00  second measurement
      -0.10  3rd+ measurement AND SIRS+ without admission (delaying intervention)
      -0.20  3rd+ measurement AND organ dysfunction without admission (very urgent)

    Leucocytes (lab test) — same logic as CRP
      +0.05 / 0.00 / -0.10 / -0.20

    LacticAcid (lab test) — clinically important to repeat, lighter penalty
      +0.05  first measurement
       0.00  second measurement
      -0.08  3rd+ AND SIRS+ without admission
      -0.15  3rd+ AND organ dysfunction without admission

    Release A–E
      +0.15  non-SIRS, no high severity (appropriate discharge)
      +0.05  SIRS+ already admitted (proper treatment completed)
      -0.25  SIRS+ not yet admitted (dangerous early discharge)
      -0.35  organ dysfunction not yet admitted (extremely dangerous)

    Return ER
      -0.25  always (patient deteriorated / readmission)

    ER Sepsis Triage
      -0.15  already appeared in prefix (exactly_one violation)
       0.00  otherwise (driven by chainresponse from ER Triage, expected)

    ER Triage
      -0.15  already appeared in prefix (exactly_one violation)

    ER Registration
       0.00  no penalty (always the very first step, no clinical decision)
    """
    has_sirs = bool(clinical.get('SIRSCriteria2OrMore', False))
    has_organ_dysfunction = any(
        bool(clinical.get(f, False))
        for f in ('Hypotensie', 'Hypoxie', 'Oligurie', 'DisfuncOrg')
    )
    has_hypotension = bool(clinical.get('Hypotensie', False))
    crp_val    = clinical.get('CRP', None)
    lactate_val = clinical.get('LacticAcid', None)
    high_severity = (
        (crp_val    is not None and crp_val    > CRP_HIGH_THRESHOLD) or
        (lactate_val is not None and lactate_val > LACTICACID_HIGH_THRESHOLD)
    )

    already_admitted = (
        ADMISSION_IC_URI in prefix_activity_uris or
        ADMISSION_NC_URI in prefix_activity_uris
    )

    # Per-test repetition counts in prefix
    crp_count      = sum(1 for a in prefix_activity_uris if a == str(EX['Activity_CRP']))
    leucocytes_count = sum(1 for a in prefix_activity_uris if a == str(EX['Activity_Leucocytes']))
    lactate_count  = sum(1 for a in prefix_activity_uris if a == str(EX['Activity_LacticAcid']))

    adj = 0.0

    # ── Admission IC ──────────────────────────────────────────────────────────
    if chosen_uri == ADMISSION_IC_URI:
        if has_sirs:              adj += 0.20
        if has_organ_dysfunction: adj += 0.15
        if high_severity:         adj += 0.10
        if not has_sirs and not high_severity and not has_organ_dysfunction:
            adj -= 0.10  # over-treatment of mild case

    # ── Admission NC ──────────────────────────────────────────────────────────
    elif chosen_uri == ADMISSION_NC_URI:
        if has_sirs and not has_organ_dysfunction: adj += 0.20
        if has_sirs and has_organ_dysfunction:     adj += 0.05  # IC would be better
        if high_severity:                          adj += 0.05
        if not has_sirs and not high_severity:     adj -= 0.10  # over-treatment

    # ── IV Antibiotics ────────────────────────────────────────────────────────
    elif chosen_uri == IV_ANTIBIOTICS_URI:
        if has_sirs:              adj += 0.15
        if high_severity:         adj += 0.05
        if has_organ_dysfunction and not already_admitted:
            adj += 0.05  # pre-escalation treatment while awaiting admission

    # ── IV Liquid ─────────────────────────────────────────────────────────────
    elif chosen_uri == IV_LIQUID_URI:
        if has_hypotension: adj += 0.15  # fluid resuscitation
        if has_sirs:        adj += 0.05
        if has_organ_dysfunction: adj += 0.05

    # ── CRP ───────────────────────────────────────────────────────────────────
    elif chosen_uri == str(EX['Activity_CRP']):
        if crp_count == 0:
            adj += 0.05  # first measurement — appropriate baseline
        elif crp_count == 1:
            adj += 0.00  # second — neutral (monitoring is ok)
        elif crp_count >= 2 and has_organ_dysfunction and not already_admitted:
            adj -= 0.20  # very urgent, should be in ICU not doing another CRP
        elif crp_count >= 2 and has_sirs and not already_admitted:
            adj -= 0.10  # delaying intervention with repeated tests

    # ── Leucocytes ────────────────────────────────────────────────────────────
    elif chosen_uri == str(EX['Activity_Leucocytes']):
        if leucocytes_count == 0:
            adj += 0.05
        elif leucocytes_count == 1:
            adj += 0.00
        elif leucocytes_count >= 2 and has_organ_dysfunction and not already_admitted:
            adj -= 0.20
        elif leucocytes_count >= 2 and has_sirs and not already_admitted:
            adj -= 0.10

    # ── LacticAcid ────────────────────────────────────────────────────────────
    elif chosen_uri == str(EX['Activity_LacticAcid']):
        if lactate_count == 0:
            adj += 0.05  # first lactate measurement is clinically important
        elif lactate_count == 1:
            adj += 0.00
        elif lactate_count >= 2 and has_organ_dysfunction and not already_admitted:
            adj -= 0.15  # lighter penalty — lactate monitoring is more justified
        elif lactate_count >= 2 and has_sirs and not already_admitted:
            adj -= 0.08

    # ── Release (A–E) ─────────────────────────────────────────────────────────
    elif chosen_uri in RELEASE_URIS:
        if not has_sirs and not high_severity:
            adj += 0.15   # appropriate discharge of mild case
        elif has_sirs and already_admitted:
            adj += 0.05   # proper treatment completed before discharge
        elif has_sirs and not already_admitted:
            adj -= 0.25   # dangerous: SIRS+ patient released without admission
        if has_organ_dysfunction and not already_admitted:
            adj -= 0.10   # extra penalty (stacks with above)

    # ── Return ER ─────────────────────────────────────────────────────────────
    elif chosen_uri == RETURN_ER_URI:
        adj -= 0.25  # always bad — patient deteriorated

    # ── AtMostOnce violations ─────────────────────────────────────────────────
    if chosen_uri in EXACTLY_ONE_URIS and chosen_uri in prefix_activity_uris:
        adj -= 0.15  # exactly_one constraint violated

    return adj


# ── Extract all Case URIs ─────────────────────────────────────────────────────
all_case_uris = sorted(
    str(s) for s, p, o in pkg.triples((None, RDF.type, BASE.Case))
)
print(f"  ✓ {len(all_case_uris)} cases")

# ── Main extraction loop ──────────────────────────────────────────────────────
print("\n[4/5] Building subgraphs and features per decision point...")

labeled_data = []
skipped_no_seq = 0
skipped_no_label = 0

for case_idx, case_uri in enumerate(all_case_uris):
    if case_idx % 100 == 0:
        print(f"  Progress: {case_idx}/{len(all_case_uris)} cases "
              f"({case_idx/len(all_case_uris)*100:.1f}%)")

    task_sequence = get_task_sequence(pkg, case_uri)
    if len(task_sequence) < 2:
        skipped_no_seq += 1
        continue

    # Case-level age
    age_val = case_age.get(case_uri, None)
    age_lo, age_hi = NUMERIC_RANGES['Age']
    age_norm = float(np.clip((age_val - age_lo) / (age_hi - age_lo), 0.0, 1.0)) \
               if age_val is not None else 0.0

    # Case-level diagnose (ordinal encoded)
    diagnose_norm = case_diagnose.get(case_uri, 0.0)

    # Full activity URI sequence for suffix quality
    activity_uri_sequence = [get_task_activity(pkg, t) for t in task_sequence]

    # Pre-load clinical values for all tasks in this case
    task_clinical = {t: get_task_clinical_values(pkg, t) for t in task_sequence}

    total_tasks = len(task_sequence)

    # One decision point per event (except last — no "next" to predict)
    for i in range(total_tasks - 1):
        chosen_activity_uri = activity_uri_sequence[i + 1]
        if chosen_activity_uri is None:
            skipped_no_label += 1
            continue

        # ── Nodes ────────────────────────────────────────────────────────
        nodes = []
        node_types = {}
        node_features = {}

        # Task nodes: history up to and including event i
        prefix_tasks = task_sequence[:i + 1]
        for pos, task_uri in enumerate(prefix_tasks):
            nodes.append(task_uri)
            node_types[task_uri] = 'Task'
            node_features[task_uri] = make_task_feature(
                task_clinical[task_uri], pos, total_tasks, age_norm, diagnose_norm
            )

        # Activity execution counts within prefix
        exec_counts = defaultdict(int)
        for t in prefix_tasks:
            a = get_task_activity(pkg, t)
            if a:
                exec_counts[a] += 1
        max_exec = max(exec_counts.values(), default=1)

        # All Activity nodes (both executed and candidates)
        for act_uri in all_activity_uris:
            nodes.append(act_uri)
            node_types[act_uri] = 'Activity'
            node_features[act_uri] = make_activity_feature(
                act_uri,
                declare_involvement.get(act_uri, 0),
                max_declare,
                exec_counts.get(act_uri, 0),
                max_exec,
                act_uri in exec_counts,
            )

        # ── Edges ────────────────────────────────────────────────────────
        node_set = set(nodes)
        edges = []

        # Task → directlyFollowedBy → Task (within prefix)
        for j in range(len(prefix_tasks) - 1):
            edges.append((prefix_tasks[j], 'directlyFollowedBy', prefix_tasks[j + 1]))

        # Task → instanceOf → Activity
        for task_uri in prefix_tasks:
            a = get_task_activity(pkg, task_uri)
            if a and a in node_set:
                edges.append((task_uri, 'instanceOf', a))

        # Activity DECLARE constraint edges
        for src, etype, dst in declare_edges:
            if src in node_set and dst in node_set:
                edges.append((src, etype, dst))

        # ── Label ────────────────────────────────────────────────────────
        # Chosen decision is the Activity of the next Task
        # Label: index of chosen_activity_uri within all_activity_uris
        if chosen_activity_uri not in all_activity_uris:
            skipped_no_label += 1
            continue

        chosen_label = get_task_activity(pkg, task_sequence[i + 1])
        # Extract readable activity name from URI
        chosen_name = chosen_activity_uri.split('Activity_')[-1].replace('%20', ' ')

        # Clinical state at this decision point (most recent task = task_sequence[i])
        current_clinical = task_clinical[task_sequence[i]]
        quality = suffix_quality(activity_uri_sequence, i, current_clinical)

        # Action-specific adjustment: reward clinically appropriate choices,
        # penalise repeated lab tests when intervention is overdue, etc.
        prefix_activity_uris = [a for a in activity_uri_sequence[:i + 1] if a is not None]
        action_adj = action_quality_adjustment(
            chosen_activity_uri, current_clinical, prefix_activity_uris, i
        )
        quality = float(np.clip(quality + action_adj, 0.0, 1.0))

        labeled_data.append({
            'decision_point': {
                'case_id':        case_uri,
                'decision_index': i,
                'chosen_decision': chosen_name,
            },
            'subgraph': {
                'nodes':               nodes,
                'edges':               edges,
                'node_types':          node_types,
                'candidate_activities': all_activity_uris,
            },
            'node_features':    node_features,
            'node_feature_dim': NODE_DIM,
            'edge_feature_dim': EDGE_DIM,
            'outcome_quality':  quality,
            'suffix_length':    total_tasks - i - 1,
            'has_release':      any(u in RELEASE_URIS for u in activity_uri_sequence),
            'has_return_er':    RETURN_ER_URI in activity_uri_sequence,
            'release_in_suffix': any(u in RELEASE_URIS for u in activity_uri_sequence[i+1:]),
            'return_in_suffix':  RETURN_ER_URI in activity_uri_sequence[i+1:],
        })

print(f"\n  ✓ {len(labeled_data)} decision points")
print(f"  Skipped (no task sequence): {skipped_no_seq}")
print(f"  Skipped (no label):         {skipped_no_label}")

# ── Vocabulary ────────────────────────────────────────────────────────────────
vocab = {
    'edge_types':       EDGE_TYPES,
    'edge_type_to_idx': EDGE_TYPE_TO_IDX,
    'activities':       all_activity_uris,
    'node_dim':         NODE_DIM,
    'edge_dim':         EDGE_DIM,
    'bool_pvs':         BOOL_PVS,
}

# ── Save ──────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving...")

with open(script_dir / 'labeled_training_data_karibdis.pkl', 'wb') as f:
    pickle.dump(labeled_data, f)

with open(script_dir / 'vocabulary_karibdis.pkl', 'wb') as f:
    pickle.dump(vocab, f)

print(f"  ✓ labeled_training_data_karibdis.pkl")
print(f"  ✓ vocabulary_karibdis.pkl")

# ── Statistics ────────────────────────────────────────────────────────────────
qualities = [ex['outcome_quality'] for ex in labeled_data]
print(f"\nOutcome quality distribution:")
print(f"  Mean: {np.mean(qualities):.3f}  Std: {np.std(qualities):.3f}")
print(f"  Min:  {np.min(qualities):.3f}  Max: {np.max(qualities):.3f}")

from collections import Counter
cands_per_dp = [len(ex['subgraph']['candidate_activities']) for ex in labeled_data]
print(f"\nCandidates per decision point: always {cands_per_dp[0]} (all {len(all_activity_uris)} activities)")

print("\n" + "=" * 70)
print("DONE — run train_gat.py next (update node_dim=32, load labeled_training_data_pkg.pkl)")
print("=" * 70)
