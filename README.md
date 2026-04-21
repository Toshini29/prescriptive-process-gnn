# Prescriptive Process Monitoring with Graph Neural Networks

Bachelor thesis implementation — Information Systems, TUM.

A framework for semantically explainable prescriptive process decision support. Process cases are represented as semantically enriched Process Knowledge Graphs (PKGs) and a Graph Attention Network is trained on those graphs using outcome-quality-weighted supervision. Recommendations are accompanied by three-level explanations: gradient-based feature attribution (Integrated Gradients), subgraph-based structural attribution (GNNExplainer), and a human-readable clinical narrative.

Instantiated and evaluated on a real-world sepsis patient event log.

---

## Repository Structure

```
prescriptive-process-gnn/
├── src/
│   ├── pipeline/
│   │   ├── build_pkg.py              # PKG construction from event log using KARIBDIS
│   │   ├── build_training_data.py    # Graph construction and OQ labelling
│   │   └── train_gat.py              # PrescriptiveGAT training
│   ├── evaluation/
│   │   ├── evaluate_oq_lift.py       # Prescriptive quality evaluation
│   │   └── evaluate_karibdis.py      # Comparison against KARIBDIS baseline
│   └── explainability/
│       ├── explainer.py              # Triple-layer explainability pipeline
│       └── compare_ig_gradients.py   # IG vs plain gradient attribution comparison
├── karibdis/                         # KARIBDIS framework files (see attribution below)
│   ├── ProcessKnowledgeGraph.py
│   ├── KnowledgeGraphBPMS.py
│   ├── KGProcessEngine.py
│   ├── KnowledgeImporter.py
│   └── utils.py
├── data/
│   └── ontologies/                   # Domain ontologies and SHACL rules
│       ├── SEPON.ttl                 # Sepsis process ontology
│       ├── additional_knowledge.ttl  # Domain-specific clinical knowledge
│       ├── base_ontology.ttl         # KARIBDIS base ontology
│       ├── base_rules.ttl            # KARIBDIS SHACL rules
│       └── declare_ontology.ttl      # DECLARE constraint ontology
└── README.md
```

---

## Pipeline

Run the steps in order:

```bash
# 1. Build the Process Knowledge Graph
python src/pipeline/build_pkg.py

# 2. Build training data (graph construction + OQ labelling)
python src/pipeline/build_training_data.py

# 3. Train the GNN
python src/pipeline/train_gat.py

# 4. Evaluate
python src/evaluation/evaluate_oq_lift.py
python src/evaluation/evaluate_karibdis.py

# 5. Generate explanations
python src/explainability/explainer.py
```

---

## Dependencies

```
torch
torch-geometric
captum
rdflib
pm4py
numpy
scipy
```

---

## Data

The event log (`Sepsis Cases - Event Log.xes`) is a real-world sepsis patient dataset from a Dutch hospital. It is not included in this repository due to data use restrictions. The dataset is publicly available at the [4TU Research Data repository](https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639).

Place the file at `data/logs/Sepsis Cases - Event Log.xes` before running the pipeline.

---

## KARIBDIS Attribution

The files in `karibdis/` are adapted from the [KARIBDIS framework](https://github.com/INSM-TUM/karibdis) developed at the Information Systems group, Technical University of Munich. KARIBDIS constructs semantically enriched Process Knowledge Graphs from event logs using ontological alignment and SHACL-based deduction. It serves as both the PKG construction infrastructure and the rule-based evaluation baseline in this work. KARIBDIS is licensed under the MIT License — Copyright (c) 2025 Leon Bein, Information Systems @ Technical University of Munich.
