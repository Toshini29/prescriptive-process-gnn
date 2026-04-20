"""
build_pkg.py - Build Complete PKG Using OnlineEventImporter
"""
import sys
from pathlib import Path

# Repo root is two levels up from src/pipeline/
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pm4py
from pm4py import discover_declare
from rdflib import Graph, URIRef, Literal, RDF, RDFS, OWL
from karibdis.ProcessKnowledgeGraph import ProcessKnowledgeGraph
from karibdis.KnowledgeGraphBPMS import KnowledgeGraphBPMS
from karibdis.KnowledgeImporter import SimpleEventLogImporter, OnlineEventImporter, ExistingOntologyImporter, TextualImporter
from karibdis.utils import BASE_PROCESS_ONTOLOGY as BPO

DATA_DIR      = REPO_ROOT / "data"
ONTOLOGY_DIR  = DATA_DIR / "ontologies"
LOG_PATH      = DATA_DIR / "logs" / "Sepsis Cases - Event Log.xes"
OUTPUT_DIR    = REPO_ROOT / "output"

from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage

class MockLLM(Runnable):
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def generate(self, prompt=None):
        if self.index < len(self.responses):
            response = self.responses[self.index]
            self.index += 1
            return response
        return ""

    def invoke(self, input, config, **kwargs):
        return BaseMessage(content=self.generate(), type='')


def build_complete_kg_with_cases():
    print("="*70)
    print("BUILDING COMPLETE SEPSIS KNOWLEDGE GRAPH WITH CASES")
    print("="*70)

    bpms = KnowledgeGraphBPMS(ProcessKnowledgeGraph())
    pkg = bpms.pkg
    engine = bpms.engine

    # Step 1: Load event log
    print("\n[1/8] Loading event log...")
    log = pm4py.read_xes(str(LOG_PATH))
    print(f"  ✓ Loaded {log['case:concept:name'].nunique()} traces")

    # Step 2: Import schema
    print("\n[2/8] Importing schema (activities, process values)...")
    schema_importer = SimpleEventLogImporter(
        pkg=pkg,
        ignore_columns=['Infusion'],
        attribute_aliases={'org:group': BPO.Resource}
    )
    schema_importer.import_event_log_entities(log=log)

    declare = discover_declare(
        log,
        allowed_templates=['init', 'chainresponse', 'exactly_one'],
        min_support_ratio=0.8,
        min_confidence_ratio=0.8
    )
    declare['exactly_one']['LacticAcid'] = False
    schema_importer.import_declare(declare)
    schema_importer.load()

    print(f"  ✓ Activities: {len(list(pkg.subjects(RDF.type, BPO.Activity)))}")

    # Step 3: Import case and task instances
    print("\n[3/8] Importing cases and tasks...")

    online_importer = OnlineEventImporter(
        pkg=pkg,
        ignore_columns=['Infusion'],
        attribute_aliases={'org:group': BPO.Resource},
        case_attributes={'Age', 'Diagnose'}
    )

    case_count = 0
    task_count = 0

    for case_name, trace_df in log.groupby('case:concept:name'):
        for _, event_row in trace_df.iterrows():
            event_dict = event_row.to_dict()
            online_importer.translate_event(event_dict)
            task_count += 1

        case_count += 1
        if case_count % 100 == 0:
            print(f"  ... processed {case_count} cases, {task_count} events")

    online_importer.load()

    print(f"  ✓ Cases: {len(list(pkg.subjects(RDF.type, BPO.Case)))}")
    print(f"  ✓ Tasks: {len(list(pkg.subjects(RDF.type, BPO.Task)))}")

    # Step 4: Import textual knowledge
    print("\n[4/8] Importing textual knowledge...")

    statements = [
        """```turtle
        log:ProcessValue_CRP a :ProcessValue ;
            rdfs:label "C-reactive protein" ;
            rdfs:comment "The mg of C-reactive protein per liter of blood." .
``````""",
        """```turtle
        log:ProcessValue_LacticAcid a :ProcessValue ;
            rdfs:label "Lactic Acid" ;
            rdfs:comment "Measures lactic acid in blood." .
`````""",
        """```turtle
        log:ProcessValue_Leucocytes a :ProcessValue ;
            rdfs:label "Leucocytes" ;
            rdfs:comment "White blood cell count." .
````""",
        """```turtle
        log:ProcessValue_Hypoxie a :ProcessValue ;
            rdfs:label "Hypoxie" ;
            rdfs:comment "Whether hypoxia has been detected." .
```"""
    ]

    mock_llm = MockLLM(statements)
    text_importer = TextualImporter(pkg, mock_llm)
    text = (DATA_DIR / 'text_input.txt').read_text()
    for line in text.splitlines():
        text_importer.import_content_from_statement(line)
    text_importer.load()

    print(f"  ✓ ProcessValues labeled")

    # Step 5: Import SEPON
    print("\n[5/8] Importing SEPON ontology...")

    sepon = Graph().parse(str(ONTOLOGY_DIR / 'SEPON.ttl'), format='turtle')
    filter_query = (ONTOLOGY_DIR / 'filter_sepon_ontology.sparql').read_text()
    sepon_filtered = sepon.query(filter_query)

    sepon_importer = ExistingOntologyImporter(pkg)
    sepon_importer.accept_filtered_result(sepon_filtered, sepon)

    alignment = [
        (URIRef('http://www.semanticweb.org/zchero/ontologies/2023/11/SepsisOntology#Leukocyte_Count'),
         OWL.sameAs, URIRef('http://example.org/ProcessValue_Leucocytes')),
        (URIRef('http://www.semanticweb.org/zchero/ontologies/2023/11/SepsisOntology#C-Reactive_Protein'),
         OWL.sameAs, URIRef('http://example.org/ProcessValue_CRP')),
        (URIRef('http://www.semanticweb.org/zchero/ontologies/2023/11/SepsisOntology#Lactate'),
         OWL.sameAs, URIRef('http://example.org/ProcessValue_LacticAcid')),
        (URIRef('http://www.semanticweb.org/zchero/ontologies/2023/11/SepsisOntology#Hypoxia'),
         OWL.sameAs, URIRef('http://example.org/ProcessValue_Hypoxie'))
    ]
    sepon_importer.apply_alignment(alignment)
    sepon_importer.load()

    print(f"  ✓ SEPON imported")

    # Step 6: Import MONDO
    print("\n[6/8] Importing MONDO ontology...")

    mondo = Graph()
    mondo.parse(str(ONTOLOGY_DIR / 'mondo-simple.owl'), format='xml')
    filter_query_mondo = (ONTOLOGY_DIR / 'filter_mondo_ontology.sparql').read_text()
    mondo_filtered = mondo.query(filter_query_mondo)

    mondo_importer = ExistingOntologyImporter(pkg)
    mondo_importer.accept_filtered_result(mondo_filtered, mondo)
    mondo_importer.load()

    print(f"  ✓ MONDO imported")

    # Step 7: Additional knowledge
    print("\n[7/8] Adding domain knowledge...")

    additional = Graph().parse(
        str(ONTOLOGY_DIR / 'additional_knowledge.ttl'),
        format='turtle'
    )
    pkg += additional

    print(f"  ✓ Additional knowledge added")

    # Step 8: Run deduction
    print("\n[8/8] Running SHACL deduction...")
    engine.deduce()
    print(f"  ✓ Deduction complete")

    # Summary
    print("\n" + "="*70)
    print("COMPLETE KNOWLEDGE GRAPH BUILT")
    print("="*70)
    print(f"Total triples: {len(pkg):,}")
    print(f"Activities: {len(list(pkg.subjects(RDF.type, BPO.Activity)))}")
    print(f"Cases: {len(list(pkg.subjects(RDF.type, BPO.Case)))}")
    print(f"Tasks: {len(list(pkg.subjects(RDF.type, BPO.Task)))}")
    print(f"ProcessValues: {len(list(pkg.subjects(RDF.type, BPO.ProcessValue)))}")
    print(f"Resources: {len(list(pkg.subjects(RDF.type, BPO.Resource)))}")
    print("="*70)

    return bpms




if __name__ == "__main__":
    bpms = build_complete_kg_with_cases()

    output_path = OUTPUT_DIR / "sepsis_complete_pkg.ttl"
    output_path.parent.mkdir(exist_ok=True)
    print(f"\nSaving to {output_path}...")
    bpms.pkg.serialize(destination=str(output_path), format='turtle')

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved: {output_path} ({size_mb:.1f} MB)")
