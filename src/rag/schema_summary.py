"""
Schema Summary Generator
========================
Extracts schema (classes, properties, sample entities) from the KB
to provide context for the NL→SPARQL LLM prompt.
"""

from rdflib import Graph, RDF, RDFS, OWL, Namespace
from collections import Counter


def extract_schema_summary(ttl_path="kg_artifacts/final_kb.ttl", max_samples=5):
    """Extract a compact schema summary from the knowledge base."""
    g = Graph()
    g.parse(ttl_path, format="turtle")

    MED = Namespace("http://example.org/medical/")
    PROP = Namespace("http://example.org/medical/prop/")
    WDT = Namespace("http://www.wikidata.org/prop/direct/")

    summary = []
    summary.append("=== Knowledge Base Schema Summary ===\n")
    summary.append(f"Total triples: {len(g)}\n")

    # Classes
    classes = set()
    for s, p, o in g.triples((None, RDF.type, None)):
        if str(o).startswith(str(MED)):
            classes.add(str(o).replace(str(MED), "med:"))

    summary.append(f"\n## Classes ({len(classes)}):")
    for c in sorted(classes):
        # Count instances
        full_uri = c.replace("med:", str(MED))
        count = len(list(g.triples((None, RDF.type, full_uri))))
        summary.append(f"  - {c} ({count} instances)")

    # Properties (private KB)
    summary.append(f"\n## Object Properties (private KB):")
    for prop_name in ["hasManufacturer", "hasActiveIngredient", "hasRoute", "hasDosageForm"]:
        prop_uri = PROP[prop_name]
        count = len(list(g.triples((None, prop_uri, None))))
        summary.append(f"  - prop:{prop_name} ({count} triples)")

    summary.append(f"\n## Data Properties (private KB):")
    for prop_name in ["brandName", "genericName"]:
        prop_uri = PROP[prop_name]
        count = len(list(g.triples((None, prop_uri, None))))
        summary.append(f"  - prop:{prop_name} ({count} triples)")

    # Wikidata predicates
    summary.append(f"\n## Key Wikidata Predicates:")
    wdt_counts = Counter()
    for s, p, o in g:
        if str(p).startswith(str(WDT)):
            pid = str(p).replace(str(WDT), "wdt:")
            wdt_counts[pid] += 1
    for pid, count in wdt_counts.most_common(10):
        summary.append(f"  - {pid} ({count} triples)")

    # Sample entities
    summary.append(f"\n## Sample Entities:")
    for cls_name in ["Drug", "Manufacturer", "ActiveIngredient"]:
        cls_uri = MED[cls_name]
        instances = list(g.triples((None, RDF.type, cls_uri)))[:max_samples]
        summary.append(f"\n  {cls_name}:")
        for s, _, _ in instances:
            label_triples = list(g.triples((s, RDFS.label, None)))
            if label_triples:
                label = str(label_triples[0][2])
            else:
                label = str(s).split("/")[-1].replace("_", " ")
            uri_short = str(s).replace(str(MED), "med:")
            summary.append(f"    - {uri_short} (label: \"{label}\")")

    # Sample values for key properties (helps LLM use correct URIs)
    summary.append(f"\n## Possible Values for Key Properties:")
    for prop_name, label in [("hasRoute", "Routes"), ("hasDosageForm", "Dosage Forms")]:
        prop_uri = PROP[prop_name]
        values = sorted(set(
            str(o).replace(str(MED), "med:")
            for s, p, o in g.triples((None, prop_uri, None))
            if str(o).startswith(str(MED))
        ))
        if values:
            summary.append(f"\n  {label} (use with prop:{prop_name}):")
            for v in values:
                summary.append(f"    - {v}")

    # Sample active ingredients
    prop_uri = PROP["hasActiveIngredient"]
    ingredients = sorted(set(
        str(o).replace(str(MED), "med:")
        for s, p, o in g.triples((None, prop_uri, None))
        if str(o).startswith(str(MED))
    ))[:15]
    if ingredients:
        summary.append(f"\n  Active Ingredients (use with prop:hasActiveIngredient):")
        for v in ingredients:
            summary.append(f"    - {v}")
        if len(list(g.triples((None, prop_uri, None)))) > 15:
            summary.append(f"    ... and more")

    # Namespace prefixes
    summary.append(f"\n## Prefixes:")
    summary.append(f"  PREFIX med: <http://example.org/medical/>")
    summary.append(f"  PREFIX prop: <http://example.org/medical/prop/>")
    summary.append(f"  PREFIX wd: <http://www.wikidata.org/entity/>")
    summary.append(f"  PREFIX wdt: <http://www.wikidata.org/prop/direct/>")
    summary.append(f"  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
    summary.append(f"  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>")
    summary.append(f"  PREFIX owl: <http://www.w3.org/2002/07/owl#>")

    schema_text = "\n".join(summary)
    return schema_text, g


def get_sparql_prompt_template():
    """Return the prompt template for NL→SPARQL conversion."""
    return """You are a SPARQL query generator. You MUST use ONLY the private KB properties listed below.

RULES (follow strictly):
1. ALWAYS use prop: properties (prop:hasRoute, prop:hasManufacturer, prop:hasActiveIngredient, prop:brandName, prop:genericName)
2. NEVER use wdt: properties. They will NOT work for answering questions.
3. Route values are URIs: med:ORAL, med:TOPICAL, med:OPHTHALMIC
4. Ingredient/manufacturer values are URIs in med: namespace, UPPERCASE with underscores (e.g., med:ACETAMINOPHEN)
5. Return ONLY the SPARQL query in a ```sparql block, no explanation.

{schema_summary}

## Examples:

Q: What drugs contain acetaminophen?
```sparql
SELECT ?drug ?label WHERE {{
    ?drug prop:hasActiveIngredient med:ACETAMINOPHEN .
    ?drug rdfs:label ?label .
}}
```

Q: Who manufactures Entresto?
```sparql
SELECT ?manufacturer ?label WHERE {{
    med:ENTRESTO prop:hasManufacturer ?manufacturer .
    ?manufacturer rdfs:label ?label .
}}
```

Q: List all oral drugs.
```sparql
SELECT ?drug ?label WHERE {{
    ?drug prop:hasRoute med:ORAL .
    ?drug rdfs:label ?label .
}}
```

Q: List all ophthalmic drugs.
```sparql
SELECT ?drug ?label WHERE {{
    ?drug prop:hasRoute med:OPHTHALMIC .
    ?drug rdfs:label ?label .
}}
```

Q: List all topical drugs.
```sparql
SELECT ?drug ?label WHERE {{
    ?drug prop:hasRoute med:TOPICAL .
    ?drug rdfs:label ?label .
}}
```

Q: How many drugs are in the knowledge base?
```sparql
SELECT (COUNT(?drug) AS ?count) WHERE {{
    ?drug a med:Drug .
}}
```

Q: What are all the routes of administration?
```sparql
SELECT DISTINCT ?route WHERE {{
    ?drug prop:hasRoute ?route .
}}
```

## User Question: {question}
"""


if __name__ == "__main__":
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)
    schema, _ = extract_schema_summary()
    print(schema)
