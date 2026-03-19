import requests
import time
import csv
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

MED = Namespace("http://example.org/medical/")
PROP = Namespace("http://example.org/medical/prop/")
WD = Namespace("http://www.wikidata.org/entity/")

g = Graph()
g.parse("private_kb.ttl", format="turtle")

alignment_graph = Graph()
alignment_graph.bind("med", MED)
alignment_graph.bind("owl", OWL)
alignment_graph.bind("wd", WD)
alignment_graph.bind("rdfs", RDFS)

mapping_rows = []
HEADERS = {"User-Agent": "KBLabProject/1.0"}

TYPE_KEYWORDS = {
    "Drug": ["pharmaceutical", "drug", "medication", "medicine", "chemical compound"],
    "Manufacturer": ["company", "enterprise", "corporation", "manufacturer", "pharmaceutical company"],
    "ActiveIngredient": ["chemical", "compound", "drug", "substance", "medication"],
    "Route": ["route of administration", "method"],
    "DosageForm": ["dosage form", "pharmaceutical", "form"],
}


def search_wikidata(label, entity_type=None):
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json",
        "limit": 5,
    }
    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        results = resp.json().get("search", [])
        if not results:
            return None, 0.0

        for result in results:
            qid = result["id"]
            desc = result.get("description", "").lower()
            result_label = result.get("label", "").lower()
            match_type = result.get("match", {}).get("type", "")

            confidence = 0.5

            if match_type == "label" and result_label == label.lower():
                confidence = 0.85
            elif match_type == "alias":
                confidence = 0.80

            if entity_type and entity_type in TYPE_KEYWORDS:
                keywords = TYPE_KEYWORDS[entity_type]
                if any(kw in desc for kw in keywords):
                    confidence = max(confidence, 0.90)

            if confidence >= 0.70:
                return qid, round(confidence, 2)

        best = results[0]
        return best["id"], 0.50

    except Exception as e:
        print(f"    ERROR: {e}")
        return None, 0.0


entity_types = {}
for s, p, o in g.triples((None, RDF.type, None)):
    if str(o).startswith(str(MED)):
        etype = str(o).replace(str(MED), "")
        entity_types[str(s)] = etype

entities = set()
for s, p, o in g:
    if isinstance(s, URIRef) and str(s).startswith(str(MED)):
        entities.add(s)
    if isinstance(o, URIRef) and str(o).startswith(str(MED)):
        entities.add(o)

classes = {MED.Drug, MED.Manufacturer, MED.ActiveIngredient, MED.Route, MED.DosageForm}
properties = {URIRef(str(PROP) + p) for p in ["brandName", "genericName", "hasActiveIngredient", "hasDosageForm", "hasManufacturer", "hasRoute"]}
entities = entities - classes - properties

print(f"Linking {len(entities)} entities to Wikidata...\n")

linked = 0
not_found = 0

for entity in sorted(entities, key=str):
    label = str(entity).replace(str(MED), "").replace("_", " ")
    etype = entity_types.get(str(entity), "")

    qid, confidence = search_wikidata(label, etype)
    time.sleep(0.3)

    if qid and confidence >= 0.50:
        wd_uri = WD[qid]
        alignment_graph.add((entity, OWL.sameAs, wd_uri))
        mapping_rows.append([str(entity), f"wd:{qid}", str(confidence), etype])
        linked += 1
        status = "LINKED" if confidence >= 0.70 else "WEAK  "
        print(f"  {status}: {label} -> wd:{qid} ({confidence})")
    else:
        if etype:
            alignment_graph.add((entity, RDF.type, MED[etype]))
            alignment_graph.add((entity, RDFS.label, Literal(label)))
        mapping_rows.append([str(entity), "NOT_FOUND", "0.0", etype])
        not_found += 1
        print(f"  NEW:    {label}")

print(f"\nResults: {linked} linked, {not_found} not found")

alignment_graph.serialize("alignment.ttl", format="turtle")
print("Saved: alignment.ttl")

with open("mapping_table.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Private Entity", "External URI", "Confidence", "Type"])
    writer.writerows(mapping_rows)

print("Saved: mapping_table.csv")
