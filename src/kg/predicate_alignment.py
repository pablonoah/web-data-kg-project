from SPARQLWrapper import SPARQLWrapper, JSON
import time

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader("User-Agent", "KBLabProject/1.0")

PRIVATE_PREDICATES = {
    "hasManufacturer": ["manufacturer", "producer"],
    "hasActiveIngredient": ["active ingredient", "ingredient", "substance"],
    "hasRoute": ["route of administration", "administration"],
    "hasDosageForm": ["dosage form", "pharmaceutical form"],
    "brandName": ["brand", "trade name"],
    "genericName": ["generic name", "international nonproprietary name"],
}


def search_predicate(keywords):
    results = []
    for kw in keywords:
        query = """
        SELECT ?property ?propertyLabel WHERE {
            ?property a wikibase:Property .
            ?property rdfs:label ?propertyLabel .
            FILTER(CONTAINS(LCASE(?propertyLabel), "%s"))
            FILTER(LANG(?propertyLabel) = "en")
        }
        LIMIT 20
        """ % kw.lower()

        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            resp = sparql.query().convert()
            for r in resp["results"]["bindings"]:
                pid = r["property"]["value"].split("/")[-1]
                label = r["propertyLabel"]["value"]
                results.append((pid, label))
        except Exception as e:
            print(f"    ERROR: {e}")
        time.sleep(1)
    return results


print("Predicate Alignment via SPARQL\n")
print("=" * 70)

alignments = {}

for pred, keywords in PRIVATE_PREDICATES.items():
    print(f"\nPrivate predicate: :{pred}")
    print(f"  Searching keywords: {keywords}")
    candidates = search_predicate(keywords)

    seen = set()
    unique = []
    for pid, label in candidates:
        if pid not in seen:
            seen.add(pid)
            unique.append((pid, label))

    for pid, label in unique[:8]:
        print(f"    Candidate: wdt:{pid} -> \"{label}\"")

    if unique:
        best_pid, best_label = unique[0]
        alignments[pred] = (best_pid, best_label)
        print(f"  >>> Best match: wdt:{best_pid} (\"{best_label}\")")

print("\n" + "=" * 70)
print("\nFinal Predicate Alignment Summary:")
print("-" * 70)

from rdflib import Graph, Namespace, URIRef, OWL

PROP = Namespace("http://example.org/medical/prop/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

pred_graph = Graph()
pred_graph.bind("prop", PROP)
pred_graph.bind("wdt", WDT)
pred_graph.bind("owl", OWL)

for pred, (pid, label) in alignments.items():
    print(f"  :{pred}  owl:equivalentProperty  wdt:{pid} (\"{label}\")")
    pred_graph.add((PROP[pred], OWL.equivalentProperty, WDT[pid]))

pred_graph.serialize("predicate_alignment.ttl", format="turtle")
print("\nSaved: predicate_alignment.ttl")
