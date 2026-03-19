from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph, Namespace, URIRef, Literal, RDF
import csv
import time

WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.addCustomHttpHeader("User-Agent", "KBLabProject/1.0")

expanded = Graph()
expanded.bind("wd", WD)
expanded.bind("wdt", WDT)


def run_query(query_str, description, limit=5000):
    print(f"\n[{description}]")
    sparql.setQuery(query_str)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        count = 0
        for b in bindings:
            s = b.get("s", b.get("entity", b.get("drug", b.get("item", {}))))
            p = b.get("p", b.get("prop", {}))
            o = b.get("o", b.get("value", {}))
            if not s or not p or not o:
                continue
            s_uri = URIRef(s["value"])
            p_uri = URIRef(p["value"])
            if o["type"] == "uri":
                o_node = URIRef(o["value"])
            else:
                o_node = Literal(o.get("value", ""))
            if "wikidata.org" in str(p_uri) or "www.w3.org" in str(p_uri):
                expanded.add((s_uri, p_uri, o_node))
                count += 1
        print(f"  Added {count} triples (total: {len(expanded)})")
    except Exception as e:
        print(f"  ERROR: {e}")


aligned_entities = []
with open("mapping_table.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["External URI"] != "NOT_FOUND" and float(row["Confidence"]) >= 0.70:
            qid = row["External URI"].replace("wd:", "")
            aligned_entities.append(qid)

print(f"Starting expansion from {len(aligned_entities)} confidently aligned entities\n")

# --- 1-HOP EXPANSION from aligned entities (batched) ---
batch_size = 10
for i in range(0, len(aligned_entities), batch_size):
    batch = aligned_entities[i:i+batch_size]
    values = " ".join(f"wd:{qid}" for qid in batch)
    query = """
    SELECT ?s ?p ?o WHERE {
        VALUES ?s { %s }
        ?s ?p ?o .
        FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
        FILTER(!isLiteral(?o) || LANG(?o) = "en" || LANG(?o) = "")
    }
    LIMIT 5000
    """ % values
    run_query(query, f"1-hop expansion batch {i//batch_size + 1}", 5000)
    time.sleep(2)

# --- PREDICATE-CONTROLLED EXPANSION ---
key_predicates = [
    ("P176", "manufacturer", 20000),
    ("P3781", "has active ingredient", 20000),
    ("P636", "route of administration", 15000),
    ("P2175", "medical condition treated", 20000),
    ("P279", "subclass of (drugs/chemicals)", 15000),
    ("P31", "instance of (for drug items)", 10000),
]

for pid, label, limit in key_predicates:
    query = """
    SELECT ?s ?p ?o WHERE {
        ?s wdt:%s ?o .
        BIND(wdt:%s AS ?p)
    }
    LIMIT %d
    """ % (pid, pid, limit)
    run_query(query, f"Predicate expansion: {label} (wdt:{pid})")
    time.sleep(3)

# --- 2-HOP: drugs -> active ingredients -> properties ---
query_2hop = """
SELECT ?s ?p ?o WHERE {
    ?drug wdt:P3781 ?ingredient .
    ?ingredient ?p ?o .
    FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
    FILTER(!isLiteral(?o) || LANG(?o) = "en" || LANG(?o) = "")
}
LIMIT 20000
"""
run_query(query_2hop, "2-hop: active ingredients -> their properties")
time.sleep(3)

# --- 2-HOP: drugs -> manufacturers -> properties ---
query_2hop_mfg = """
SELECT ?s ?p ?o WHERE {
    ?drug wdt:P176 ?mfg .
    ?mfg ?p ?o .
    FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/"))
    FILTER(!isLiteral(?o) || LANG(?o) = "en" || LANG(?o) = "")
}
LIMIT 20000
"""
run_query(query_2hop_mfg, "2-hop: manufacturers -> their properties")
time.sleep(3)

# --- CLEANING ---
print("\n\nCleaning...")

to_remove = []
for s, p, o in expanded:
    if isinstance(o, Literal) and len(str(o)) > 500:
        to_remove.append((s, p, o))
    if isinstance(o, URIRef) and "wikimedia.org" in str(o):
        to_remove.append((s, p, o))

for triple in to_remove:
    expanded.remove(triple)

print(f"Removed {len(to_remove)} noisy triples")

entities = set()
predicates = set()
for s, p, o in expanded:
    if isinstance(s, URIRef):
        entities.add(s)
    if isinstance(o, URIRef):
        entities.add(o)
    predicates.add(p)

print(f"\n{'='*50}")
print(f"FINAL EXPANDED KB STATISTICS")
print(f"{'='*50}")
print(f"Triples:   {len(expanded)}")
print(f"Entities:  {len(entities)}")
print(f"Relations: {len(predicates)}")
print(f"{'='*50}")

expanded.serialize("expanded_kb.nt", format="nt")
expanded.serialize("expanded_kb.ttl", format="turtle")
print("\nSaved: expanded_kb.nt, expanded_kb.ttl")
