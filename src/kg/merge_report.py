from rdflib import Graph, Namespace, URIRef, Literal, RDF
from collections import Counter

MED = Namespace("http://example.org/medical/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")

final = Graph()
final.bind("med", MED)
final.bind("wd", WD)
final.bind("wdt", WDT)

print("Loading private KB...")
private = Graph()
private.parse("private_kb.ttl", format="turtle")
for t in private:
    final.add(t)
print(f"  {len(private)} triples")

print("Loading alignment...")
alignment = Graph()
alignment.parse("alignment.ttl", format="turtle")
for t in alignment:
    final.add(t)
print(f"  {len(alignment)} triples")

print("Loading predicate alignment...")
pred_align = Graph()
pred_align.parse("predicate_alignment.ttl", format="turtle")
for t in pred_align:
    final.add(t)
print(f"  {len(pred_align)} triples")

print("Loading expanded KB...")
expanded = Graph()
expanded.parse("expanded_kb.nt", format="nt")
for t in expanded:
    final.add(t)
print(f"  {len(expanded)} triples")

entities = set()
predicates = set()
for s, p, o in final:
    if isinstance(s, URIRef):
        entities.add(s)
    if isinstance(o, URIRef):
        entities.add(o)
    predicates.add(p)

top_preds = Counter()
for s, p, o in final:
    top_preds[str(p)] += 1

report = []
report.append("=" * 60)
report.append("KNOWLEDGE BASE - STATISTICS REPORT")
report.append("=" * 60)
report.append("")
report.append(f"Total triples:    {len(final)}")
report.append(f"Total entities:   {len(entities)}")
report.append(f"Total relations:  {len(predicates)}")
report.append("")
report.append("--- Sources ---")
report.append(f"Private KB:            {len(private)} triples")
report.append(f"Entity alignment:      {len(alignment)} triples")
report.append(f"Predicate alignment:   {len(pred_align)} triples")
report.append(f"Expanded (Wikidata):   {len(expanded)} triples")
report.append("")
report.append("--- Top 20 Predicates ---")
for pred, count in top_preds.most_common(20):
    short = pred.split("/")[-1] if "/" in pred else pred
    report.append(f"  {short:50s} {count}")
report.append("")
report.append("=" * 60)

for line in report:
    print(line)

with open("statistics_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print("\nSaved: statistics_report.txt")

final.serialize("final_kb.nt", format="nt")
final.serialize("final_kb.ttl", format="turtle")
print("Saved: final_kb.nt, final_kb.ttl")
