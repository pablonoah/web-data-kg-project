import requests
import json
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

MED = Namespace("http://example.org/medical/")
PROP = Namespace("http://example.org/medical/prop/")

g = Graph()
g.bind("med", MED)
g.bind("prop", PROP)
g.bind("owl", OWL)
g.bind("rdfs", RDFS)

g.add((MED.Drug, RDF.type, OWL.Class))
g.add((MED.Manufacturer, RDF.type, OWL.Class))
g.add((MED.ActiveIngredient, RDF.type, OWL.Class))
g.add((MED.Route, RDF.type, OWL.Class))
g.add((MED.DosageForm, RDF.type, OWL.Class))

g.add((PROP.hasManufacturer, RDF.type, OWL.ObjectProperty))
g.add((PROP.hasManufacturer, RDFS.domain, MED.Drug))
g.add((PROP.hasManufacturer, RDFS.range, MED.Manufacturer))

g.add((PROP.hasActiveIngredient, RDF.type, OWL.ObjectProperty))
g.add((PROP.hasActiveIngredient, RDFS.domain, MED.Drug))
g.add((PROP.hasActiveIngredient, RDFS.range, MED.ActiveIngredient))

g.add((PROP.hasRoute, RDF.type, OWL.ObjectProperty))
g.add((PROP.hasRoute, RDFS.domain, MED.Drug))
g.add((PROP.hasRoute, RDFS.range, MED.Route))

g.add((PROP.hasDosageForm, RDF.type, OWL.ObjectProperty))
g.add((PROP.hasDosageForm, RDFS.domain, MED.Drug))
g.add((PROP.hasDosageForm, RDFS.range, MED.DosageForm))

g.add((PROP.brandName, RDF.type, OWL.DatatypeProperty))
g.add((PROP.genericName, RDF.type, OWL.DatatypeProperty))


def clean_uri(text):
    return text.strip().replace(" ", "_").replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace("/", "_")


url = "https://api.fda.gov/drug/label.json?limit=100"
response = requests.get(url)
data = response.json()

for result in data.get("results", []):
    openfda = result.get("openfda", {})

    brand_names = openfda.get("brand_name", [])
    generic_names = openfda.get("generic_name", [])
    manufacturers = openfda.get("manufacturer_name", [])
    routes = openfda.get("route", [])
    dosage_forms = openfda.get("dosage_form", [])
    substances = openfda.get("substance_name", [])

    if not brand_names:
        continue

    drug_uri = MED[clean_uri(brand_names[0])]
    g.add((drug_uri, RDF.type, MED.Drug))

    for bn in brand_names:
        g.add((drug_uri, PROP.brandName, Literal(bn, datatype=XSD.string)))

    for gn in generic_names:
        g.add((drug_uri, PROP.genericName, Literal(gn, datatype=XSD.string)))

    for m in manufacturers:
        m_uri = MED[clean_uri(m)]
        g.add((m_uri, RDF.type, MED.Manufacturer))
        g.add((m_uri, RDFS.label, Literal(m)))
        g.add((drug_uri, PROP.hasManufacturer, m_uri))

    for r in routes:
        r_uri = MED[clean_uri(r)]
        g.add((r_uri, RDF.type, MED.Route))
        g.add((r_uri, RDFS.label, Literal(r)))
        g.add((drug_uri, PROP.hasRoute, r_uri))

    for d in dosage_forms:
        d_uri = MED[clean_uri(d)]
        g.add((d_uri, RDF.type, MED.DosageForm))
        g.add((d_uri, RDFS.label, Literal(d)))
        g.add((drug_uri, PROP.hasDosageForm, d_uri))

    for s in substances:
        s_uri = MED[clean_uri(s)]
        g.add((s_uri, RDF.type, MED.ActiveIngredient))
        g.add((s_uri, RDFS.label, Literal(s)))
        g.add((drug_uri, PROP.hasActiveIngredient, s_uri))

entities = set()
for s, p, o in g:
    if isinstance(s, URIRef) and str(s).startswith(str(MED)):
        entities.add(s)
    if isinstance(o, URIRef) and str(o).startswith(str(MED)):
        entities.add(o)

print(f"Triples: {len(g)}")
print(f"Entities: {len(entities)}")

g.serialize("private_kb.ttl", format="turtle")
g.serialize("private_kb.nt", format="nt")
print("Saved: private_kb.ttl, private_kb.nt")
