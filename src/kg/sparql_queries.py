"""
Part 4: SPARQL with Python
Loads the family ontology and executes all SPARQL queries from Part 3.
Requires: pip install rdflib
"""

from rdflib import Graph, Namespace

NS = Namespace("http://www.semanticweb.org/family#")
RDF = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")


def load_ontology(filepath, use_reasoning=True):
    graph = Graph()
    graph.parse(filepath, format="xml")
    graph.bind("ns", NS)
    print(f"Ontology loaded: {len(graph)} triples")

    if use_reasoning:
        try:
            import owlrl
            owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(graph)
            print(f"After OWL-RL reasoning: {len(graph)} triples")
        except ImportError:
            print("owlrl not installed, skipping reasoning (pip install owlrl)")

    print()
    return graph


def run_query(graph, title, sparql):
    print(f"{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    results = graph.query(sparql)
    if len(results) == 0:
        print("  (no results)")
    else:
        for row in results:
            values = [str(v).replace(str(NS), "") for v in row]
            print("  " + " | ".join(values))
    print()


def main():
    graph = load_ontology("family_lab_completed.owl")

    prefix = """
        PREFIX ns: <http://www.semanticweb.org/family#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX owl: <http://www.w3.org/2002/07/owl#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    """

    # Query 1: How old is Peter?
    run_query(graph, "Q1: How old is Peter?", prefix + """
        SELECT ?age WHERE {
            ns:Peter ns:age ?age .
        }
    """)

    # Query 2: Who are Sylvie's parents?
    run_query(graph, "Q2: Who are Sylvie's parents?", prefix + """
        SELECT ?parent WHERE {
            ns:Sylvie ns:isChildOf ?parent .
        }
    """)

    # Query 3: Women over 30 years
    run_query(graph, "Q3: Women over 30?", prefix + """
        SELECT ?person ?age WHERE {
            ?person rdf:type ns:Female .
            ?person ns:age ?age .
            FILTER (?age > 30)
        }
    """)

    # Query 4: Instances of Person
    run_query(graph, "Q4: Instances of Person", prefix + """
        SELECT ?person WHERE {
            ?person rdf:type ns:Person .
        }
    """)

    # Query 5: Instances of Son
    run_query(graph, "Q5: Instances of Son", prefix + """
        SELECT ?son WHERE {
            ?son rdf:type ns:Son .
        }
    """)

    # Query 6: Instances of Daughter
    run_query(graph, "Q6: Instances of Daughter", prefix + """
        SELECT ?daughter WHERE {
            ?daughter rdf:type ns:Daughter .
        }
    """)

    # Query 7: Instances of Parent
    run_query(graph, "Q7: Instances of Parent", prefix + """
        SELECT ?parent WHERE {
            ?parent rdf:type ns:Parent .
        }
    """)

    # Query 8: Instances of Father
    run_query(graph, "Q8: Instances of Father", prefix + """
        SELECT ?father WHERE {
            ?father rdf:type ns:Father .
        }
    """)

    # Query 9: Instances of Mother
    run_query(graph, "Q9: Instances of Mother", prefix + """
        SELECT ?mother WHERE {
            ?mother rdf:type ns:Mother .
        }
    """)

    # Query 10: Instances of Grandmother
    run_query(graph, "Q10: Instances of Grandmother", prefix + """
        SELECT ?grandmother WHERE {
            ?grandmother rdf:type ns:Grandmother .
        }
    """)

    # Query 11: Instances of Grandfather
    run_query(graph, "Q11: Instances of Grandfather", prefix + """
        SELECT ?grandfather WHERE {
            ?grandfather rdf:type ns:Grandfather .
        }
    """)

    # Query 12: Instances of Brother
    run_query(graph, "Q12: Instances of Brother (class-based)", prefix + """
        SELECT ?brother WHERE {
            ?brother rdf:type ns:Brother .
        }
    """)

    # Query 12-alt: Brothers via shared parent
    run_query(graph, "Q12-alt: Brothers (shared parent pattern)", prefix + """
        SELECT DISTINCT ?brother ?sibling WHERE {
            ?brother rdf:type ns:Male .
            ?brother ns:isChildOf ?parent .
            ?sibling ns:isChildOf ?parent .
            FILTER (?brother != ?sibling)
        }
    """)

    # Query 13: Instances of Sister
    run_query(graph, "Q13: Instances of Sister (class-based)", prefix + """
        SELECT ?sister WHERE {
            ?sister rdf:type ns:Sister .
        }
    """)

    # Query 13-alt: Sisters via shared parent
    run_query(graph, "Q13-alt: Sisters (shared parent pattern)", prefix + """
        SELECT DISTINCT ?sister ?sibling WHERE {
            ?sister rdf:type ns:Female .
            ?sister ns:isChildOf ?parent .
            ?sibling ns:isChildOf ?parent .
            FILTER (?sister != ?sibling)
        }
    """)

    # Query 14: Children of Peter (name, age)
    run_query(graph, "Q14: Children of Peter (name, age)", prefix + """
        SELECT ?childName ?childAge WHERE {
            ?child ns:isChildOf ns:Peter .
            ?child ns:name ?childName .
            ?child ns:age ?childAge .
        }
    """)

    # Query 15: Persons whose father is more than 40 years old
    run_query(graph, "Q15: Persons whose father is >40 years old", prefix + """
        SELECT ?personName ?fatherName ?fatherAge WHERE {
            ?father ns:isFatherOf ?person .
            ?father ns:age ?fatherAge .
            ?person ns:name ?personName .
            ?father ns:name ?fatherName .
            FILTER (?fatherAge > 40)
        }
    """)

    # Query 15-alt: Using isParentOf + Male type (works with OWL-RL)
    run_query(graph, "Q15-alt: Persons whose father is >40 (via isParentOf+Male)", prefix + """
        SELECT ?personName ?fatherName ?fatherAge WHERE {
            ?father rdf:type ns:Male .
            ?father ns:isParentOf ?person .
            ?father ns:age ?fatherAge .
            ?person ns:name ?personName .
            ?father ns:name ?fatherName .
            FILTER (?fatherAge > 40)
        }
    """)

    # Query 16: French citizens + spouse
    run_query(graph, "Q16: French citizens (name, age, spouse)", prefix + """
        SELECT ?personName ?personAge ?spouseName WHERE {
            ?person ns:nationality "French"^^xsd:string .
            ?person ns:name ?personName .
            ?person ns:age ?personAge .
            OPTIONAL {
                ?person ns:isMarriedWith ?spouse .
                ?spouse ns:name ?spouseName .
            }
        }
    """)

    # Query 17: Persons who are brother of someone
    run_query(graph, "Q17: Brothers (via isBrotherOf)", prefix + """
        SELECT DISTINCT ?brotherName WHERE {
            ?brother ns:isBrotherOf ?someone .
            ?brother ns:name ?brotherName .
        }
    """)

    # Query 17-alt: Brothers via shared parent
    run_query(graph, "Q17-alt: Brothers (shared parent)", prefix + """
        SELECT DISTINCT ?brotherName WHERE {
            ?brother rdf:type ns:Male .
            ?brother ns:isChildOf ?parent .
            ?sibling ns:isChildOf ?parent .
            ?brother ns:name ?brotherName .
            FILTER (?brother != ?sibling)
        }
    """)

    # Query 18: Persons who are daughter of someone
    run_query(graph, "Q18: Daughters", prefix + """
        SELECT DISTINCT ?daughterName WHERE {
            ?daughter ns:isDaughterOf ?someone .
            ?daughter ns:name ?daughterName .
        }
    """)

    # Query 19: Persons who are uncle of someone
    run_query(graph, "Q19: Uncles (class-based)", prefix + """
        SELECT DISTINCT ?uncleName WHERE {
            ?uncle rdf:type ns:Uncle .
            ?uncle ns:name ?uncleName .
        }
    """)

    # Query 19-alt: Uncles via property chain
    run_query(graph, "Q19-alt: Uncles (property chain)", prefix + """
        SELECT DISTINCT ?uncleName WHERE {
            ?uncle rdf:type ns:Male .
            ?uncle ns:isChildOf ?grandparent .
            ?parent ns:isChildOf ?grandparent .
            ?parent ns:isParentOf ?child .
            ?uncle ns:name ?uncleName .
            FILTER (?uncle != ?parent)
        }
    """)

    # Query 20: Persons who are married
    run_query(graph, "Q20: Married persons", prefix + """
        SELECT DISTINCT ?personName WHERE {
            ?person ns:isMarriedWith ?spouse .
            ?person ns:name ?personName .
        }
    """)


if __name__ == "__main__":
    main()
