"""
KGE Data Preparation
====================
Converts RDF triples to train/valid/test splits for KGE training.
Produces entity2id.txt, relation2id.txt, train.txt, valid.txt, test.txt
"""

import os
import random
from collections import Counter
from rdflib import Graph, URIRef, Literal

random.seed(42)


def load_triples_from_nt(filepath):
    """Load triples from NT file, filtering to only URI-based triples."""
    g = Graph()
    print(f"Loading {filepath}...")
    g.parse(filepath, format="nt")
    print(f"  Loaded {len(g)} raw triples")

    triples = []
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            triples.append((str(s), str(p), str(o)))

    print(f"  Kept {len(triples)} URI-only triples")
    return triples


def shorten_uri(uri):
    """Shorten URI for readability."""
    prefixes = {
        "http://www.wikidata.org/entity/": "wd:",
        "http://www.wikidata.org/prop/direct/": "wdt:",
        "http://example.org/medical/": "med:",
        "http://example.org/medical/prop/": "prop:",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf:",
        "http://www.w3.org/2000/01/rdf-schema#": "rdfs:",
        "http://www.w3.org/2002/07/owl#": "owl:",
    }
    for full, short in prefixes.items():
        if uri.startswith(full):
            return short + uri[len(full):]
    return uri


def clean_triples(triples, min_entity_freq=2, min_relation_freq=5):
    """Remove rare entities/relations to improve KGE training quality."""
    entity_count = Counter()
    relation_count = Counter()

    for s, p, o in triples:
        entity_count[s] += 1
        entity_count[o] += 1
        relation_count[p] += 1

    # Filter OWL/RDF schema triples that are not useful for KGE
    schema_predicates = {
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://www.w3.org/2000/01/rdf-schema#label",
        "http://www.w3.org/2000/01/rdf-schema#subClassOf",
        "http://www.w3.org/2002/07/owl#sameAs",
        "http://www.w3.org/2002/07/owl#equivalentProperty",
        "http://www.w3.org/2002/07/owl#equivalentClass",
    }

    cleaned = []
    for s, p, o in triples:
        if p in schema_predicates:
            continue
        if entity_count[s] >= min_entity_freq and entity_count[o] >= min_entity_freq:
            if relation_count[p] >= min_relation_freq:
                cleaned.append((s, p, o))

    print(f"  After cleaning: {len(cleaned)} triples")
    return cleaned


def split_triples(triples, train_ratio=0.8, valid_ratio=0.1):
    """Split triples into train/valid/test sets."""
    random.shuffle(triples)
    n = len(triples)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train = triples[:train_end]
    valid = triples[train_end:valid_end]
    test = triples[valid_end:]

    return train, valid, test


def create_id_mappings(triples):
    """Create entity2id and relation2id mappings."""
    entities = set()
    relations = set()

    for s, p, o in triples:
        entities.add(s)
        entities.add(o)
        relations.add(p)

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    relation2id = {r: i for i, r in enumerate(sorted(relations))}

    return entity2id, relation2id


def save_triples(triples, filepath, use_short=True):
    """Save triples in tab-separated format (h\tr\tt)."""
    with open(filepath, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            if use_short:
                f.write(f"{shorten_uri(s)}\t{shorten_uri(p)}\t{shorten_uri(o)}\n")
            else:
                f.write(f"{s}\t{p}\t{o}\n")


def save_id_mapping(mapping, filepath):
    """Save id mapping to file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{len(mapping)}\n")
        for key, idx in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{shorten_uri(key)}\t{idx}\n")


def create_size_subsets(triples, sizes=[20000, 50000]):
    """Create subsets of different sizes for size-sensitivity analysis."""
    subsets = {}
    for size in sizes:
        if size < len(triples):
            subsets[size] = random.sample(triples, size)
        else:
            subsets[size] = triples[:]
    subsets[len(triples)] = triples[:]
    return subsets


def prepare_kge_data(input_nt="kg_artifacts/final_kb.nt", output_dir="data/kge"):
    """Full pipeline: load, clean, split, save."""
    os.makedirs(output_dir, exist_ok=True)

    # Load
    triples = load_triples_from_nt(input_nt)

    # Clean
    triples = clean_triples(triples)

    if len(triples) < 100:
        print("WARNING: Very few triples after cleaning. Relaxing filters...")
        triples = load_triples_from_nt(input_nt)
        triples = clean_triples(triples, min_entity_freq=1, min_relation_freq=1)

    # Create ID mappings
    entity2id, relation2id = create_id_mappings(triples)

    print(f"\n--- KGE Dataset Statistics ---")
    print(f"Entities:  {len(entity2id)}")
    print(f"Relations: {len(relation2id)}")
    print(f"Triples:   {len(triples)}")

    # Split
    train, valid, test = split_triples(triples)
    print(f"\nSplit: train={len(train)}, valid={len(valid)}, test={len(test)}")

    # Save full dataset
    save_triples(train, os.path.join(output_dir, "train.txt"))
    save_triples(valid, os.path.join(output_dir, "valid.txt"))
    save_triples(test, os.path.join(output_dir, "test.txt"))
    save_id_mapping(entity2id, os.path.join(output_dir, "entity2id.txt"))
    save_id_mapping(relation2id, os.path.join(output_dir, "relation2id.txt"))

    # Save size subsets
    subsets = create_size_subsets(triples, sizes=[20000, 50000])
    for size, subset in subsets.items():
        subset_dir = os.path.join(output_dir, f"subset_{size}")
        os.makedirs(subset_dir, exist_ok=True)
        tr, va, te = split_triples(subset)
        save_triples(tr, os.path.join(subset_dir, "train.txt"))
        save_triples(va, os.path.join(subset_dir, "valid.txt"))
        save_triples(te, os.path.join(subset_dir, "test.txt"))
        print(f"  Subset {size}: train={len(tr)}, valid={len(va)}, test={len(te)}")

    # Print top relations
    rel_count = Counter()
    for s, p, o in triples:
        rel_count[shorten_uri(p)] += 1
    print(f"\nTop 10 relations:")
    for rel, count in rel_count.most_common(10):
        print(f"  {rel}: {count}")

    return train, valid, test, entity2id, relation2id


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)
    prepare_kge_data()
