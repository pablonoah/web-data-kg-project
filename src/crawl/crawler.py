"""
=============================================================================
Lab Session 1: From Unstructured Web to Structured Entities
=============================================================================
Course: Web Mining & Semantics
Domain: Medical Research - Cancer Cure
Student: 4eme annee Ingenieur Data/IA

This script combines both phases:
- Phase 1: Web Crawling & Cleaning (trafilatura)
- Phase 2: Information Extraction (spaCy NER)
=============================================================================
"""

import json
import pandas as pd
from datetime import datetime
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

# ============================================================================
# PHASE 1: WEB CRAWLING
# ============================================================================

# Required imports for crawling
# pip install trafilatura httpx
import trafilatura
from trafilatura import fetch_url, extract

# Seed URLs for Cancer Research domain
SEED_URLS = [
    "https://www.cancer.gov/research/areas/treatment",
    "https://www.cancer.org/cancer/managing-cancer/treatment-types.html",
    "https://www.nature.com/subjects/cancer-therapy",
    "https://www.who.int/news-room/fact-sheets/detail/cancer",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9436517/",  # Cancer immunotherapy review
    "https://www.mayoclinic.org/diseases-conditions/cancer/diagnosis-treatment/drc-20370594",
]

MIN_WORDS = 500
USER_AGENT = "LabCrawler/1.0 (Educational Purpose)"


def can_fetch(url, user_agent=USER_AGENT):
    """
    Check if crawling is allowed by robots.txt.
    Respects web ethics as required by the lab.
    """
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:
        return True  # If no robots.txt or error, allow crawling


def is_useful_page(text, min_words=MIN_WORDS):
    """Check if page has enough content (>500 words)."""
    if not text:
        return False
    return len(text.split()) >= min_words


def extract_content(url):
    """
    Fetch and extract main content using trafilatura.
    Trafilatura automatically removes boilerplate:
    - Navigation menus
    - Ads and sidebars
    - Footer content
    """
    downloaded = fetch_url(url)
    if not downloaded:
        return None
    
    text = extract(
        downloaded,
        include_comments=False,
        include_tables=False,
    )
    return text


def crawl_urls(urls, output_file="crawler_output.jsonl"):
    """Crawl URLs and save to JSONL format."""
    results = []

    for url in urls:
        print(f"[CRAWL] {url}")

        # Check robots.txt before crawling (web ethics)
        if not can_fetch(url):
            print(f"  -> Skipped (blocked by robots.txt)")
            continue

        text = extract_content(url)

        if text and is_useful_page(text):
            record = {
                "url": url,
                "text": text,
                "word_count": len(text.split()),
                "crawled_at": datetime.now().isoformat()
            }
            results.append(record)
            print(f"  -> {record['word_count']} words extracted")
        else:
            print(f"  -> Skipped (insufficient content)")
    
    # Save JSONL: one JSON object per line
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    return results


# ============================================================================
# PHASE 2: INFORMATION EXTRACTION
# ============================================================================

# Required: python -m spacy download en_core_web_trf
import spacy

# Target entity types for Knowledge Graph
# Standard spaCy labels: PERSON, ORG, GPE, DATE, PRODUCT, EVENT, WORK_OF_ART
TARGET_LABELS = {"PERSON", "ORG", "GPE", "DATE", "PRODUCT", "EVENT", "WORK_OF_ART"}


def load_nlp_model():
    """Load spaCy model with fallback."""
    try:
        return spacy.load("en_core_web_trf")  # Transformer model
    except OSError:
        return spacy.load("en_core_web_sm")   # Fallback


def extract_entities(doc, source_url):
    """
    Named Entity Recognition (NER) using standard spaCy labels:
    - PERSON: Researchers, doctors (Dr. James Allison, Carl June)
    - ORG: Institutions (NIH, WHO, MD Anderson, Pfizer)
    - GPE: Locations (Houston, Boston, Switzerland)
    - DATE: Dates (2018, Phase III trials)
    - PRODUCT: Drugs/Treatments (Keytruda, CAR-T)
    - EVENT: Events (conferences, trials)
    - WORK_OF_ART: Publications, studies
    """
    entities = []

    for ent in doc.ents:
        if ent.label_ in TARGET_LABELS:
            entities.append({
                "entity": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source_url": source_url
            })

    return entities


def extract_relations(doc):
    """
    Relation Extraction using Dependency Parsing.

    Find Subject-Verb-Object patterns for medical research:
    "Pembrolizumab treats melanoma"
    -> (Pembrolizumab, treat, melanoma)

    "FDA approved Keytruda"
    -> (FDA, approve, Keytruda)

    "CAR-T therapy targets leukemia cells"
    -> (CAR-T therapy, target, leukemia cells)

    Dependency tags used:
    - nsubj: nominal subject
    - dobj: direct object
    - pobj: object of preposition
    """
    relations = []
    
    for sent in doc.sents:
        sent_ents = [e for e in sent.ents if e.label_ in TARGET_LABELS]
        
        if len(sent_ents) < 2:
            continue
        
        for token in sent:
            if token.pos_ == "VERB":
                subj, obj = None, None
                
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        for ent in sent_ents:
                            if child.i >= ent.start and child.i < ent.end:
                                subj = ent.text
                                break
                    
                    if child.dep_ in ("dobj", "pobj"):
                        for ent in sent_ents:
                            if child.i >= ent.start and child.i < ent.end:
                                obj = ent.text
                                break
                
                if subj and obj:
                    relations.append({
                        "subject": subj,
                        "predicate": token.lemma_,
                        "object": obj,
                        "sentence": sent.text.strip()
                    })
    
    return relations


def process_documents(input_file="crawler_output.jsonl"):
    """Process all crawled documents for NER and RE."""
    nlp = load_nlp_model()
    all_entities = []
    all_relations = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            url = record["url"]
            text = record["text"]
            
            print(f"[NER] Processing: {url}")
            doc = nlp(text)
            
            entities = extract_entities(doc, url)
            all_entities.extend(entities)
            
            relations = extract_relations(doc)
            all_relations.extend(relations)
    
    # Save results
    df_entities = pd.DataFrame(all_entities)
    df_entities.to_csv("extracted_knowledge.csv", index=False)
    
    df_relations = pd.DataFrame(all_relations)
    df_relations.to_csv("extracted_relations.csv", index=False)
    
    print(f"\n[DONE] {len(all_entities)} entities saved")
    print(f"[DONE] {len(all_relations)} relations saved")
    
    return df_entities, df_relations


# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LAB 1: Web Crawling & Information Extraction")
    print("Domain: Cancer Research & Treatment")
    print("=" * 60)
    
    # Phase 1
    print("\n[PHASE 1] Web Crawling...")
    crawl_urls(SEED_URLS)
    
    # Phase 2
    print("\n[PHASE 2] Information Extraction...")
    entities_df, relations_df = process_documents()
    
    # Preview results
    print("\n" + "=" * 60)
    print("SAMPLE ENTITIES:")
    print(entities_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("SAMPLE RELATIONS (Knowledge Graph Triples):")
    print(relations_df.head(5).to_string(index=False))
