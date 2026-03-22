# Knowledge Graph Construction, Alignment, Reasoning & RAG

**Course**: Web Mining & Semantics — ESILV A4
**Domain**: Medical / Pharmaceutical (FDA Drug Labels + Wikidata)

End-to-end pipeline: from web crawling to a RAG-powered question-answering system over a knowledge graph.

## Project Structure

```
project/
├── src/
│   ├── crawl/          # Web crawler (trafilatura)
│   ├── ie/             # NER & relation extraction (spaCy)
│   ├── kg/             # RDF KB construction, alignment, expansion
│   ├── reason/         # SWRL reasoning (OWLReady2)
│   ├── kge/            # Knowledge Graph Embeddings (TransE, ComplEx)
│   └── rag/            # RAG pipeline (NL→SPARQL + self-repair)
├── data/
│   ├── samples/        # Crawler output, extracted entities
│   ├── kge/            # Train/valid/test splits
│   └── sparql_queries.txt
├── kg_artifacts/
│   ├── family_lab_completed.owl
│   ├── private_kb.ttl
│   ├── alignment.ttl
│   ├── predicate_alignment.ttl
│   ├── expanded_kb.ttl
│   └── statistics_report.txt
├── reports/
│   └── final_report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/noahlips/web-data-kg-project.git
cd web-data-kg-project

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (for NER)
python -m spacy download en_core_web_sm

# Install Ollama (for RAG module)
# Download from https://ollama.ai then:
ollama pull llama3.2
```

## Hardware Requirements

- **CPU**: Any modern processor
- **RAM**: 8 GB minimum (16 GB recommended for KGE on full dataset)
- **Disk**: ~500 MB for KB artifacts + embeddings
- **GPU**: Not required (CPU training is supported)
- **Ollama**: Required only for the RAG module (~4 GB for llama3.2)

## How to Run Each Module

### 1. Web Crawling & NER (Lab 1)

```bash
python src/crawl/crawler.py
```
Crawls cancer research pages, extracts text with trafilatura, runs spaCy NER.
Output: `data/samples/crawler_output.jsonl`, `data/samples/extracted_knowledge.csv`

### 2. KB Construction & Alignment (Lab 4)

```bash
# Build private KB from FDA API
python src/kg/build_kb.py

# Entity linking to Wikidata
python src/kg/entity_linking.py

# Predicate alignment
python src/kg/predicate_alignment.py

# SPARQL expansion from Wikidata
python src/kg/expansion.py

# Merge all sources + statistics
python src/kg/merge_report.py
```

### 3. SPARQL Queries (Lab 2-3)

```bash
python src/kg/sparql_queries.py
```
Runs 20 SPARQL queries on the family ontology with OWL-RL reasoning.

### 4. SWRL Reasoning

```bash
python src/reason/swrl_reasoning.py
```
- Part 1: SWRL rules on `family.owl` (grandparent, uncle inference)
- Part 2: SWRL rule on medical KB (drugs sharing active ingredients)

### 5. Knowledge Graph Embeddings

```bash
# Prepare data (train/valid/test splits)
python src/kge/prepare_data.py

# Train and evaluate TransE + ComplEx
python src/kge/train_evaluate.py
```
Outputs: metrics (MRR, Hits@1/3/10), t-SNE plot, size sensitivity analysis.

### 6. RAG Demo (NL → SPARQL)

```bash
# Start Ollama first
ollama serve

# Interactive CLI
python src/rag/rag_pipeline.py

# Run evaluation (baseline vs RAG)
python src/rag/rag_pipeline.py --evaluate
```

**Example interaction:**
```
You: What drugs contain acetaminophen?
  Generating SPARQL query...
  Generated query: SELECT ?drug ?label WHERE { ?drug prop:hasActiveIngredient med:ACETAMINOPHEN ...
  Results (3 rows):
    basic_care_acetaminophen | basic care acetaminophen
    Pain_Reliever_Extra_Strength | Pain Reliever Extra Strength
    ...
```

![RAG Demo Screenshot](reports/rag_demo_screenshot.png)

## Screenshots

### KGE Model Comparison
![KGE Comparison](reports/kge_comparison.png)

### t-SNE Entity Embeddings
![t-SNE Embeddings](reports/tsne_embeddings.png)

### Size Sensitivity Analysis
![Size Sensitivity](reports/size_sensitivity.png)

## KB Statistics

| Metric | Value |
|--------|-------|
| Total triples | 62,178 |
| Entities | 53,920 |
| Relations | 365 |
| Private KB | 355 triples |
| Aligned entities | 138 |
| Wikidata expansion | 61,733 triples |

## Technologies

- **Python 3.10+**
- **rdflib** — RDF parsing and SPARQL
- **spaCy** — Named Entity Recognition
- **OWLReady2** — OWL reasoning + SWRL
- **NumPy / scikit-learn** — KGE training + t-SNE
- **Ollama** — Local LLM for RAG
- **trafilatura** — Web content extraction
- **SPARQLWrapper** — Wikidata SPARQL endpoint
