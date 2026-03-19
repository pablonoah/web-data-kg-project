"""
RAG Pipeline: NL → SPARQL with Self-Repair
===========================================
Uses Ollama (local LLM) to convert natural language questions
to SPARQL queries, executes them, and self-repairs on errors.

Usage:
    python src/rag/rag_pipeline.py                  # Interactive CLI
    python src/rag/rag_pipeline.py --evaluate        # Run evaluation
"""

import os
import re
import sys
import json
import argparse
import requests
from rdflib import Graph

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.rag.schema_summary import extract_schema_summary, get_sparql_prompt_template


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # or mistral, codellama, etc.


def call_ollama(prompt, model=None):
    """Call Ollama API for LLM completion."""
    model = model or OLLAMA_MODEL
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"  Ollama error: {response.status_code}")
            return None
    except requests.ConnectionError:
        print("  ERROR: Cannot connect to Ollama. Make sure it is running:")
        print("    ollama serve")
        print(f"    ollama pull {model}")
        return None
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def extract_sparql(text):
    """Extract SPARQL query from LLM response."""
    # Try to find query in code blocks
    patterns = [
        r"```sparql\s*(.*?)```",
        r"```\s*(SELECT.*?)```",
        r"```\s*(PREFIX.*?)```",
        r"(SELECT\s+.*?WHERE\s*\{.*?\})",
        r"(PREFIX.*?SELECT\s+.*?WHERE\s*\{.*?\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If text looks like a query itself
    if "SELECT" in text.upper() and "WHERE" in text.upper():
        return text.strip()

    return None


def execute_sparql(graph, query):
    """Execute SPARQL query on the RDF graph."""
    try:
        results = graph.query(query)
        rows = []
        for row in results:
            rows.append([str(val) for val in row])
        return rows, None
    except Exception as e:
        return None, str(e)


def self_repair_query(question, failed_query, error_msg, schema_summary, max_attempts=2):
    """Attempt to repair a failed SPARQL query using the LLM."""
    repair_prompt = f"""The following SPARQL query failed with an error. Fix it.

## Schema:
{schema_summary}

## Original Question: {question}

## Failed Query:
```sparql
{failed_query}
```

## Error:
{error_msg}

## Instructions:
- Fix the syntax or semantic error in the query.
- Return ONLY the corrected SPARQL query in a ```sparql code block.
- Make sure all prefixes are declared.
- Make sure all variables in SELECT are bound in WHERE.
"""

    for attempt in range(max_attempts):
        print(f"  Self-repair attempt {attempt + 1}/{max_attempts}...")
        response = call_ollama(repair_prompt)
        if response:
            repaired = extract_sparql(response)
            if repaired:
                return repaired
    return None


def answer_question(question, graph, schema_summary, verbose=True):
    """Full RAG pipeline: NL → SPARQL → Execute → Self-repair if needed."""
    if verbose:
        print(f"\nQ: {question}")

    # Step 1: Generate SPARQL
    prompt_template = get_sparql_prompt_template()
    prompt = prompt_template.format(schema_summary=schema_summary, question=question)

    if verbose:
        print("  Generating SPARQL query...")

    llm_response = call_ollama(prompt)
    if not llm_response:
        return {"question": question, "query": None, "results": None, "error": "LLM unavailable"}

    query = extract_sparql(llm_response)
    if not query:
        if verbose:
            print("  Could not extract SPARQL from LLM response")
        return {"question": question, "query": None, "results": None, "error": "No SPARQL extracted"}

    if verbose:
        print(f"  Generated query:\n    {query[:200]}...")

    # Step 2: Execute
    results, error = execute_sparql(graph, query)

    # Step 3: Self-repair if error
    if error:
        if verbose:
            print(f"  Query failed: {error}")
            print("  Attempting self-repair...")

        repaired_query = self_repair_query(question, query, error, schema_summary)
        if repaired_query:
            if verbose:
                print(f"  Repaired query:\n    {repaired_query[:200]}...")
            results, error2 = execute_sparql(graph, repaired_query)
            if error2:
                if verbose:
                    print(f"  Repair failed: {error2}")
                return {"question": question, "query": repaired_query, "results": None, "error": error2}
            query = repaired_query

    # Step 4: Format results
    if results is not None:
        if verbose:
            print(f"  Results ({len(results)} rows):")
            for row in results[:10]:
                shortened = [v.split("/")[-1] if "/" in v else v for v in row]
                print(f"    {' | '.join(shortened)}")
            if len(results) > 10:
                print(f"    ... and {len(results) - 10} more rows")

    return {"question": question, "query": query, "results": results, "error": error}


# ============================================================================
# Baseline (no RAG - direct SPARQL without LLM)
# ============================================================================

def baseline_keyword_search(question, graph):
    """Simple keyword-based search without LLM (baseline for comparison)."""
    MED = "http://example.org/medical/"
    PROP = "http://example.org/medical/prop/"

    # Extract keywords
    words = question.lower().split()

    # Try to find matching entities
    results = []
    for s, p, o in graph:
        s_str = str(s).lower()
        for word in words:
            if len(word) > 3 and word in s_str:
                results.append(str(s))
                break
        if len(results) >= 10:
            break

    return results


# ============================================================================
# Evaluation
# ============================================================================

EVAL_QUESTIONS = [
    {
        "question": "What drugs contain acetaminophen?",
        "expected_pattern": "hasActiveIngredient.*ACETAMINOPHEN",
    },
    {
        "question": "Who manufactures Entresto?",
        "expected_pattern": "hasManufacturer",
    },
    {
        "question": "List all drugs administered orally.",
        "expected_pattern": "hasRoute.*ORAL",
    },
    {
        "question": "What active ingredients does Betadine contain?",
        "expected_pattern": "hasActiveIngredient",
    },
    {
        "question": "How many drugs are in the knowledge base?",
        "expected_pattern": "COUNT|Drug",
    },
    {
        "question": "What are the topical drugs?",
        "expected_pattern": "hasRoute.*TOPICAL",
    },
    {
        "question": "Which manufacturers produce nicotine products?",
        "expected_pattern": "hasManufacturer.*NICOTINE|hasActiveIngredient.*NICOTINE",
    },
]


def run_evaluation(graph, schema_summary):
    """Run evaluation comparing RAG vs baseline."""
    print("\n" + "=" * 60)
    print("RAG EVALUATION")
    print("=" * 60)

    results = []
    for i, q in enumerate(EVAL_QUESTIONS):
        print(f"\n--- Question {i+1}/{len(EVAL_QUESTIONS)} ---")

        # RAG answer
        rag_result = answer_question(q["question"], graph, schema_summary)

        # Baseline
        baseline_results = baseline_keyword_search(q["question"], graph)

        rag_success = rag_result["results"] is not None and len(rag_result["results"]) > 0
        baseline_has = len(baseline_results) > 0

        results.append({
            "question": q["question"],
            "rag_success": rag_success,
            "rag_results_count": len(rag_result["results"]) if rag_result["results"] else 0,
            "rag_query": rag_result["query"],
            "baseline_results_count": len(baseline_results),
            "rag_error": rag_result["error"],
        })

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Question':<50} {'RAG':>6} {'Base':>6}")
    print("-" * 62)
    for r in results:
        rag_str = f"{r['rag_results_count']}" if r["rag_success"] else "FAIL"
        base_str = f"{r['baseline_results_count']}"
        print(f"{r['question'][:48]:<50} {rag_str:>6} {base_str:>6}")

    rag_success_rate = sum(1 for r in results if r["rag_success"]) / len(results)
    print(f"\nRAG Success Rate: {rag_success_rate:.0%}")

    # Save evaluation results
    eval_path = "reports/rag_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved: {eval_path}")

    return results


# ============================================================================
# Interactive CLI
# ============================================================================

def interactive_cli(graph, schema_summary):
    """Interactive command-line interface for the RAG system."""
    print("\n" + "=" * 60)
    print("  Medical KB Question Answering (RAG)")
    print("  Type your question or 'quit' to exit")
    print("=" * 60)
    print("\nExample questions:")
    print("  - What drugs contain acetaminophen?")
    print("  - Who manufactures Entresto?")
    print("  - List all oral drugs")
    print("  - What active ingredients does Betadine have?")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer_question(question, graph, schema_summary, verbose=True)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline: NL → SPARQL")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--kb", default="kg_artifacts/final_kb.ttl", help="Path to KB (TTL)")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    args = parser.parse_args()

    global OLLAMA_MODEL
    OLLAMA_MODEL = args.model

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)

    print("Loading knowledge base and schema...")
    schema_summary, graph = extract_schema_summary(args.kb)
    print(f"KB loaded: {len(graph)} triples\n")

    if args.evaluate:
        run_evaluation(graph, schema_summary)
    else:
        interactive_cli(graph, schema_summary)


if __name__ == "__main__":
    main()
