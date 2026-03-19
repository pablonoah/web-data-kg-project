"""
KGE Training & Evaluation
==========================
Trains TransE and ComplEx models using PyKEEN.
Evaluates with MRR, Hits@1, Hits@3, Hits@10.
Includes size-sensitivity analysis and t-SNE visualization.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict


def load_triples(filepath):
    """Load triples from tab-separated file."""
    triples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def build_mappings(train, valid, test):
    """Build entity and relation ID mappings from all splits."""
    entities = set()
    relations = set()
    for triples in [train, valid, test]:
        for h, r, t in triples:
            entities.add(h)
            entities.add(t)
            relations.add(r)
    ent2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(relations))}
    return ent2id, rel2id


def triples_to_numpy(triples, ent2id, rel2id):
    """Convert triples to numpy array of IDs."""
    arr = []
    for h, r, t in triples:
        if h in ent2id and r in rel2id and t in ent2id:
            arr.append([ent2id[h], rel2id[r], ent2id[t]])
    return np.array(arr, dtype=np.int64)


# ============================================================================
# PyKEEN-based Training
# ============================================================================

def train_with_pykeen(data_dir, model_name="TransE", epochs=100, embedding_dim=100):
    """Train a KGE model using PyKEEN."""
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    train = load_triples(os.path.join(data_dir, "train.txt"))
    valid = load_triples(os.path.join(data_dir, "valid.txt"))
    test = load_triples(os.path.join(data_dir, "test.txt"))

    # Build shared entity/relation mappings from all splits
    ent2id, rel2id = build_mappings(train, valid, test)

    # Create labeled triples as numpy string arrays
    train_arr = np.array(train, dtype=str)
    valid_arr = np.array(valid, dtype=str)
    test_arr = np.array(test, dtype=str)

    # Use from_labeled_triples with shared mappings
    tf_train = TriplesFactory.from_labeled_triples(
        train_arr, entity_to_id=ent2id, relation_to_id=rel2id,
    )
    tf_valid = TriplesFactory.from_labeled_triples(
        valid_arr, entity_to_id=ent2id, relation_to_id=rel2id,
    )
    tf_test = TriplesFactory.from_labeled_triples(
        test_arr, entity_to_id=ent2id, relation_to_id=rel2id,
    )

    print(f"\n{'='*60}")
    print(f"Training {model_name} (dim={embedding_dim}, epochs={epochs})")
    print(f"  Train: {len(train_arr)} | Valid: {len(valid_arr)} | Test: {len(test_arr)}")
    print(f"  Entities: {len(ent2id)} | Relations: {len(rel2id)}")
    print(f"{'='*60}")

    result = pipeline(
        training=tf_train,
        validation=tf_valid,
        testing=tf_test,
        model=model_name,
        model_kwargs={"embedding_dim": embedding_dim},
        training_kwargs={"num_epochs": epochs, "batch_size": 256},
        evaluation_kwargs={"batch_size": 256},
        random_seed=42,
    )

    # Extract metrics
    metrics = result.metric_results.to_dict()
    mrr = metrics.get("both", {}).get("realistic", {}).get("inverse_harmonic_mean_rank", 0)
    h1 = metrics.get("both", {}).get("realistic", {}).get("hits_at_1", 0)
    h3 = metrics.get("both", {}).get("realistic", {}).get("hits_at_3", 0)
    h10 = metrics.get("both", {}).get("realistic", {}).get("hits_at_10", 0)

    print(f"\n--- {model_name} Results ---")
    print(f"  MRR:      {mrr:.4f}")
    print(f"  Hits@1:   {h1:.4f}")
    print(f"  Hits@3:   {h3:.4f}")
    print(f"  Hits@10:  {h10:.4f}")

    return result, {"model": model_name, "MRR": mrr, "Hits@1": h1, "Hits@3": h3, "Hits@10": h10}


# ============================================================================
# Lightweight fallback (no PyKEEN)
# ============================================================================

def train_transe_manual(data_dir, epochs=200, embedding_dim=50, lr=0.01, margin=1.0):
    """Manual TransE implementation as fallback."""
    train = load_triples(os.path.join(data_dir, "train.txt"))
    valid = load_triples(os.path.join(data_dir, "valid.txt"))
    test = load_triples(os.path.join(data_dir, "test.txt"))

    ent2id, rel2id = build_mappings(train, valid, test)
    n_ent = len(ent2id)
    n_rel = len(rel2id)

    train_np = triples_to_numpy(train, ent2id, rel2id)
    test_np = triples_to_numpy(test, ent2id, rel2id)

    # Initialize embeddings
    ent_emb = np.random.randn(n_ent, embedding_dim).astype(np.float32) * 0.1
    rel_emb = np.random.randn(n_rel, embedding_dim).astype(np.float32) * 0.1

    # Normalize entity embeddings
    ent_emb = ent_emb / (np.linalg.norm(ent_emb, axis=1, keepdims=True) + 1e-8)

    print(f"\nTraining TransE (manual): {n_ent} entities, {n_rel} relations, {len(train_np)} triples")

    for epoch in range(epochs):
        np.random.shuffle(train_np)
        total_loss = 0.0

        for i in range(0, len(train_np), 128):
            batch = train_np[i:i+128]
            h_idx, r_idx, t_idx = batch[:, 0], batch[:, 1], batch[:, 2]

            h = ent_emb[h_idx]
            r = rel_emb[r_idx]
            t = ent_emb[t_idx]

            # Positive score
            pos_dist = np.sum((h + r - t) ** 2, axis=1)

            # Negative sampling (corrupt tail)
            neg_t_idx = np.random.randint(0, n_ent, size=len(batch))
            neg_t = ent_emb[neg_t_idx]
            neg_dist = np.sum((h + r - neg_t) ** 2, axis=1)

            # Margin loss
            loss = np.maximum(0, margin + pos_dist - neg_dist)
            total_loss += np.sum(loss)

            # Gradient update (simplified SGD)
            mask = (loss > 0).astype(np.float32)
            grad_h = 2 * (h + r - t) * mask[:, None]
            grad_r = 2 * (h + r - t) * mask[:, None]
            grad_t = -2 * (h + r - t) * mask[:, None]
            grad_neg_t = 2 * (h + r - neg_t) * mask[:, None]

            ent_emb[h_idx] -= lr * grad_h
            rel_emb[r_idx] -= lr * grad_r
            ent_emb[t_idx] -= lr * grad_t
            ent_emb[neg_t_idx] += lr * grad_neg_t

            # Re-normalize
            ent_emb[h_idx] = ent_emb[h_idx] / (np.linalg.norm(ent_emb[h_idx], axis=1, keepdims=True) + 1e-8)
            ent_emb[t_idx] = ent_emb[t_idx] / (np.linalg.norm(ent_emb[t_idx], axis=1, keepdims=True) + 1e-8)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}")

    # Evaluate
    metrics = evaluate_embeddings(test_np, ent_emb, rel_emb, n_ent)
    print(f"\n--- TransE (manual) Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return ent_emb, rel_emb, ent2id, rel2id, metrics


def train_complex_manual(data_dir, epochs=200, embedding_dim=50, lr=0.01):
    """Manual ComplEx implementation as fallback."""
    train = load_triples(os.path.join(data_dir, "train.txt"))
    valid = load_triples(os.path.join(data_dir, "valid.txt"))
    test = load_triples(os.path.join(data_dir, "test.txt"))

    ent2id, rel2id = build_mappings(train, valid, test)
    n_ent = len(ent2id)
    n_rel = len(rel2id)

    train_np = triples_to_numpy(train, ent2id, rel2id)
    test_np = triples_to_numpy(test, ent2id, rel2id)

    # ComplEx uses complex embeddings (real + imaginary parts)
    ent_re = np.random.randn(n_ent, embedding_dim).astype(np.float32) * 0.1
    ent_im = np.random.randn(n_ent, embedding_dim).astype(np.float32) * 0.1
    rel_re = np.random.randn(n_rel, embedding_dim).astype(np.float32) * 0.1
    rel_im = np.random.randn(n_rel, embedding_dim).astype(np.float32) * 0.1

    print(f"\nTraining ComplEx (manual): {n_ent} entities, {n_rel} relations, {len(train_np)} triples")

    def complex_score(h_re, h_im, r_re, r_im, t_re, t_im):
        return np.sum(
            r_re * h_re * t_re + r_re * h_im * t_im +
            r_im * h_re * t_im - r_im * h_im * t_re,
            axis=1
        )

    for epoch in range(epochs):
        np.random.shuffle(train_np)
        total_loss = 0.0

        for i in range(0, len(train_np), 128):
            batch = train_np[i:i+128]
            h_idx, r_idx, t_idx = batch[:, 0], batch[:, 1], batch[:, 2]

            h_r, h_i = ent_re[h_idx], ent_im[h_idx]
            r_r, r_i = rel_re[r_idx], rel_im[r_idx]
            t_r, t_i = ent_re[t_idx], ent_im[t_idx]

            pos_score = complex_score(h_r, h_i, r_r, r_i, t_r, t_i)

            # Negative sampling
            neg_t_idx = np.random.randint(0, n_ent, size=len(batch))
            nt_r, nt_i = ent_re[neg_t_idx], ent_im[neg_t_idx]
            neg_score = complex_score(h_r, h_i, r_r, r_i, nt_r, nt_i)

            # Logistic loss
            loss = -np.log(1 / (1 + np.exp(-pos_score)) + 1e-8) - np.log(1 / (1 + np.exp(neg_score)) + 1e-8)
            total_loss += np.sum(loss)

            # Simplified gradient update
            pos_sig = 1 / (1 + np.exp(-pos_score))
            neg_sig = 1 / (1 + np.exp(neg_score))

            grad_scale_pos = -(1 - pos_sig)
            grad_scale_neg = neg_sig

            ent_re[h_idx] -= lr * (grad_scale_pos[:, None] * (r_r * t_r + r_i * t_i) * 0.01)
            ent_im[h_idx] -= lr * (grad_scale_pos[:, None] * (r_r * t_i - r_i * t_r) * 0.01)

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss:.2f}")

    # Evaluate using concatenated embeddings
    ent_emb_full = np.concatenate([ent_re, ent_im], axis=1)
    rel_emb_full = np.concatenate([rel_re, rel_im], axis=1)

    # Custom evaluation for ComplEx
    metrics = evaluate_complex(test_np, ent_re, ent_im, rel_re, rel_im, n_ent)
    print(f"\n--- ComplEx (manual) Results ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return ent_re, ent_im, rel_re, rel_im, ent2id, rel2id, metrics


def evaluate_embeddings(test_triples, ent_emb, rel_emb, n_ent, max_eval=500):
    """Evaluate TransE embeddings with ranking metrics."""
    ranks = []
    n_eval = min(len(test_triples), max_eval)

    for idx in range(n_eval):
        h, r, t = test_triples[idx]
        h_emb = ent_emb[h]
        r_emb = rel_emb[r]

        # Score all entities as tail
        scores = np.sum((h_emb + r_emb - ent_emb) ** 2, axis=1)
        rank = np.sum(scores <= scores[t])
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float32)
    mrr = np.mean(1.0 / ranks)
    h1 = np.mean(ranks <= 1)
    h3 = np.mean(ranks <= 3)
    h10 = np.mean(ranks <= 10)

    return {"MRR": mrr, "Hits@1": h1, "Hits@3": h3, "Hits@10": h10}


def evaluate_complex(test_triples, ent_re, ent_im, rel_re, rel_im, n_ent, max_eval=500):
    """Evaluate ComplEx embeddings."""
    ranks = []
    n_eval = min(len(test_triples), max_eval)

    for idx in range(n_eval):
        h, r, t = test_triples[idx]

        h_r, h_i = ent_re[h], ent_im[h]
        r_r, r_i = rel_re[r], rel_im[r]

        # Score all entities as tail
        scores = np.sum(
            r_r * h_r * ent_re + r_r * h_i * ent_im +
            r_i * h_r * ent_im - r_i * h_i * ent_re,
            axis=1
        )
        rank = np.sum(scores >= scores[t])
        ranks.append(rank)

    ranks = np.array(ranks, dtype=np.float32)
    mrr = np.mean(1.0 / np.maximum(ranks, 1))
    h1 = np.mean(ranks <= 1)
    h3 = np.mean(ranks <= 3)
    h10 = np.mean(ranks <= 10)

    return {"MRR": mrr, "Hits@1": h1, "Hits@3": h3, "Hits@10": h10}


# ============================================================================
# Visualization
# ============================================================================

def plot_tsne(ent_emb, ent2id, output_path="reports/tsne_embeddings.png", n_points=2000):
    """Generate t-SNE visualization of entity embeddings."""
    from sklearn.manifold import TSNE

    n = min(n_points, len(ent_emb))
    indices = np.random.choice(len(ent_emb), n, replace=False)
    subset = ent_emb[indices]

    print(f"Computing t-SNE for {n} entities...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n - 1))
    coords = tsne.fit_transform(subset)

    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=5, alpha=0.5, c='steelblue')
    plt.title("t-SNE Visualization of Entity Embeddings")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE plot: {output_path}")


def plot_nearest_neighbors(ent_emb, ent2id, queries=None, k=5):
    """Find and display nearest neighbors for query entities."""
    id2ent = {v: k for k, v in ent2id.items()}

    if queries is None:
        # Pick some random entities
        queries = list(ent2id.keys())[:5]

    print(f"\n--- Nearest Neighbors (k={k}) ---")
    for query in queries:
        if query not in ent2id:
            continue
        q_id = ent2id[query]
        q_emb = ent_emb[q_id]

        dists = np.sum((ent_emb - q_emb) ** 2, axis=1)
        nearest = np.argsort(dists)[1:k+1]

        print(f"\n  Query: {query}")
        for rank, idx in enumerate(nearest):
            print(f"    {rank+1}. {id2ent.get(idx, f'entity_{idx}')} (dist={dists[idx]:.4f})")


def plot_metrics_comparison(all_results, output_path="reports/kge_comparison.png"):
    """Bar chart comparing models on all metrics."""
    models = [r["model"] for r in all_results]
    metrics = ["MRR", "Hits@1", "Hits@3", "Hits@10"]

    x = np.arange(len(metrics))
    width = 0.35
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, result in enumerate(all_results):
        values = [result[m] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=result["model"])

    ax.set_ylabel("Score")
    ax.set_title("KGE Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved comparison plot: {output_path}")


def plot_size_sensitivity(size_results, output_path="reports/size_sensitivity.png"):
    """Plot metrics vs dataset size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for model_name, results in size_results.items():
        sizes = sorted(results.keys())
        mrr_vals = [results[s]["MRR"] for s in sizes]
        h10_vals = [results[s]["Hits@10"] for s in sizes]

        axes[0].plot(sizes, mrr_vals, 'o-', label=model_name)
        axes[1].plot(sizes, h10_vals, 's-', label=model_name)

    axes[0].set_xlabel("Dataset Size (triples)")
    axes[0].set_ylabel("MRR")
    axes[0].set_title("MRR vs Dataset Size")
    axes[0].legend()

    axes[1].set_xlabel("Dataset Size (triples)")
    axes[1].set_ylabel("Hits@10")
    axes[1].set_title("Hits@10 vs Dataset Size")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved size sensitivity plot: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def run_full_kge_pipeline(data_dir="data/kge", use_pykeen=True):
    """Run full KGE training and evaluation pipeline."""
    all_results = []
    size_results = defaultdict(dict)

    if use_pykeen:
        try:
            # Train with PyKEEN
            for model_name in ["TransE", "ComplEx"]:
                _, metrics = train_with_pykeen(data_dir, model_name=model_name, epochs=100)
                all_results.append(metrics)

            # Size sensitivity
            for subset_dir in sorted(os.listdir(data_dir)):
                if subset_dir.startswith("subset_"):
                    size = int(subset_dir.split("_")[1])
                    subset_path = os.path.join(data_dir, subset_dir)
                    for model_name in ["TransE", "ComplEx"]:
                        _, metrics = train_with_pykeen(subset_path, model_name=model_name, epochs=50)
                        size_results[model_name][size] = metrics

        except ImportError:
            print("PyKEEN not available, falling back to manual implementation...")
            use_pykeen = False

    if not use_pykeen:
        # Manual TransE
        ent_emb_t, rel_emb_t, ent2id_t, rel2id_t, metrics_t = train_transe_manual(data_dir)
        metrics_t["model"] = "TransE"
        all_results.append(metrics_t)

        # Manual ComplEx
        ent_re, ent_im, rel_re, rel_im, ent2id_c, rel2id_c, metrics_c = train_complex_manual(data_dir)
        metrics_c["model"] = "ComplEx"
        all_results.append(metrics_c)

        # t-SNE
        try:
            plot_tsne(ent_emb_t, ent2id_t)
        except ImportError:
            print("sklearn not available, skipping t-SNE")

        # Nearest neighbors
        plot_nearest_neighbors(ent_emb_t, ent2id_t)

        # Size sensitivity with manual
        for subset_dir_name in sorted(os.listdir(data_dir)):
            if subset_dir_name.startswith("subset_"):
                size = int(subset_dir_name.split("_")[1])
                subset_path = os.path.join(data_dir, subset_dir_name)
                _, _, _, _, m_t = train_transe_manual(subset_path, epochs=100)
                m_t["model"] = "TransE"
                size_results["TransE"][size] = m_t

                _, _, _, _, _, _, m_c = train_complex_manual(subset_path, epochs=100)
                m_c["model"] = "ComplEx"
                size_results["ComplEx"][size] = m_c

    # Generate plots
    if all_results:
        plot_metrics_comparison(all_results)

    if size_results:
        plot_size_sensitivity(size_results)

    # Save results to JSON
    results_path = os.path.join("reports", "kge_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "models": all_results,
            "size_sensitivity": {
                model: {str(k): v for k, v in sizes.items()}
                for model, sizes in size_results.items()
            }
        }, f, indent=2)
    print(f"\nResults saved: {results_path}")

    return all_results


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(base_dir)
    run_full_kge_pipeline(use_pykeen=False)
