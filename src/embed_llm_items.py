"""Embed LLM-generated item descriptions with sentence-transformers.

Falls back to a bag-of-words + tf-idf representation if sentence-transformers
is not available. Produces outputs/item_llm_emb.npy (shape n_movies x d),
indexed in the same order as data.load_movies().
"""
import argparse
import json
from pathlib import Path

import numpy as np

from config import OUTPUT_DIR
from data import load_movies


def _st_encode(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Encode with sentence-transformers if available."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    return embs.astype(np.float32)


def _tfidf_encode(texts, dim=384):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                         stop_words="english", lowercase=True, min_df=2)
    X = tf.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(dim, X.shape[1] - 1), random_state=0)
    Z = svd.fit_transform(X)
    # L2 normalize
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (Z / norms).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descriptions", type=str,
                    default=str(OUTPUT_DIR / "item_descriptions.json"))
    ap.add_argument("--out", type=str,
                    default=str(OUTPUT_DIR / "item_llm_emb.npy"))
    ap.add_argument("--model", type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--dim_tfidf", type=int, default=384)
    args = ap.parse_args()

    movies = load_movies()
    with open(args.descriptions) as f:
        desc_map = json.load(f)
    # Align to load_movies() order
    mids = list(movies.index)
    texts = []
    for m in mids:
        d = desc_map.get(str(int(m)), desc_map.get(int(m), ""))
        if not d:
            row = movies.loc[m]
            d = f"{row['title_clean']} ({row['year']}) — genres: {', '.join(row['genres_list'] or [])}"
        texts.append(d)
    print(f"Encoding {len(texts)} item descriptions...")
    try:
        embs = _st_encode(texts, model_name=args.model)
        print(f"  using sentence-transformers ({args.model})")
    except Exception as e:
        print(f"  sentence-transformers unavailable ({e}); falling back to TF-IDF+SVD")
        embs = _tfidf_encode(texts, dim=args.dim_tfidf)
    print(f"  shape: {embs.shape}")
    np.save(args.out, embs)
    # Also save the movie_id order for downstream consumers
    np.save(args.out.replace(".npy", "_ids.npy"), np.asarray(mids, dtype=np.int64))
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
