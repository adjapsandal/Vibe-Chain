import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from fetch_metadata import extract_video_id
from transcript import get_transcript
from embedder import embed_video


def build_embeddings(csv_path: Path, cache_path: Path):
    df = pd.read_csv(csv_path)
    video_ids = df['video_id'].tolist()
    embs = []
    for vid in tqdm(video_ids, desc="Embedding videos"):
        if 'url' in df.columns:
            url = df.loc[df['video_id'] == vid, 'url'].iloc[0]
        else:
            url = f"https://youtu.be/{vid}"
        try:
            emb = embed_video(url)
        except Exception as e:
            print(f"[Warning] Не удалось влить видео {vid}: {e}")
            emb = np.zeros(1280, dtype=np.float32)
        embs.append(emb)
    embs = np.vstack(embs)
    with open(cache_path, 'wb') as f:
        pickle.dump((video_ids, embs), f)
    print(f"Сохранено {len(video_ids)} эмбеддингов в {cache_path}")


def load_embeddings(cache_path: Path):
    with open(cache_path, 'rb') as f:
        video_ids, embs = pickle.load(f)
    return video_ids, embs


def recommend(video_id: str, video_ids, embs, top_k=5):
    query_url = f"https://youtu.be/{video_id}"
    try:
        q_emb = embed_video(query_url).reshape(1, -1)
    except Exception as e:
        raise RuntimeError(f"Не удалось встраивать видео {video_id}: {e}")

    sims = cosine_similarity(q_emb, embs)[0]

    order = np.argsort(-sims)
    picks = [i for i in order][:top_k]
    return [(video_ids[i], float(sims[i])) for i in picks]


def main():
    parser = argparse.ArgumentParser(description="Recommend similar YouTube videos")
    parser.add_argument("video_id", help="ID YouTube-видео для запроса")
    parser.add_argument("--corpus", default="data/small_trending.csv", help="CSV с колонками video_id,url")
    parser.add_argument("--cache", default="data/embeddings.pkl", help="Файл для хранения эмбеддингов")
    parser.add_argument("--rebuild", action="store_true", help="Перестроить эмбеддинги заново")
    parser.add_argument("--topk", type=int, default=5, help="Сколько рекомендаций вывести")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    cache_path = Path(args.cache)

    if args.rebuild or not cache_path.exists():
        build_embeddings(corpus_path, cache_path)

    video_ids, embs = load_embeddings(cache_path)
    recs = recommend(args.video_id, video_ids, embs, top_k=args.topk)

    print(f"Рекомендации для {args.video_id}:")
    for vid, score in recs:
        print(f"https://youtu.be/{vid}\t(score={score:.4f})")


if __name__ == "__main__":
    main()
