#!/usr/bin/env python3
"""
Load the pre-embedded BGE-M3 parquet from Hugging Face into Qdrant.

Dataset: https://huggingface.co/datasets/ferMorales/Gaceta_UNAM_BGE_M3_V2

The parquet already contains dense (1024-d) and sparse vectors, so this
script only creates the collection and upserts the points — no embedding.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

HF_REPO_ID = "ferMorales/Gaceta_UNAM_BGE_M3_V2"
HF_FILENAME = "embeddings_bgem3.parquet"
EMBEDDING_DIM = 1024
COLLECTION_NAME = "rag_documents"
QDRANT_HOST = "localhost"
QDRANT_GRPC_PORT = 6334


def download_parquet(local_path: Path | None) -> Path:
    if local_path and local_path.is_file():
        logger.info("Using local parquet: %s", local_path)
        return local_path
    logger.info("Downloading %s from %s ...", HF_FILENAME, HF_REPO_ID)
    path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset"
    )
    logger.info("Downloaded to %s", path)
    return Path(path)


def ensure_collection(client: QdrantClient, name: str, recreate: bool) -> None:
    exists = client.collection_exists(name)
    if exists and recreate:
        logger.info("Recreating collection '%s'.", name)
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config={
                "dense": VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
        )
        logger.info("Created collection '%s'.", name)


def build_point_id(chunk_id: str, source_file: str, chunk_index: int) -> str:
    key = chunk_id.strip() or f"{source_file}:{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def row_to_point(row: pd.Series) -> PointStruct:
    return PointStruct(
        id=build_point_id(
            str(row["chunk_id"]), str(row["source_file"]), int(row["chunk_index"])
        ),
        vector={
            "dense": [float(x) for x in row["embedding"]],
            "sparse": SparseVector(
                indices=[int(i) for i in row["sparse_indices"]],
                values=[float(v) for v in row["sparse_values"]],
            ),
        },
        payload={
            "doc_id": row["doc_id"],
            "chunk_id": row["chunk_id"],
            "chunk_index": int(row["chunk_index"]),
            "corpus": row["corpus"],
            "decade": row["decade"],
            "issue_date": row["issue_date"],
            "source_pdf": row["source_pdf"],
            "source_file": row["source_file"],
            "text": row["text"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load the BGE-M3 parquet from Hugging Face into Qdrant."
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Optional local path to the parquet (otherwise downloaded from HF).",
    )
    parser.add_argument("--qdrant-host", type=str, default=QDRANT_HOST)
    parser.add_argument("--qdrant-grpc-port", type=int, default=QDRANT_GRPC_PORT)
    parser.add_argument("--collection-name", type=str, default=COLLECTION_NAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recreate-collection", action="store_true")
    args = parser.parse_args()

    parquet_path = download_parquet(args.parquet_path)

    logger.info("Reading parquet ...")
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from parquet.", len(df))

    client = QdrantClient(
        host=args.qdrant_host,
        grpc_port=args.qdrant_grpc_port,
        prefer_grpc=True,
        timeout=300,
    )
    logger.info(
        "Connected to Qdrant at %s:%d (gRPC)", args.qdrant_host, args.qdrant_grpc_port
    )
    ensure_collection(client, args.collection_name, args.recreate_collection)

    total = len(df)
    for start in tqdm(range(0, total, args.batch_size), desc="Upsert", unit="batch"):
        end = min(start + args.batch_size, total)
        batch = [row_to_point(df.iloc[i]) for i in range(start, end)]
        client.upsert(collection_name=args.collection_name, points=batch, wait=True)

    info = client.get_collection(args.collection_name)
    logger.info(
        "Done. Collection '%s' has %d points.",
        args.collection_name,
        info.points_count,
    )


if __name__ == "__main__":
    main()
