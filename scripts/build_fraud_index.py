"""
Build the ChromaDB vector index for the Fraud Detection Knowledge Base.

Usage:
    python scripts/build_fraud_index.py           # build (skips if already exists)
    python scripts/build_fraud_index.py --rebuild  # force full rebuild

Requires:
    chromadb
    openai (OPENAI_API_KEY in .env)

Estimated cost: ~$0.01 one-time (25 docs x ~150 tokens x text-embedding-3-small pricing).
Index is saved to data/chroma_fraud/ and loaded automatically by FraudRAGAgent at runtime.
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

load_dotenv()


def build_index(rebuild: bool = False) -> None:
    try:
        import chromadb
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    except ImportError:
        print("ERROR: chromadb is not installed. Run: pip install chromadb")
        sys.exit(1)

    import os

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key.startswith("sk-your-"):
        print("ERROR: OPENAI_API_KEY not set. Please add it to your .env file.")
        sys.exit(1)

    from src.agent.rag_agent import FRAUD_KNOWLEDGE_BASE, CHROMA_PATH, COLLECTION_NAME

    chroma_path = Path(CHROMA_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)

    embed_fn = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",
        dimensions=256,
    )

    client = chromadb.PersistentClient(path=str(chroma_path))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("Existing collection deleted.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing = collection.count()
    if existing > 0 and not rebuild:
        print(f"Index already contains {existing} documents. Use --rebuild to force.")
        return

    print(f"Embedding {len(FRAUD_KNOWLEDGE_BASE)} knowledge base documents...")
    batch_size = 10
    for i in range(0, len(FRAUD_KNOWLEDGE_BASE), batch_size):
        batch = FRAUD_KNOWLEDGE_BASE[i : i + batch_size]
        collection.upsert(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[
                {"category": d["category"], "source": d["source"]} for d in batch
            ],
        )
        print(f"  Batch {i // batch_size + 1}: {len(batch)} documents embedded.")

    print(f"\nIndex built successfully. Total documents: {collection.count()}")
    print(f"Saved to: {chroma_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ChromaDB fraud knowledge index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force full rebuild of the index.",
    )
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)
