"""Shared configuration and Chroma collection safeguards."""

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "vectorstore"
COLLECTION = "proyecto_faro"


def embedding_model_name() -> str:
    load_dotenv(ROOT / ".env")
    return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def require_api_key() -> None:
    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "replace-me":
        raise RuntimeError("Configurá OPENAI_API_KEY en .env antes de ejecutar el proyecto.")


def open_store(*, create: bool = False) -> Chroma:
    """Open the persisted collection with the model used at ingest."""
    require_api_key()
    model = embedding_model_name()
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        collection = client.get_collection(COLLECTION)
    except chromadb.errors.NotFoundError:
        if not create:
            raise RuntimeError("No hay base indexada. Ejecutá primero: python ingest.py") from None
        collection = client.get_or_create_collection(
            COLLECTION, metadata={"embedding_model": model}
        )
    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model != model:
        raise RuntimeError(
            f"La colección usa embeddings {stored_model!r}, pero la configuración indica "
            f"{model!r}. Restaurá el modelo original o ejecutá python ingest.py --rebuild."
        )
    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=OpenAIEmbeddings(model=model),
    )
