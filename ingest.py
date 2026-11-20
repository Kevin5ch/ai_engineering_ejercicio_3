"""Ingest the sample documents into a persistent Chroma collection."""

import argparse
import hashlib

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import COLLECTION, DATA_DIR, DB_DIR, embedding_model_name, open_store, require_api_key


def load_documents() -> list[Document]:
    paths = sorted((*DATA_DIR.glob("*.md"), *DATA_DIR.glob("*.txt")))
    if not paths:
        raise RuntimeError(f"No se encontraron archivos .md o .txt en {DATA_DIR}")
    return [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in paths
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=600,
        chunk_overlap=50,
        separators=[" ", ""],
    )
    chunks = splitter.split_documents(documents)
    counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        counters[source] = counters.get(source, 0) + 1
        chunk.metadata["reference"] = f"{source}#chunk-{counters[source]}"
    return chunks


def ingest(*, rebuild: bool = False) -> int:
    require_api_key()
    if rebuild:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        try:
            client.delete_collection(COLLECTION)
        except chromadb.errors.NotFoundError:
            pass
    store = open_store(create=True)
    count = store._collection.count()
    if count:
        print(f"La colección ya tiene {count} fragmentos; se omite la ingesta.")
        return count
    chunks = split_documents(load_documents())
    ids = [hashlib.sha256(chunk.metadata["reference"].encode()).hexdigest() for chunk in chunks]
    store.add_documents(chunks, ids=ids)
    print(f"Guardados {len(chunks)} fragmentos en {DB_DIR} con {embedding_model_name()}.")
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Borra e indexa de nuevo")
    args = parser.parse_args()
    ingest(rebuild=args.rebuild)
