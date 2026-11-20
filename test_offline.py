"""Exercise the complete pipeline without network access or API charges."""

import asyncio
import hashlib
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import ingest
import rag
import settings


class LocalEmbeddings(Embeddings):
    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * 256
        for term in re.findall(r"\w+", text.lower()):
            index = int(hashlib.sha256(term.encode()).hexdigest(), 16) % len(vector)
            vector[index] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def local_model(**_kwargs):
    def answer(prompt):
        message = prompt.to_string()
        if "código secreto de lanzamiento" in message:
            return AIMessage(content='{"texto":"No lo sé","referencias":[]}')
        assert "48 horas" in message, "La recuperación debe incluir la respuesta conocida"
        return AIMessage(
            content='{"texto":"48 horas","referencias":["01_operacion.md#chunk-1"]}'
        )

    return RunnableLambda(answer)


async def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}),
            patch.object(settings, "DB_DIR", Path(directory)),
            patch.object(ingest, "DB_DIR", Path(directory)),
            patch.object(settings, "OpenAIEmbeddings", lambda model: LocalEmbeddings()),
            patch.object(rag, "ChatOpenAI", local_model),
        ):
            count = ingest.ingest()
            assert count == 6
            assert ingest.ingest() == count, "La segunda ingesta debe reutilizar la colección"
            known = await rag.get_rag_response("¿Cuánto dura la ventana de observación de Faro?")
            trap = await rag.get_rag_response("¿Cuál es el código secreto de lanzamiento de Faro?")
            assert known.texto == "48 horas" and known.referencias
            assert trap.texto == "No lo sé" and trap.referencias == []
            print("Prueba local OK: ingesta persistente, respuesta conocida y pregunta trampa.")


if __name__ == "__main__":
    asyncio.run(main())
