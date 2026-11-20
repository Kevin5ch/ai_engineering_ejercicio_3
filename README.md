# Pre-entrega 3: RAG local con ChromaDB

Este proyecto responde preguntas sobre el protocolo ficticio Proyecto Faro usando exclusivamente los documentos de `data/`. La base vectorial se guarda en `vectorstore/`; embeddings y generación usan OpenAI.

## Requisitos

- Python 3.14 (probado con el `.venv` de este repositorio)
- Clave de API de OpenAI

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Editá `.env` y completá `OPENAI_API_KEY`. La clave y `vectorstore/` están excluidos de Git.

## Ingesta

```bash
python ingest.py
```

El script lee `.txt` y `.md` de `data/`, los divide con `RecursiveCharacterTextSplitter` en fragmentos de hasta 600 tokens y 50 tokens de solapamiento, y los guarda en ChromaDB. Si la colección tiene datos, evita repetir las llamadas de embeddings. Al cambiar el dataset o el modelo de embeddings, reconstruí la colección:

```bash
python ingest.py --rebuild
```

El modelo de embeddings queda registrado en la colección. La consulta se detiene si `OPENAI_EMBEDDING_MODEL` no coincide con el modelo usado al indexar.

## Consultas

```bash
python rag.py "¿Cuánto dura la ventana de observación de Faro?"
```

`get_rag_response(query: str)` es asíncrona y devuelve un `RAGResponse` de Pydantic con `texto` y `referencias`. Recupera como máximo 4 fragmentos. La cadena LCEL formatea el contexto, llama al LLM de forma asíncrona y procesa la salida con `PydanticOutputParser`. Si la respuesta no está en el contexto, el prompt exige `No lo sé` y referencias vacías.

## Dos pruebas de extremo a extremo

Para comprobar la estructura del flujo sin clave ni red:

```bash
python test_offline.py
```

Esta comprobación reemplaza los servicios de OpenAI por simulaciones locales y usa una base temporal. Verifica la persistencia, la recuperación y el parser; no evalúa el comportamiento real del modelo.

Con `.env` configurado y la ingesta hecha:

```bash
python test_rag.py
```

La prueba consulta un dato presente y otro ausente. Verifica que la primera respuesta mencione **48 horas** y cite una fuente, y que la pregunta trampa responda exactamente **No lo sé** sin referencias. Requiere red, consume tokens y puede variar según el modelo.

## Estructura

- `ingest.py`: carga, fragmentación y persistencia.
- `rag.py`: recuperación y cadena LCEL asíncrona.
- `test_rag.py`: dos consultas de comprobación.
- `test_offline.py`: comprobación local sin API.
- `data/`: tres documentos de ejemplo sobre Proyecto Faro.
- `.env.example`: variables configurables sin secretos.
