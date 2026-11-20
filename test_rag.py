"""Two live end-to-end checks; requires an indexed database and an OpenAI key."""

import asyncio

from rag import get_rag_response


async def main() -> None:
    known = await get_rag_response("¿Cuánto dura la ventana de observación de Faro?")
    print("Pregunta conocida:", known.model_dump_json(indent=2))
    assert "48 horas" in known.texto.lower(), "La respuesta conocida debe mencionar 48 horas"
    assert known.referencias, "La respuesta conocida debe citar al menos un fragmento"

    trap = await get_rag_response("¿Cuál es el código secreto de lanzamiento de Faro?")
    print("Pregunta trampa:", trap.model_dump_json(indent=2))
    assert trap.texto == "No lo sé", "La pregunta trampa debe rechazarse"
    assert trap.referencias == [], "Una respuesta desconocida no debe incluir referencias"
    print("Ambas pruebas pasaron.")


if __name__ == "__main__":
    asyncio.run(main())
