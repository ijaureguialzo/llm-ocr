import base64
import re
import unicodedata
from pathlib import Path

import fitz  # pymupdf
import requests


TARGET_WIDTH = 2000
DATOS_DIR = Path("./datos")
LLM_URL = "http://localhost:1234/api/v1/chat"
LLM_MODEL = "allenai/olmocr-2-7b"


def slugify(text: str) -> str:
    """Convierte un texto en un slug válido para nombre de directorio."""
    # Normalizar caracteres unicode a su forma ASCII más cercana
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    # Reemplazar cualquier carácter que no sea alfanumérico o guion por guion bajo
    text = re.sub(r"[^\w]+", "_", text)
    # Eliminar guiones bajos al inicio y al final
    text = text.strip("_")
    return text


def call_llm(image_path: Path) -> str:
    """Envía una imagen al LLM en base64 y devuelve el texto de la respuesta."""
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": LLM_MODEL,
        "input": [
            {
                "type": "image",
                "data_url": f"data:image/png;base64,{image_data}",
            }
        ],
    }
    response = requests.post(LLM_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    # Intentar extraer el texto de la respuesta
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return str(data)


def convert_pdf_to_images(pdf_path: Path, output_base: Path) -> None:
    """Convierte cada página de un PDF en una imagen PNG de TARGET_WIDTH píxeles de ancho."""
    slug = slugify(pdf_path.stem)
    output_dir = output_base / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_base / f"{slug}.md"

    print(f"Procesando: {pdf_path.name}  →  {output_dir}/")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    with markdown_path.open("w", encoding="utf-8") as md_file:
        md_file.write(f"# {pdf_path.stem}\n\n")

        for page_number in range(total_pages):
            page = doc[page_number]
            # Calcular el factor de escala para obtener TARGET_WIDTH píxeles de ancho
            scale = TARGET_WIDTH / page.rect.width
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            filename = f"pagina_{page_number + 1:03d}.png"
            output_path = output_dir / filename
            pix.save(str(output_path))
            print(f"  Guardado: {filename} — llamando al LLM...", end=" ", flush=True)

            try:
                text = call_llm(output_path)
            except Exception as e:
                text = f"[Error al procesar esta página: {e}]"

            md_file.write(f"## Página {page_number + 1}\n\n{text}\n\n")
            md_file.flush()
            print("OK")

    doc.close()
    print(f"  {total_pages} página(s) procesada(s). Markdown: {markdown_path}\n")


def main() -> None:
    pdf_files = sorted(DATOS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron archivos PDF en '{DATOS_DIR}'.")
        return

    print(f"Se encontraron {len(pdf_files)} archivo(s) PDF.\n")

    for pdf_path in pdf_files:
        convert_pdf_to_images(pdf_path, DATOS_DIR)


if __name__ == "__main__":
    main()

