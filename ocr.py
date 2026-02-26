import re
import unicodedata
from pathlib import Path

import fitz  # pymupdf


TARGET_WIDTH = 2000
DATOS_DIR = Path("./datos")


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


def convert_pdf_to_images(pdf_path: Path, output_base: Path) -> None:
    """Convierte cada página de un PDF en una imagen PNG de TARGET_WIDTH píxeles de ancho."""
    slug = slugify(pdf_path.stem)
    output_dir = output_base / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Procesando: {pdf_path.name}  →  {output_dir}/")

    doc = fitz.open(str(pdf_path))

    for page_number in range(len(doc)):
        page = doc[page_number]
        # Calcular el factor de escala para obtener TARGET_WIDTH píxeles de ancho
        scale = TARGET_WIDTH / page.rect.width
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        filename = f"pagina_{page_number + 1:03d}.png"
        output_path = output_dir / filename
        pix.save(str(output_path))
        print(f"  Guardado: {filename}")

    doc.close()
    print(f"  {len(doc)} página(s) procesada(s).\n")


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

