import base64
import re
import select
import sys
import termios
import threading
import tty
import unicodedata
from pathlib import Path

import fitz  # pymupdf
import requests

MAX_LONG_SIDE = 1288
DATOS_DIR = Path("./datos")
LLM_URL = "http://localhost:1234/api/v1/chat"
LLM_MODEL = "allenai/olmocr-2-7b"

# Flag compartido: se activa cuando el usuario pulsa Escape
stop_requested = threading.Event()
# Flag interno: señal para que el hilo listener termine sin que el usuario haya pulsado Escape
_listener_exit = threading.Event()


def _keyboard_listener() -> None:
    """Hilo que espera la tecla Escape sin poner el terminal en modo raw."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # cbreak: entrega caracteres de uno en uno pero mantiene \r\n intactos
        while not _listener_exit.is_set():
            # Esperar hasta 200 ms para no bloquear indefinidamente
            ready, _, _ = select.select([sys.stdin], [], [], 0.2)
            if ready:
                ch = sys.stdin.read(1)
                if ch == "\x1b":  # tecla Escape
                    print("\n[Escape] Deteniendo tras la página actual...")
                    stop_requested.set()
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def slugify(text: str) -> str:
    """Convierte un texto en un slug válido para nombre de directorio."""
    # Normalizar caracteres unicode a su forma ASCII más cercana
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    # Reemplazar cualquier carácter que no sea alfanumérico o guion por guion bajo
    text = re.sub(r"\W+", "_", text)
    # Eliminar guiones bajos al inicio y al final
    text = text.strip("_")
    return text


def call_llm(image_bytes: bytes) -> str:
    """Envía una imagen en bytes al LLM en base64 y devuelve el texto de la respuesta."""
    image_data = base64.b64encode(image_bytes).decode("utf-8")
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
        text = data["output"][0]["content"]
    except (KeyError, IndexError):
        text = str(data)

    # Eliminar el bloque de metadatos al inicio (delimitado por ---)
    text = re.sub(r"^---\n.*?---\n", "", text, flags=re.DOTALL)

    return text


def get_last_processed_page(markdown_path: Path) -> int:
    """Devuelve el número de la última página procesada en el Markdown, o 0 si no hay ninguna."""
    content = markdown_path.read_text(encoding="utf-8")
    matches = re.findall(r"^## Página (\d+)", content, flags=re.MULTILINE)
    if matches:
        return max(int(m) for m in matches)
    return 0


def convert_pdf_to_images(pdf_path: Path, output_base: Path) -> None:
    """Convierte cada página de un PDF en una imagen PNG en memoria y la envía al LLM."""
    slug = slugify(pdf_path.stem)

    markdown_path = output_base / f"{slug}.md"

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    # Determinar desde qué página continuar
    if markdown_path.exists():
        last_page = get_last_processed_page(markdown_path)
        if last_page >= total_pages:
            print(f"Saltando (ya completo): {pdf_path.name}  →  {markdown_path}\n")
            doc.close()
            return
        start_page = last_page  # índice 0-based: la siguiente página a procesar
        file_mode = "a"
        print(f"Reanudando desde página {last_page + 1}: {pdf_path.name}  →  {markdown_path}")
    else:
        start_page = 0
        file_mode = "w"
        print(f"Procesando: {pdf_path.name}  →  {markdown_path}")

    with markdown_path.open(file_mode, encoding="utf-8") as md_file:
        if file_mode == "w":
            md_file.write(f"# {pdf_path.stem}\n\n")

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 3

        for page_number in range(start_page, total_pages):
            if stop_requested.is_set():
                doc.close()
                return

            page = doc[page_number]
            # Calcular el factor de escala para que el lado más largo no supere MAX_LONG_SIDE
            long_side = max(page.rect.width, page.rect.height)
            scale = MAX_LONG_SIDE / long_side
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Obtener los bytes PNG directamente en memoria (sin escribir a disco)
            image_bytes = pix.tobytes("png")

            print(f"  Página {page_number + 1}/{total_pages} — llamando al LLM...", end=" ", flush=True)

            try:
                text = call_llm(image_bytes)
                consecutive_errors = 0  # reiniciar contador en éxito
            except Exception as e:
                consecutive_errors += 1
                print(f"  [Página {page_number + 1} omitida — error {consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}]")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"  {MAX_CONSECUTIVE_ERRORS} errores consecutivos. Deteniendo procesado de: {pdf_path.name}")
                    doc.close()
                    return
                continue

            md_file.write(f"## Página {page_number + 1}\n\n{text}\n\n")
            md_file.flush()
            print("OK")

    doc.close()
    pages_processed = total_pages - start_page
    print(f"  {pages_processed} página(s) procesada(s). Markdown: {markdown_path}\n")


def main() -> None:
    pdf_files = sorted(DATOS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No se encontraron archivos PDF en '{DATOS_DIR}'.")
        return

    print(f"Se encontraron {len(pdf_files)} archivo(s) PDF.")
    print("Pulsa Escape en cualquier momento para detener el procesamiento.\n")

    listener = threading.Thread(target=_keyboard_listener, daemon=True)
    listener.start()

    for pdf_path in pdf_files:
        if stop_requested.is_set():
            break
        convert_pdf_to_images(pdf_path, DATOS_DIR)

    # Restaurar el terminal si el listener sigue vivo (salida normal)
    if not stop_requested.is_set():
        _listener_exit.set()  # señal para que el hilo termine sin marcar parada de usuario
    listener.join(timeout=1)

    if stop_requested.is_set():
        print("Procesamiento detenido por el usuario.")


if __name__ == "__main__":
    main()
