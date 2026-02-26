import base64
import os
import re
import select
import sys
import termios
import threading
import tty
import unicodedata
from pathlib import Path

import fitz  # pymupdf
import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MAX_LONG_SIDE = int(os.getenv("MAX_LONG_SIDE", "1288"))
DATOS_DIR = Path(os.getenv("DATOS_DIR", "./datos"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "allenai/olmocr-2-7b")
STREAM_CHUNK_TIMEOUT = int(os.getenv("STREAM_CHUNK_TIMEOUT", "60"))
MAX_CONSECUTIVE_ERRORS = int(os.getenv("MAX_CONSECUTIVE_ERRORS", "3"))

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
    """Envía una imagen en bytes al LLM en base64 usando streaming SSE y devuelve el texto.

    Lanza TimeoutError si la petición completa tarda más de STREAM_CHUNK_TIMEOUT segundos.
    """
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": LLM_MODEL,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                ],
            }
        ],
    }

    result: dict = {"text": None, "error": None, "generation_id": None}
    # Cliente httpx compartido: cerrarlo desde el hilo principal interrumpe el socket
    http_client = httpx.Client(timeout=None)

    def _do_request() -> None:
        try:
            chunks: list[str] = []
            with http_client.stream(
                "POST",
                f"{LLM_BASE_URL}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer lm-studio"},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        import json
                        data = json.loads(data_str)
                    except Exception:
                        continue
                    if result["generation_id"] is None:
                        result["generation_id"] = data.get("id")
                    delta = (data.get("choices") or [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        chunks.append(content)
            result["text"] = "".join(chunks)
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=_do_request, daemon=True)
    thread.start()
    thread.join(timeout=STREAM_CHUNK_TIMEOUT)

    if thread.is_alive():
        # Cerrar el cliente httpx: interrumpe el socket y hace que iter_lines() lance una excepción
        http_client.close()
        # Pedir al servidor que detenga la generación
        generation_id = result["generation_id"]
        if generation_id:
            try:
                httpx.post(
                    f"{LLM_BASE_URL}/chat/completions/{generation_id}/cancel",
                    headers={"Authorization": "Bearer lm-studio"},
                    timeout=5,
                )
            except Exception:
                pass
        raise TimeoutError(
            f"La petición al LLM superó el límite de {STREAM_CHUNK_TIMEOUT}s sin completarse"
        )

    http_client.close()

    if result["error"] is not None:
        raise result["error"]

    text = result["text"] or ""

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
                print(f"{e}")
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
