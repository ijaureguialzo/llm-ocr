import base64
import json
import os
import re
import select
import sys
import termios
import threading
import time
import tty
import unicodedata
from collections.abc import Callable
from pathlib import Path

import fitz  # pymupdf
import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MAX_LONG_SIDE = int(os.getenv("MAX_LONG_SIDE", "1288"))
DATOS_DIR = Path(os.getenv("DATOS_DIR", "./datos"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "allenai/olmocr-2-7b")
STREAM_CHUNK_TIMEOUT = int(os.getenv("STREAM_CHUNK_TIMEOUT", "300"))
MAX_CONSECUTIVE_ERRORS = int(os.getenv("MAX_CONSECUTIVE_ERRORS", "3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
DEBUG = os.getenv("DEBUG", "false").strip().lower() in {"1", "true", "yes"}

# Flag compartido: se activa cuando el usuario pulsa Escape
stop_requested = threading.Event()
# Flag interno: señal para que el hilo listener termine sin que el usuario haya pulsado Escape
_listener_exit = threading.Event()

# Estado de la petición LLM en curso (accedido desde el listener de teclado y call_llm)
_current_http_client: httpx.Client | None = None
_current_generation_id: str | None = None
_current_lock = threading.Lock()


def _cancel_current_request() -> None:
    """Interrumpe la petición LLM en curso cerrando el socket y notificando al servidor."""
    with _current_lock:
        client = _current_http_client
        generation_id = _current_generation_id

    if client is not None:
        try:
            client.close()
        except httpx.HTTPError:
            pass

    if generation_id is not None:
        try:
            httpx.post(
                f"{LLM_BASE_URL}/chat/completions/{generation_id}/cancel",
                headers={"Authorization": "Bearer lm-studio"},
                timeout=5,
            )
        except httpx.HTTPError:
            pass


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
                    print("\n\n  [Escape] Deteniendo el proceso...")
                    stop_requested.set()
                    _cancel_current_request()
                    return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _fmt(seconds: float) -> str:
    """Formatea una duración en segundos como m:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _print_banner() -> None:
    print(r"""
  ██╗     ██╗     ███╗   ███╗       ██████╗  ██████╗██████╗
  ██║     ██║     ████╗ ████║      ██╔═══██╗██╔════╝██╔══██╗
  ██║     ██║     ██╔████╔██║      ██║   ██║██║     ██████╔╝
  ██║     ██║     ██║╚██╔╝██║      ██║   ██║██║     ██╔══██╗
  ███████╗███████╗██║ ╚═╝ ██║      ╚██████╔╝╚██████╗██║  ██║
  ╚══════╝╚══════╝╚═╝     ╚═╝       ╚═════╝  ╚═════╝╚═╝  ╚═╝
""")


def slugify(text: str) -> str:
    """Convierte un texto en un slug válido para nombre de directorio.

    Conserva caracteres no ASCII (kanji, hiragana, katakana, etc.) y solo
    reemplaza los caracteres que son inválidos o problemáticos en nombres de archivo.
    """
    # Normalizar a forma NFC para consistencia sin eliminar caracteres no ASCII
    text = unicodedata.normalize("NFC", text)
    # Reemplazar caracteres inválidos en nombres de archivo y espacios por guion bajo
    # Se conservan letras, dígitos, guiones, puntos y cualquier carácter unicode de palabra (incluye CJK)
    text = re.sub(r'[<>:"/\\|?*\s]+', "_", text)
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
        "max_tokens": MAX_TOKENS,
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

    global _current_http_client, _current_generation_id
    with _current_lock:
        _current_http_client = http_client
        _current_generation_id = None

    def _do_request() -> None:
        try:
            current_max_tokens = payload["max_tokens"]

            while True:
                payload["max_tokens"] = current_max_tokens
                finish_reason: str | None = None
                chunks = []

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
                        if DEBUG:
                            print(f"\n  [DEBUG] {data_str}", flush=True)
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                        except (json.JSONDecodeError, KeyError):
                            continue
                        if result["generation_id"] is None:
                            result["generation_id"] = data.get("id")
                        choice = (data.get("choices") or [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content")
                        if content:
                            chunks.append(content)
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]

                if finish_reason != "length":
                    # Respuesta completa o error del modelo: salir del bucle
                    break

                # El modelo cortó por límite de tokens: duplicar y reintentar
                current_max_tokens *= 2
                print(
                    f"\n  [tokens] Límite alcanzado ({current_max_tokens // 2} tokens). "
                    f"Reintentando con {current_max_tokens}...",
                    flush=True,
                )
                # Reiniciar el generation_id para que el hilo principal lo actualice
                # con el id de la nueva petición
                result["generation_id"] = None

            result["text"] = "".join(chunks)
        except (httpx.HTTPError, OSError) as exc:
            result["error"] = exc

    thread = threading.Thread(target=_do_request, daemon=True)
    thread.start()

    deadline = STREAM_CHUNK_TIMEOUT
    while deadline > 0 and thread.is_alive():
        thread.join(timeout=0.1)
        deadline -= 0.1
        # Propagar el generation_id al estado global en cuanto esté disponible
        if result["generation_id"] is not None:
            with _current_lock:
                _current_generation_id = result["generation_id"]

    if thread.is_alive():
        _cancel_current_request()
        raise TimeoutError(
            f"La petición al LLM superó el límite de {STREAM_CHUNK_TIMEOUT}s sin completarse"
        )

    with _current_lock:
        _current_http_client = None
        _current_generation_id = None
    try:
        http_client.close()
    except OSError:
        pass

    # Si la petición fue cancelada externamente (Escape o timeout ya gestionado), ignorar el error
    if stop_requested.is_set():
        raise InterruptedError("Petición cancelada por el usuario")

    if result["error"] is not None:
        raise result["error"]

    text = result["text"] or ""

    # Eliminar el bloque de metadatos al inicio (delimitado por ---).
    # Acepta espacios/tabuladores tras los delimitadores y bloque al final del texto.
    text = re.sub(r"^---[ \t]*\n.*?---[ \t]*(?:\n|$)", "", text, flags=re.DOTALL)
    text = text.strip()

    # Si tras eliminar el bloque no queda contenido útil, devolver cadena vacía
    # para que el llamador pueda decidir si escribe o no la página.
    return text


def get_last_processed_page(markdown_path: Path) -> int:
    """Devuelve el número de la última página procesada en el Markdown, o 0 si no hay ninguna."""
    content = markdown_path.read_text(encoding="utf-8")
    matches = re.findall(r"^## Página (\d+)", content, flags=re.MULTILINE)
    if matches:
        return max(int(m) for m in matches)
    return 0


def _process_pages(
    title: str,
    markdown_path: Path,
    total_pages: int,
    start_page: int,
    file_mode: str,
    get_image_bytes: Callable[[int], bytes],
) -> None:
    """Bucle común de procesado de páginas: llama al LLM y escribe el Markdown.

    get_image_bytes(page_number) debe devolver los bytes PNG de la página indicada.
    """
    with markdown_path.open(file_mode, encoding="utf-8") as md_file:
        if file_mode == "w":
            md_file.write(f"# {title}\n\n")

        consecutive_errors = 0
        error_pages: list[int] = []
        page_times: list[float] = []

        for page_number in range(start_page, total_pages):
            if stop_requested.is_set():
                break

            image_bytes = get_image_bytes(page_number)

            t_start = time.monotonic()
            done_event = threading.Event()

            def _display_timer(pn: int = page_number) -> None:
                while not done_event.wait(timeout=1.0):
                    elapsed_time = time.monotonic() - t_start
                    print(f"\r  Página {pn + 1}/{total_pages} — llamando al LLM... {_fmt(elapsed_time)}",
                          end="", flush=True)

            print(f"  Página {page_number + 1}/{total_pages} — llamando al LLM...", end="", flush=True)
            timer_thread = threading.Thread(target=_display_timer, daemon=True)
            timer_thread.start()

            try:
                text = call_llm(image_bytes)
                consecutive_errors = 0
            except InterruptedError:
                done_event.set()
                timer_thread.join()
                break
            except (TimeoutError, httpx.HTTPError, OSError) as e:
                elapsed = time.monotonic() - t_start
                done_event.set()
                timer_thread.join()
                consecutive_errors += 1
                error_pages.append(page_number + 1)
                print(f"\r  Página {page_number + 1}/{total_pages} — ERROR ({_fmt(elapsed)}) — {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print(f"\n  {MAX_CONSECUTIVE_ERRORS} errores consecutivos. Deteniendo el proceso...")
                    break
                continue
            finally:
                done_event.set()

            elapsed = time.monotonic() - t_start
            page_times.append(elapsed)

            pages_done = len(page_times)
            pages_left = total_pages - (start_page + pages_done)
            avg = sum(page_times) / pages_done
            eta = avg * pages_left

            print(f"\r  Página {page_number + 1}/{total_pages} — OK ({_fmt(elapsed)}) — "
                  f"media {_fmt(avg)}/pág — estimado restante: {_fmt(eta)}")

            if text:
                md_file.write(f"## Página {page_number + 1}\n\n{text}\n\n")
            else:
                md_file.write(f"## Página {page_number + 1}\n\nSin contenido.\n\n")
                print(f"  Página {page_number + 1}/{total_pages} — sin contenido.")
            md_file.flush()

    pages_processed = len(page_times)
    print(f"\n  {pages_processed} página(s) procesada(s) correctamente.")
    if error_pages:
        print(f"  {len(error_pages)} página(s) no procesada(s) por error: {error_pages}")
    print()


def convert_pdf_to_images(pdf_path: Path, output_base: Path) -> None:
    """Convierte cada página de un PDF en una imagen PNG en memoria y la envía al LLM."""
    slug = slugify(pdf_path.stem)
    markdown_path = output_base / f"{slug}.md"

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if markdown_path.exists():
        last_page = get_last_processed_page(markdown_path)
        if last_page >= total_pages:
            print(f"Saltando (ya completo): {pdf_path.name}  →  {markdown_path}\n")
            doc.close()
            return
        start_page = last_page
        file_mode = "a"
        print(f"Reanudando desde página {last_page + 1}: {pdf_path.name}  →  {markdown_path}")
    else:
        start_page = 0
        file_mode = "w"
        print(f"Procesando: {output_base}/{pdf_path.name}")
        print(f"Markdown:   {markdown_path}\n")

    def get_image_bytes(page_number: int) -> bytes:
        page = doc[page_number]
        long_side = max(page.rect.width, page.rect.height)
        scale = MAX_LONG_SIDE / long_side
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")

    _process_pages(pdf_path.stem, markdown_path, total_pages, start_page, file_mode, get_image_bytes)
    doc.close()


def process_image_dir(dir_path: Path, output_base: Path) -> None:
    """Procesa todas las imágenes PNG/JPEG de un subdirectorio como si fuera un proyecto."""
    slug = slugify(dir_path.name)
    markdown_path = output_base / f"{slug}.md"

    image_files = sorted(
        f for f in dir_path.iterdir()
        if f.is_file() and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )

    if not image_files:
        return

    total_pages = len(image_files)

    if markdown_path.exists():
        last_page = get_last_processed_page(markdown_path)
        if last_page >= total_pages:
            print(f"Saltando (ya completo): {dir_path.name}/  →  {markdown_path}\n")
            return
        start_page = last_page
        file_mode = "a"
        print(f"Reanudando desde imagen {last_page + 1}: {dir_path.name}/  →  {markdown_path}")
    else:
        start_page = 0
        file_mode = "w"
        print(f"Procesando: {output_base}/{dir_path.name}/")
        print(f"Markdown:   {markdown_path}\n")

    def get_image_bytes(page_number: int) -> bytes:
        img_path = image_files[page_number]
        # Reescalar con fitz para respetar MAX_LONG_SIDE igual que con los PDFs
        doc = fitz.open(str(img_path))
        page = doc[0]
        long_side = max(page.rect.width, page.rect.height)
        scale = MAX_LONG_SIDE / long_side
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        doc.close()
        return pix.tobytes("png")

    _process_pages(dir_path.name, markdown_path, total_pages, start_page, file_mode, get_image_bytes)


def main() -> None:
    _print_banner()
    pdf_files = sorted(DATOS_DIR.glob("*.pdf"))
    image_dirs = sorted(
        d for d in DATOS_DIR.iterdir()
        if d.is_dir() and any(
            f.suffix.lower() in {".png", ".jpg", ".jpeg"} for f in d.iterdir() if f.is_file()
        )
    )

    if not pdf_files and not image_dirs:
        print(f"No se encontraron archivos PDF ni directorios con imágenes en '{DATOS_DIR}'.")
        return

    total = len(pdf_files) + len(image_dirs)
    print(
        f"Se encontraron {len(pdf_files)} PDF(s) y {len(image_dirs)} directorio(s) con imágenes ({total} proyecto(s) en total).")
    print("Pulsa Escape en cualquier momento para detener el procesamiento.\n")

    listener = threading.Thread(target=_keyboard_listener, daemon=True)
    listener.start()

    for pdf_path in pdf_files:
        if stop_requested.is_set():
            break
        convert_pdf_to_images(pdf_path, DATOS_DIR)

    for dir_path in image_dirs:
        if stop_requested.is_set():
            break
        process_image_dir(dir_path, DATOS_DIR)

    # Restaurar el terminal si el listener sigue vivo (salida normal)
    if not stop_requested.is_set():
        _listener_exit.set()
    listener.join(timeout=1)

    if stop_requested.is_set():
        print("\nProcesamiento detenido por el usuario.")


if __name__ == "__main__":
    main()
