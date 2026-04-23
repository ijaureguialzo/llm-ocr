# AGENTS.md — llm-ocr

## Descripción del proyecto

`llm-ocr` es una herramienta de línea de comandos que extrae texto de PDFs e imágenes (PNG/JPEG) y lo convierte a Markdown utilizando un LLM local con una API compatible con OpenAI (por defecto, [LM Studio](https://lmstudio.ai/) con el modelo `allenai/olmocr-2-7b`).

Toda la lógica reside en un único fichero: **`ocr.py`**.

---

## Arquitectura

```
ocr.py
├── Configuración vía .env               (variables de entorno con python-dotenv)
├── Lazy-imports                         (_fitz, _httpx: se cargan al primer uso)
├── _TeeWriter                           (duplica stdout/stderr a consola y fichero de log)
├── call_llm(image_bytes)                (envía imagen al LLM en streaming SSE, devuelve texto)
├── convert_pdf_to_images(pdf, out)      (convierte cada página de un PDF a PNG y llama al LLM)
├── process_image_dir(dir, out)          (procesa directorio de imágenes PNG/JPEG)
├── _process_pages(...)                  (bucle principal de procesado: llama al LLM y escribe Markdown)
│   ├── Fase 1: recupera páginas hueco  (gaps de ejecuciones previas interrumpidas)
│   └── Fase 2: procesa páginas nuevas (desde la última página ya procesada)
├── _collect_items(root)                 (descubre PDFs e image_dirs recursivamente con rglob)
├── _keyboard_listener()                 (hilo daemon: Escape para detener, S para saltar página)
└── main() / _main_inner()              (punto de entrada, descubre PDFs y directorios)
```

### Flujo de datos

1. El usuario coloca PDFs o directorios con imágenes en `DATOS_DIR` (por defecto `./datos`).
2. `_collect_items()` recorre el árbol **recursivamente** con `rglob`: encuentra todos los PDFs en cualquier nivel y todos los directorios que contienen directamente imágenes PNG/JPEG.
3. `main()` lanza `convert_pdf_to_images` o `process_image_dir` por cada elemento encontrado. El Markdown de salida se guarda **junto al fichero fuente** (en el mismo directorio que el PDF o el directorio de imágenes).
4. Cada página se renderiza a PNG en memoria (a través de PyMuPDF/fitz), escalada para que el lado largo no supere `MAX_LONG_SIDE` píxeles.
5. La imagen PNG se codifica en base64 y se envía al LLM vía `POST /v1/chat/completions` con streaming SSE.
6. El texto extraído se escribe en un fichero Markdown junto al fichero de entrada, con una sección `## Página N` por página.
7. Si la ejecución se interrumpe, la próxima ejecución **reanuda** desde la última página procesada y **rellena** los huecos.

### Concurrencia

- Un **hilo de teclado** (`_keyboard_listener`) escucha Escape y S sin bloquear el hilo principal.
- La petición HTTP al LLM se ejecuta en un **hilo interno** dentro de `call_llm` para poder aplicar un timeout.
- `stop_requested` y `skip_page_requested` son `threading.Event` compartidos.
- `_current_http_client` y `_current_generation_id` están protegidos por `_current_lock`.

---

## Entorno y dependencias

| Herramienta | Versión mínima | Propósito |
|---|---|---|
| Python | 3.14 | Intérprete |
| Poetry | 2.x | Gestión de dependencias y entornos virtuales |
| PyMuPDF (`fitz`) | ≥ 1.24 | Renderizado de PDFs e imágenes |
| httpx | ≥ 0.27 | Cliente HTTP con soporte streaming |
| python-dotenv | ≥ 1.0 | Carga de `.env` |
| PyInstaller | ≥ 6.0 (dev) | Compilación a ejecutable standalone |

### Instalación del entorno de desarrollo

```bash
poetry install          # instala dependencias de producción
poetry install --with dev  # incluye PyInstaller
```

---

## Configuración

Copia `env-example` a `.env` y ajusta los valores:

```
LLM_BASE_URL=http://localhost:1234/v1   # URL base de la API compatible con OpenAI
LLM_MODEL=allenai/olmocr-2-7b           # identificador del modelo en el servidor LLM
DATOS_DIR=./datos                        # directorio con los archivos de entrada
LOGS_DIR=./logs                          # directorio donde se guardan los logs
MAX_LONG_SIDE=1288                       # píxeles del lado largo al escalar imágenes
STREAM_CHUNK_TIMEOUT=300                 # segundos máximos de espera por respuesta
MAX_CONSECUTIVE_ERRORS=3                 # errores consecutivos antes de abortar
MAX_TOKENS=4096                          # tokens máximos de salida por página
DEBUG=false                              # activa el log de chunks SSE en crudo
```

**Nota:** el `.env` debe situarse junto al binario (`sys.executable`) cuando se usa el ejecutable compilado, o junto a `ocr.py` en desarrollo.

---

## Comandos principales

```bash
# Ejecutar el OCR (usa DATOS_DIR del .env o ./datos por defecto)
make ocr

# Equivalente directo
poetry run python ocr.py

# Pasar un directorio concreto
poetry run python ocr.py /ruta/a/mis/pdfs

# Compilar a ejecutable standalone (genera dist/llm-ocr/)
make executable
```

---

## Salida generada

- **Markdown**: un fichero `.md` por cada PDF o directorio de imágenes, guardado en el mismo directorio que la fuente. El nombre se deriva del nombre del fichero/directorio tras pasar por `slugify()`.
- **Logs**: un fichero `.txt` con timestamp en `LOGS_DIR` que registra toda la salida de la sesión (sin secuencias `\r` de progreso).

---

## Convenciones de código

- **Un solo fichero**: toda la lógica vive en `ocr.py`. No crear módulos adicionales sin motivo justificado.
- **Lazy-imports**: `fitz` y `httpx` se importan mediante los helpers `_fitz()` y `_httpx()` para acelerar el arranque y hacer el bundle de PyInstaller más eficiente.
- **Type hints**: usar anotaciones de tipo en todas las funciones públicas. Utilizar `TYPE_CHECKING` para importaciones que solo se usan en anotaciones.
- **Idioma del código**: español para comentarios, mensajes al usuario y docstrings (el proyecto está orientado a usuarios hispanohablantes).
- **Nombres internos**: funciones y variables privadas/internas con prefijo `_`.
- **Formato**: seguir el estilo estándar de Python (PEP 8). Sin linter configurado explícitamente; mantener consistencia con el código existente.

---

## Comportamiento de reanudación

El programa es **idempotente y reanudable**:

- Si el Markdown ya existe y está **completo** (todas las páginas presentes), el archivo se **salta**.
- Si está **incompleto**, se procesan primero los **huecos** (páginas que faltan dentro del rango ya procesado) y luego se continúa desde la última página guardada.
- No se re-procesa ninguna página ya existente en el Markdown.

Al modificar la lógica de escritura del Markdown, asegurarse de que `get_last_processed_page`, `get_missing_pages` e `_insert_page_into_markdown` siguen siendo consistentes entre sí.

---

## Límites y notas importantes

- El programa usa `termios`/`tty`/`select` (Unix only). **No es compatible con Windows** sin cambios.
- El LLM debe estar corriendo y accesible en `LLM_BASE_URL` antes de ejecutar el programa.
- Si `finish_reason == "length"`, el programa **dobla automáticamente** `MAX_TOKENS` y reintenta la misma página.
- La cancelación de una petición en curso se realiza cerrando el socket HTTP y enviando una petición `POST .../cancel` al servidor LLM (específico de LM Studio).
- Los metadatos YAML (bloque `---`) que el modelo puede incluir al inicio de la respuesta se eliminan automáticamente antes de escribir el Markdown.
