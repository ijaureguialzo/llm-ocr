#!make

help: _header
	${info }
	@echo Opciones:
	@echo -----------------------
	@echo ocr
	@echo executable
	@echo -----------------------

_header:
	@echo -------
	@echo LLM OCR
	@echo -------

ocr:
	@poetry run python ocr.py

executable:
	@poetry install --with dev
	@poetry run pyinstaller \
    --onedir \
    --noconfirm \
    --name llm-ocr \
    --exclude-module tkinter \
    --exclude-module _tkinter \
    --collect-all fitz \
    --collect-all pymupdf \
    ocr.py
	@echo ""
	@echo "✅ Ejecutable listo en: dist/llm-ocr/llm-ocr"
	@echo "   Para distribuir, copia la carpeta dist/llm-ocr/ completa."
