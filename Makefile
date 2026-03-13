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
    --onefile \
    --name llm-ocr \
    --collect-all fitz \
    --collect-all pymupdf \
    ocr.py
	@echo ""
	@echo "✅ Ejecutable generado en: dist/llm-ocr"
