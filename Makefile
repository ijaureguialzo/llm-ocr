#!make

help: _header
	${info }
	@echo Opciones:
	@echo -----------------------
	@echo ocr
	@echo -----------------------

_header:
	@echo -----------------------------------
	@echo OCR de PDF a Markdown con LLM local
	@echo -----------------------------------

ocr:
	@poetry run python ocr.py
