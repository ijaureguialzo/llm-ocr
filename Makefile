#!make

help: _header
	${info }
	@echo Opciones:
	@echo -----------------------
	@echo ocr
	@echo -----------------------

_header:
	@echo -------
	@echo LLM OCR
	@echo -------

ocr:
	@poetry run python ocr.py
