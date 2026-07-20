# Build the paper. Requires a TeX install with acmart.cls (TeX Live / MiKTeX / Overleaf).
#   make        -> main.pdf (latexmk if available, else manual pdflatex+bibtex passes)
#   make clean  -> remove build artifacts
MAIN=main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex refs.bib
	@if command -v latexmk >/dev/null 2>&1; then \
		latexmk -pdf -interaction=nonstopmode $(MAIN).tex; \
	else \
		pdflatex -interaction=nonstopmode $(MAIN).tex; \
		bibtex $(MAIN); \
		pdflatex -interaction=nonstopmode $(MAIN).tex; \
		pdflatex -interaction=nonstopmode $(MAIN).tex; \
	fi

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).fdb_latexmk $(MAIN).fls sections/*.aux

.PHONY: all clean
