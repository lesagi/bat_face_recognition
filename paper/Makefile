# -------------------------------------------------------
#  Makefile for the bat face recognition paper
# -------------------------------------------------------
MAIN   = main
PDF    = $(MAIN).pdf

# tectonic is a self-contained compiler (auto-downloads packages)
# fallback: latexmk -pdf $(MAIN).tex
LATEX  = tectonic

.PHONY: all clean

all: $(PDF)

$(PDF): $(MAIN).tex preamble.tex references.bib \
        sections/*.tex appendix/*.tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f $(PDF) $(MAIN).xdv $(MAIN).aux $(MAIN).log \
	      $(MAIN).out $(MAIN).toc $(MAIN).bbl $(MAIN).blg \
	      $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).synctex.gz
