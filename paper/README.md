# Open-Set Bat Face Recognition — LaTeX Paper

## Building the PDF

### Option A — latexmk (recommended once texlive is installed)

```bash
cd paper/
make          # produces main.pdf
make clean    # remove build artefacts
```

### Option B — manual pdflatex + bibtex

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option C — tectonic (single binary, auto-downloads packages)

```bash
cd paper/
tectonic main.tex    # produces main.pdf in-place
```

Install tectonic on Ubuntu: `cargo install tectonic` or download the binary from
https://tectonic-typesetting.github.io.

## Installing texlive (Ubuntu/Debian)

```bash
sudo apt-get install -y \
  texlive-latex-recommended \
  texlive-latex-extra \
  texlive-pictures \
  texlive-fonts-recommended \
  texlive-science \
  latexmk
```

## Structure

```
paper/
  main.tex              root document
  preamble.tex          packages, TikZ/pgfplots setup, macros
  references.bib        BibTeX bibliography
  sections/             one .tex file per section (00–08)
  appendix/             A_architecture.tex, B_technology.tex
  figures/              (reserved for standalone figure fragments)
  Makefile
  README.md
```

## All figures are native LaTeX

No external image files are required.  All diagrams and charts are produced by TikZ
(architecture/pipeline diagrams) and pgfplots (bar charts, learning curves).  The
paper compiles to a fully self-contained PDF.
