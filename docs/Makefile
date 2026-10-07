# Build the guide with pdflatex (TeX Live).
#   make        diagrams (draws/*.tex -> figs/*.pdf) + book (tex/main.tex -> kaair_sim_guide.pdf)
#   make figs   diagrams only
#   make clean  remove build/ (keeps figs/ and the PDF)
# Vietnamese uses the T5 encoding from vntex; install it once per account with
#   bash tools/install_vntex.sh      (-> ~/texmf, no sudo)
DRAWS := $(filter-out draws/_style.tex,$(wildcard draws/*.tex))
FIGS  := $(patsubst draws/%.tex,figs/%.pdf,$(DRAWS))
TEX   := $(wildcard tex/*.tex tex/chapters/*.tex)
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

all: kaair_sim_guide.pdf
figs: $(FIGS)

figs/%.pdf: draws/%.tex draws/_style.tex
	@mkdir -p build/draws
	cd draws && $(LATEXMK) -outdir=../build/draws $*.tex >/dev/null
	cp build/draws/$*.pdf $@

kaair_sim_guide.pdf: $(FIGS) $(TEX) $(wildcard figs/*.png figs/*.jpg)
	cd tex && $(LATEXMK) -outdir=../build main.tex >/dev/null
	cp build/main.pdf $@

clean:
	rm -rf build

.PHONY: all figs clean
