# kaair_sim docs

Beginner guide (Vietnamese) to the kaair sim → dataset → VLA pipeline.

```
docs/
├── tex/      the book: main.tex + chapters/*.tex   (only \includegraphics, no drawing code)
├── figs/     every figure the book uses: sim captures (*.png) and diagrams exported from draws/ (*.pdf)
├── draws/    TikZ sources of the diagrams (one standalone file each, shared style in _style.tex)
├── tools/    install_vntex.sh — Vietnamese (T5) support for pdflatex, once per account
├── demo/     kaair_demo.ipynb — step-by-step train + run demo for VS Code web (see demo/README.md)
└── Makefile  make → figs/*.pdf from draws/, then kaair_sim_guide.pdf
```

Build with pdflatex (TeX Live):

```bash
bash docs/tools/install_vntex.sh   # once per account: vntex -> ~/texmf (already done for trung@A6000)
cd docs && make                    # or: make figs (diagrams only)
cd docs/tex && latexmk -pdf main.tex   # the book alone (e.g. from an editor), once figs/ is built
```

Add a figure: sim capture → drop the PNG in `figs/`; diagram → new `draws/<name>.tex` (copy an existing
one for the header), `make figs`, then `\includegraphics{<name>.pdf}` in a chapter.
