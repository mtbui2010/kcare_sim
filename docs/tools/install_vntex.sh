#!/bin/bash
# Install CTAN vntex (T5 Vietnamese encoding, font definitions, fonts) into the
# per-user TeX tree (TEXMFHOME, usually ~/texmf). Run once per account:
#     bash docs/tools/install_vntex.sh
# The system TeX Live has no vntex (texlive-lang-other not installed). After this,
# every LaTeX project of this account can use
#   \usepackage[T5]{fontenc}   with pdflatex.   No sudo, system TeX Live untouched.
set -e
URL=https://ftp.jaist.ac.jp/pub/CTAN/install/language/vietnamese/vntex.tds.zip
HOME_TEXMF=$(kpsewhich -var-value TEXMFHOME)
mkdir -p "$HOME_TEXMF" /tmp/vntex_tds
curl -sfL -o /tmp/vntex_tds/vntex.tds.zip "$URL"
unzip -oq /tmp/vntex_tds/vntex.tds.zip -d "$HOME_TEXMF"
rm -rf /tmp/vntex_tds
echo "installed into $HOME_TEXMF:"; ls "$HOME_TEXMF"; du -sh "$HOME_TEXMF"
