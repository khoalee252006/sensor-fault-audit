r"""Build the Springer LNCS version (paper_lncs/, TAIMMA 2027 / Springer CCIS) from the IEEE version (paper_overleaf/).
Text, numbers and tables are copied unchanged; only layout is adapted for one column:
  figure*/table* -> figure/table, half-width figures -> 0.68 textwidth, tables in \scriptsize with tighter columns and
  scaled down only if still wider than the text block. paper_lncs/main.tex (title page, keywords, bib style) is kept
  as written by hand and never overwritten. Run after scripts/21_paper_numbers.py or any edit in paper_overleaf/.
usage: python scripts/28_to_lncs.py"""
import os, re, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC, DST = os.path.join(ROOT, "paper_overleaf"), os.path.join(ROOT, "paper_lncs")
SKIP = {"main.tex"}                                   # LNCS front matter is maintained by hand


def to_lncs(name, text):
    text = text.replace("figure*", "figure").replace("table*", "table")
    text = text.replace(r"width=\columnwidth", r"width=0.68\textwidth")
    # Fig. 4: one-column version with larger text (scripts/29_paper_figures_embed.py)
    text = text.replace(r"width=\textwidth]{sc_line}", r"width=\textwidth]{sc_line_lncs}")
    text = text.replace(r"$\Delta = \CHtwo$", r"\mbox{$\Delta = \CHtwo$}")   # no line break after "="
    text = text.replace(r"width=0.68\textwidth]{sc_contrast}", r"width=0.55\textwidth]{sc_contrast}")
    if name.startswith("tab_"):                       # supplement tables stay as they are (article class)
        text = text.replace(r"\centering\footnotesize", r"\centering\scriptsize")
        text = re.sub(r"\\setlength\{\\tabcolsep\}\{\d+pt\}", r"\\setlength{\\tabcolsep}{3pt}", text)
        text = re.sub(r"\\begin\{tabular\}(\{[^}]*\})", r"\\begin{adjustbox}{max width=\\textwidth}\\begin{tabular}\1", text)
        text = text.replace(r"\end{tabular}", r"\end{tabular}\end{adjustbox}")
    return text


def main():
    os.makedirs(DST, exist_ok=True)
    for name in sorted(os.listdir(SRC)):
        src = os.path.join(SRC, name)
        if name.endswith(".tex") and name not in SKIP:
            text = open(src, encoding="utf-8").read()
            open(os.path.join(DST, name), "w", encoding="utf-8", newline="\n").write(to_lncs(name, text))
        elif name == "refs.bib":                      # Springer style: DOI only when an entry has one
            entries = open(src, encoding="utf-8").read().split("\n@")
            entries = [e if not re.search(r"^\s*doi\s*=", e, re.M) else
                       re.sub(r"^\s*url\s*=.*\n", "", e, flags=re.M) for e in entries]
            open(os.path.join(DST, name), "w", encoding="utf-8", newline="\n").write("\n@".join(entries))
        elif name == "figs":
            shutil.copytree(src, os.path.join(DST, "figs"), dirs_exist_ok=True)
    print("paper_lncs/ updated from paper_overleaf/ (main.tex kept)")


if __name__ == "__main__":
    main()
