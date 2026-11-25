# Paper source

LaTeX source of the manuscript (Springer Nature `sn-jnl` template, `sn-nature` style) prepared for *Scientific Reports*.

* Main file: `sn-article.tex` (bibliography: `sn-bibliography.bib`, figures in `figures/`).
* **Revision highlighting.** Changes made for the Scientific Reports resubmission are marked in the PDF:
  blue = added text (`\rev{}`), red strikethrough = removed text (`\revdel{}`), magenta = notes for the authors (`\revnote{}`).
  Set `\highlightfalse` in the preamble to compile the clean version for submission.
  The commit "Add paper source as submitted to npj Digital Medicine" holds the original version, so `git diff` against it shows every change.
* **Overleaf.** Import the GitHub repository into Overleaf and set *Menu → Main document* to `paper/sn-article.tex`.
  The `latexmkrc` at the repository root makes the class, bibliography style and figures in `paper/` visible to the compiler.
