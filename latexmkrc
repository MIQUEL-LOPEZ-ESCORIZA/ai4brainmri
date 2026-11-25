# Overleaf / latexmk configuration so the paper can live in paper/ while
# Overleaf compiles from the project root. Set the main document to
# paper/sn-article.tex in Overleaf (Menu -> Main document).
$do_cd = 1;
ensure_path('TEXINPUTS', './paper//');
ensure_path('BIBINPUTS', './paper//');
ensure_path('BSTINPUTS', './paper//');
$pdf_mode = 1;
