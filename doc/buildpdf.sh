xelatex mt_inversion.tex
bibtex mt_inversion.aux
xelatex mt_inversion.tex
xelatex mt_inversion.tex
#dvipdfm mt_inversion.dvi

rm -rf *.aux
rm -rf *.dvi
rm -rf *.log
rm -rf *.toc
rm -rf *.bbl
rm -rf *.blg
rm -rf *.out
rm -rf *.fff
rm -rf *.lof
rm *~

evince mt_inversion.pdf &
