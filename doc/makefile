# from https://tex.stackexchange.com/a/40759/297129


.PHONY: compbiol.pdf all clean

all: compbiol.pdf

%.tex: %.raw
	./raw2tex $< > $@

%.tex: %.dat
		./dat2tex $< > $@

compbiol.pdf: compbiol.tex
		latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make compbiol.tex

clean:
	latexmk -CA