CONVERT_IPYNB = jupyter nbconvert --to html

# Note: old pandoc (e.g. version 2.9) doesn't embed mathjax properly.
CONVERT_MD = pandoc --from=markdown+backtick_code_blocks+smart --to=html5 --standalone --mathjax --css=pandoc.css

all: ex1.html pytorch.html

ex1.html: ex1.ipynb
	$(CONVERT_IPYNB) $*

pytorch.html: pytorch.md
	$(CONVERT_MD) -o pytorch.html pytorch.md

