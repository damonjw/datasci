CONVERT_IPYNB = jupyter nbconvert --to html

# Note: old pandoc (e.g. version 2.9) doesn't embed mathjax properly.
CONVERT_MD = pandoc --from=markdown+backtick_code_blocks+smart --to=html5 --standalone --mathjax --css=pandoc.css

all: ex1.html pytorch.html climate.html particlefilter.html

ex1.html: ex1.ipynb
	$(CONVERT_IPYNB) ex1.ipynb

climate.html: climate.ipynb
	$(CONVERT_IPYNB) climate.ipynb

particlefilter.html: particlefilter.ipynb
	$(CONVERT_IPYNB) particlefilter.ipynb

pytorch.html: pytorch.md
	$(CONVERT_MD) -o pytorch.html pytorch.md

