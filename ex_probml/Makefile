CONVERT = jupyter nbconvert --to html
SRCS1 = $(wildcard coursework*.ipynb)

%.html: %.ipynb
	$(CONVERT) $*

all: $(SRCS1:.ipynb=.html) 

