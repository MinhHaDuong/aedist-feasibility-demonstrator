# Warning: make does not handle filenames containing the space character.

TEST_PDFS = $(wildcard test/*.pdf)
PROD_PDFS = $(wildcard prod/*.pdf)

TEST_MDS = $(TEST_PDFS:.pdf=.md)
PROD_MDS = $(PROD_PDFS:.pdf=.md)

.PHONY: all test prod clean

all: test prod

test: $(TEST_MDS)

prod: $(PROD_MDS)

%.md: %.pdf
	python3 pdfOCR2md.py "$<"

clean:
	rm -f $(TEST_MDS) $(PROD_MDS)

.PARALLEL: $(TEST_MDS) $(PROD_MDS)
