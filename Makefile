include Makefile.conf

MEXSRC = mex
OUTDIR = util

all: cholinv cholsolve

cholinv: $(OUTDIR)/cholinv.$(MEXEXT)

cholsolve: $(OUTDIR)/cholsolve.$(MEXEXT)

$(OUTDIR)/%.$(MEXEXT): $(MEXSRC)/%.c
	$(MEX) $< CFLAGS="$(CFLAGS)" COPTIMFLAGS="$(COPTIMFLAGS)" $(MEXFLAGS) $(MEXLIBS) -output $@

.PHONY: clean

clean:
	rm $(OUTDIR)/*.$(MEXEXT)
