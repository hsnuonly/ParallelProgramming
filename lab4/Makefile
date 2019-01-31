NVCCFLAGS  = -O2 -std=c++11 -Xptxas=-v -arch=sm_61 --default-stream per-thread
NVCC     = nvcc 
LDFLAGS = -lpng -lz

TARGETS = sobel

.PHONY: all
all: $(TARGETS)

sobel: sobel.cu
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
