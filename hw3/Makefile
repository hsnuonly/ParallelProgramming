CC = mpicc
CXX = mpicxx 
LDLIBS = -pthread
CFLAGS = -O3 -fopenmp -g
CXXFLAGS = -O3 -fopenmp -g -std=c++14 
TARGETS = sssp apsp partition

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
