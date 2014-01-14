# Make sure you have installed the OPM core library and that it's visible to GCC.
# You might want to set CPLUS_INCLUDE_PATH and LIBRARY_PATH in your shell session
# as follows:
#
#     export CPLUS_INCLUDE_PATH=/opt/opm/include
#     export LIBRARY_PATH=/opt/opm/lib/x86_64-linux-gnu
#
# Likewise for the other dependencies (Boost, UMFPACK, LAPACK, BLAS, etc.)

LIBS = -lopmcore -lboost_filesystem -lboost_system -lboost_program_options -lumfpack -llapack -lblas

.PHONY: all
all:
	@echo "$$(tput bold; tput setaf 2)Building OPM-based blackoil simulator$$(tput sgr0)"
	g++ -std=c++11 -Wall -Wextra -pedantic -O3 -march=native simulator.cpp $(LIBS) -o simulator

.PHONY: clean
clean:
	rm -f simulator

.PHONY: help
help:
	@echo
	@echo "all: (default target)"
	@echo "     build the reservoir simulator"
	@echo
	@echo "clean:"
	@echo "     delete binaries"
	@echo
	@echo "help:"
	@echo "     print this message"
	@echo
