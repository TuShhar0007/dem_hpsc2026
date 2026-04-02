# =============================================================
#  Makefile  –  DEM Solver (HPSC 2026 Assignment 1)
#
#  Targets
#  -------
#    make              →  build optimised serial binary
#    make omp          →  build OpenMP binary
#    make all          →  build both
#    make tests        →  run verification tests (serial)
#    make scaling      →  run full scaling study (OMP)
#    make plot         →  run Python verification & plot scripts
#    make clean        →  remove binaries and CSV outputs
#
#  Compiler flags
#  --------------
#    CXXFLAGS_BASE  – flags applied to every build
#    OMP_FLAG       – flag that enables OpenMP (GCC/Clang)
#
#  Usage
#  -----
#    make               # serial
#    make omp           # OpenMP (uses all available cores by default)
#    OMP_NUM_THREADS=4 ./dem_omp scaling
# =============================================================

CXX           := g++
SRC           := dem_solver.cpp
BIN_SERIAL    := dem_serial
BIN_OMP       := dem_omp

# Optimisation level (change to -O0 for debugging / profiling)
OPT           := -O2

CXXFLAGS_BASE := -std=c++17 $(OPT) -march=native -Wall -Wextra \
                  -Wno-unused-parameter

OMP_FLAG      := -fopenmp

# ──────────────────────────────────────────────────────────────
# Default target: serial
# ──────────────────────────────────────────────────────────────
.PHONY: serial
serial: $(BIN_SERIAL)

$(BIN_SERIAL): $(SRC)
	$(CXX) $(CXXFLAGS_BASE) -o $@ $<
	@echo "  Built serial binary: $@"

# ──────────────────────────────────────────────────────────────
# OpenMP target
# ──────────────────────────────────────────────────────────────
.PHONY: omp
omp: $(BIN_OMP)

$(BIN_OMP): $(SRC)
	$(CXX) $(CXXFLAGS_BASE) $(OMP_FLAG) -o $@ $<
	@echo "  Built OpenMP binary: $@"

# ──────────────────────────────────────────────────────────────
# Build both
# ──────────────────────────────────────────────────────────────
.PHONY: all
all: serial omp

# ──────────────────────────────────────────────────────────────
# Run verification tests (serial binary)
# ──────────────────────────────────────────────────────────────
.PHONY: tests
tests: serial
	@echo "\n=== Running verification tests ==="
	./$(BIN_SERIAL) tests

# ──────────────────────────────────────────────────────────────
# Run scaling study (OpenMP binary)
# ──────────────────────────────────────────────────────────────
.PHONY: scaling
scaling: omp
	@echo "\n=== Running scaling study ==="
	./$(BIN_OMP) scaling

# ──────────────────────────────────────────────────────────────
# Quick single-N runs for manual inspection
# ──────────────────────────────────────────────────────────────
.PHONY: run200 run1000 run5000
run200: omp
	./$(BIN_OMP) all 200

run1000: omp
	./$(BIN_OMP) all 1000

run5000: omp
	./$(BIN_OMP) all 5000

# ──────────────────────────────────────────────────────────────
# Generate publication-quality plots
# ──────────────────────────────────────────────────────────────
.PHONY: plot
plot:
	@echo "\n=== Generating figures ==="
	python3 verify_plots.py
	@echo "  Figures written to ./figures/"

# ──────────────────────────────────────────────────────────────
# Profiling build (gprof)
# ──────────────────────────────────────────────────────────────
.PHONY: profile
profile: $(SRC)
	$(CXX) $(CXXFLAGS_BASE) -pg -o dem_profile $<
	./dem_profile serial 1000
	gprof dem_profile gmon.out > profile_report.txt
	@echo "  Profile report: profile_report.txt"

# ──────────────────────────────────────────────────────────────
# Clean
# ──────────────────────────────────────────────────────────────
.PHONY: clean
clean:
	rm -f $(BIN_SERIAL) $(BIN_OMP) dem_profile gmon.out
	rm -f *.csv
	rm -rf figures/
	@echo "  Clean complete."
