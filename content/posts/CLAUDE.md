---
title: "Claude Reference"
date: 2025-08-22T13:13:51.346Z
draft: false
tags: []
---
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a research workspace containing multiple machine learning and high-performance computing projects, with emphasis on:

- **Machine Learning Research**: LLM quantization, mixed precision computing, weight decay studies
- **High-Performance Computing**: CUDA programming, MPI, parallel computing
- **Academic Projects**: Course materials, homework assignments, and research experiments
- **Personal Documents**: Academic credentials, research papers, and personal files

## Key Projects Structure

### Core Research Projects

**`/research/mustafar/`** - LLM inference optimization with sparse KV cache pruning
**`/unstructured_mixture_precision_demo/`** - Mixed precision FP16/FP8 computing library  
**`/research/mix_weight_decay/`** - Weight decay regularization studies
**`/research/mnist/`** - MNIST optimization experiments
**`/research/weight_norm_research/`** - Weight normalization analysis

### Development Areas

**`/cuda/`** - CUDA kernel development (int8 GEMM, tensor cores)
**`/temp/`** - CUDA practice and experimental kernels
**`/mpi/`** - MPI parallel programming exercises
**`/parallel_programming/`** - OpenMP, CUDA, MPI implementations

### Academic Materials

**`/archive/`** - Course materials (MSBD 5001, 5002, 5003, 5004, 5007, 5014, 5017, 5018)
**`/comp6211j/`** - Advanced machine learning systems course materials

## Development Commands

### Python Projects

For Python-based research projects, use these patterns:

```bash
# Install dependencies
pip install -r requirements.txt

# Development installation (for libraries)
pip install -e .

# Run tests (when available)
pytest
pytest --cov=project_name

# Code formatting and quality
black .
mypy .
flake8 .
isort .
```

### CUDA Development

```bash
# Compile CUDA kernels
nvcc -o program kernel.cu
nvcc -arch=sm_70 -o optimized_program kernel.cu

# For projects with Makefiles
cd kernel/build
make -j8
```

### C++/MPI Projects

```bash
# MPI compilation
mpicc -o program source.c
mpicxx -o program source.cpp

# CMake projects
mkdir build && cd build
cmake ..
make -j8

# Run MPI programs
mpirun -np 4 ./program
```

### Research Project Specific Commands

**Mustafar (LLM KV Cache Pruning)**:
```bash
# Install dependencies and build kernels
pip install -r requirements.txt
cd kernel/build && make -j8
cd ../kernel_wrapper && pip install -e .

# Run LongBench evaluation
bash long_test.sh 0.7 0.7 meta-llama/Llama-2-7b-hf mustafar

# Evaluate results
python eval_long_bench.py --model Llama-2-7b-hf_4096_K_0.7_V_0.7
```

**Mixed Precision Library**:
```bash
# Install and test
pip install -e .
pytest --cov=mixed_precision_lib

# Run CLI tools
mixed-precision-benchmark conversion --size 128 --threshold 2.0
mixed-precision-demo
```

## Architecture Patterns

### Research Project Structure

Most research projects follow this pattern:
- **Main implementation** in root directory (.py files)
- **Requirements** specified in requirements.txt or pyproject.toml
- **Scripts** for running experiments (.sh files)
- **Data/Results** in separate directories
- **README.md** with detailed setup and usage instructions

### CUDA Project Structure

CUDA projects typically have:
- **Kernel implementation** (.cu files)
- **Host code** (.cpp/.c files) 
- **Build scripts** (Makefile or CMakeLists.txt)
- **Compilation flags** for specific GPU architectures

### Python Library Structure

Well-structured Python libraries follow:
```
project_name/
├── project_name/          # Main package
│   ├── core/             # Core functionality
│   ├── utils/            # Utilities
│   └── cli/              # Command-line tools
├── tests/                # Test suite
├── examples/             # Usage examples
├── requirements.txt      # Dependencies
└── pyproject.toml        # Build configuration
```

## Testing Patterns

### Python Testing
- Use **pytest** as the primary testing framework
- Test files follow `test_*.py` or `*_test.py` naming
- Coverage reporting with `pytest-cov`
- Separate unit, integration, and benchmark tests

### CUDA Testing
- Build separate test executables
- Use simple assertions for correctness
- Compare with CPU reference implementations
- Test different input sizes and edge cases

## Development Environment Notes

### GPU Computing
- Projects expect NVIDIA GPUs with CUDA 12.x+
- Many experiments use specific architectures (sm_70, sm_80)
- CUDA kernels often require compilation with appropriate compute capability flags

### Python Environment
- Most projects use Python 3.8+
- PyTorch-based machine learning projects
- Mixed precision and quantization libraries require specific PyTorch versions
- Virtual environments recommended for dependency isolation

### Academic Workflow
- Experiments often generate CSV files with results
- LaTeX documents for academic papers and reports
- Jupyter notebooks for data analysis and visualization
- Git repositories for version control (when initialized)

## Common Development Tasks

### Running Experiments
1. Check README.md for specific setup instructions
2. Install dependencies from requirements.txt
3. Run provided shell scripts for batch experiments
4. Monitor results in generated output directories

### Adding New Research Code
1. Follow existing project structure patterns
2. Add appropriate requirements.txt or pyproject.toml
3. Include README.md with setup and usage instructions
4. Add shell scripts for common operations

### CUDA Development
1. Start with simple kernels in `/temp/` for experimentation
2. Use appropriate compilation flags for target GPU architecture
3. Test against CPU reference implementations
4. Optimize for specific use cases (memory bandwidth, compute throughput)
