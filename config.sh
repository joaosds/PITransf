#!/bin/bash
# =============================================================================
# PHYSICS PARAMETERS
# =============================================================================

# System size (number of particles/sites)
N=6

# Hopping parameter
t=0.18

# Basis type: "hf" (Hartree-Fock), "chiral", or "band"
basis="hf"

# Hartree-Fock convergence accuracy
accuracy=1e-10

# Use detuned HF basis (true/false)
detuned=false

# Enable Hartree-Fock calculations (set to false to skip HF and use existing data)
hf_enabled=true

# System size limit for Exact Diagonalization (ED becomes expensive for large N)
ed_limit=11

# =============================================================================
# TRANSFORMER PARAMETERS
# =============================================================================

# Training iterations
niter=100

# Batch size
nbatch=100000000

# Number of unique samples
nunique=17000

# Secondary batch size
sec_batch=20

# Number of attention heads
nhead=2

# Number of decoder layers
ndeclayer=2

# Embedding dimension
demb=300

# Sampler type
sampler="tree"

# Run identifier (for distinguishing different runs)
identifier="default_run"

# CUDA device number (0, 1, 2, etc.)
cuda_device=0

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

# Conda environment names
# Make sure these environments exist and have the required packages
venv_name="pitransf"

# =============================================================================
# EXECUTION MODE
# =============================================================================

# Execution mode: "train" or "eval"
option="train"
