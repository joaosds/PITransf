#!/bin/bash

# Quantum State Transformer - Main Execution Script
# This script orchestrates Hartree-Fock calculations, Exact Diagonalization, 
# and Transformer training for quantum many-body systems

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default configuration - can be overridden by config file or command line
CONFIG_FILE="${CONFIG_FILE:-config.sh}"

# Load configuration if it exists
if [[ -f "$CONFIG_FILE" ]]; then
    source "$CONFIG_FILE"
    echo "✓ Loaded configuration from $CONFIG_FILE"
else
    echo "ℹ No config file found at $CONFIG_FILE, using defaults"
fi

# Core parameters (with defaults)
N=${N:-6}                           # System size
t=${t:-0.09}                        # Hopping parameter
basis=${basis:-"hf"}                # Basis type: hf, chiral, or band
option=${option:-"train"}           # Mode: train or eval
hf_enabled=${hf_enabled:-true}      # Enable Hartree-Fock calculations
detuned=${detuned:-false}          # Use detuned HF basis
accuracy=${accuracy:-1e-10}         # HF convergence threshold

# Transformer parameters
niter=${niter:-20000}               # Training iterations
nbatch=${nbatch:-100000000}         # Batch size
nunique=${nunique:-17000}           # Unique samples
sec_batch=${sec_batch:-20}          # Secondary batch size
nhead=${nhead:-1}                   # Attention heads
ndeclayer=${ndeclayer:-1}           # Decoder layers
demb=${demb:-300}                   # Embedding dimension
sampler=${sampler:-"tree"}          # Sampler type
identifier=${identifier:-"default"} # Run identifier
cuda_device=${cuda_device:-0}       # CUDA device

# System limits
ed_limit=${ed_limit:-11}            # ED calculation limit for system size

# Environment names (can be overridden)
env=${env:-"pitransf"}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_colored() {
    local color=$1
    local text=$2
    printf "\033[${color}m${text}\033[0m\n"
}

print_header() {
    local text=$1
    printf "\n"
    print_colored "1;34" "===== $text ====="
    printf "\n"
}

print_success() {
    print_colored "1;32" "✓ $1"
}

print_info() {
    print_colored "1;33" "ℹ $1"
}

print_error() {
    print_colored "1;31" "✗ ERROR: $1" >&2
}

check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "conda not found. Please install Anaconda/Miniconda."
        exit 1
    fi
}

check_env() {
    local env_name=$1
    if ! conda info --envs | grep -q "^$env_name "; then
        print_error "Conda environment '$env_name' not found."
        echo "Please create it or check the environment name in your config."
        echo "Available environments:"
        conda info --envs
        exit 1
    fi
}

activate_env() {
    local env_name=$1
    print_info "Activating conda environment: $env_name"
    
    # Initialize conda for bash (if not already done)
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    
    if ! conda activate "$env_name" 2>/dev/null; then
        print_error "Failed to activate environment '$env_name'"
        exit 1
    fi
}

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Define paths relative to project root
HF_DIR="$PROJECT_ROOT/hartree-fock"
TRANSFORMER_DIR="$PROJECT_ROOT/transformer_quantum_state/"
RESULTS_DIR="$PROJECT_ROOT/results"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

print_header "Project Setup"
print_success "Project root: $PROJECT_ROOT"
print_success "Results directory: $RESULTS_DIR"

# Verify required directories exist
for dir in "$HF_DIR" "$TRANSFORMER_DIR"; do
    if [[ ! -d "$dir" ]]; then
        print_error "Required directory not found: $dir"
        exit 1
    fi
done

# =============================================================================
# HARTREE-FOCK CALCULATIONS
# =============================================================================

run_hartree_fock() {
    local mode=$1  # "complete", "energy", or "detuned"
    
    print_header "Hartree-Fock Calculation ($mode mode)"
    
    check_env "$env"
    activate_env "$env"
    
    # Set up Python path for HF modules
    export PYTHONPATH="${PYTHONPATH}:$HF_DIR/masterproject-develop/"
    cd "$HF_DIR/masterproject-develop/"
    
    print_info "Running HF calculation with parameters: t=$t, N=$N"
    # Save HF results to transformer directory where they're expected
    python3 -u HartreeFock/HF_main.py "$t" "$N" "$TRANSFORMER_DIR" "$mode" "$accuracy"
    print_success "HF calculation completed"
    
    # Generate basis plots/data (also save to transformer directory)
    print_info "Generating HF basis data"
    python3 -u Plotting/plot_HF.py "$TRANSFORMER_DIR" "$N" "$t" > /dev/null 2>&1
    print_success "HF basis data generated"
}

run_basis_energy() {
    local basis_type=$1
    
    print_header "Energy Calculation for $basis_type Basis"
    
    check_env "$env"
    activate_env "$env"
    
    export PYTHONPATH="${PYTHONPATH}:$HF_DIR/masterproject-develop/"
    cd "$HF_DIR/masterproject-develop/"
    
    # Save results to transformer directory
    python3 -u Supplementary/energy_of_given_wf2.py "$t" "$N" "$TRANSFORMER_DIR" "$basis_type"
    print_success "$basis_type basis energy calculated"
}

run_exact_diagonalization() {
    if [[ $N -ge $ed_limit ]]; then
        print_info "Skipping ED: system size N=$N exceeds limit ($ed_limit)"
        return
    fi
    
    print_header "Exact Diagonalization"
    
    # Use same environment as HF
    export PYTHONPATH="${PYTHONPATH}:$HF_DIR/masterproject-develop/"
    cd "$HF_DIR/masterproject-develop/"
    
    # Save results to transformer directory
    python3 -u Supplementary/exactDiagFermionsExtended.py "$t" "$N" "$TRANSFORMER_DIR"
    print_success "ED calculation completed"
}

# =============================================================================
# TRANSFORMER TRAINING/EVALUATION
# =============================================================================

run_transformer() {
    local mode=$1  # "train" or "eval"
    
    print_header "Transformer $mode"
    
    check_env "$env"
    activate_env "$env"
    
    cd "$TRANSFORMER_DIR"
    
    # Export the results directory so the Python script can use it
    export RESULTS_DIR="$RESULTS_DIR"
    
    # Verify HF files are present
    if [[ -f "enhf.txt" && -f "ed.npy" ]]; then
        print_success "HF data files found in transformer directory"
    else
        print_info "HF data files not found, transformer will use defaults"
    fi
    
    if [[ "$mode" == "train" ]]; then
        python3 -u main.py "$t" "$N" "$basis" "$niter" "$nbatch" "$nunique" \
                "$identifier" "$sec_batch" "$sampler" "$cuda_device" \
                "$demb" "$nhead" "$nenclayer" "$RESULTS_DIR"
    elif [[ "$mode" == "eval" ]]; then
        python3 -u evalmain.py "$t" "$N" "$basis" "$niter" "$nbatch" "$nunique" \
                "$identifier" "$sec_batch" "$sampler" "$cuda_device" \
                "$demb" "$nhead" "$nenclayer" "$RESULTS_DIR"
    else
        print_error "Invalid transformer mode: $mode"
        exit 1
    fi
    
    print_success "Transformer $mode completed"
}

# =============================================================================
# =============================================================================

print_info "Mode: $option"
print_info "System size: N=$N, Hopping: t=$t, Basis: $basis"

# Check prerequisites
check_conda

# Only run physics calculations in train mode
if [[ "$option" == "train" ]]; then
    
    if [[ "$hf_enabled" == true ]]; then
        
        if [[ "$detuned" == true ]]; then
            # Detuned basis workflow
            print_info "Using detuned HF basis workflow"
            run_hartree_fock "detuned"
            run_hartree_fock "energy"  # Follow up with energy calculation
            run_exact_diagonalization
            
        else
            # Standard workflow based on basis type
            case "$basis" in
                "hf")
                    run_hartree_fock "complete"
                    ;;
                "chiral"|"band")
                    run_basis_energy "$basis"
                    ;;
                *)
                    print_error "Invalid basis type: $basis. Valid options: hf, chiral, band"
                    exit 1
                    ;;
            esac
            
            run_exact_diagonalization
        fi
    else
        print_info "Hartree-Fock calculations disabled"
    fi
    
    # Run transformer training
    run_transformer "train"
    
elif [[ "$option" == "eval" ]]; then
    # Evaluation mode - only run transformer
    run_transformer "eval"
    
else
    print_error "Invalid option: $option. Valid options: train, eval"
    exit 1
fi

print_info "Results saved in: $RESULTS_DIR"
