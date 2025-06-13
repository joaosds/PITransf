#!/bin/bash
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
nenclayer=${nenclayer:-1}           # Encoder layers
demb=${demb:-300}                   # Embedding dimension
sampler=${sampler:-"tree"}          # Sampler type
identifier=${identifier:-"default"} # Run identifier
cuda_device=${cuda_device:-0}       # CUDA device

# System limits
ed_limit=${ed_limit:-11}            # ED calculation limit for system size

# Virtual environment name (single environment for all components)
venv_name=${venv_name:-"pitransf"}

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

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found. Please install Python 3.8+."
        exit 1
    fi
    
    local python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_info "Python version: $python_version"
}

check_venv() {
    local venv_path="$1"
    if [[ ! -d "$venv_path" ]]; then
        print_error "Virtual environment not found at: $venv_path"
        echo "Please create it with: python3 -m venv $venv_name"
        echo "Then install dependencies with: pip install -r requirements.txt"
        exit 1
    fi
}

activate_venv() {
    local venv_path="$1"
    print_info "Activating virtual environment: $venv_path"
    
    # Determine activation script based on OS
    local activate_script
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        activate_script="$venv_path/Scripts/activate"
    else
        # Linux/Mac
        activate_script="$venv_path/bin/activate"
    fi
    
    if [[ ! -f "$activate_script" ]]; then
        print_error "Activation script not found: $activate_script"
        exit 1
    fi
    
    # Source the activation script
    source "$activate_script"
    
    # Verify activation
    if [[ "$VIRTUAL_ENV" == "$venv_path" ]]; then
        print_success "Virtual environment activated successfully"
        print_info "Using Python: $(which python3)"
    else
        print_error "Failed to activate virtual environment"
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
VENV_PATH="$PROJECT_ROOT/$venv_name"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

print_header "Project Setup"
print_success "Project root: $PROJECT_ROOT"
print_success "Results directory: $RESULTS_DIR"
print_success "Virtual environment: $VENV_PATH"

# Verify required directories exist
for dir in "$HF_DIR" "$TRANSFORMER_DIR"; do
    if [[ ! -d "$dir" ]]; then
        print_error "Required directory not found: $dir"
        exit 1
    fi
done

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

setup_environment() {
    print_header "Environment Setup"
    
    check_python
    check_venv "$VENV_PATH"
    activate_venv "$VENV_PATH"
    
    # Verify key packages are installed
    print_info "Verifying package installation..."
    python3 -c "
import sys
try:
    import torch
    import numpy
    import scipy
    import matplotlib
    print('✓ Core packages verified')
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'✗ Missing package: {e}')
    print('Please install requirements: pip install -r requirements.txt')
    sys.exit(1)
" || exit 1
}

# =============================================================================
# HARTREE-FOCK CALCULATIONS
# =============================================================================

run_hartree_fock() {
    local mode=$1  # "complete", "energy", or "detuned"
    
    print_header "Hartree-Fock Calculation ($mode mode)"
    
    # Set up Python path for HF modules
    export PYTHONPATH="${PYTHONPATH}:$HF_DIR/masterproject-develop/"
    cd "$HF_DIR/masterproject-develop/"
    
    print_info "Current directory: $(pwd)"
    print_info "Running HF calculation with parameters: t=$t, N=$N"
    print_info "Command: python3 -u HartreeFock/HF_main.py $t $N $TRANSFORMER_DIR $mode $accuracy"
    
    # Save HF results to transformer directory where they're expected
    if python3 -u HartreeFock/HF_main.py "$t" "$N" "$TRANSFORMER_DIR" "$mode" "$accuracy"; then
        print_success "HF calculation completed"
    else
        print_error "HF calculation failed with exit code $?"
        return 1
    fi
    
    # Generate basis plots/data (also save to transformer directory)
    print_info "Generating HF basis data"
    if python3 -u Plotting/plot_HF.py "$TRANSFORMER_DIR" "$N" "$t" > /dev/null 2>&1; then
        print_success "HF basis data generated"
    else
        print_info "HF basis data generation failed (non-critical)"
    fi
    
    # Return to project root
    cd "$PROJECT_ROOT"
}

run_basis_energy() {
    local basis_type=$1
    
    print_header "Energy Calculation for $basis_type Basis"
    
    export PYTHONPATH="${PYTHONPATH}:$HF_DIR/masterproject-develop/"
    cd "$HF_DIR/masterproject-develop/"
    
    print_info "Current directory: $(pwd)"
    print_info "Command: python3 -u Supplementary/energy_of_given_wf2.py $t $N $TRANSFORMER_DIR $basis_type"
    
    # Save results to transformer directory
    if python3 -u Supplementary/energy_of_given_wf2.py "$t" "$N" "$TRANSFORMER_DIR" "$basis_type"; then
        print_success "$basis_type basis energy calculated"
    else
        print_error "$basis_type basis energy calculation failed with exit code $?"
        return 1
    fi
    
    # Return to project root
    cd "$PROJECT_ROOT"
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
    
    print_info "Current directory: $(pwd)"
    print_info "Command: python3 -u Supplementary/exactDiagFermionsExtended.py $t $N $TRANSFORMER_DIR"
    
    # Save results to transformer directory
    if python3 -u Supplementary/exactDiagFermionsExtended.py "$t" "$N" "$TRANSFORMER_DIR"; then
        print_success "ED calculation completed"
    else
        print_error "ED calculation failed with exit code $?"
        return 1
    fi
    
    # Return to project root
    cd "$PROJECT_ROOT"
}

# =============================================================================
# TRANSFORMER TRAINING/EVALUATION
# =============================================================================

run_transformer() {
    local mode=$1  # "train" or "eval"
    
    print_header "Transformer $mode"
    
    cd "$TRANSFORMER_DIR"
    print_info "Current directory: $(pwd)"
    
    # Export the results directory so the Python script can use it
    export RESULTS_DIR="$RESULTS_DIR"
    
    # Verify HF files are present
    print_info "Checking for HF data files..."
    if [[ -f "enhf.txt" && -f "ed.npy" ]]; then
        print_success "HF data files found in transformer directory"
        print_info "enhf.txt content: $(cat enhf.txt 2>/dev/null || echo 'Could not read')"
        print_info "ed.npy exists: $(ls -la ed.npy 2>/dev/null || echo 'File not found')"
    else
        print_info "HF data files not found, transformer will use defaults"
        print_info "Files in transformer directory:"
        ls -la . | head -10
    fi
    
    local cmd_args=("$t" "$N" "$basis" "$niter" "$nbatch" "$nunique" "$identifier" "$sec_batch" "$sampler" "$cuda_device" "$demb" "$nhead" "$nenclayer" "$RESULTS_DIR")
    
    if [[ "$mode" == "train" ]]; then
        print_info "Command: python3 -u main.py ${cmd_args[*]}"
        if python3 -u main.py "${cmd_args[@]}"; then
            print_success "Transformer training completed"
        else
            print_error "Transformer training failed with exit code $?"
            return 1
        fi
    elif [[ "$mode" == "eval" ]]; then
        print_info "Command: python3 -u evalmain.py ${cmd_args[*]}"
        if python3 -u evalmain.py "${cmd_args[@]}"; then
            print_success "Transformer evaluation completed"
        else
            print_error "Transformer evaluation failed with exit code $?"
            return 1
        fi
    else
        print_error "Invalid transformer mode: $mode"
        return 1
    fi
    
    # Return to project root
    cd "$PROJECT_ROOT"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

print_info "Mode: $option"
print_info "System size: N=$N, Hopping: t=$t, Basis: $basis"
print_info "HF enabled: $hf_enabled, Detuned: $detuned"

# Setup environment once for all operations
setup_environment

# Track execution flow
execution_successful=true

# Only run physics calculations in train mode
if [[ "$option" == "train" ]]; then
    
    if [[ "$hf_enabled" == true ]]; then
        
        if [[ "$detuned" == true ]]; then
            # Detuned basis workflow
            print_info "Using detuned HF basis workflow"
            
            if ! run_hartree_fock "detuned"; then
                print_error "Detuned HF calculation failed"
                execution_successful=false
            elif ! run_hartree_fock "energy"; then  # Follow up with energy calculation
                print_error "Energy calculation failed"
                execution_successful=false
            fi
            
        else
            # Standard workflow based on basis type
            case "$basis" in
                "hf")
                    if ! run_hartree_fock "complete"; then
                        print_error "HF calculation failed"
                        execution_successful=false
                    fi
                    ;;
                "chiral"|"band")
                    if ! run_basis_energy "$basis"; then
                        print_error "Basis energy calculation failed"
                        execution_successful=false
                    fi
                    ;;
                *)
                    print_error "Invalid basis type: $basis. Valid options: hf, chiral, band"
                    exit 1
                    ;;
            esac
        fi
        
        # Only run ED if previous steps succeeded
        if [[ "$execution_successful" == true ]]; then
            if ! run_exact_diagonalization; then
                print_error "Exact diagonalization failed"
                execution_successful=false
            fi
        else
            print_info "Skipping ED due to previous failures"
        fi
        
    else
        print_info "Hartree-Fock calculations disabled"
    fi
    
    # Run transformer training (even if physics calculations failed, in case using defaults)
    if [[ "$execution_successful" == true ]] || [[ "$hf_enabled" == false ]]; then
        if ! run_transformer "train"; then
            print_error "Transformer training failed"
            execution_successful=false
        fi
    else
        print_error "Skipping transformer training due to previous failures"
    fi
    
elif [[ "$option" == "eval" ]]; then
    # Evaluation mode - only run transformer
    if ! run_transformer "eval"; then
        print_error "Transformer evaluation failed"
        execution_successful=false
    fi
    
else
    print_error "Invalid option: $option. Valid options: train, eval"
    exit 1
fi

if [[ "$execution_successful" == true ]]; then
    print_success "All operations completed successfully!"
    print_info "Results saved in: $RESULTS_DIR"
else
    print_error "Some operations failed. Check the output above for details."
    exit 1
fi
