#!/bin/bash

# Note: This file is unfortunately necessary since both parts of the project were developed independently. 
# A better fix in the future is actually having everything in one single main python file

# Export path for running Michael's codes
# 
hfoption="no"
N=6
t=0.06
pathsavefiles="/Users/jass/Documents/oldlinux/phd/projects/transf/final/transformer_quantum_state/"


if [ "$hfoption" == "yes" ]; then
  source /Users/jass/miniconda3/bin/activate perle
  export PYTHONPATH="${PYTHONPATH}:/Users/jass/Documents/oldlinux/phd/projects/perle2/masterproject-develop/"
  cd /Users/jass/Documents/oldlinux/phd/projects/perle2/masterproject-develop/

  # Calculate ED # t N as arguments
  echo "-------------"
  python HartreeFock/HF_main.py  $t $N $pathsavefiles

  echo "-------------"

  # Generate Hartree-Fock-basis for occupation
  python Plotting/plot_HF.py $pathsavefiles > /dev/null 2>&1


  echo "Calculated Nk in HF basis" 

  echo "-------------"
  # Calculate EHF
  python Supplementary/exactDiagFermionsExtended.py $t $N $pathsavefiles

  echo "-------------"
  echo "Calculated Ehf energy"
fi

# Transformers , give which basis you want computations

echo "-------------"
echo "Transformers stuff"
basis="band" # chiral, band or hf
niter=1
nbatch=100000
nunique=64
pathtransf="/Users/jass/Documents/oldlinux/phd/projects/transf/final/transformer_quantum_state"


source /Users/jass/miniconda3/bin/activate transf
cd "$pathtransf"
echo "entered folder $pathtransf"
python main.py $t $N $basis $niter $nbatch $nunique
