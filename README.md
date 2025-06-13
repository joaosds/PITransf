# Physics-informed Transformers for Electronic Quantum States

<p align="center">
<img src="logo.pdf" width=50% height=50%>
</p>

---

This repository contains all the scripts used to generate and analyze the data in the paper "Transformer in variational bases for electronic quantum states".


**Authors:** João Augusto Sobral (University of Stuttgart), Michael Perle (University of Innsbruck) and Mathias S. Scheurer (University of Stuttgart).

The main code is adapted from [Yuanhangzhang98/transformer_quantum_state](https://github.com/yuanhangzhang98/transformer_quantum_state) and [SOAP](https://github.com/nikhilvyas/SOAP).

João is thankful for insightful discussions with Yuan-Hang Zhang (also for making his code for the transformer_quantum_state open source).

If you have any questions please reach out by joao.sobral@itp3.uni-stuttgart.de.

---

- [Install from Github](#install-from-github)
- [Short Instructions](#short-instructions)
- [Citation](#citation)
- [License](#license)


## Install from Github

```bash
git clone https://github.com/your-username/quantum-state-transformer
cd quantum-state-transformer
```

### Environment Setup

Create the required conda environments:

```bash
# Create Hartree-Fock environment
conda env create -f environment_hf.yml

# Create Transformer environment  
conda env create -f environment_transformer.yml
```

## Short Instructions

### Basic Usage

Run with default parameters:
```bash
./main.sh
```

Run with custom parameters:
```bash
N=8 t=0.1 basis="hf" ./main.sh
```

Use a configuration file:
```bash
CONFIG_FILE=my_config.sh ./main.sh
```

### Configuration

Copy the template and modify:
```bash
cp config.sh my_config.sh
# Edit my_config.sh with your parameters
```

**Key Parameters:**
- `N`: System size (number of particles/sites)
- `t`: Hopping parameter  
- `basis`: Basis type (`"hf"`, `"chiral"`, or `"band"`)
- `niter`: Training iterations (default: 20000)
- `embedding_size`: Embedding dimension (default: 300)

### Project Structure

```
.
├── main.sh                          # Main execution script
├── config.sh                        # Configuration template
├── environment_hf.yml               # HF environment
├── environment_transformer.yml      # Transformer environment
├── hartree-fock/                    # HF calculation code
├── transformer_quantum_state/       # Transformer code
│   ├── main.py                      # Main transformer script
│   ├── plot.py                      # Plotting functions
│   ├── model.py                     # Transformer model
│   ├── Hamiltonian.py              # Hamiltonian definitions
│   └── optimizer.py                # Optimization routines
└── results/                         # Output directory
```


## Citation

If you use this code in your work, please cite the associated paper with:

```bibtex
@article{Sobral2024Dec,
	author = {Sobral, Jo{\ifmmode\tilde{a}\else\~{a}\fi}o Augusto and Perle, Michael and Scheurer, Mathias S.},
	title = {{Physics-informed Transformers for Electronic Quantum States}},
	journal = {arXiv},
	year = {2024},
	month = dec,
	eprint = {2412.12248},
	doi = {10.48550/arXiv.2412.12248}
}
```

In this case, please also cite the original Transformer Quantum State paper:

```bibtex
@article{Zhang2023Feb,
	author = {Zhang, Yuan-Hang and Di Ventra, Massimiliano},
	title = {{Transformer quantum state: A multipurpose model for quantum many-body problems}},
	journal = {Phys Rev B},
	volume = {107},
	number = {7},
	pages = {075147},
	year = {2023},
	month = feb,
	publisher = {American Physical Society},
	doi = {10.1103/PhysRevB.107.075147}
}
```
## License

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
