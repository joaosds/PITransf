# Transformer in variational bases for electronic quantum states

This repository contains all the scripts used to generate and analyze the data in the paper "Transformer in variational bases for electronic quantum states". 

arxiv: -


Authors: João Augusto Sobral (University of Stuttgart), Michael Perle (University of Innsbruck) and Mathias S. Scheurer (University of Stuttgart).

If you have any questions please reach out by joao.sobral@itp3.uni-stuttgart.de.

arXiv link: [arXiv:2208.01758](https://arxiv.org/abs/2208.01758)

This repository contains code adapted from the [Yuanhangzhang98/transformer_quantum_state](https://github.com/yuanhangzhang98/transformer_quantum_state).

João is thankful for insightful discussions with Yuan-Hang Zhang (also for making his code for the transformer_quantum_state open_source), Sayan Banerjee, Lucas Pupim, Vitor Dantas and Michael Muehlbauer.

# Things to do 

- Implement reduced loss.  [Done]
- Dynamical cutting of trees.  [Done]
- Vectorize Hnl and Hloc a bit more  [Done]
- Clean up the code 
- Fix the contribution of occupation to each extreme-basis  
- Add option for the energies of reference for the chiral and band basis.
- Learn how to use the saved model
- Make attention maps plot 
- Add the option for different samplers 
- Remove the copy flags. 
- Implement dropout of states that do not contribute up to some percentage on the sampler. 
- Implement division of batches for memory efficiency [Done]
- Check one more time if there are any typos on the occupation observable
- Add a final function that calculates the energy and occupation at the end 
- Test effect of directly cutting the HF from the tree-sampler
- Final optimization of the code 
- Fix the detuned case + obtaining the energy of the polarized state for chiral and band basis
- Add an option to start training from a model file on the main file + script. Save a backup of the model every 500 epochs.
- Add option to either train from previous result (if there is one) or start from scratch.

- Test paper from markus heyl, stochastic reconfiguration and linear algebra trick from becca.
- Play around with other embeddings
- also think about the attention mechanism itself
