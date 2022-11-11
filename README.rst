###########################################
MUSCIT: Multiscale Ion Transport Simulation 
###########################################

This package aims to simulate ion motion on large time and length scales.  While local transfer rates are sampled from atomistic (ab initio) molecular dynamics simulations,
the actual long-term propagation of particles is carried out probabilistically among a (dynamically evolving) grid of lattice atoms.
Depending on the ion type and the conduction mechanism, ion transport can be simulated by applying a transition matrix (Markov chain) or by a Lattice Monte Carlo approach.

In addition, this package provides a variety of tools, scripts, and code snippets for a) basic analysis of trajectories, b) creation of custom analysis functions, and c) analysis and visualization of ion motion.

======
How To
======
see ``docs/`` for documentation
see ``example/`` for a example calculation


============
Installation
============
clone this git repository and install the package with pip using a virtual environment::

    python3 -m venv ~/muscit_venv
    source ~/muscit_venv/bin/activate
    pip install --upgrade pip

    git clone https://github.com/muscit-project/muscit.git 
    pip install -e muscit 

