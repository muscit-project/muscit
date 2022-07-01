# MUSCIT: Multiscale Ion Transport Simualtion 

This package aims at the simulation of ionic motion   at large time and length scales.  While local transfer rates are sampled from atomistic (ab initio) molecular dynamics  simulations,
the actual long term propagation of particles is acheived/done  probabilistically  among a (dynamically evolving) grid of lattice atoms.
Dependent on the secific type of ion and conduction mechanism, ion transport can be simulated via the application of a transition matrix (Markov chain) or via a Lattice Monte Carlo approach. 

Furthermore, this package provides a variety of tools, scripts and code snipets for a) basic analysis of trajectories, b)  creation of own analysis function and  c) analysis and visualization  of ionic motion. 



## How To
see `path/to/docs` for documention of the package
see `path/to/example` for a example calculation


## installation
clone this git repository and install the package with pipi using a virtual environment:
```
python3 -m venv ~/muscit_venv
source ~/muscit_venv/bin/activate
pip install --upgrade pip

git clone https://github.com/muscit-project/muscit.git 
pip install -e muscit 
```

