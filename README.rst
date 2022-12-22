###########################################
MUSCIT: Multiscale Ion Transport Simulation 
###########################################

.. image:: https://readthedocs.org/projects/muscit/badge/?version=latest
    :target: https://muscit.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

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
The package can be installed by invoking pip from the main directory. Performance critical parts can be compiled using pythran/transonic for considerable speed-up. To this end a Makefile is supplied::
    
    git clone https://github.com/muscit-project/muscit.git
    cd muscit
    make

Alternatively, you can call the contents of the Makefile by hand::

    git clone https://github.com/muscit-project/muscit.git
    cd muscit
    python3 -m pip install .
    python3 setup.py build_ext --build-lib=.

==========
References
==========

