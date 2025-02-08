.. causarray documentation master file, created by
   sphinx-quickstart on Mon Jan 13 17:38:13 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Overview
=======================


Ensemble Cross-validation is a Python package for performing specialized cross-validation on ensemble models, such as extrapolated cross-validation (ECV), generalized cross-validation (GCV), and etc. The implementation of ensemble models are based on scikit-learn.

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   readme_link.md


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Main functions

   main_function/gcate
   main_function/lfc


.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Tutorials

   tutorials/basics.ipynb
   tutorials/gcv.ipynb
   tutorials/cgcv_l1_huber.ipynb
   tutorials/multitask.ipynb
   tutorials/random_forests.ipynb
