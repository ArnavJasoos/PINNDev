"""PINN core: architectures, solvers, losses, training, and evaluation.

This layer is pure numerics (torch/numpy/scipy) with no LLM or agent dependency, so
it can be proven end-to-end on hand-written problems before any model is wired in.
"""
