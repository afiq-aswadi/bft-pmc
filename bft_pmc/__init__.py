"""Predictive Monte Carlo (PMC) analysis of Bayes-filtered transformers.

Research code for *What does a Bayes-filtered transformer believe? A
predictive Monte Carlo approach* (NeurIPS 2026). The package is organised by
task family, with the pieces the families share kept at the top level:

``bft_pmc.linear_regression``
    In-context linear regression with a finite Gaussian task pool.
``bft_pmc.balls_and_urns``
    Balls and urns (Dirichlet-multinomial) with a finite task pool, plus the
    continuous-prior Beta-Bernoulli experiment.
``bft_pmc.markov``
    Finite-state 1-Markov chains with a finite pool of transition matrices.
``bft_pmc.analysis``
    Cross-family analyses: checkpoint and rollout I/O, the PMC reliability
    diagnostics (path stability, recovery against rollout length, count and
    dimension, next-token fidelity) and the reliability-appendix figures.
``bft_pmc.plotting``
    Shared paper typography and marginal-density rendering.
``bft_pmc.metrics``
    Distribution distances: energy distance, sliced Wasserstein, symmetrised KL.
``bft_pmc.pos_encoding``
    The learned / rotary / no positional-encoding variants.

Every runnable module is reachable through the repository's ``train.py`` and
``eval.py``; see :mod:`bft_pmc.cli` for the full command list.
"""

__version__ = "0.1.0"
