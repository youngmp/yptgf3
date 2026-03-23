dat should contain all necessary data files for figure generation.

To obtain agent-based model simulations, I used agents.py on a cluster to run a bunch of simulations in parallel.

When using xppaut on fp_equal.ode, use CVODE with atol and rtol set to at most 2e-5. Set parameters correctly for each panel. In particular, we need ze large enough for there to exist a middle branch (like ze=8). Set numerics so that ds=0.02, dsmin=0.00001, dsmax=0.05, parmin=0, parmax=15.