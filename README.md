dat should contain all necessary data files for figure generation. I apologize, it is very bloated -- there is on the order of 25MB of actual data spread across tens of thousands of small text files, which makes the file size on the order of gigabytes. I will make this repository more space-friendly if there is demand.

To obtain agent-based model simulations, I used agents.py on a cluster (HiPerGator) to run a bunch of simulations in parallel. See the folder slurm for examples of shell scripts used.

Use fp.ode to obtain bifurcations for the Fokker-Planck equation when motor numbers are not equal. Use fp_equal.ode when calculating bifurcations in the case that the attached up and down motors are equal (zero velocity).

When using xppaut on fp_equal.ode, use CVODE with atol and rtol set to at most 2e-5. Set parameters correctly for each panel. In particular, we need ze large enough for there to exist a middle branch (like ze=8). Set numerics so that ds=0.02, dsmin=0.00001, dsmax=0.05, parmin=0, parmax=15.

If you have any trouble running this code, please feel free to reach out or leave an issue.