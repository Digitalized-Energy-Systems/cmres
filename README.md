# CMRES

Complex Multi-Energy System Resilience Simulation

# Installation

```bash
pip install -e .
pip install -r requirement.txt
```

# Running the resilience simulation

```bash
python experiments/re/cp_cn_py $RUN_ID_FOR_PARALLELIZATION
```

The $RUN_ID can be any positive integer (typically 1-300) and specifies the density and resilience parameter set. To run the whole parameter set you need execute this command with every integer between 1 and 300 (including).

# Results

The results are writen to `data/res/MoneeResilienceExperiment-*$PARAMETER_SET`. Here you will find a monee network pickled (network.p), the failure.csv in which every component failure is written, the repair.csv in which every components' repair step is written, and performance.csv in which the performance decrease in every time step is written (`0`=electricity,`1`=heat,`2`=gas). The column `id` specifies the run and can be used to join the datasets.

To evaluate the results, there is the script `experiments/re/cp_cn_evaluation.py`. This script calculates all network and resilience metrics for every experiment and run. It will also generate all necessary plots. Note that the evaluation need around a day, due to the individual impact metric calculation.