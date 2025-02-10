# Pew Surveys Experiments (Section 7.1 and Appendix E)
To reproduce the Pew Surveys Experiments,

1. Install the required packages in `requirements.txt`.
2. Download all the required Pew surveys (check the required files from the dictionary `SURVEYS` from `utils.constants`). See the links provided below. 
3. To reproduce Figure 1, run 
```
python3 EV_plot.py
```
4. To reproduce experiments in 7.1, run 
```
python3 nbc_sensitivity.py
```
5. To reproduce the figures in Appendix E, run 
```
python3 avg_reward_vs_nbc.py
```