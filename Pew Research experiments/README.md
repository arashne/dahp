# Pew Surveys Experiments (Section 7.1 and Appendix E)
To reproduce the Pew Surveys Experiments,

1. Install the required packages in `requirements.txt`.
2. Download all the required Pew surveys (check the required files from the dictionary `SURVEYS` from `utils.constants`). See the links provided in the [Pew Research Surveys Datasets](#pew-research-surveys-datasets) section below.
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

## Pew Research Surveys Datasets

Make sure to download the datasets and place them in the appropriate directory as specified in the `SURVEYS` dictionary from `utils.constants`.

The required surveys are:

- [American Trends Panel Wave 132](https://www.pewresearch.org/dataset/american-trends-panel-wave-132/)
- [American Trends Panel Wave 131](https://www.pewresearch.org/dataset/american-trends-panel-wave-131/)
- [American Trends Panel Wave 130](https://www.pewresearch.org/dataset/american-trends-panel-wave-130/)
- [American Trends Panel Wave 129](https://www.pewresearch.org/dataset/american-trends-panel-wave-129/)
- [American Trends Panel Wave 128](https://www.pewresearch.org/dataset/american-trends-panel-wave-128/)
- [American Trends Panel Wave 127](https://www.pewresearch.org/dataset/american-trends-panel-wave-127/)
- [American Trends Panel Wave 126](https://www.pewresearch.org/dataset/american-trends-panel-wave-126/)
- [American Trends Panel Wave 121](https://www.pewresearch.org/dataset/american-trends-panel-wave-121/)
- [American Trends Panel Wave 120](https://www.pewresearch.org/dataset/american-trends-panel-wave-120/)
- [American Trends Panel Wave 119](https://www.pewresearch.org/dataset/american-trends-panel-wave-119/)
- [American Trends Panel Wave 114](https://www.pewresearch.org/dataset/american-trends-panel-wave-114/)
- [American Trends Panel Wave 112](https://www.pewresearch.org/dataset/american-trends-panel-wave-112/)
- [American Trends Panel Wave 111](https://www.pewresearch.org/dataset/american-trends-panel-wave-111/)
- [American Trends Panel Wave 109](https://www.pewresearch.org/dataset/american-trends-panel-wave-109/)
- [American Trends Panel Wave 99](https://www.pewresearch.org/dataset/american-trends-panel-wave-99/)
- [American Trends Panel Wave 83](https://www.pewresearch.org/dataset/american-trends-panel-wave-83/)
- [American Trends Panel Wave 79](https://www.pewresearch.org/dataset/american-trends-panel-wave-79/)
- [American Trends Panel Wave 52](https://www.pewresearch.org/dataset/american-trends-panel-wave-52/)
- [American Trends Panel Wave 35](https://www.pewresearch.org/dataset/american-trends-panel-wave-35/)