# Semi-Synthetic Experiments (Section 7.3)
To reproduce the Llama-3-8B fine-tuning experiments,

0. Install the required packages with conda.
1. Align CausalLM models.
2. Learn reward models.
***
## 0. Package installation with conda
Run the following command to install the required conda packages:
```
conda env create -f spec-file.yml
```
---
## 1. Direct alignment experiments
To align a model with preferences sampled from anonymous random users (ignoring heterogeneity), run the following command:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_obj=dpo --train_dataset=rnd_rew
```
To align a model with a preference dataset with maximum annotator information (modeling heterogeneity), run the following command:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_obj=dpo --train_dataset=sum_rew
```
---
## 2. Reward learning experiments
To learn a reward model of preferences sampled from anonymous random users (ignoring heterogeneity), run the following command:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_obj=rew --train_dataset=rnd_rew
```
To learn a reward model of preferences with maximum annotator information (modeling heterogeneity), run the following command:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --train_obj=rew --train_dataset=sum_rew
```
