# MisCC
Misinformation detection on counterfactual claims. Data is available in the file: data/miscc_tv_both_kg_info.csv

## Current stage
All code is under development so not ready to be used yet. The plan of this work will be in two steps and I am in the first step currently. <br />
S1. Query LLMs for a classification task, e.g., the entailment between two claims or the truth value of a claim, with the zero-shot method. <br />
S2. Fine-tuning LLMs for the same classification task for better performance. <br />
I am planning to use Llama3 as the first LLM but more LLMs may be applied in future. <br />

### Current focus: entailment truth values
Llama3 is queried to classify the entailment of a pair of the antecedent and the consequent in a counterfactual claim being true (1), false (-1) or unknown (0). The script is **ent_tv.py**. 

### Requirement.txt
This file includes packages that are necessary for the experiments. However, it will be updated throughout the project as it will be decided based on errors in experiments (hopefully not too many errors).
