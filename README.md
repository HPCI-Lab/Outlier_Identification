# 🚀 Outlier Side-Channel Attacks (IBM Cloudstars)

## 📌 Overview
This project aims to analyze how likely a sample is to be used as an attack vector for membership inference attacks without requiring a full training process. 

### 🔍 Key Questions:
- Are outliers used in the training process of the model?
- Can we determine if outliers are included in the training dataset?
- Can we identify whether a sample is an outlier?
- Can side-channel data provide useful signals about training dynamics?

Outliers in training can have both positive and negative effects:
- ✅ **Benefits:** Reduced overfitting, better generalization, and often highly descriptive samples.
- ⚠️ **Risks:** Higher computational costs and increased vulnerability to membership inference attacks due to distinguishable processing patterns.

🧪 One hypothesis is that outliers may introduce higher delta gradients, leading to increased energy consumption. By analyzing side-channel data, we may determine if a model is being trained with outliers.

---

## 🏃‍♂️ How to run: 

I'm assuming that in the src directory there is a run.lsf file which looks something like this: 

```bash
jbsub -q x86_1h -cores 1+1 -mem 20g python $@
```

Then running seq to seq finetuning for the albert model can be run like this: 

```bash
./run.lsf IBM/lightning_routine.py IBM/c_albert.yaml
```

The table below shows the major options for the yaml file: 

| Name      | Options     |
| ------------- | ------------- |
|  run -> optimizer   | "adam" |
| run -> criterion |  "CrossEntropyLoss", "MSELoss" |
| model -> type | "albert", "bert", "granite", "swin", ... |
|   dataset -> type  | "med", "imdb", "mnist", "cifar100", ... |
|   dataset -> samples_per_class | 2000 or 7000 usually |
|  dataset -> classes   | 3 or 10 usually |


while this table shows the resource configs used for all runs: 

| Model | Dataset      | GPUs | Mem | Hrs     |
| ------------- | ------------- | - | - |- |
|   SwinT-V2  | CIFAR | 2 | 50 | 6 |
| All other vision | All | 1 | 50 | 1 |
| Smaller than Bert | Imdb | 1 | 90 | 1 |
| Bert and larger | Med | 2-3-4 | 100+ | 6-12 |
| Granite and larger | Fineweb | ? | ? | ? |


## 📖 Methodology
The methodology involves training a model while tracking various metrics related to energy consumption and computational performance. These metrics will be correlated to the presence of outliers in training batches.

### 📌 Steps:
1. **Training Phase:**
   - Train a model while tracking specific side-channel metrics.
   - Introduce labeled outliers to observe their impact.
   
2. **📊 Metric Tracking:**
   - 📉 Training loss
   - ⚡ GPU power usage during the training step
   - 🔋 GPU energy consumption per training step
   - ⏳ Training step duration
   - 🔄 Backpropagation time

3. **🕵️ Outlier Detection:**
   - Apply multiple statistical and clustering methods to detect outliers in batches.
   - Investigate the correlation between outlier presence and tracked metrics.

4. **📈 Analysis & Evaluation:**
   - Compare side-channel information with the presence of known outliers.
   - Cross-reference indices across epochs (since batches are shuffled) to identify persistent outliers.
   - Determine if outliers impact specific computational aspects such as processing time and energy usage.

---

## 🔍 Outlier Detection Techniques (Batch-Level)
To detect outliers at the batch level, we use multiple techniques:
- 📏 **Mahalanobis Distance:** Measures how far a sample is from the mean, considering covariance.
- 🔎 **Local Outlier Factor (LOF):** Identifies density-based anomalies by comparing local densities.
- 🧩 **DBSCAN (Density-Based Spatial Clustering):** Detects clusters and identifies points in low-density regions as outliers.
- 📉 **Weighted Least-Frequent (WLF) Detection:** Identifies samples that appear infrequently across batches.
- 📊 **Z-score:** Standard statistical approach for measuring how many standard deviations a sample deviates from the mean.

<!-- 
<p align="center">
  <img src="imgs/LOF_mistral_chat_doctor_0.01.png" width="45%">
  <img src="imgs/WLF_mistral_chat_doctor_0.01.png" width="45%">
</p> -->

---

## 🎯 Outlier Identification (Within a Batch)
Once a batch is identified as outlier-heavy, we analyze individual samples:
- 🔄 Cross-reference indices across multiple epochs (since batches are shuffled each epoch).
- 🔍 Identify recurring outlier samples that consistently appear in flagged batches.
- 📊 Examine the impact of these samples on training metrics.
