# 🫀 Fine-Tuning MedGemma for Cardiovascular Disease Detection

This project showcases the fine-tuning of **MedGemma 4B Instruction-Tuned** using **LoRA** on a structured clinical dataset to predict cardiovascular disease risk. It was developed as part of the **Google Solve for Healthcare & Life Sciences with Gemma Hackathon**.


## 🚀 Objective

Enhance MedGemma’s ability to predict cardiovascular risk from tabular patient data (e.g., age, life habits,...) to assist with early triage and pre-diagnosis before a medical consultation.


## 🧠 Base Model

- **Model**: `google/medgemma-4b-it`
- **Source**: [MedGemma on HuggingFace](https://huggingface.co/google/medgemma-4b-it)


## 🗃️ Dataset

### Name: [Kaggle - Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/colewelkins/cardiovascular-disease)

### Features

| Feature               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `ID`                  | Unique identifier for each patient                                          |
| `age`                 | Age in days                                                                 |
| `age_years`           | Age in years (derived from `age`)                                           |
| `gender`              | Gender (1 = Female, 2 = Male)                                                |
| `height`              | Height in centimeters                                                       |
| `weight`              | Weight in kilograms                                                         |
| `ap_hi`               | Systolic blood pressure                                                     |
| `ap_lo`               | Diastolic blood pressure                                                    |
| `cholesterol`         | Cholesterol level (1 = Normal, 2 = Above Normal, 3 = Well Above Normal)     |
| `gluc`                | Glucose level (1 = Normal, 2 = Above Normal, 3 = Well Above Normal)         |
| `smoke`               | Smoking status (0 = Non-smoker, 1 = Smoker)                                 |
| `alco`                | Alcohol consumption (0 = No, 1 = Yes)                                       |
| `active`              | Physical activity (0 = Inactive, 1 = Active)                                |
| `cardio`              | Target variable — cardiovascular disease (0 = No, 1 = Yes)                  |
| `bmi`                 | Body Mass Index = weight (kg) / (height (m))²                               |
| `bp_category`         | Blood pressure category: "Normal", "Elevated", "Stage 1", "Stage 2", "Crisis" |
| `bp_category_encoded` | Encoded form of `bp_category`                                               |


### Description 

We used the Cardiovascular Disease Dataset as published on Kaggle by Cole Welkins, which aggregates data from multiple public health sources. The dataset includes approximately 81,846 anonymized patient records and is designed for binary classification of cardiovascular disease risk (cardio: 0 = no disease, 1 = presence of disease).

**Provenance and Data Sources**

This dataset is a composite built from two open-access repositories:

- **UCI Machine Learning Repository – Heart Disease Dataset**

Originally assembled from four different hospitals (Cleveland, Hungary, Switzerland, and the VA Long Beach), this dataset provides detailed diagnostic measurements including clinical and lifestyle data.

- **Kaggle - Cardiovascular Disease Dataset**

This source includes data collected during routine medical checkups, focusing on lifestyle and biometric attributes (e.g., BMI, smoking status, activity level).

This dataset has been widely used for benchmarking cardiovascular disease prediction with classical machine learning and transformer-based models. Its tabular structure and clearly defined target make it suitable for classification, risk scoring, and model interpretability studies.

## 🔬 Methodology 

We split the data into train, validation and test sets with 34.102, 13.641 and 34.103 samples respectively. Starting from the `google/medgemma-4b-it` base model, we applied Low-Rank Adaptation (LoRA) with rank r = 16, scaling factor α = 16 and dropout = 0.05. Training was carried out with a learning rate of 2e-4 using AdamW optimizer, a batch size of 32 distributed over 2 x NVIDIA Tesla A100 GPUs. For time constraints, we ran only 1 epoch, performing a validation pass every 50 training steps to monitor progress.


## 📈 Evaluation and Results

We present here our results on the test set of our dataset (34K samples):


| Metric     | Baseline (Zero-shot) | Fine-tuned |
|------------|----------------------|------------|
| Accuracy   | 27.41%               | 66.43%     |
| F1-score   | 38.16%               | 66.09%     |
| Precision  | 63.34%               | 66.95%     |
| Recall     | 27.41%               | 66.43%     |


We evaluated the model on our test set containing 34,000 samples. The comparison between the zero-shot baseline and the fine-tuned model reveals the following:

Fine-tuning yields a dramatic increase in every metric. Accuracy more than doubles (from 27.4 % to 66.4 %).

- The zero-shot model is highly precise (63.3 %) but suffers from extremely low recall (27.4 %). In practical terms, it rarely produces false positives, but it misses roughly 73 % of the true instances. We improved the model to make it more balanced in its predictions, increasing recall to 66.4 % while maintaining a high precision of 66.95 %.

- The F1-score goes up from 38.2 % to 66.1 %. This gain confirms that the fine-tuned model is not only more aggressive in finding positives (high recall) but does so without sacrificing correctness (maintaining high precision).

Since in real case where missing a positive is costly the baseline’s low recall would be unacceptable, our fine-tuned model exhibits more robust and deployable performance.


In summary, fine-tuning has transformed a weak zero-shot baseline into a well-balanced classifier, significantly improving both its ability to detect positive instances and its overall accuracy.


- The accuracy goes up more than twice as the baseline model showing a significan improvement.
- In both the base model and the fine-tuned version the model

- We more than doubles accuracy from 27.4 % to 66.4 % and drives recall up by 39 points (27.4 % → 66.4 %).

- Precision also improves modestly (63.3 % → 67.0 %), showing the model remains reliable when identifying positives.

- The F1-score jumps from 38.2 % to 66.1 %, reflecting a balanced gain in both precision and recall.


The fine-tuned MedGemma-4B model achieves significant performance gains across all core metrics, especially in terms of recall and overall classification balance (F1). The baseline model, while showing superficially high precision, failed to generalize well and suffered from low sensitivity.

These results validate the effectiveness of domain-specific fine-tuning using a structured clinical dataset.


## 📦 Reproduction

All code is available in the notebook `medgemma_4b_it_sft_lora_cardiovascular_disease.ipynb` and the fine tuned model is available into the HuggingFace Hub (https://huggingface.co/acours/medgemma-4b-it-sft-lora-cardiovascular-disease).

All training and validation metrics are available in the HuggingFace Repository of the model in Tensorboard through the `Metrics` tab.