# Homework 4

Homework 4 considers multivariate time series (multi input multi output). The datasets used here
are `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Electricity`, `Traffic`, `Weather`, `Exchange`, `ILI`.

## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model MeanForecast
```

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model TsfKNN --n_neighbors 1 --msas MIMO --distance euclidean
```

## Part 1 Global-Local Model (40 pts)

path: `trainer.py`

**Introduction:**
The
paper ["Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality"](https://arxiv.org/abs/2008.00444)
delves into the methodologies for forecasting multiple time series, focusing on the comparison and application of local
and global forecasting algorithms.
It contrasts global methods, where a single forecasting model is applied to all time series, with local methods, which
treat each time series individually. The paper reveals that **global methods can be as effective as local methods, even
without assuming similarities among the series**. This finding challenges the common belief that global models are more
restrictive and suggests their broader applicability in a variety of forecasting scenarios.

**Objective:** 
Implement the global model, and compare it with local model. Datasets used 
in the global model can be chosen from the datasets mentioned above. Models used here are `DLinear` and `TsfKNN`.

Tips: Use channel-independent models to implement global models.


## Part 2 SPIRIT (60 pts)

path: `trainer.py`
path: `src/models/SPIRIT.py`
path: ``

**Objective:** Implement the [SPIRIT](https://www.cs.cmu.edu/~jimeng/papers/spirit_vldb05.pdf) model which 
is introduced in our class, and use DLinear as the forcast model signed below.
![streaming.jpg](imgs%2Fstreaming.jpg)

## Part 3 Evaluation

**Instructions:**

**1. Apply your models to the datasets specified at the start of this project:**

Tips: You can choose the best model on one dateset and use it to predict the other datasets.

The experimental settings used here are the same as [TimesNet](https://arxiv.org/abs/2210.02186). You can easily compare
your model with past SOTA models.

| Dataset | pred_len | Model           | MSE | MAE |
 |---------|----------|-----------------|-----|-----|
| ETTh1   | 96       | TsfKNN (Global) |     |     |
| ...     |          |                 |     |     |

## Submission

**1. Modified Code:**

- Provide the modified code for all components of the task.
- Include a `README.md` file in Markdown format that covers the entire task. This file should contain:
    - how to install any necessary dependencies for the entire task.
    - how to run the code and scripts to reproduce the reported results.
    - datasets used for testing in all parts of the task.

**2. PDF Report:**

- Create a detailed PDF report that encompasses the entire task. The report should include sections for each component
  of the task.

**3. Submission Format:**

- Submit the entire task, including all code and scripts, along with the `README.md` file and the PDF report, in a
  compressed archive (.zip).

**4. Submission Deadline:**
2024-01-15 23:55
