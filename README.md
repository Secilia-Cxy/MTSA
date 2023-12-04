# Homework 3

Homework 3 considers multivariate time series (multi input multi output). The datasets used here are `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Electricity`, `Traffic`, `Weather`, `Exchange`, `ILI`.
## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model MeanForecast
```

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model TsfKNN --n_neighbors 1 --msas MIMO --distance euclidean
```

## Part 1 Decomposition (20 pts)
path: `src/utils/decomposition.py`

**Objective:** Implement STL and X11 decomposition methods to separate the trend and seasonal components from the original time series data.


## Part 2 Model (20 pts)

path: `src/models/ARIMA.py`
path: `src/models/ThetaMethod.py`

**Objective:** Implement the ARIMA and Theta forecasting models.

## Part 3 ResidualModel (60 pts)

path: `src/models/ResidualModel.py`

**Objective:**
The objective is to create an ensemble forecasting model that integrates various individual 
models you've implemented earlier, such as LR, ETS, DLinear, TSFKNN, ARIMA, and ThetaMethod. 
The aim is to leverage the strengths of each model for improved forecasting accuracy.

**Instructions:**

**1. Decomposition-Based Forecasting:**

Implement time series decomposition to separate trend and seasonal components from the original data.
Apply different forecasting models to predict the trend and seasonal components independently.
Combine these predictions to form a comprehensive forecast.

**2. Residual Network Approach:**

Employ a residual network strategy, inspired by [N-Beats](https://arxiv.org/pdf/1905.10437.pdf), which uses multiple MLPs, and each MLP in the network aims to predict the residuals (errors) of the preceding MLP.
For example, the residual from a TSFKNN prediction could be modeled using DLinear.
The final forecast is the cumulative sum of predictions from all models.

**3. Diverse Prediction Methods:**

Experiment with various forecasting methods, such as recursive, non-recursive, direct, and indirect approaches.

**4. Combining Methods for Enhanced Accuracy:**

You can combine the above methods to get a better result, not necessary to use all of them.


## Part 4 Evaluation

**Instructions:**

**1. Apply your ResidualModels to the datasets specified at the start of this project:**

Tips: You can choose the best model on one dateset and use it to predict the other datasets.

The experimental settings used here are the same as [TimesNet](https://arxiv.org/abs/2210.02186). You can easily compare your model with past SOTA models.
If your model is better than SOTA, you can get 15 pts extra.

| Dataset | pred_len | Models | Decomposition | MSE  | MAE  |
 |---------|----------|--------|------------| ----- | ----- |
| ETTh1   | 96       | TsfKNN | MA     |      |      |
 | ...     |          |        |      |      |      |

 

## Submission

**1. Modified Code:**

- Provide the modified code for all components of the task.
- Include a `README.md` file in Markdown format that covers the entire task. This file should contain:
  - how to install any necessary dependencies for the entire task.
  - how to run the code and scripts to reproduce the reported results.
  - datasets used for testing in all parts of the task.

**2. PDF Report:**

- Create a detailed PDF report that encompasses the entire task. The report should include sections for each component of the task.

**3. Submission Format:**

- Submit the entire task, including all code and scripts, along with the `README.md` file and the PDF report, in a compressed archive (.zip).

**4. Submission Deadline:**
  2024-01-15 23:55
