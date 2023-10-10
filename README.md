# Homework 1

Homework 1 involves the development of a comprehensive time series analysis framework, spanning various components including dataset handling, data transformations, metric calculations, and forecasting models.

## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --target OT --model MeanForecast
```

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --target OT --model TsfKNN --n_neighbors 1 --msas MIMO --distance euclidean
```

## Part 1. Dataset(20 pts)

path: `src/dataset/dataset.py`, `src/dataset/data_visualizer.py`

**Objective:** In this part, you will implement a custom dataset class, `CustomDataset`, to handle various time series datasets. Your custom dataset class will inherit from a base class, `DatasetBase`, and provide functionality to read and preprocess the data. This assignment aims to enhance your understanding of dataset handling in machine learning and time series analysis.

**Instructions:**

**1. DatasetBase Class:**

- You are provided with a base class named `DatasetBase`. This class has methods and attributes to manage the data reading and dataset splitting process.
- The `read_data` method is marked as `NotImplementedError`, indicating that it needs to be implemented in subclasses.

**2. CustomDataset Class:**

- Your task is to implement the `CustomDataset` class, a subclass of `DatasetBase`, specifically tailored to handle various time series datasets.

- The `CustomDataset` class should have the following attributes:

  - `data_path`: A string specifying the path to the dataset file.
  - `target`: A string specifying the target column name.
  - `type`: A string indicating the dataset type.

- Implement the `read_data`  method in the `CustomDataset` class:

  - This method should read the data from the provided `data_path`.
  - It should handle different dataset formats (CSV files) by providing a general mechanism for reading and preprocessing the data.
  - Extract relevant columns and timestamps from the dataset.
  - Create a numpy array to store the data, where the last channel should contain the target values.
  - Note that the format and structure of the dataset may vary, so ensure your implementation is flexible.

- Implement the `split_data`  method in the `CustomDataset` class:

  - In the `CustomDataset` class, override the `split_data` method to implement the dataset splitting logic.

  - Set the `split` attribute to `True` to indicate that the data has been split.

  - Split the data into training, validation, and test sets based on the ratios provided in the `args` object. You can use the `ratio_train`, `ratio_val`, and `ratio_test` attributes from the base class.

  - Ensure that the splits are done correctly and that the data is stored appropriately in the class attributes (`train_data`, `val_data`, `test_data`).

**3. `get_dataset` Function:**

- The provided `get_dataset` function allows you to create instances of different dataset classes based on the `args.dataset` attribute.
- Ensure that your `CustomDataset` class is correctly integrated into this function so that it can be used to load custom datasets.

**4. `data_visualize` Function:**

- Implement the `data_visualize`  method in `data_visualizer`
- This function should take two parameters:
  - `dataset`: Dataset to visualize. Dataset.data might have shape (n_samples, timesteps, channels) or (n_samples, timesteps), depending on whether the dataset has multiple channels.
  - `t`: An integer representing the number of continuous time points to visualize.
- The purpose of this function is to choose `t` continuous time points from the dataset and visualize them.
- Ensure that your implementation is flexible enough to handle both single-channel and multi-channel datasets.

**5. Testing:**

- Test your implementation by creating an instance of the `CustomDataset` class and loading a custom dataset using the `get_dataset` function.
- Verify that the data is read and split correctly by accessing the relevant attributes of the dataset object (e.g., `dataset.train_data`, `dataset.val_data`, `dataset.test_data`).
- After loading the dataset, call the `data_visualize` function on the dataset object with a specific value of `t` to visualize the data. 

<font color= #FF0000>Note: Plot the datasets in your report.</font>

## Part 2. Transform(30 pts)

path: `src/utils/transforms.py`

**Objective:** In this part, you will implement various data transformation techniques for preprocessing time series data. These transformations will help prepare the data for machine learning models. You will create custom transformation classes that inherit from the `Transform` base class and implement the necessary methods.

**Instructions:**

**1. Transform Base Class:**

You are provided with a base class named `Transform`, which defines the structure for data transformations. This class has two abstract methods: `transform` and `inverse_transform`. You will create custom transformation classes that inherit from this base class and implement these methods.

**2. Custom Transformation Classes:**

You need to implement the following custom transformation classes, each inheriting from the `Transform` base class:

**a. Normalization**

**b. Standardization**

**c. MeanNormalization**

**d. BoxCox**

<font color= #FF0000>Note: Write down the mathematical formula for each transform in your report.</font>

## Part 3 Metrics(20 pts)

path: `src/utils/metrics.py`

**Objective:** In this programming assignment, you will implement several metrics commonly used to evaluate the performance of predictive models. These metrics will help you assess the accuracy and quality of predictions made by models. Your task is to implement the Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (SMAPE), and Mean Absolute Scaled Error (MASE) metrics.

<font color= #FF0000>Note: Write down the mathematical formula for each metric in your report.</font>

## Part 4 Models(30 pts)

path: `src/models/baselines.py`

**Objective:** In this part, you will design and implement two forecasting models: Linear Regression and Exponential Smoothing. These models will take historical time series data of length `seq_len` as input and generate predictions of length `pred_len`. Your task is to create classes for both models, inheriting from the `MLForecastModel` base class, and implement the necessary methods. Additionally, you will test these models on two datasets: ETT and Custom, and present the evaluation metrics in a table format.

**Instructions:**

**1. MLForecastModel Base Class:**

You are provided with a base class named `MLForecastModel`, which defines the structure for forecasting models. This class includes methods for training (`fit`) and forecasting (`forecast`). You will create custom forecasting model classes that inherit from this base class and implement the `_fit` and `_forecast` methods.

**2. Custom Models:**

**a. LinearRegressionForecast (MLForecastModel):**

**b. ExponentialSmoothingForecast (MLForecastModel):**

**3. Testing:**

- Test both `LinearRegressionForecast` and `ExponentialSmoothingForecast` on two different datasets: ETT and Custom. You can select 2-3 datasets from these sources.
- Load the selected datasets, split them into training and testing sets, and apply each forecasting model to make predictions on the testing data.
- Calculate and record relevant metrics for each model and dataset pair, including Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), Symmetric Mean Absolute Percentage Error (SMAPE), and Mean Absolute Scaled Error (MASE).
- Present the evaluation metrics in a table format, showcasing the performance of each model on each dataset.

<font color= #FF0000>Note: Write down the mathematical formula for each algorithm in your report, and fill the tables below.</font>

| Dataset  | Model | Transform | MSE  | MAE  | MAPE | SMAPE | MASE |
| -------- | ----- | --------- | ----- | ----- | ----- | ----- | ----- |
| dataset1 | AR    | None      |      |      |      |       |      |
|          |       | Normalize |      |      |      |       |      |
|          |       | Box-Cox   |      |      |      |       |      |
|          |       | ...       |      |      |      |       |      |
| dataset2 | EMA   | None      |      |      |      |       |      |
|          |       | Normalize |      |      |      |       |      |
|          |       | Box-Cox   |      |      |      |       |      |
|          |       | ...       |      |      |      |       |      |





## Part 5 TsfKNN*(optional 20pts)

path: `src/models/TsfKNN.py`

**objective:** In this part, you will work with a basic version of the K-Nearest Neighbors (KNN) Time Series Forecasting Model (`TsfKNN`). Your goal is to enhance and improve the model's performance by implementing various improvements. The initial model takes historical time series data as input and generates predictions of length `pred_len` using the KNN algorithm. You will explore two main areas for improvement: enhancing distance calculation and reducing computational complexity. A simple introduction to this method can be found [here](https://cran.r-project.org/web/packages/tsfknn/vignettes/tsfknn.html#:~:text=Given%20a%20new%20example%2C%20KNN,values%20associated%20with%20its%20nearest).

**Instructions:**

**1. Current Implementation:**

You are provided with a basic implementation of the `TsfKNN` class, which uses the KNN algorithm for time series forecasting. The model currently supports two options: `MIMO` and `recursive` for multi-step ahead forecasting.

**2. Distance Calculation Enhancement:**

**a. Explore Different Distance Metrics:**

- Currently, the model uses the Euclidean distance (`euclidean`) for KNN. Enhance the model by allowing the use of different distance metrics such as Manhattan distance, Chebyshev distance, or others.
- Implement distance functions for these metrics and provide an option for users to choose the desired distance metric when creating an instance of the `TsfKNN` class.

**b. Experiment with Custom Distance Metrics:**

- Explore the use of custom distance metrics tailored to the characteristics of time series data. Custom distance metrics might consider factors such as trend, seasonality, or cyclic patterns.
- Implement one or more custom distance metrics and allow users to choose them as an option when creating an instance of the `TsfKNN` class.

**3. Computational Complexity Reduction:**

**a. Approximate KNN Search:**

- The current implementation uses an exact KNN search, which can be computationally expensive for large datasets. Investigate and implement approximate KNN search algorithms, such as locality-sensitive hashing (LSH), to reduce the computational complexity while maintaining reasonable accuracy.
- Provide options for users to enable or disable approximate KNN search when creating an instance of the `TsfKNN` class.

**b. Parallelization:**

- Implement parallelization techniques to speed up the KNN search process. You can use libraries like NumPy or parallel processing libraries to distribute the workload across multiple CPU cores.
- Ensure that the parallelized code is well-optimized and provides substantial speedup for KNN search.

**4. Testing:**

- Test the enhanced `TsfKNN` model on real-world time series datasets, such as ETT and Custom datasets, to evaluate its performance.
- Use appropriate evaluation metrics (e.g., MAE, MAPE, SMAPE, MASE) to measure the model's accuracy and compare it with the basic version of the model.
- Compare the computational time required for various operations between the basic and enhanced versions.

<font color= #FF0000>Note: Include the detailed implement of your algorithm and comparison of time and accuracy in your report.</font>

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

