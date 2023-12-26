# Homework 5

Homework 5 considers multivariate time series (multi input multi output). The datasets used here are `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Electricity`, `Traffic`, `Weather`, `Exchange`, `ILI`. Homework 5 is the final examination of our course. You need to choose one of the questions below to complete. Note that use PatchTST as the baseline model for all questions. PatchTST and Transformer have been implemented in our code. **Don't change the way the model is evaluated. You can only modify the model structure and the way the model is trained.**

## Usage examples

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model PatchTST
```

```
python main.py --data_path ./dataset/ETT/ETTh1.csv --dataset ETT --model Transformer
```

## Question 1 time series distillation (100 pts)

**Introduction:**

The main idea of knowledge distillation (available at [link](https://arxiv.org/abs/1503.02531)) is to transfer the knowledge from a teacher model to a student model, allowing the student model to acquire the capabilities of the teacher model. In some cases, the performance of the student model can even surpass that of the teacher model. As introduced in class, the PatchTST model (available at [link](https://arxiv.org/abs/2211.14730)), based on the Transformer architecture, achieves better performance compared to the original Transformer. In this context, we can treat PatchTST as a teacher model and transfer its knowledge to the traditional Transformer. Specifically, starting from a traditional teacher-student model framework, we can explore various distillation methods, such as utilizing data augmentation strategies to leverage the capabilities of the teacher model, aligning different network layers (e.g., output layer, hidden layers), modifying the structure of Transformer and using different distill loss functions, among others.

Baseline: PatchTST, Transformer

## Question 2 time series forecasting based on LLMs (100 pts)

**Introduction:**

The primary obstacle impeding the advancement of pre-trained models for time series analysis lies in the scarcity of extensive training data. However, can we harness the power of Large Language Models (LLMs), which have undergone training on billions of tokens, to elevate our model's performance? A notable example, "One Fits All" (available at [link](https://arxiv.org/abs/2302.11939)), illustrates how a pre-trained LLM, when equipped with input embedding and output head layers, can be employed to construct a highly effective time series forecasting model, yielding remarkable results. Although we may not possess the same level of hardware resources as outlined in their work, we can make use of chatGPT's capabilities via an interactive dialogue-based approach, thereby aiding us in achieving superior predictions, as what"PromptCast" (available at [link](https://arxiv.org/abs/2210.08964)) does. Within this framework, we have the opportunity to delve into time series forecasting methodologies that leverage the prowess of large language models. In this context, you need to explore different ways to incorporate the pre-trained LLMs into your time series forecasting model.

![streaming.jpg](imgs%2FPromptCast.png)

Baseline: PatchTST

## Question 3 time series transfer (100 pts)

**Introduction:**

In homework 4, we simply train models in different time series to attain global model. Nevertheless, apart from this approach, there are several alternative methods we can explore to harness the potential of external datasets. For instance, we can select a few time series which are useful for the target task from external datasets, instead of using all of them. Additionally, we can delve into adapting models that have been pre-trained on external datasets. This entails taking advantage of models that have already learned valuable features and patterns from diverse data sources. By fine-tuning or transfer learning, we can align these pre-trained models with our specific task, enhancing their effectiveness in capturing domain-specific insights.
Furthermore, the challenge of processing heterogeneous data is an important consideration. External datasets often come in various formats and structures, making it imperative to develop strategies for harmonizing and integrating this diverse data effectively. Techniques such as data preprocessing, feature engineering, and data fusion can play a vital role in this context.

Baseline: PatchTST

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
2024-01-25 23:55
