# Predicting Recessions Using Machine Learning Techniques

## Overview

This project aims to predict whether the U.S. economy is in a recession using machine learning techniques. The analysis utilizes both micro and macroeconomic data from the 2008 and 2020 recessions to create a classification model. The primary goal is to build an accurate model that can predict economic recession periods based on these indicators, providing insights into economic trends that may help investors and policymakers make better decisions.

## Table of Contents
- [Project Objective](#project-objective)
- [Approach and Methodology](#approach-and-methodology)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

## Project Objective

The goal of this project is to accurately predict if the U.S. economy is in a recession using various economic indicators. The prediction is based on historical data collected from the 2008 financial crisis and the 2020 COVID-19 recession. By applying machine learning models, the project aims to predict the current recession status based on micro and macroeconomic data.

## Approach and Methodology

The project follows a typical machine learning pipeline:
1. **Data Collection**: Gather economic data from trusted sources such as Yahoo Finance and the Federal Reserve Economic Data (FRED).
2. **Data Preprocessing**: Clean, normalize, and fill missing data. Forward-fill is used for slow-changing indicators, while NaN values are dropped where appropriate.
3. **Feature Engineering**: Create new features from the raw data, such as price percentiles and historical data from previous days.
4. **Model Training**: Train several classification models using supervised learning. These models include Logistic Regression, Support Vector Machines (SVM), Decision Trees, Random Forest, Gradient Boosting, and an ensemble Voting model.
5. **Cross-Validation**: 5-fold cross-validation is used to evaluate the models and ensure that each model is generalized effectively.
6. **Feature Selection**: Apply techniques such as variance thresholding and ablation testing to identify the most relevant features for model accuracy.
7. **Evaluation**: Evaluate the models using metrics such as accuracy, precision, recall, and F1-score on the test data from the 2020 recession period.

## Requirements

The project requires the following dependencies:

- Python 3.x
- pandas
- numpy
- scikit-learn
- yfinance
- matplotlib
- seaborn

You can install the necessary dependencies using the following command:
```bash
pip install pandas numpy scikit-learn yfinance matplotlib seaborn
```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/recession-prediction.git
   ```

2. **Navigate to the Project Directory**:
   ```bash
   cd recession-prediction
   ```

3. **Install Dependencies**:
   Install the required dependencies using `pip` as mentioned above:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook**:
   Start the Jupyter notebook to run the code and see the analysis:
   ```bash
   jupyter notebook
   ```

5. **Explore the Data**:
   Open the notebook titled `Predicting Recessions Using Machine Learning Techniques.ipynb` to explore the data and models.

## Data Sources

- **Yahoo Finance** (`yfinance`): Used to collect micro-level data on banking securities.
- **Federal Reserve Economic Data (FRED)**: Used to collect macroeconomic indicators such as the Federal Bond Rate, government spending, and volatility.

## Feature Engineering

Two main features were engineered from the raw data:
- **Price Percentiles**: The percentile of a security's closing price in relation to the last 30 days.
- **Previous Day’s Price Information**: Additional features that include data from the previous day to enrich the dataset without using time-series analysis.

The final dataset had 23 features, and Min-Max Scaling was applied to normalize the data. 

## Model Selection

The following machine learning models were tested:
- **Logistic Regression** (with and without regularization)
- **Support Vector Machine (SVM)** (with Linear and Polynomial kernels)
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **Voting Ensemble**

The Random Forest model was selected as the best-performing model based on cross-validation results.

## Results

- **Random Forest Model**: Achieved the highest validation accuracy of 99.5% during cross-validation. However, out-of-sample testing on the 2020 recession dataset showed lower accuracy, primarily due to imbalanced class distribution.
- **Precision-Recall**: The model performed better at predicting non-recession periods than recession periods, suggesting a tendency to overpredict recessionary periods.

## Conclusion

While the machine learning models performed well on the training data from the 2008 recession, their performance significantly dropped when tested on the 2020 data. This suggests that economic factors influencing recessions are not entirely consistent across different recessions, necessitating continuous adaptation of models to new data.

## Acknowledgments

This project was developed as part of the final project for the course **I320D - Applied Machine Learning with Python** under the guidance of **Dr. Mishra**. Special thanks to my team members **Michael Chen** and **Yuning Zhang** for their contributions to this project.

