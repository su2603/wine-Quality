# Wine Quality Analysis Documentation

## Overview

This script performs comprehensive analysis and classification on a wine quality dataset. It employs data exploration, visualization, and machine learning techniques to predict wine quality based on various physicochemical properties.

## Features

- Data exploration and visualization
- Feature importance analysis
- Multiple classification models comparison
- Prediction of wine quality (binary classification)

## Requirements

The script requires the following Python libraries:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost (optional)

## Dataset

The script uses the "winequality-red.csv" dataset which contains various attributes of red wine samples along with their quality ratings (scale of 0-10).

### Dataset Attributes

1. **fixed acidity**: Most acids involved with wine or fixed or nonvolatile
2. **volatile acidity**: The amount of acetic acid in wine (high amounts can lead to an unpleasant vinegar taste)
3. **citric acid**: Found in small quantities, can add freshness and flavor
4. **residual sugar**: Amount of sugar remaining after fermentation stops
5. **chlorides**: Amount of salt in the wine
6. **free sulfur dioxide**: Prevents microbial growth and wine oxidation
7. **total sulfur dioxide**: Amount of free and bound forms of SO2
8. **density**: The density of wine is close to that of water depending on alcohol and sugar content
9. **pH**: Describes how acidic or basic a wine is (0 - very acidic; 14 - very basic)
10. **sulphates**: A wine additive which contributes to SO2 levels
11. **alcohol**: Percentage of alcohol content
12. **quality**: Score between 0 and 10 (target variable)

## Script Structure

### Data Loading and Exploration

The script begins by loading the dataset and performing basic exploratory data analysis:
- Displaying the first 5 rows
- Computing statistics for each column
- Checking for missing values
- Computing correlations between variables
- Grouping data by quality scores

```python
wine = pd.read_csv(wine_file_path)
print(wine.head())
print(wine.describe())
print(wine.isna().sum())
correlation = wine.corr()
print(wine.groupby('quality').mean())
```

### Data Visualization

The script creates several visualizations to better understand the data:
- Distribution of wine quality ratings
- Distribution of alcohol content
- Box plots comparing quality vs. alcohol content
- Correlation heatmap between all variables 
- Box plots of quality vs. fixed acidity
- Box plots of quality vs. volatile acidity

These visualizations are saved as 'wine_analysis_plots.png'.

### Feature Engineering

For classification purposes, the script creates a binary target variable:
- "goodquality": 1 if quality score ≥ 7, otherwise 0

```python
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]
```

### Feature Importance Analysis

The script uses the ExtraTreesClassifier to determine which features are most important for predicting wine quality:

```python
feature_selector = ExtraTreesClassifier(random_state=42)
feature_selector.fit(X, y)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_selector.feature_importances_
}).sort_values(by='Importance', ascending=False)
```

A visualization of feature importance is saved as 'feature_importance.png'.

### Model Training and Evaluation

The script trains and evaluates several classification models:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Gaussian Naive Bayes
6. Random Forest
7. XGBoost (if installed)

For each model, the script reports:
- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

### Model Comparison

Finally, the script compares all models based on their accuracy:
- Creates a DataFrame with models and their accuracy scores
- Identifies the best performing model
- Creates a bar chart visualizing the accuracy comparison
- Saves the comparison chart as 'model_comparison.png'

## Running the Script

1. Place the "winequality-red.csv" file in the same directory as the script
2. Run the script using Python:
   ```
   python quality-Wine.py
   ```

## Output

The script produces:
1. Console output with statistics and model performance metrics
2. Three visualization files:
   - wine_analysis_plots.png: Various data visualizations
   - feature_importance.png: Bar chart of feature importance
   - model_comparison.png: Comparison of model accuracies

## Interpreting Results

### Data Insights
The exploratory analysis and visualizations can reveal insights about wine quality:
- Which chemical properties correlate most strongly with quality
- Distribution patterns in high-quality vs. low-quality wines
- Outliers and their potential significance

### Classification Performance
The model comparison helps identify:
- Which algorithm works best for this specific classification problem
- The overall predictability of wine quality from chemical attributes
- Potential areas for model improvement

## Troubleshooting

- **FileNotFoundError**: Ensure the CSV file is in the correct directory
- **ImportError**: Install any missing libraries using pip:
  ```
  pip install numpy pandas matplotlib seaborn scikit-learn xgboost
  ```
- **Memory Issues**: For very large datasets, consider using data sampling or incremental learning
