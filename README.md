# ASD-Traits-Prediction-Dashboard-using-R-and-MongoDB

## Overview

This Shiny dashboard allows users to explore, train, and predict Autism Spectrum Disorder (ASD) traits using a dataset stored in MongoDB. The app offers:

- **Data Loading:** Connects to a MongoDB database and loads ASD-related data.
- **Exploratory Data Analysis (EDA):** Visualizes class distribution and feature density plots.
- **Model Training:** Trains an XGBoost binary classifier to predict ASD traits.
- **Prediction Interface:** Provides dynamic input fields for predicting ASD traits on new data points.

---

## Features

- **MongoDB Integration:** Fetches data directly from a MongoDB collection.
- **Interactive Visualization:** Bar plots and density plots for understanding the dataset.
- **Machine Learning:** Uses `xgboost` to build a predictive model.
- **Real-time Prediction:** Users can input feature values and receive ASD trait predictions instantly.
- **Performance Metrics:** Displays accuracy, confusion matrix, and ROC curve of the trained model.

---

## Prerequisites

- R (version 4.0 or higher recommended)
- Required R packages:
  - shiny
  - shinydashboard
  - mongolite
  - ggplot2
  - dplyr
  - caret
  - xgboost
  - DT
  - pROC

Install missing packages using:

```r
install.packages(c("shiny", "shinydashboard", "mongolite", "ggplot2", "dplyr", "caret", "xgboost", "DT", "pROC"))
