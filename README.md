# Breast Cancer Prediction using Machine Learning and Deep Learning

## Table of Contents
- [Breast Cancer Prediction using Machine Learning and Deep Learning](#breast-cancer-prediction-using-machine-learning-and-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Dataset](#dataset)
  - [Libraries Used](#libraries-used)
  - [Installation](#installation)

## Project Overview
This project aims to develop machine learning and deep learning models to predict breast cancer diagnosis (benign or malignant) based on various features extracted from digitized images of breast tissue. The models utilize logistic regression for classification and a deep learning neural network to explore the effectiveness of advanced machine learning techniques in medical diagnosis.

## Features
- **Data Preprocessing**: Data cleaning, normalization, and handling of missing values to ensure model accuracy.
- **Feature Selection**: Utilization of relevant features to improve model interpretability and performance.
- **Machine Learning Model**: Implementation of logistic regression for baseline performance evaluation.
- **Deep Learning Model**: Construction of a neural network using TensorFlow/Keras for capturing complex patterns in the data.
- **Evaluation Metrics**: Comprehensive performance evaluation using metrics like accuracy, precision, recall, and F1-score.

## Dataset
The dataset used in this project consists of various measurements and characteristics of breast tumors, including:
- `id`
- `diagnosis` (M = malignant, B = benign)
- `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, etc.


## Libraries Used
This project relies on several Python libraries, including:
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical operations.
- `scikit-learn`: For implementing machine learning algorithms and preprocessing.
- `tensorflow`: For building and training deep learning models.
- `keras`: A high-level API for simplifying the construction of neural networks.
- `matplotlib`: For visualizing data and model performance.
- `seaborn`: For enhanced data visualization.

## Installation
To run this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-prediction.git
