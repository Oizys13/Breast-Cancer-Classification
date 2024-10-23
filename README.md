# Breast Cancer Prediction using Machine Learning and Deep Learning

## Table of Contents
- [Breast Cancer Prediction using Machine Learning and Deep Learning](#breast-cancer-prediction-using-machine-learning-and-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Dataset](#dataset)
  - [Libraries Used](#libraries-used)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Training](#model-training)
  - [Results](#results)

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
   git clone https://github.com/Oizys13/Breast-Cancer-Classification.git
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   
## Usage
Prepare your dataset by placing it in the project directory or update the file path in the code.

## Model Training
The project includes two models for breast cancer prediction:

### Logistic Regression:
1. Data Loading: Load the dataset into a Pandas DataFrame.
2. Data Preprocessing: Normalize feature values using StandardScaler or MinMaxScaler for better model performance.
3. Train-Test Split: Split the dataset into training and testing sets using train_test_split from scikit-learn.
4. Model Fitting: Train the logistic regression model using the training data.
5. Evaluation: Assess model performance using metrics like accuracy, precision, recall, and F1-score.

### Deep Learning Model
1. Model Architecture: Create a neural network with the following layers:
2. Input layer with neurons matching the number of features.
3. One or more hidden layers with activation functions like ReLU.
4. Output layer with one neuron using a sigmoid activation function for binary classification.
5. Model Compilation: Use a binary cross-entropy loss function and an optimizer like Adam for model training.
6. Training: Train the model using the training dataset and validate it using the testing dataset.
7. Evaluation: Compare the performance of the deep learning model against the logistic regression model using metrics such as accuracy, precision, recall, and F1-score.

## Results
The models' performances will be compared, showcasing metrics like accuracy, precision, recall, and F1-score for both logistic regression and deep learning approaches. Detailed metrics can be found in the classification report generated after model evaluation.
       
