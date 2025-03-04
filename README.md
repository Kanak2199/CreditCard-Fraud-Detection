# Credit Card Fraud Detection
This project aims to detect fraudulent credit card transactions using a variety of machine learning techniques. The project employs both supervised learning models and unsupervised anomaly detection techniques to identify fraudulent transactions in a dataset.

## Project Overview
The goal of this project is to build a fraud detection system that can accurately predict whether a given transaction is fraudulent or not. The dataset consists of credit card transactions, where each transaction is labeled as either fraudulent (1) or non-fraudulent (0). We will apply several machine learning algorithms to this dataset to detect fraudulent transactions.

## Supervised Learning Models
1. Logistic Regression: A linear model for binary classification that predicts the probability of fraud based on the features.
Random Forest: An ensemble learning method that combines multiple decision trees to improve prediction accuracy.
2. Support Vector Classifier (SVC): A classification method that finds the hyperplane which best separates fraudulent and non-fraudulent transactions.
3. Gradient Boosting: A boosting technique that combines weak learners (typically decision trees) to create a strong predictive model.
4. Neural Network (Sequential): A deep learning approach where multiple layers of neurons are used to learn patterns in the data and predict fraudulent transactions.

## Anomaly Detection Models
1. Isolation Forest: A tree-based algorithm that isolates observations by randomly selecting a feature and a split value, making it efficient for detecting anomalies.
2. Local Outlier Factor (LOF): A density-based algorithm that identifies anomalies based on the local density of data points compared to their neighbors.
3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise): A clustering algorithm that detects outliers based on the density of data points, classifying points that do not belong to any cluster as noise (anomalies).

## Dataset
The dataset used in this project consists of credit card transactions from - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download 

## Installation
To run the project, make sure you have Python installed on your system. You can install the required dependencies using the requirements.txt file.

bash
Copy
Edit
## Clone the repository
git clone https://github.com/kanak2199/credit-card-fraud-detection.git

## Install dependencies
pip install -r requirements.txt

bash
Copy
Edit
jupyter notebook
This will open the Jupyter Notebook interface in your web browser.

Open the Notebook: In the Jupyter Notebook interface, navigate to the notebooks folder and open fraud_detection.ipynb to start running the code.

## Evaluation Metrics
**For the supervised models:**

1. Accuracy: Measures the proportion of correctly classified transactions.
2. Precision, Recall, F1-score: Measures the model's performance, particularly for imbalanced classes like fraud detection.
3. Confusion Matrix: Visualizes the true positives, true negatives, false positives, and false negatives.
For anomaly detection models:

**Anomaly Detection Performance:** Visualize and interpret the results by identifying the number of anomalies detected by each model.

## Results and Findings
The evaluation of the models can be found in the notebook notebooks/fraud_detection.ipynb. The results include:

1. The performance comparison of all supervised models.
2. The identification of anomalies using unsupervised methods such as Isolation Forest, LOF, and DBSCAN.

## Conclusion
This project demonstrates the use of multiple machine learning techniques for credit card fraud detection. The supervised models (Logistic Regression, Random Forest, SVC, Gradient Boosting, and Neural Networks) perform classification based on labeled data. Additionally, the anomaly detection models (Isolation Forest, LOF, and DBSCAN) help identify fraudulent transactions without the need for labeled data. Both approaches provide valuable insights and can be used for further refinement and deployment in real-world fraud detection systems.