# FraudGuard ReadME
---
## Title: FraudGuard: Unmasking Transactional Deception
---
![FraudGuard Diagram](https://github.com/J4smith0601/FraudGuard-Unmasking-Transactional-Deception/blob/main/FraudGuard_banner_image.jpg)
---
**Introduction:**

In the rapidly evolving landscape of digital finance, the detection and prevention of fraudulent transactions have become critical challenges for institutions worldwide. As financial systems grow more complex, so too do the methods employed by fraudsters, necessitating innovative approaches to safeguard assets and maintain trust. This project, titled **FraudGuard: Unmasking Transactional Deception**, seeks to address these challenges by leveraging the IEEE-CIS Fraud Detection dataset to enhance the identification and prevention of fraudulent activities within transaction data.
---
**Project Aims:**

The IEEE-CIS Fraud Detection dataset pertains to online payment fraud, with a specific emphasis on credit card transactions. Developed in collaboration between IEEE and the Consumer Internet Services (CIS) industry, the dataset is designed to identify fraudulent activities within online payment systems. The dataset primarily addresses Card-Not-Present (CNP) fraud, a form of fraud that occurs when transactions are executed without the physical presentation of the card, typically in online or telephonic payment scenarios.

The core objective of this project is to uncover the underlying patterns and factors that contribute to fraudulent transactions. By analysing the dataset, we aim to develop predictive models capable of estimating the likelihood of fraud in real-time, thus providing a robust tool for financial institutions to pre-empt and counteract fraudulent behaviours. Additionally, this project will explore temporal trends and patterns within the data, offering insights into how fraudulent activities evolve over time and how they differ from legitimate transactions.
---
**Models Developed:**

In this project, four distinct models were developed and evaluated for their ability to detect fraudulent transactions:
- **Boosted Decision Stumps**
- **Boosted Logistic Regression**
- **A Stacked Model**: Combining K-Nearest Neighbors, Decision Tree, and Logistic Regression with XGBoost as the meta-classifier
- **Feedforward Neural Network (FNN)**
---
## Technologies Used

This project employs a range of technologies and libraries for data analysis, visualisation, and machine learning:

- **NumPy**: Fundamental package for scientific computing and numerical operations.
- **Pandas**: Applied for efficient data manipulation and analysis.
- **Matplotlib**: Utilised for creating static, animated, and interactive visualisations.
- **Seaborn**: Provides advanced statistical data visualisation.
- **Plotly Express**: Enables high-level interactive visualisations.
- **sweetviz**: Automatically generates visualisations and detailed EDA reports.
- **dataprep.eda**: Automates the creation of comprehensive EDA reports.
- **SciPy**:
  - **skew, kurtosis, shapiro, norm**: Performs statistical assessments and normality tests.
  - **ttest_rel, wilcoxon**: Conducts significance testing.
- **sklearn.preprocessing**:
  - **LabelEncoder**: Converts categorical variables into numerical labels.
  - **OneHotEncoder**: Creates binary vectors for categorical features.
  - **StandardScaler**: Normalises numeric features to mean 0 and standard deviation 1.
- **sklearn.impute**:
  - **IterativeImputer**: Provides advanced imputation techniques for missing data.
  - **SimpleImputer**: Imputes missing values with basic strategies.
- **sklearn.model_selection**:
  - **train_test_split**: Splits data into training and testing sets.
  - **ColumnTransformer**: Applies different preprocessing to different columns.
  - **Pipeline**: Sequentially applies a list of transforms and a final estimator.
- **sklearn.ensemble**:
  - **AdaBoostClassifier**: Implements AdaBoost for boosting classification.
  - **StackingClassifier**: Combines multiple models to improve predictions.
- **sklearn.tree**:
  - **DecisionTreeClassifier**: Provides a decision tree classifier.
- **sklearn.linear_model**:
  - **LogisticRegression**: Performs binary classification using logistic regression.
- **sklearn.neighbors**:
  - **KNeighborsClassifier**: Implements the k-nearest neighbours algorithm.
- **xgboost**: Provides an efficient and scalable implementation of gradient boosting.
- **sklearn.metrics**:
  - **accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss, confusion_matrix, classification_report**: Metrics for evaluating model performance.
  - **roc_curve, precision_recall_curve, auc**: Calculates and plots ROC and Precision-Recall curves.
  - **learning_curve**: Visualises the training and validation score as a function of the training set size.
- **sklearn.preprocessing.label_binarize**: Binarises labels for multiclass classification.
- **TensorFlow**:
  - **Sequential**: Provides a sequential model for deep learning.
  - **Dense, Dropout**: Layers for building neural networks.
  - **Adam**: Optimiser for training deep learning models.
  - **BinaryCrossentropy**: Loss function for binary classification.
  - **EarlyStopping**: Prevents overfitting during training.
- **Python**: The primary programming language for data analysis and processing.
- **Jupyter Notebook**: For interactive analysis and documenting the process.
- **VS Code**: The IDE used for coding and managing the project.
- **Git**: For version control and tracking changes.
- **GitHub**: Hosts the repository and supports collaboration.
---
## Installation Instructions

To run this project locally, follow these steps:

**1. Clone the Repository:**
Open a terminal or command prompt and run the following command to clone the repository:
```bash
git clone https://github.com/J4smith0601/FraudGuard-Unmasking-Transactional-Deception.git

2. Navigate to the Project Directory:
Change your directory to the project folder:
cd FraudGuard-Unmasking-Transactional-Deception

3. Install Dependencies:
Ensure you have Python installed on your system. Then, install the required libraries using pip:

pip install numpy pandas matplotlib seaborn plotly sweetviz dataprep scipy scikit-learn xgboost tensorflow

4. Set Up Data:

	•	Download the IEEE-CIS Fraud Detection dataset from Kaggle or the dataset source.
	•	Store the data files in a directory accessible by the project. Update the file paths in the scripts or notebooks to reflect where you have saved the dataset.

5. Verify Installation:
To verify that the dependencies are correctly installed, you can run a Python script or open a Jupyter Notebook and import the libraries to ensure they are available.
---

## Results and Analysis

The project evaluated four distinct models for detecting fraudulent transactions. Although the actual results from the external competition will be available at the end of September, predictions have been submitted for evaluation. Preliminary analysis based on training and test metrics indicates the following:

- **Boosted Decision Stumps**: 
  - **Performance**: Achieved a test accuracy of 95.1% with a balanced precision and recall, resulting in an F1 score of approximately 0.603. The AUC-ROC and AUC-PR values were 0.924 and 0.692, respectively.
  - **Strengths**: Simplicity and interpretability, effective at distinguishing between classes, though it may struggle with complex patterns.

- **Boosted Logistic Regression**: 
  - **Performance**: Attained a test accuracy of 94.3%, with a lower precision and recall, resulting in an F1 score of around 0.559. The AUC-ROC and AUC-PR were 0.897 and 0.625, respectively.
  - **Strengths**: Efficiency and interpretability, but its linear nature limits its ability to capture non-linear relationships.

- **Stacked Model (KNN, Decision Tree, Logistic Regression with XGBoost as Meta-Classifier)**: 
  - **Performance**: Demonstrated high performance with a test accuracy of 96.64%, precision of 86.76%, and recall of 68.14%, resulting in an F1 score of 0.763. The AUC-ROC and AUC-PR were 0.9519 and 0.8277, respectively.
  - **Strengths**: Combines the strengths of multiple base classifiers, with XGBoost enhancing its ability to handle imbalanced data. Offers a robust balance between complexity and performance.

- **Feedforward Neural Network (FNN)**: 
  - **Performance**: Achieved a test accuracy of 96.31%, with high precision at 89.62% and lower recall of 60.67%, resulting in an F1 score of 0.7236. The AUC-ROC and AUC-PR were 0.9440 and 0.8047, respectively.
  - **Strengths**: Effective at capturing complex patterns and non-linear relationships, though it showed signs of overfitting, indicating potential issues with generalization.

### Comparison and Best Model Selection

- **Complexity vs. Interpretability**: Boosted Decision Stumps and Logistic Regression models are more interpretable but less capable of capturing complex patterns compared to the Stacked Model and FNN.
- **Precision and Recall Balance**: The Stacked Model offers the best balance between precision and recall, which is crucial in fraud detection where both false positives and negatives have significant consequences. The FNN, while strong in precision, has lower recall, making it less reliable for identifying all fraudulent cases.
- **Generalization Ability**: The Stacked Model exhibits strong generalization with high test accuracy and robust AUC scores. The FNN shows potential but struggles with overfitting, which could affect its reliability.
- **Robustness in Imbalanced Data**: The Stacked Model is particularly well-suited for handling imbalanced datasets, a common issue in fraud detection. Its combination of multiple base models and XGBoost as the meta-classifier makes it the most robust option.

### Best Model for the Project Objective

The Stacked Model emerges as the best choice due to its strong balance between precision, recall, and overall predictive power. It is well-suited for real-time fraud detection and provides valuable insights into fraudulent patterns. The model's ability to generalize across various patterns and handle imbalanced datasets aligns perfectly with the project's objective of providing a reliable tool for financial institutions to pre-empt and counteract fraudulent behaviours.

