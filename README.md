🚢 Titanic Survival Prediction using Decision Tree Classification
This repository contains a machine learning project that predicts the survival of passengers from the Titanic dataset using a Decision Tree Classifier. It is hosted at https://github.com/mudasarsabir/Titanic-Competition-By-Decision-Tree-Classification and serves as a submission for the Kaggle Titanic - Machine Learning from Disaster competition. The implementation is detailed in a Kaggle notebook: View Notebook.

📂 Repository Contents

Decision-Tree-Classification.ipynb: The main Jupyter notebook containing all steps from data preprocessing to model prediction.
titanic_clean.csv: A preprocessed version of the Kaggle training dataset (train.csv).
competition_clean.csv: A preprocessed version of the Kaggle test dataset (test.csv).
submission.csv: A sample submission file with model predictions for Kaggle.
LICENSE: The Apache License 2.0 governing the project.
README.md: Project description and instructions.


🧠 Objective
To build a classification model that can predict whether a passenger survived the Titanic shipwreck using a decision tree approach.

📊 Dataset
The dataset is sourced from Kaggle’s Titanic competition and is not included in its raw form in this repository. You must download train.csv and test.csv separately from Kaggle Titanic Dataset. The repository includes preprocessed versions of these datasets:

titanic_clean.csv: A cleaned version of train.csv, with missing values handled (e.g., Age imputed with median, Embarked with mode), irrelevant columns dropped (e.g., PassengerId, Name, Ticket, Cabin), and categorical variables encoded (e.g., Sex, Embarked).
competition_clean.csv: A cleaned version of test.csv, preprocessed similarly to titanic_clean.csv for model predictions.
train.csv (external): Training data with passenger details and survival outcomes.
test.csv (external): Test data for generating survival predictions.

Key features in the raw dataset include:

PassengerId: ID of each passenger
Survived: Whether the passenger survived (1) or not (0)
Pclass: Ticket class (1st, 2nd, 3rd)
Name, Sex, Age: Demographics
SibSp: Number of siblings/spouses aboard
Parch: Number of parents/children aboard
Ticket, Fare: Ticket details
Cabin, Embarked: Cabin and port of embarkation


🛠 Technologies & Libraries

Python 3.x
Jupyter Notebook
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn


🚀 Workflow Overview
The project follows a structured workflow, as implemented in the Kaggle notebook:

Import Libraries and Load Data: Load train.csv and test.csv using Pandas.
Exploratory Data Analysis (EDA): Visualize survival rates by features like Pclass, Sex, and Age using Matplotlib and Seaborn.
Handle Missing Values: Impute missing Age values (e.g., with median) and Embarked (e.g., with mode); drop Cabin due to high missingness.
Encode Categorical Variables: Convert Sex and Embarked to numerical values using one-hot encoding or label encoding.
Feature Engineering: Drop irrelevant columns like PassengerId, Name, and Ticket; optionally create new features (e.g., family size from SibSp + Parch).
Split Data into Train/Test Sets: Split training data into training and validation sets for model evaluation.
Train a Decision Tree Classifier: Use scikit-learn’s DecisionTreeClassifier with default or tuned parameters.
Evaluate Model Accuracy: Compute accuracy on the validation set (typically 75–80% for a basic Decision Tree).
Generate Submission File for Kaggle: Predict survival for test.csv and save as submission.csv.


📈 Model Performance
The model uses a basic Decision Tree Classifier, achieving an accuracy of approximately 75–80% on the validation set, based on standard Titanic competition results. The submission.csv file contains predictions for the Kaggle test set.
Further improvements can be achieved by:

Hyperparameter tuning (e.g., adjusting max_depth or min_samples_split)
Using cross-validation for robust evaluation
Trying ensemble models like Random Forest or Gradient Boosting


▶️ How to Run the Project
To run the notebook locally:

Clone the repositorygit clone https://github.com/mudasarsabir/Titanic-Competition-By-Decision-Tree-Classification.git


Navigate to the project directorycd Titanic-Competition-By-Decision-Tree-Classification


Install required dependenciespip install pandas numpy matplotlib seaborn scikit-learn jupyter


Download the dataset from Kaggle and place train.csv and test.csv in the project directory, if you wish to preprocess the raw data yourself. Alternatively, use titanic_clean.csv and competition_clean.csv provided in the repository or download them from the Releases.
Launch Jupyter Notebook and open the project notebookjupyter notebook Decision-Tree-Classification.ipynb


Run all cells in the notebook to execute the workflow and generate predictions.


📦 Releases
Stable versions of this project, including the notebook, preprocessed datasets, and other files, are available for download from the Releases page. Each release includes release notes and packaged files for easy access.

👋 Muhammad Muhammad Mudasar Sabir
I’m a Machine Learning and Deep Learning enthusiast with a strong interest in Computer Vision and Generative AI. I enjoy solving real-world problems using intelligent, data-driven approaches. My focus areas include:

🤖 Machine Learning & Deep Learning  
🧠 Computer Vision & Generative Models  
📊 Data Analysis & Feature Engineering  
🚀 Model Evaluation & Deployment


🔗 Connect with Me

🧠 Kaggle: https://www.kaggle.com/mudasarsabir  
💻 GitHub: https://github.com/mudasarsabir  
🔗 LinkedIn: https://www.linkedin.com/in/mudasarsabir/  
🌐 Portfolio: https://muddasarsabir.netlify.app


📌 Featured Project
🎯 Titanic Survival Prediction - Decision Tree

GitHub: View Repository  
Kaggle: View Notebook

A beginner-friendly ML project applying Decision Tree classification to predict Titanic passenger survival, including EDA, preprocessing, and model evaluation.

📬 Get in Touch
I'm open to collaboration, research, and AI-focused opportunities. Let’s connect!

📜 License
This project is open-source and available under the Apache License, Version 2.0. See the LICENSE file for details.
