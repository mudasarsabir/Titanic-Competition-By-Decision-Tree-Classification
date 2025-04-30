# 🚢 Titanic Survival Prediction using Decision Tree Classification

This repository contains a machine learning project that predicts the survival of passengers from the Titanic dataset using a **Decision Tree Classifier**. It is hosted at [https://github.com/mudasarsabir/Titanic-Competition-By-Decision-Tree-Classification](https://github.com/mudasarsabir/Titanic-Competition-By-Decision-Tree-Classification) and serves as a submission for the [Kaggle Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) competition. The implementation is detailed in a Kaggle notebook: [View Notebook](https://www.kaggle.com/code/mudasarsabir/titanic-competition-by-decision-tree).

---

## 📂 Repository Contents

- `Decision-Tree-Classification.ipynb`: The main Jupyter notebook containing all steps from data preprocessing to model prediction.
- `titanic_clean.csv`: A preprocessed version of the Kaggle training dataset (`train.csv`).
- `competition_clean.csv`: A preprocessed version of the Kaggle test dataset (`test.csv`).
- `submission.csv`: A sample submission file with model predictions for Kaggle.
- `LICENSE`: The Apache License 2.0 governing the project.
- `README.md`: Project description and instructions.

---

## 🧠 Objective

To build a classification model that can predict whether a passenger survived the Titanic shipwreck using a decision tree approach.

---

## 📊 Dataset

The dataset is sourced from Kaggle’s Titanic competition and is not included in its raw form in this repository. You must download `train.csv` and `test.csv` separately from [Kaggle Titanic Dataset](https://www.kaggle.com/competitions/titanic/data). The repository includes preprocessed versions of these datasets:

- `titanic_clean.csv`: A cleaned version of `train.csv`, with missing values handled (e.g., `Age` imputed with median, `Embarked` with mode), irrelevant columns dropped (e.g., `PassengerId`, `Name`, `Ticket`, `Cabin`), and categorical variables encoded (e.g., `Sex`, `Embarked`).
- `competition_clean.csv`: A cleaned version of `test.csv`, preprocessed similarly to `titanic_clean.csv` for model predictions.
- `train.csv` (external): Training data with passenger details and survival outcomes.
- `test.csv` (external): Test data for generating survival predictions.

Key features in the raw dataset include:
- `PassengerId`: ID of each passenger
- `Survived`: Whether the passenger survived (1) or not (0)
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Name`, `Sex`, `Age`: Demographics
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Ticket`, `Fare`: Ticket details
- `Cabin`, `Embarked`: Cabin and port of embarkation

---

## 🛠 Technologies & Libraries

- Python 3.x
- Jupyter Notebook
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/)

---

## 🚀 Workflow Overview

The project follows a structured workflow, as implemented in the Kaggle notebook:

1. **Import Libraries and Load Data**: Load `train.csv` and `test.csv` using Pandas.
2. **Exploratory Data Analysis (EDA)**: Visualize survival rates by features like `Pclass`, `Sex`, and `Age` using Matplotlib and Seaborn.
3. **Handle Missing Values**: Impute missing `Age` values (e.g., with median) and `Embarked` (e.g., with mode); drop `Cabin` due to high missingness.
4. **Encode Categorical Variables**: Convert `Sex` and `Embarked` to numerical values using one-hot encoding or label encoding.
5. **Feature Engineering**: Drop irrelevant columns like `PassengerId`, `Name`, and `Ticket`; optionally create new features (e.g., family size from `SibSp` + `Parch`).
6. **Split Data into Train/Test Sets**: Split training data into training and validation sets for model evaluation.
7. **Train a Decision Tree Classifier**: Use scikit-learn’s `DecisionTreeClassifier` with default or tuned parameters.
8. **Evaluate Model Accuracy**: Compute accuracy on the validation set (typically 75–80% for a basic Decision Tree).
9. **Generate Submission File for Kaggle**: Predict survival for `test.csv` and save as `submission.csv`.

---

## 📈 Model Performance

The model uses a basic Decision Tree Classifier, achieving an accuracy of approximately 75–80% on the validation set, based on standard Titanic competition results. The `submission.csv` file contains predictions for the Kaggle test set.

Further improvements can be achieved by:
- Hyperparameter tuning (e.g., adjusting `max_depth` or `min_samples_split`)
- Using cross-validation for robust evaluation
- Trying ensemble models like Random Forest or Gradient Boosting

---

## ▶️ How to Run the Project

To run the notebook locally:

1. **Clone the repository**
   ```bash
   git clone https://github.com/mudasarsabir/Titanic-Competition-By-Decision-Tree-Classification.git
   ```
2. **Navigate to the project directory**
   ```bash
   cd Titanic-Competition-By-Decision-Tree-Classification
   ```
3. **Install required dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```
4. **Download the dataset** from [Kaggle](https://www.kaggle.com/competitions/titanic/data) and place `train.csv` and `test.csv` in the project directory, if you wish to preprocess the raw data yourself. Alternatively, use `titanic_clean.csv` and `competition_clean.csv` provided in the repository.
5. **Launch Jupyter Notebook and open the project notebook**
   ```bash
   jupyter notebook Decision-Tree-Classification.ipynb
   ```
6. **Run all cells** in the notebook to execute the workflow and generate predictions.

---

## 👋 Muhammad Muhammad Mudasar Sabir

I’m a Machine Learning and Deep Learning enthusiast with a strong interest in Computer Vision and Generative AI. I enjoy solving real-world problems using intelligent, data-driven approaches. My focus areas include:

- 🤖 Machine Learning & Deep Learning  
- 🧠 Computer Vision & Generative Models  
- 📊 Data Analysis & Feature Engineering  
- 🚀 Model Evaluation & Deployment  

---

### 🔗 Connect with Me

- 🧠 **Kaggle**: [https://www.kaggle.com/mudasarsabir](https://www.kaggle.com/mudasarsabir)  
- 💻 **GitHub**: [https://github.com/mudasarsabir](https://github.com/mudasarsabir)  
- 🔗 **LinkedIn**: [https://www.linkedin.com/in/mudasarsabir/](https://www.linkedin.com/in/mudasarsabir/)  
- 🌐 **Portfolio**: [https://muddasarsabir.netlify.app](https://muddasarsabir.netlify.app)

---

### 📌 Featured Project

#### 🎯 Titanic Survival Prediction - Decision Tree
- GitHub: [View Repository](https://github.com/mudasarsabir/Titanic-Competition-By-Decision-Tree-Classification/tree/main)  
- Kaggle: [View Notebook](https://www.kaggle.com/code/mudasarsabir/titanic-competition-by-decision-tree)  

A beginner-friendly ML project applying Decision Tree classification to predict Titanic passenger survival, including EDA, preprocessing, and model evaluation.

---

### 📬 Get in Touch

I'm open to collaboration, research, and AI-focused opportunities. Let’s connect!

---

## 📜 License

This project is open-source and available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](https://github.com/mudasarsabir/Titanic-Competition-By-Decision-Tree-Classification/blob/main/LICENSE) file for details.
