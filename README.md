# 🚢 Titanic Survival Prediction — Beginner Data Science Project

A beginner-friendly machine learning project to predict which passengers survived the Titanic disaster. This project covers the core foundations of Data Science: data exploration, data cleaning, feature engineering, and model building.

---

## 📁 Project Structure

```
titanic-project/
│
├── train.csv          # Dataset (download from Kaggle)
├── titanic.ipynb      # Main Jupyter Notebook
└── README.md          # This file
```

---

## 🎯 Goal

Predict whether a passenger **survived (1)** or **did not survive (0)** based on features like age, gender, ticket class, and port of embarkation.

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python 3 | Programming language |
| Pandas | Data manipulation and cleaning |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning model |
| Jupyter Notebook | Interactive coding environment |

Install everything with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 📊 Dataset

Download from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

### Key Columns

| Column | Description |
|--------|-------------|
| `Survived` | Target variable — 0 = No, 1 = Yes |
| `Pclass` | Passenger class (1st, 2nd, 3rd) |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Fare` | Ticket price |
| `Embarked` | Port of embarkation (S, C, Q) |

---

## 🔄 Project Steps

### Step 1 — Load the Data
```python
import pandas as pd
df = pd.read_csv('train.csv')
df.head()
```

### Step 2 — Explore the Data (EDA)
```python
df.info()
df.describe()
df['Survived'].value_counts()
```

### Step 3 — Visualize
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()
```

### Step 4 — Clean the Data
```python
# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop irrelevant columns
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True, errors='ignore')

# Convert text to numbers
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

### Step 5 — Build the Model
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

---

## ✅ Expected Results

A good beginner accuracy is between **78% and 83%**. Don't worry if you don't hit it right away — the learning process is more important than the score!

---

## 💡 What You'll Learn

- How to load and explore a real-world dataset
- How to identify and handle missing data
- How to convert text/categorical data into numbers
- How to train and evaluate a classification model
- The basics of the full Data Science workflow

---

## 🚀 Next Steps (After This Project)

- Try other models: Logistic Regression, Decision Trees, Gradient Boosting
- Create new features (e.g. family size = SibSp + Parch)
- Submit your predictions to the Kaggle leaderboard
- Move on to the **House Price Prediction** project

---

## 📚 Resources

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
