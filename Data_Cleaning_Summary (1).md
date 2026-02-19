# 🧹 Data Cleaning — Beginner's Summary Guide

Data cleaning is one of the most important skills in Data Science. Real-world data is always messy — it has missing values, wrong formats, useless columns, and text where numbers should be. This guide summarizes the core concepts using the Titanic project as an example.

---

## 🤔 What is Data Cleaning?

Data cleaning (also called **data preprocessing**) is the process of transforming raw, messy data into a clean format that a machine learning model can understand and learn from.

> **Rule of thumb:** Data Scientists spend about 80% of their time cleaning data and only 20% building models.

---

## 🔍 Step 1 — Understand Your Data First

Before cleaning anything, always explore your data:

```python
df.info()       # See column types and how many values are missing
df.describe()   # Basic statistics (mean, min, max, etc.)
df.isnull().sum()  # Count missing values per column
```

Ask yourself:
- Which columns have missing values?
- Which columns are text instead of numbers?
- Which columns are irrelevant to the prediction?

---

## 🩹 Step 2 — Handle Missing Values

Missing values appear as `NaN` in Pandas. You have 3 options:

### Option A — Fill with a statistical value (most common)
```python
# Fill with median (good for numbers like Age)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill with mean
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Fill with most common value / mode (good for categories like Embarked)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
```

### Option B — Drop the column (when too many values are missing)
```python
# Cabin has 77% missing — not worth keeping
df.drop('Cabin', axis=1, inplace=True, errors='ignore')
```

### Option C — Drop the rows (when very few rows are missing)
```python
df.dropna(subset=['Embarked'], inplace=True)
```

**When to use each:**

| Strategy | When to use |
|----------|-------------|
| Fill with median | Numerical columns with some missing values |
| Fill with mode | Categorical columns with a few missing values |
| Drop column | More than 40–50% of the column is missing |
| Drop rows | Less than 1–2% of rows are affected |

---

## 🗑️ Step 3 — Remove Irrelevant Columns

Not every column helps the model. Remove columns that:
- Are unique identifiers (PassengerId, Name, Ticket)
- Have too many missing values (Cabin)
- Don't carry predictive information

```python
df.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True, errors='ignore')
```

---

## 🔢 Step 4 — Convert Text to Numbers

Machine learning models only understand numbers. There are two ways to convert text:

### Binary Encoding (for 2 categories)
```python
# male → 0, female → 1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
```

### One-Hot Encoding (for 3+ categories)
```python
# Embarked has 3 values: S, C, Q
# This creates new columns: Embarked_Q, Embarked_S
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
```

**Why not just use 0, 1, 2 for 3 categories?**
Because the model would think C (2) > S (1) > Q (0), which is meaningless. One-Hot Encoding avoids this problem.

---

## 👀 Step 5 — Always Check Your Work

After every cleaning step, verify the result:

```python
df.info()          # Check no more missing values
df.head()          # Preview the cleaned data
df.isnull().sum()  # Confirm all nulls are handled
```

---

## 📋 Data Cleaning Checklist

Use this checklist for any project:

- [ ] Loaded the data and explored it with `.info()` and `.describe()`
- [ ] Identified all columns with missing values
- [ ] Filled or dropped missing values appropriately
- [ ] Removed irrelevant or redundant columns
- [ ] Converted all text/categorical columns to numbers
- [ ] Verified the cleaned dataframe has no nulls
- [ ] The data is ready for model training

---

## 🧠 Key Terms to Remember

| Term | Meaning |
|------|---------|
| `NaN` | Missing value (Not a Number) |
| `fillna()` | Fill missing values |
| `dropna()` | Remove rows/columns with missing values |
| `median` | Middle value — not affected by extreme outliers |
| `mean` | Average — affected by extreme outliers |
| `mode` | Most frequent value |
| `One-Hot Encoding` | Converting categories into binary columns |
| `Binary Encoding` | Mapping 2 categories to 0 and 1 |
| `drop()` | Remove a column or row |

---

## ⚠️ Common Mistakes to Avoid

1. **Running cleaning cells more than once** — restart the kernel if something looks wrong
2. **Using `inplace=True` with newer Pandas** — use `df['col'] = df['col'].method()` instead
3. **Forgetting to check for missing values after cleaning**
4. **Dropping columns that might actually be useful**
5. **Using numbers (0,1,2) for categories with no order** — use One-Hot Encoding instead

---

*This guide was created as part of the Titanic Survival Prediction beginner project.*
