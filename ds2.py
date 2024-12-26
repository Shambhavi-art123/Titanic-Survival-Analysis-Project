import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
gender_submission_df = pd.read_csv('gender_submission.csv')


 # Data Preprocessing
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Select only numeric columns for correlation matrix
numeric_train_df = train_df.select_dtypes(include='number')


# Merge test dataset with gender_submission to get survival predictions
test_df = test_df.merge(gender_submission_df, on='PassengerId', how='left')

# Data Cleaning:
train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
train_df = train_df.drop(columns=['Cabin'])
test_df = test_df.drop(columns=['Cabin'])


# 1. Pie Chart: Distribution of Passenger Classes
plt.figure(figsize=(6, 6))
train_df['Pclass'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#66c2a5','#fc8d62','#8da0cb'])
plt.title('Passenger Class Distribution')
plt.ylabel('')
plt.show()

# 2. Spider Chart: Comparison of Attributes (Survival Rate by Gender and Class)
# Prepare data for radar chart: Survival rate by gender and passenger class
attributes = ['Pclass1', 'Pclass2', 'Pclass3']
survival_by_class_gender = train_df.groupby(['Sex', 'Pclass'])['Survived'].mean().unstack().fillna(0)

# Normalize data for radar chart
female_stats = survival_by_class_gender.loc['female'].values
male_stats = survival_by_class_gender.loc['male'].values
categories = list(survival_by_class_gender.columns)

# Number of variables we're plotting
N = len(categories)

# Set up angles and radar chart
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)
plt.xticks(angles[:-1], categories)

# Radar chart for females
female_stats = np.append(female_stats, female_stats[0])
ax.plot(angles, female_stats, linewidth=2, linestyle='solid', label="Female")
ax.fill(angles, female_stats, color='blue', alpha=0.1)

# Radar chart for males
male_stats = np.append(male_stats, male_stats[0])
ax.plot(angles, male_stats, linewidth=2, linestyle='solid', label="Male")
ax.fill(angles, male_stats, color='red', alpha=0.1)

plt.title('Survival Rate by Gender and Passenger Class')
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.show()

# 3. Summary Statistics for Key Variables
summary_statistics = train_df[['Age', 'Fare', 'Pclass']].describe()
print("Summary Statistics:\n", summary_statistics)

# 4. Correlation Matrix (Matrix Chart)

plt.figure(figsize=(10, 6))
sns.heatmap(numeric_train_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

# 5. Additional Analysis: Survival Rate by Gender
plt.figure(figsize=(6, 6))
sns.countplot(x='Sex', hue='Survived', data=train_df)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(['Not Survived', 'Survived'])
plt.show()

# 6. Age Distribution of Survived vs. Not Survived
plt.figure(figsize=(10, 6))
sns.kdeplot(data=train_df[train_df['Survived'] == 1], x='Age', label='Survived', shade=True)
sns.kdeplot(data=train_df[train_df['Survived'] == 0], x='Age', label='Not Survived', shade=True)
plt.title('Age Distribution of Survived vs. Not Survived')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# Generate a heatmap for the correlation matrix, using only numeric columns
plt.figure(figsize=(10, 6))
numeric_columns = train_df.select_dtypes(include=['number'])  # Select only numeric columns
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()


# Optional: Save cleaned data
train_df.to_csv('cleaned_train.csv', index=False)
test_df.to_csv('cleaned_test.csv', index=False)
