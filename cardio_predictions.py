# %% [markdown]
# # Import libraries

# %%
import tensorflow as tf

import sklearn
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

import scipy as sp
from scipy import stats

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno

import pickle

import os

# %% [markdown]
# # User-Defined Function

# %%
def categorical_matrix_display(df, columns):
    dim = len(columns)
    array = np.zeros((dim, dim))          

    for i, name1 in enumerate(columns):
        for j, name2 in enumerate(columns):
            logit = LogisticRegression()
            logit.fit(df[name1].values.reshape(-1, 1), df[name2])
            score = logit.score(df[name1].values.reshape(-1, 1), df[name2])
            array[i, j] = score

    arrayFrame = pd.DataFrame(data=array, columns=columns)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(arrayFrame, annot=True, ax=ax, yticklabels=columns, vmin=0, vmax=1)


def cramers_V(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


def cramersVMatrix(df, col):
    len_cat = len(col)
    array  = np.zeros((len_cat, len_cat))

    for i, name1 in enumerate(col):
        for j, name2 in enumerate(col):
            cross_tab = pd.crosstab(df[name1], df[name2]).to_numpy()
            value = cramers_V(cross_tab)
            array[i, j] = value

    array_frame = pd.DataFrame(data=array, columns=col)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(array_frame, annot=True, ax=ax, yticklabels=col, vmin=0, vmax=1)

# %% [markdown]
# # Load data

# %%
path = os.path.join(os.getcwd(), 'data', 'cardio_train.csv')
df = pd.read_csv(path, delimiter = ';')
df 

# %% [markdown]
# - Age | Objective Feature | age | int (days)
# - Height | Objective Feature | height | int (cm) |
# - Weight | Objective Feature | weight | float (kg) |
# - Gender | Objective Feature | gender | categorical code |
# - Systolic blood pressure | Examination Feature | ap_hi | int |
# - Diastolic blood pressure | Examination Feature | ap_lo | int |
# - Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |
# - Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |
# - Smoking | Subjective Feature | smoke | binary |
# - Alcohol intake | Subjective Feature | alco | binary |
# - Physical activity | Subjective Feature | active | binary |
# - Presence or absence of cardiovascular disease | Target Variable | cardio | binary |

# %% [markdown]
# ## drop 'id' column

# %%
df = df.drop('id', axis=1)
df

# %% [markdown]
# # Rename columns

# %%
df = df.rename(columns = {'ap_hi': 'sys', 'ap_lo': 'dia'})
df

# %% [markdown]
# # Convert values in 'age' column from days to years

# %%
df['age'] = round(df['age']/365).astype(int)
df

# %% [markdown]
# # Get general infos

# %%
df.info()

# %% [markdown]
# # Find out how many NAs in each column

# %%
df.isnull().sum()

# %% [markdown]
# # Display rows with the NAs

# %% [markdown]
# ## 'height' column

# %%
df[df['height'].isna()]

# %% [markdown]
# ## 'weight' column

# %%
df[df['weight'].isna()]

# %% [markdown]
# ## 'cholesterol' column

# %%
df[df['cholesterol'].isna()]

# %% [markdown]
# ## 'alco' column

# %%
df[df['alco'].isna()]

# %% [markdown]
# # Handling NAs

# %% [markdown]
# ## fill NAs of numerical column with its median value

# %%
df[['height', 'weight']] = df[['height', 'weight']].fillna(df[['height', 'weight']].median())
df.isnull().sum()

# %% [markdown]
# ## drop rows that have NAs in any of its categorical feature

# %%
df = df.dropna(axis=0, subset=['cholesterol', 'alco'])
df

# %% [markdown]
# # Handle duplicates

# %% [markdown]
# ## check for how many duplicates

# %%
df.duplicated().sum()

# %% [markdown]
# ## show which rows are duplicated

# %%
df[df.duplicated()]

# %% [markdown]
# ## Drop duplicates

# %%
df = df.drop_duplicates()
df

# %% [markdown]
# # Summary statistics

# %%
df.describe()

# %% [markdown]
# # Outliers Handling (numerical variables)

# %% [markdown]
# ## Get general view of outliers for each numerical feature

# %%
df[['age', 'height', 'weight', 'sys', 'dia']].plot.box(subplots=True, 
                                                       layout=(2, 3), 
                                                       figsize=(20, 8), 
                                                       vert=False, 
                                                       sharex=False)

# %% [markdown]
# ## Defining InterQuartile Range (IQR), lower bounds, and upper bounds of each feature

# %%
# IQRs
iqrAge = np.quantile(df['age'], 0.75) - np.quantile(df['age'], 0.25)
iqrHeight = np.quantile(df['height'], 0.75) - np.quantile(df['height'], 0.25)
iqrWeight = np.quantile(df['weight'], 0.75) - np.quantile(df['weight'], 0.25)
iqrSys = np.quantile(df['sys'], 0.75) - np.quantile(df['sys'], 0.25)
iqrDia = np.quantile(df['dia'], 0.75) - np.quantile(df['dia'], 0.25)

#---------------------------------------------------------------------------
# Upper and lower bounds for each feature
lowerAge = np.quantile(df['age'], 0.25) - 1.5*iqrAge

lowerHeight = np.quantile(df['height'], 0.25) - 1.5*iqrHeight
upperHeight = np.quantile(df['height'], 0.75) + 1.5*iqrHeight

lowerWeight = np.quantile(df['weight'], 0.25) - 1.5*iqrWeight
upperWeight = np.quantile(df['weight'], 0.75) + 1.5*iqrWeight

lowerSys = np.quantile(df['sys'], 0.25) - 1.5*iqrSys
upperSys = np.quantile(df['sys'], 0.75) + 1.5*iqrSys

lowerDia = np.quantile(df['dia'], 0.25) - 1.5*iqrDia
upperDia = np.quantile(df['dia'], 0.75) + 1.5*iqrDia

# %% [markdown]
# ## filter out all outliers in each feature

# %%
columns = ('age', 'height', 'weight', 'sys', 'dia')

ageValue = (lowerAge, 0)
heightValue = (lowerHeight, upperHeight)
weightValue = (lowerWeight, upperWeight)
sysValue = (lowerSys, upperSys)
diaValue = (lowerDia, upperDia)

values = (ageValue, heightValue, weightValue, sysValue, diaValue)

for col, value in zip(columns, values):
    if col == 'age':
        df = df.loc[df[col]>=value[0]]
    else:
        df = df.loc[(df[col]>=value[0]) & (df[col]<=value[1])]

df

# %% [markdown]
# # Correlation heat map between numerical features

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
sns.heatmap(df[['age', 'height', 'weight', 'sys', 'dia']].corr(), ax=ax, annot=True)

# %% [markdown]
# # Visualing distribution of categorical features

# %% [markdown]
# ## Gender

# %%
ax = sns.countplot(x=df['gender'], 
                   order=df['gender'].value_counts(ascending=False).index)
        
abs_values = df['gender'].value_counts(ascending=False)
rel_values = df['gender'].value_counts(ascending=False, normalize=True).values*100
lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

ax.bar_label(container=ax.containers[0], labels=lbls)

# %% [markdown]
# ## Cholesterol

# %%
ax = sns.countplot(x=df['cholesterol'], 
                   order=df['cholesterol'].value_counts(ascending=False).index)
        
abs_values = df['cholesterol'].value_counts(ascending=False)
rel_values = df['cholesterol'].value_counts(ascending=False, normalize = True).values*100
lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

ax.bar_label(container=ax.containers[0], labels=lbls)

# %% [markdown]
# ## Glucose

# %%
ax = sns.countplot(x=df['gluc'], 
                   order=df['gluc'].value_counts(ascending=False).index)
        
abs_values = df['gluc'].value_counts(ascending=False)
rel_values = df['gluc'].value_counts(ascending=False, normalize=True).values*100
lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

ax.bar_label(container=ax.containers[0], labels=lbls)

# %% [markdown]
# ## Smoke

# %%
ax = sns.countplot(x=df['smoke'], 
                   order=df['smoke'].value_counts(ascending=False).index)
        
abs_values = df['smoke'].value_counts(ascending=False)
rel_values = df['smoke'].value_counts(ascending=False, normalize=True).values*100
lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

ax.bar_label(container=ax.containers[0], labels=lbls)	

# %% [markdown]
# ## Alcohol

# %%
ax = sns.countplot(x=df['alco'], 
                   order=df['alco'].value_counts(ascending=False).index)
        
abs_values = df['alco'].value_counts(ascending=False)
rel_values = df['alco'].value_counts(ascending=False, normalize=True).values*100
lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

ax.bar_label(container=ax.containers[0], labels=lbls)		

# %% [markdown]
# ## Active

# %%
ax = sns.countplot(x=df['active'], 
                   order=df['active'].value_counts(ascending=False).index)
        
abs_values = df['active'].value_counts(ascending=False)
rel_values = df['active'].value_counts(ascending=False, normalize=True).values*100
lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

ax.bar_label(container=ax.containers[0], labels=lbls)	

# %% [markdown]
# ## Cardio

# %%
ax = sns.countplot(x=df['cardio'], 
                   order=df['cardio'].value_counts(ascending=False).index)
        
abs_values = df['cardio'].value_counts(ascending=False)
rel_values = df['cardio'].value_counts(ascending=False, normalize=True).values*100
lbls = [f'{p[0]} ({p[1]:.0f}%)' for p in zip(abs_values, rel_values)]

ax.bar_label(container=ax.containers[0], labels=lbls) 	

# %% [markdown]
# Because the categories of the target are approximately balanced, no need to filter by target anymore.

# %% [markdown]
# ## Categorical features correlation (Using logistic regression)

# %%
cat_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
categorical_matrix_display(df, cat_columns)

# %% [markdown]
# ## Categorical features correlation (Using Cramer's V)

# %%
cat_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
cramersVMatrix(df, cat_columns)

# %% [markdown]
# # (Optional | Code are commented out | For owner's reference) Sample are sampled for each categorical features such that all the countplots above have balanced categories
# ## Will HEAVILY reduce the sample size

# %%
# groupbyGender = df.groupby('gender')
# df = groupbyGender.sample(20000, random_state = 42)
# df

# %%
# groupbyCholesterol = df.groupby('cholesterol')
# df = groupbyCholesterol.sample(4393, random_state = 42)
# df

# %%
# groupbyGluc = df.groupby('gluc')
# df = groupbyGluc.sample(1536, random_state = 42)
# df

# %%
# groupbySmoke = df.groupby('smoke')
# df = groupbySmoke.sample(572, random_state = 42)
# df

# %%
# groupbyAlco = df.groupby('alco')
# df = groupbyAlco.sample(201, random_state = 42)
# df

# %%
# groupbyActive = df.groupby('active')
# df = groupbyActive.sample(80, random_state = 42)
# df

# %%
# groupbyCardio = df.groupby('cardio')
# df = groupbyCardio.sample(66, random_state = 42)
# df

# %% [markdown]
# # Separating features from target

# %%
X = df.drop('cardio', axis=1)
X

# %%
y = df['cardio']
y

# %% [markdown]
# # train-test splitting

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %% [markdown]
# # training 3 different classifier with the same scaler (standard scaler)

# %%
# pipes
logreg_cv_steps = [('scaler', StandardScaler()), 
                   ('logreg', LogisticRegression(random_state=42))]
logreg_cv_pipe = Pipeline(steps=logreg_cv_steps)

dt_cv_steps = [('scaler', StandardScaler()), 
               ('dt', DecisionTreeClassifier(random_state = 42))]
dt_cv_pipe = Pipeline(steps=dt_cv_steps)

rf_cv_steps = [('scaler', StandardScaler()), 
               ('rf', RandomForestClassifier(random_state=42))]
rf_cv_pipe = Pipeline(steps=rf_cv_steps)

# dict of different pipes
pipes = {'Logistic Regression': logreg_cv_pipe,
         'Decision Tree': dt_cv_pipe,
         'Random Forest': rf_cv_pipe}

best_score = 0.0

fig, ax = plt.subplots(1, 3, figsize=(20, 4))

for index, (name, pipe) in enumerate(pipes.items()):
    pipe.fit(X_train, y_train)
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, normalize='all', ax=ax[index])
    ax[index].set_title(f'{name}')
    acc_score = pipe.score(X_test, y_test)
    class_report = classification_report(y_test, pipe.predict(X_test))
    if acc_score > best_score:
        best_score = acc_score
        best_pipe = pipe
        best_name = name
    print(name)
    print(acc_score)
    print(class_report)
    print('-----------------------------------------')

    if index == 2:
        plt.tight_layout() 
        plt.show()

print(f'Best model is {best_name} with accuracy score: {best_score}')

# %%
# pipes
logreg_cv_steps = [('scaler', StandardScaler()), 
                   ('logreg', LogisticRegression(random_state=42))]
logreg_cv_pipe = Pipeline(steps=logreg_cv_steps)

dt_cv_steps = [('scaler', StandardScaler()), 
               ('dt', DecisionTreeClassifier(random_state = 42))]
dt_cv_pipe = Pipeline(steps=dt_cv_steps)

rf_cv_steps = [('scaler', StandardScaler()), 
               ('rf', RandomForestClassifier(random_state=42))]
rf_cv_pipe = Pipeline(steps=rf_cv_steps)

# parameters grid
params1 = {'logreg__penalty': ['l2', 'l1', 'elasticnet'],
          'logreg__C': [0.1, 1.0, 10]}

params2 = {'dt__min_samples_leaf': [1, 10, 100], 
          'dt__min_samples_split': [2, 20, 200]}

params3 = {'rf__n_estimators': [10, 100, 500],
          'rf__min_samples_split': [2, 20, 200]}

# dict of different pipes
pipes = {'Logistic Regression': logreg_cv_pipe,
         'Decision Tree': dt_cv_pipe,
         'Random Forest': rf_cv_pipe}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

best_score = 0.0

fig, ax = plt.subplots(1, 3, figsize=(20, 4))

for index, (name, pipe) in enumerate(pipes.items()):

    if name == 'Logistic Regression':
        gridSearchCV = GridSearchCV(estimator=pipe, param_grid=params1, cv=kf, scoring='accuracy')
        gridSearchCV.fit(X_train, y_train)
        ConfusionMatrixDisplay.from_estimator(gridSearchCV, X_test, y_test, normalize='all', ax=ax[index])
        ax[index].set_title(f'{name}')
        acc_score = gridSearchCV.score(X_test, y_test)
        class_report = classification_report(y_test, gridSearchCV.predict(X_test))
       
        if acc_score > best_score:
            best_score = acc_score
            best_pipe = gridSearchCV
            best_name = name

        print(name)
        print(acc_score)
        print(class_report)
        
        print('-----------------------------------------')

    elif name == 'Decision Tree':
        gridSearchCV = GridSearchCV(estimator=pipe, param_grid=params2, cv=kf, scoring='accuracy')
        gridSearchCV.fit(X_train, y_train)
        ConfusionMatrixDisplay.from_estimator(gridSearchCV, X_test, y_test, normalize='all', ax=ax[index])
        ax[index].set_title(f'{name}')
        acc_score = gridSearchCV.score(X_test, y_test)
        class_report = classification_report(y_test, gridSearchCV.predict(X_test))

        if acc_score > best_score:
            best_score = acc_score
            best_pipe = pipe
            best_name = name

        print(name)
        print(acc_score)
        print(class_report)
        
        print('-----------------------------------------')

    elif name == 'Random Forest':
        model = pipe
        model.fit(X_train, y_train)
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, normalize='all', ax=ax[index])
        ax[index].set_title(f'{name}')
        acc_score = model.score(X_test, y_test)
        class_report = classification_report(y_test, model.predict(X_test))

        if acc_score > best_score:
            best_score = acc_score
            best_pipe = pipe
            best_name = name

        print(name)
        print(acc_score)
        print(class_report)
        
        plt.tight_layout() 
        plt.show()

    else:
        print('ERROR!!!')

print(f'Best model is {best_name} with accuracy score: {best_score}')

# %% [markdown]
# # Saving/Pickling the trained model

# %%
# saving
MODEL_PATH = os.path.join(os.getcwd(), 'model', 'model.pkl')
with open(MODEL_PATH, 'wb') as file:
    pickle.dump(best_pipe, file)

# loading
with open(MODEL_PATH, 'rb') as file:
    loaded_model = pickle.load(file)


