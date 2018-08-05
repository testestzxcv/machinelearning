import sklearn

import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

path_train = r"C:\Users\core_\Documents\all/train.csv"

df_train = pd.read_csv(path_train)
df_train.head()

df_train.info()

df_train.describe()

df_train.isnull().sum()

df_train['Cabin'] = df_train['Cabin'].fillna('N')

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

df_train['Embarked'].value_counts()

df_train['Embarked'] = df_train['Embarked'].fillna('S')

df_train.isnull().sum()

df_train.head(10)

df_train['Cabin'] = df_train['Cabin'].apply(lambda x: x[0])

df_train['Cabin'].value_counts()

sns.set_style("darkgrid")

sns.set_palette(sns.color_palette("Set2", 10))

sns.factorplot('Sex', kind='count', data=df_train)

sns.factorplot('Pclass', kind='count', hue='Sex', data=df_train)

df_train['Age'].hist()

sns.factorplot('Cabin', kind='count', data=df_train)

sns.factorplot('Cabin', kind='count', data=df_train.loc[df_train['Cabin']!='N'])

sns.factorplot('Embarked', kind = 'count', data = df_train)

df_train['Survivor'] = df_train['Survived'].map({0 : 'no', 1: 'yes'})

sns.factorplot('Survivor', kind='count', hue='Sex', data=df_train)

sns.factorplot('Pclass', kind='count', hue='Survived', data=df_train)

sns.factorplot('Sex', kind='count', hue='Survivor', data=df_train)

sns.distplot(df_train['Age'].loc[df_train['Sex']=='male'])
sns.distplot(df_train['Age'].loc[df_train['Sex']=='female'])

sns.lmplot('Age', 'Survived', hue='Sex', data=df_train)

sns.factorplot('Age', kind='count', hue='Survivor', data=df_train)

sns.factorplot('Age', kind='count', hue='Survivor', data=df_train.loc[df_train['Age'] < 16])

sns.factorplot('Age', kind='count', hue='Survivor', data=df_train.loc[df_train['Age'] >=50 ] )

sns.factorplot('Embarked', kind='count', hue='Survivor', data=df_train)

sns.factorplot('Cabin', kind='count', hue='Survivor', data=df_train)

sns.lmplot('Age', 'Survived', hue='Sex', data=df_train.loc[df_train['Age']>=16])

sns.lmplot('SibSp', 'Survived', hue='Sex', data=df_train)

sns.lmplot('Parch', 'Survived', hue='Sex', data=df_train)

df_train['Family_Size'] = df_train['SibSp'] + df_train['Parch']

sns.lmplot('Family_Size', 'Survived', hue='Sex', data=df_train)

sns.lmplot('Fare', 'Survived', hue='Sex', data=df_train)

df_train['Ticket'].value_counts()

path_train = r"C:\Users\core_\Documents\all/train.csv"
path_test = r"C:\Users\core_\Documents\all/test.csv"

df_train = pd.read_csv(path_train)
df_test = pd.read_csv(path_test)

y_train = df_train['Survived']
df_train.drop('Survived', axis=1, inplace=True)

df_combined = df_train.append(df_test)
df_combined.reset_index(inplace=True)
df_combined.drop('index', axis=1, inplace=True)
df_combined.head()

df_combined.shape

df_combined['Name'].head(20)

name_title = list(set(df_combined['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())))

name_title

df_combined['Title'] = df_combined['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())

df_combined.groupby('Title').count()

dict_title={
    'Capt' : 'officer',
    'Col': 'officer',
    'Dr': 'officer',
    'Major': 'officer',
    'Rev': 'officer',
    'Master': 'master',
    'Don' : 'noble',
    'Dona': 'noble',
    'Jonkheer': 'noble',
    'Lady' : 'noble',
    'the Countess' : 'noble',
    'Sir' : 'noble',
    'Miss' : 'miss',
    'Mlle' : 'miss',
    'Mme' : 'mrs',
    'Mrs' : 'mrs',
    'Ms' : 'mrs',
    'Mr' : 'mr',
}

df_combined['Title'] = df_combined['Title'].map(dict_title)

df_combined.groupby('Title').count()

grouped_df_train = df_combined.head(891).groupby(['Sex', 'Pclass', 'Title']).median()
grouped_df_test = df_combined.iloc[891:].groupby(['Sex', 'Pclass', 'Title']).median()
grouped_df_train

grouped_df_test

grouped_df_test.loc['female', 1, 'miss']['Age']

def fill_age(row, grouped_median):

    if row['Sex']== 'female' and row['Pclass'] == 1:
        if row['Title'] == 'miss':
            return grouped_median.loc['female', 1, 'miss']['Age']
        elif row['Title'] == 'mrs':
            return grouped_median.loc['female', 1, 'mrs']['Age']
        elif row['Title'] == 'noble':
            return grouped_median.loc['female', 1, 'noble']['Age']
        elif row['Title'] == 'officer':
            return grouped_median.loc['female', 1, 'officer']['Age']

    elif row['Sex']== 'female' and row['Pclass'] == 2:
        if row['Title'] == 'miss':
            return grouped_median.loc['female', 2, 'miss']['Age']
        elif row['Title'] == 'mrs':
            return grouped_median.loc['female', 2, 'mrs']['Age']
        elif row['Title'] == 'noble':
            return grouped_median.loc['female', 2, 'noble']['Age']
        elif row['Title'] == 'officer':
            return grouped_median.loc['female', 2, 'officer']['Age']

    elif row['Sex']== 'female' and row['Pclass'] == 3:
        if row['Title'] == 'miss':
            return grouped_median.loc['female', 3, 'miss']['Age']
        elif row['Title'] == 'mrs':
            return grouped_median.loc['female', 3, 'mrs']['Age']
        elif row['Title'] == 'noble':
            return grouped_median.loc['female', 3, 'noble']['Age']
        elif row['Title'] == 'officer':
            return grouped_median.loc['female', 3, 'officer']['Age']

    elif row['Sex']== 'male' and row['Pclass'] == 1:
        if row['Title'] == 'master':
            return grouped_median.loc['male', 1, 'master']['Age']
        elif row['Title'] == 'mr':
            return grouped_median.loc['male', 1, 'mr']['Age']
        elif row['Title'] == 'noble':
            return grouped_median.loc['male', 1, 'noble']['Age']
        elif row['Title'] == 'officer':
            return grouped_median.loc['male', 1, 'officer']['Age']

    elif row['Sex']== 'male' and row['Pclass'] == 2:
        if row['Title'] == 'master':
            return grouped_median.loc['male', 2, 'master']['Age']
        elif row['Title'] == 'mr':
            return grouped_median.loc['male', 2, 'mr']['Age']
        elif row['Title'] == 'noble':
            return grouped_median.loc['male', 2, 'noble']['Age']
        elif row['Title'] == 'officer':
            return grouped_median.loc['male', 2, 'officer']['Age']

    elif row['Sex']== 'male' and row['Pclass'] == 3:
        if row['Title'] == 'master':
            return grouped_median.loc['male', 3, 'master']['Age']
        elif row['Title'] == 'mr':
            return grouped_median.loc['male', 3, 'mr']['Age']
        elif row['Title'] == 'noble':
            return grouped_median.loc['male', 3, 'noble']['Age']
        elif row['Title'] == 'officer':
            return grouped_median.loc['male', 3, 'officer']['Age']

df_combined.head(891).Age = df_combined.head(891).apply(lambda x: fill_age(x, grouped_df_train) if np.isnan(x['Age']) else x['Age'] , axis=1)

df_combined.iloc[891:]['Age'] = df_combined.iloc[891:].apply(lambda x: fill_age(x, grouped_df_test) if np.isnan(x['Age']) else x['Age'] , axis=1)

df_combined.isnull().sum()

df_combined['Embarked'] = df_combined['Embarked'].fillna('S')

df_combined['Cabin'] = df_combined['Cabin'].fillna('N') # N = None

df_combined['Cabin'] = df_combined['Cabin'].apply(lambda x : x[0])

df_combined['Fare'] = df_combined['Fare'].fillna(df_combined['Fare'].mean())

df_combined.isnull().sum()

df_combined.head()

df_combined.drop('Name', axis=1, inplace=True)

df_combined['Family_size'] = df_combined['SibSp'] + df_combined['Parch']

df_combined['Ticket'] = df_combined['Ticket'].apply(lambda x: x.replace('/', '').replace('.', '').split()[0])

df_combined['Ticket'] = df_combined['Ticket'].apply(lambda x: 'Digit' if x.isdigit() else x)

titles_dummies = pd.get_dummies(df_combined['Title'], prefix='Title')
df_combined = pd.concat([df_combined, titles_dummies], axis=1)
df_combined.drop('Title', axis=1, inplace=True)
df_combined.shape

cabin_dummies = pd.get_dummies(df_combined['Cabin'], prefix='Cabin')
df_combined = pd.concat([df_combined, cabin_dummies], axis=1)
df_combined.drop('Cabin', axis=1, inplace=True)
df_combined.shape

embarked_dummies = pd.get_dummies(df_combined['Embarked'], prefix='Embarked')
df_combined = pd.concat([df_combined, embarked_dummies], axis=1)
df_combined.drop('Embarked', axis=1, inplace=True)
df_combined.shape

pclass_dummies = pd.get_dummies(df_combined['Pclass'], prefix='Pclass')
df_combined = pd.concat([df_combined, pclass_dummies], axis=1)
df_combined.drop('Pclass', axis=1, inplace=True)
df_combined.shape

ticket_dummies = pd.get_dummies(df_combined['Ticket'], prefix='Ticket')
df_combined = pd.concat([df_combined, ticket_dummies], axis=1)
df_combined.drop('Ticket', axis=1, inplace=True)
df_combined.shape

df_combined['Person'] = df_combined.apply(lambda x : 'child' if x['Age'] < 16 else x['Sex'] , axis=1)

person_dummies = pd.get_dummies(df_combined['Person'], prefix = 'Person')
df_combined = pd.concat([df_combined, person_dummies], axis=1)
df_combined.drop('Person', axis=1, inplace=True)
df_combined.shape

df_combined['Sex'] = df_combined['Sex'].map({'male':1, 'female':0})

df_combined['Alone'] = df_combined['Family_size'].map(lambda x: 1 if x < 1 else 0)
df_combined['Small_Family'] = df_combined['Family_size'].map(lambda x: 1 if 4 >= x >= 1 else 0)
df_combined['Bag_Family'] = df_combined['Family_size'].map(lambda x: 1 if x > 4 else 0)

df_combined.head()

eliminate_features = ['PassengerId']

df_combined.drop(eliminate_features, axis=1, inplace=True)
df_combined.head()

from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score

y_train = pd.read_csv(path_train)['Survived']

X_train = df_combined.iloc[:891]
X_test = df_combined.iloc[891:]

X_train.shape, X_test.shape, y_train.shape

from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X_train, y_train)

df_features = pd.DataFrame({'Features' : X_train.columns})
df_features['Importance'] = clf.feature_importances_
df_features.sort_values(by=['Importance'], ascending=True,inplace=True)
df_features.set_index('Features', inplace=True)

df_features.plot(kind='barh', figsize=(20, 20))

model = SelectFromModel(clf, prefit = True, threshold='0.75*mean')
X_train_reduced = model.transform(X_train)
X_test_reduced = model.transform(X_test)

X_train_reduced.shape, X_test_reduced.shape

lr = LogisticRegression()

lr_acc = cross_val_score(lr, X_train_reduced, y_train, cv=5, scoring='accuracy')
lr_acc = np.mean(lr_acc)

svc = SVC()

svc_acc = cross_val_score(svc, X_train_reduced, y_train, cv=5, scoring='accuracy')
svc_acc = np.mean(svc_acc)

knn = KNeighborsClassifier()

knn_acc = cross_val_score(knn, X_train_reduced, y_train, cv=5, scoring='accuracy')
knn_acc = np.mean(knn_acc)

gs = GaussianNB()

gs_acc = cross_val_score(gs, X_train_reduced, y_train, cv=5, scoring='accuracy')
gs_acc = np.mean(gs_acc)

tr = DecisionTreeClassifier()

tr_acc = cross_val_score(tr, X_train_reduced, y_train, cv=5, scoring='accuracy')
tr_acc = np.mean(tr_acc)

rf = RandomForestClassifier()

rf_acc = cross_val_score(rf, X_train_reduced, y_train, cv=5, scoring='accuracy')
rf_acc = np.mean(rf_acc)

gbc = GradientBoostingClassifier()

gbc_acc = cross_val_score(gbc, X_train_reduced, y_train, cv=5, scoring='accuracy')
gbc_acc = np.mean(gbc_acc)

mlp = MLPClassifier()

mlp_acc = cross_val_score(mlp, X_train_reduced, y_train, cv=10, scoring='accuracy')
mlp_acc = np.mean(mlp_acc)

models = pd.DataFrame({
    'Model' : ['SVM', 'Linear Regression', 'Random Forest', 'Decision Tree',
               'Gradient Boosting Classifier','KNN', 'Gausian Naive Bayes'],
    'Score': [svc_acc, lr_acc, rf_acc, tr_acc, gbc_acc, knn_acc, gs_acc]
})

models.sort_values(by='Score', ascending=False)

# param_grid = {
#     'n_estimators' : [10, 50, 100, 500],
#     'subsample' : [0.1, 0.3, 0.5, 0.7],
#     'max_features': ['sqrt', 'auto', 'log2'],
#     'learning_rate' : [0.1, 0.01, 0.001],
#     'max_depth' : [4, 6, 8],
#     'min_samples_split': [2, 7, 15],
#     'min_samples_leaf': [1, 5, 15],
# }

param_grid = {
    'max_depth' : [6, 8, 10, 15],
    'n_estimators': [1200],
    'max_features': ['sqrt'],
    'min_samples_split': [2, 7, 15, 30],
    'min_samples_leaf': [1, 15, 30, 60],
    'bootstrap': [True],
    }

cv = StratifiedKFold(y_train, n_folds=10)

grid_search = GridSearchCV(rf, scoring='accuracy', param_grid=param_grid, cv=cv)

grid_search.fit(X_train_reduced, y_train)

best_params = grid_search.best_params_

best_model = RandomForestClassifier(**best_params)

cv_result = cross_val_score(best_model, X_train_reduced, y_train, cv=5, scoring='accuracy' )

np.mean(cv_result)

0.83278336271498521

best_model.fit(X_train_reduced, y_train)

y_pred = best_model.predict(X_test_reduced)

# lr = LogisticRegression()
# lr.fit(X_train_reduced, y_train)
# y_pred = lr.predict(X_test_reduced)

df_submission = pd.DataFrame({
    "PassengerId": df_test["PassengerId"],
    "Survived": y_pred
})

df_submission.to_csv(r'C:\Users\core_\Documents\all\submission_all_rf_10th.csv', index=False)

