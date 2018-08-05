#데이터 탐색 및 전처리
import matplotlib
import pandas as pd
import os

from tensorflow.contrib.distributions.python.ops.bijectors import inline

os.chdir(r"F:\[내문서]\[프로그래밍]\교육\4_딥러닝\all")
os.getcwd()

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# train = pd.read_csv('train.csv', index_col='PassengerId')
# test = pd.read_csv('test.csv', index_col='PassengerId')

train.head(5)

#from IPython.display import Image

test.head(5)

train.shape

test.shape

train.info()

test.info()

train.isnull().sum()

# 데이터 시각화

import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
sns.set() # setting seaborn default for plots - 디폴트 값으로 설정

def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    print(survived)
    dead = train[train['Survived']==0][feature].value_counts()
    print(dead)
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10,5))

bar_chart('Sex')# 성별에 따른 사망통계

# 남자가 많이 죽음

bar_chart('Pclass') # 클래스에 따른 사망통계.

# 3등석(저렴한 좌석)일수록 많이 사망.

bar_chart('SibSp') # 가족수에 따른 사망 통계

# 혼자 탄 경우 조금 더 많이 사망.

bar_chart('Embarked') # 승선산 선착장에 따른 사망 통계

# S에서 탔을 경우 사망확률이 높을 가능성 있음.

train.head(5)

train_test_data = [train, test]

train_test_data

# 정규표현식을 사용해 Mr. Mrs. 등 Title 추출
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand = False)

train['Title'].value_counts()

test['Title'].value_counts()

title_mapping = {"Mr":0, "Miss":1, "Mrs": 2, "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3,
                "Mile": 3, "Countess": 3, "Ms": 3, "Lady": 3, "Jonkheer":3, "Don": 3, "Dona": 3, "Mme": 3,
                "Capt": 3, "Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

train.head(5)

test.head(5)

# Name 칼럼 삭제
train.drop('Name', axis=1, inplace=True)
test.drop('Name', axis=1, inplace=True)

train.head(5)

test.head(5)

bar_chart('Title')

# Name에 대한 전처리 종료. 이제 성별로 진행.

#성별 매핑
sex_mapping = {"male": 0, "female": 1}
for dataset in train_test_data:
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

bar_chart('Sex')

train.head(5)

# Age는 Nan값을 해당 그룹이 속하는 Median값으로 대체
train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace=True)

test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace=True)

train.head(5)

#Age를 Binning
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 26), 'Age'] = 1,
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 36), 'Age'] = 2,
    dataset.loc[(dataset['Age'] > 36) & (dataset['Age'] <= 62), 'Age'] = 3,
    dataset.loc[ dataset['Age'] > 62, 'Age'] = 4

train.head(5)

bar_chart('Age')

# embarked는 탑승한 선착장에 관한 정보. 고소득 거주자 지역에서 탑승했으면 1등석일 확률이 높고 생존할 확률이 높아진다.
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()

df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train.head(5)

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

train.head(5)

# Nan인 인스턴스가 속한 Pclass의 median값을 해당 결측지를 가진 인스턴스에 넣어준다. Age와 동일
train["Fare"].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace=True)
test["Fare"].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace=True)

train.head(5)

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 17, 'Fare'] = 0,
    dataset.loc[(dataset['Fare'] > 17) & (dataset['Fare'] <= 30), 'Fare'] = 1,
    dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 100), 'Fare'] = 2,
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

train.head(5)

# 앞의 알파벳만 따온다.
train.Cabin.value_counts()

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind='bar', stacked=True, figsize=(10,5))

# 1등석에서 ABCDE로 시작하는 cabin이 많지만 2등석 3등석은 아예 없다.
# Cabin은 객실을 뜻하는 것이고 알파벳과 숫자의 조합으로 이루어진다.
# 제일 앞에 있는 알파벳만 추출하여 연관성을 보기위해 시각화를 한것이다.

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

# Familysize

train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

train["FamilySize"].max()

test["FamilySize"].max()

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

train.head(5)

features_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(features_drop, axis=1)
test = test.drop(features_drop, axis=1)
train = train.drop(['PassengerId'], axis=1)

train.head(5)

train_data = train.drop('Survived', axis=1)
target = train['Survived']

train_data.shape, target.shape

# 기존에 SibSP와 Parch 두개로 나누어져있던 칼럼을 Familysize 하나로 통합한다.
# 그후 해당 두 칼럼을 drop
# 전처리 종료
# 모델학습 시작

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import numpy as np


train.info()

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print("score == ", score)

round(np.mead(score)*100, 2)

clf = DecisionTreeClassifier()

clf

scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

round(np.mean(score)*100, 2)

clf = RandomForestClassifier(n_estimators=13)
clf

scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

round(np.mean(score)*100, 2)

clf = GaussianNB()
clf

scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

round(np.mean(score)*100, 2)

clf = SVC(C=1, kernel='rbf', coef0=1)
clf

scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

round(np.mean(score)*100, 2)

clf = SVC(C=1, kernel='rbf', coef0=1)
clf.fit(train_data, target)

test_data = test.drop("PassengerId", axis=1).copy()
prediction = clf.predict(test_data)

import collections, numpy

collections.Counter(prediction)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": prediction
})

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv("submission.csv")
submission.head(5)

