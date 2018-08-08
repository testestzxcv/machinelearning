import pandas
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

# clf = svm.SVC() # 알고리즘 선택
# clf.fit()   # 학습
# predict = clf.predict() # 예측
# score = metrics.accuracy_score()    # 정답률 구하는 함수
# predict("정답률:", score)

csv = pandas.read_csv("iris.csv")
data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
label = csv["Name"]

train_data, test_data, train_label, test_label = \
    train_test_split(data, label)

clf = svm.SVC()
clf.fit(train_data, train_label)
results = clf.predict(test_data)

score = metrics.accuracy_score(results, test_label)
print("정답률:", score)

