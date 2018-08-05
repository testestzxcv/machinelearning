import pandas
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

csv = pandas.read_csv("iris_o.csv")
data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
label = csv["Name"]

clf = svm.SVC()
clf.fit(data, label)
results = clf.predict([
    [5.1, 3.0, 1.3, 0.2]
])
print(results)


