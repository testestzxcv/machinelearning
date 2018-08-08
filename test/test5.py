from sklearn import model_selection, svm, metrics
import pandas

train_csv = pandas.read_csv("./mnist/train.csv", header=None)
tk_csv = pandas.read_csv("./mnist/t10k.csv", header=None)

def test(l):
    output = []
    for i in l:
        output.append(float(i) / 256)
    return output

train_csv_data = list(map(test, train_csv.iloc[:, 1:].values))
tk_csv_data = list(map(test, tk_csv.iloc[:, 1:].values))
train_csv_label = train_csv[0].values
tk_csv_label = tk_csv[0].values


clf = svm.SVC() # 알고리즘 선택
clf.fit(train_csv_data, train_csv_label)   # 학습
predict = clf.predict(tk_csv_data) # 예측
score = metrics.accuracy_score(tk_csv_label, predict)    # 정답률 구하는 함수
print("정답률:", score)