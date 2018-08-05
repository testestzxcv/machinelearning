from sklearn import svm, metrics
datas = [   #데이터
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]
labels = [0, 100, 100, 0]   # 데이터에 대한 답
examples = [    # 테스트
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
]
examples_label = [0, 100, 100, 0]   # 테스트에 대한 답

clf = svm.SVC()
clf.fit(datas, labels)      # 데이터, 답 // 숫자
results = clf.predict(examples)     # 이것에 대한 답을 추측하라

print(results)

score = metrics.accuracy_score(examples_label, results)     # 답, 예측 결과
print("정답률:", score)