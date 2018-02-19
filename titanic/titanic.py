import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler as SC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# データを読み込む
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# データ整形用関数の定義
def format_data(data):
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Sex"] = data["Sex"].replace(["male", "female"], [0, 1])
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data["Embarked"] = data["Embarked"].fillna("S").replace(["C", "Q", "S"], [0, 1, 2])
    data = data.drop(["Name", "Ticket", "Cabin"], axis=1)
    return data

# データを整形する
train_data = format_data(train)
test_data = format_data(test)

clf = LR()

# 使用すると特徴
use_variable = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

train_X = train_data[use_variable]
train_y = train_data["Survived"]

test_X = test_data[use_variable]

# 標準化
sc = SC()
train_X_std = sc.fit_transform(train_X)
test_X_std = sc.fit_transform(test_X)

#clf.fit(train_X_std, train_y)
diparameter = {
    "C": [10**i for i in range(-10, 20)]
}
gsc = GridSearchCV(clf, param_grid=diparameter, cv=3)
gsc.fit(train_X_new, train_y)

print(gsc.score(train_X_new,train_y))

#predict = model.predict(test_X_std)
predict = gsc.predict(test_X_std)

submission = pd.DataFrame({
    "PassengerId": test_data["PassengerId"],
    "Survived": predict
})

submission.to_csv("submission_std_gsc.csv", index=False)