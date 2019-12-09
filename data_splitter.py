import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data = pd.read_csv("data/mushrooms.csv")
    label_encoder = LabelEncoder()

    for column in data.columns:
        data[column] = label_encoder.fit_transform(data[column])
    #对标签进行数字化编码
    labels = data["class"]
    features = data.drop(["class"], axis=1).astype(float)
    #删除表中的某一行或者某一列
    Features_train, Features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)
    #分成数据集（0.2测试集，0.8训练集）
    Features_train.to_csv(r"data/Features_train.csv", index=None, header=True)
    Features_test.to_csv(r"data/Features_test.csv", index=None, header=True)
    labels_train.to_csv(r"data/labels_train.csv", index=None, header=True)
    labels_test.to_csv(r"data/labels_test.csv", index=None, header=True)
