from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


# 计算信息熵
def get_entropy(data_df, columns=None):
    pe_value_array = data_df[columns].unique()
    ent = 0.0
    for x_value in pe_value_array:
        p = float(data_df[data_df[columns] == x_value].shape[0]) / data_df.shape[0]
        logp = np.log(p)
        ent -= p * logp
    return ent


# 提取训练集特征和标签
domain = []
label = []
feature = []

with open('train.txt') as f:
    for dom in f:
        dom = dom.strip('\n')
        line = dom.split(',')
        domain.append(line[0])
        if line[-1] == 'dga':
            label.append(1)
        else:
            label.append(0)

# 使用长度、数字、信息熵3个特征
for it in domain:
    cur = [len(it), sum(c.isdigit() for c in it), get_entropy(pd.DataFrame(list(it)), 0)]
    feature.append(cur)

# 训练模型
pd.DataFrame(feature).to_csv("tmp.csv")
clf = RandomForestClassifier(128)
clf.fit(feature, label)


# 进行测试
test_case = []
domain_name = []
with open("test.txt") as f:
    for dom in f:
        dom = dom.strip('\n')
        domain_name.append(dom)
        cur = [len(dom), sum(c.isdigit() for c in dom), get_entropy(pd.DataFrame(list(dom)), 0)]
        test_case.append(cur)

# 输出测试结果
result = clf.predict(test_case)
with open("result.txt", "w+") as f:
    for i in range(len(domain_name)):
        if result[i] == 1:
            f.write(domain_name[i] + ",dga\n")
        else:
            f.write(domain_name[i] + ",notdga\n")