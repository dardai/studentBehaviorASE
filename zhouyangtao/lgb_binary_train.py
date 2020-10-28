# -*- coding: utf-8 -*-
import csv
import json

import lightgbm as lgb
import math
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# 对 dataframe 结果按 item_id 和 course_id 进行分组去重
def group_and_merge(list):
    t = pd.DataFrame(list)
    data = t[2].groupby([t[0],t[1]]).max()
    #print(data)
    result = pd.DataFrame(data).reset_index().values.tolist()
    #print(result)
    return result

def Get_Data():
    print("加载训练数据···")
    # 822123条训练数据
    train_data = pd.read_csv("../newfeature/new_Processed_train.csv")
    train_data = train_data.drop("ID", axis=1)# watching_count_sum
    #train_data = train_data.drop("watching_count_sum", axis=1)
    train_text_data = pd.read_csv("final_train.csv")
    train_text_data = train_text_data.drop("id", axis=1)
    train_text_data = train_text_data.drop("f1", axis=1)
    train_text_data = train_text_data.drop("f2", axis=1)
    train_data = pd.concat([train_data, train_text_data], axis=1)
    train_label = pd.read_csv("../zhangyan/train.csv")
    train_label = train_label[['drop']]
    print("训练数据加载成功！")
    print("加载测试数据···")
    # 91347条测试数据
    test_data = pd.read_csv("../newfeature/new_Processed_test.csv")
    test_data = test_data.drop("ID", axis=1)
    #test_data = test_data.drop("watching_count_sum", axis=1)
    test_text_data = pd.read_csv("final_test.csv")
    test_text_data = test_text_data.drop("id", axis=1)
    test_text_data = test_text_data.drop("f1", axis=1)
    test_text_data = test_text_data.drop("f2", axis=1)
    test_data = pd.concat([test_data, test_text_data], axis=1)
    test_label = pd.read_csv("../zhangyan/test.csv")
    test_label = test_label[['drop']]
    print("测试数据加载成功！")
    pre_data = pd.read_csv("../newfeature/new_Processed_predict.csv")
    pre_data = pre_data.drop("ID", axis=1)
    #pre_data = pre_data.drop("watching_count_sum", axis=1)
    pre_text_data = pd.read_csv("predict_result.csv")
    pre_text_data = pre_text_data.drop("id", axis=1)
    pre_text_data = pre_text_data.drop("f1", axis=1)
    pre_text_data = pre_text_data.drop("f2", axis=1)
    pre_data = pd.concat([pre_data, pre_text_data], axis=1)
    return train_data, train_label, test_data, test_label, pre_data

def get_pre_result(ypred):
    result = []
    for row in ypred:
        #print(row)
        #print(np.argmax(row))
        if row > 0.5:
            result.append(1)
        else:
            result.append(0)
            #result.append(np.argmax(row))
    return result

def get_total_result(result):
    #print('score: ', metrics.accuracy_score(result, y_valid.values.tolist()))
    # 读取预测结果中的 item_id 和 course_id，与预测结果进行匹配
    file_name = '../zhangyan/user_video_act_val_triple_withId_noLabel_1.csv'
    with open(file_name, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
        vector = np.array(data)
        vector = np.delete(vector, 0, axis=0)
        vector = np.delete(vector, 0, axis=1)
        #vector = vector.astype(np.float)
    itemid = [row[0] for row in vector]
    course_id = [row[1] for row in vector]
    data = list(zip(itemid, course_id, result))
    data = group_and_merge(data)
    temp ={}
    for row in data:
        if row[0] not in temp:
           temp[row[0]] = [row[2]]
        else:
            temp[row[0]].append(row[2])
    return temp

def save_json(temp):
    print("json结果保存")
    for key, value in temp.items():
        output = {}
        output["label_list"] = value
        output["item_id"] = key
        filename = 'new_gbm_result.json'
        with open(filename,'a') as file_obj:
            file_obj.write(json.dumps(output,cls=NpEncoder))
            file_obj.write('\n')
    print("保存完成！")

x_train, y_train, x_test, y_test, pre_data = Get_Data()
x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=0)
print(y_valid)
train_data = lgb.Dataset(data=x_train, label=y_train)
test_data = lgb.Dataset(data=x_test, label=y_test)

params = {
    'boosting_type' : 'gbdt',
    'objective' : 'binary',
    'metric' : {'binary_logloss', 'auc'},
    'max_depth' : 20,#25
    'num_leaves' : 40,#40
    'min_child_samples' : 21,#18
    'min_child_weight' : 0.001,#0.001
    'feature_fraction' : 1,#1
    'bagging_fraction' : 1,#0.8
    'bagging_freq' : 2,#2
    'lambda_l1' : 0.1,#0.4
    'lambda_l2' : 15,#0
    'cat_smooth' : 0,
    'learning_rate' : 0.01,#0.05
    'seed' : 2020,
    #"num_class" : 2,
}

num_round = 20000
print("")
bst = lgb.train(params, train_data, num_round, valid_sets=test_data)
bst.save_model('model.txt')
#ypred = bst.predict(x_valid, num_iteration=bst.best_iteration )
ypred = bst.predict(pre_data, num_iteration=bst.best_iteration)
#print(ypred)
result = get_pre_result(ypred)
temp = get_total_result(result)
save_json(temp)