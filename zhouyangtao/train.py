# -*- coding: utf-8 -*-
import datetime
import json
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers,Sequential,optimizers,metrics
import csv

# 解决不同数据格式存储成json时报错的问题
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

# 加载特征向量
def load_characteristic_vector(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
        vector = np.array(data)
        print(len(vector))
        vector = np.delete(vector, 0, axis=0)
        vector = np.delete(vector, 0, axis=1)
        vector = vector.astype(np.float)
    return vector

# 对预测结果按课程id进行分组去重再合并
def group_and_merge(list):
    t = pd.DataFrame(list)
    data = t[2].groupby([t[0], t[1]]).max()
    result = pd.DataFrame(data).reset_index().values.tolist()
    return result

# 加载数据集的标签
def load_label(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        data = [row[11] for row in reader]
        vector = np.array(data)
        vector = np.delete(vector, 0, axis=0)
        vector = vector.astype(np.int)
        return vector

# 模型定义
def create_model():
    return Sequential([
        #layers.Dense(256, activation='relu'),
        #layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        #layers.Dense(16, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

# 模型训练过程
def train_step(x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        loss = tf.reduce_sum(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_accuracy.update_state(y_train, predictions)

# 模型测试过程
def test_step(x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    loss = tf.reduce_sum(loss)
    test_loss(loss)
    test_accuracy.update_state(y_test, predictions)

# 加载数据
print("开始加载数据")

# 加载训练数据的离散特征向量、字幕信息向量和标签
train_data = load_characteristic_vector('../newfeature/New_Processed_train.csv')
train_text_data = load_characteristic_vector('final_train.csv')
train_label = load_label('../zhangyan/train.csv')

# 加载测试数据的离散特征向量、字幕信息向量和标签
test_data = load_characteristic_vector('../newfeature/New_Processed_test.csv')
test_text_data = load_characteristic_vector('final_test.csv')
test_label = load_label('../zhangyan/test.csv')

# 加载预测数据的离散特征向量和字幕信息向量
pre_data = load_characteristic_vector('../newfeature/new_Processed_predict.csv')
pre_text_data = load_characteristic_vector('predict_result.csv')

# 将训练数据、测试数据和预测数据的离散特征向量和字幕信息向量进行拼接
x_train = tf.convert_to_tensor(train_data, dtype=tf.float32)
x_text_train = tf.convert_to_tensor(train_text_data, dtype=tf.float32)
y_train = tf.convert_to_tensor(train_label, dtype=tf.int32)
x_test = tf.convert_to_tensor(test_data, dtype=tf.float32)
x_text_test  = tf.convert_to_tensor(test_text_data, dtype=tf.float32)
y_test = tf.convert_to_tensor(test_label, dtype=tf.int32)
result_pre = tf.convert_to_tensor(pre_data, dtype=tf.float32)
text_pre = tf.convert_to_tensor(pre_text_data, dtype=tf.float32)
x_train = tf.concat([x_train, x_text_train], axis=1)
x_test = tf.concat([x_test, x_text_test], axis=1)
result_pre = tf.concat([result_pre, text_pre], axis=1)

print("数据加载成功！")

# 创建模型
model = create_model()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
train_dataset = train_dataset.shuffle(60000).batch(128)
test_dataset = test_dataset.batch(128)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam(lr=0.001)

train_loss = metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = metrics.SparseCategoricalAccuracy('test_accuracy')

# 创建日志目录，以供模型训练完使用tensorboard进行查看
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# 设置训练epoch
EPOCHS = 100
# 设置是否训练的标志符，“1”表示进行训练，“0”表示不训练直接加载模型
TRAIN = 1
if (TRAIN==1):
    print("开始模型训练")
    for epoch in range(EPOCHS):
        for (x_train, y_train) in train_dataset:
            train_step(x_train, y_train)
        # 将训练数据集的损失值和准确率保存到日志中
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for (x_test, y_test) in test_dataset:
            test_step(x_test, y_test)
        # 将测试数据集的损失值和准确率保存到日志中
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1, train_loss.result(), train_accuracy.result(), test_loss.result(),test_accuracy.result()))

        # 清空容器状态
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
    print("训练完成，在命令行输入tensorboard --logdir logs/gradient_tape的存储路径")
    # 模型保存
    print("保存模型")
    filename = current_time + 'model.h5'
    model.save(filename)

else:
    # 模型加载
    print("开始加载模型")
    model = tf.keras.models.load_model('20200925-153905model.h5')

# 计算预测结果
print("开始计算结果")
result = model(result_pre)
drop_result = tf.argmax(result, axis=1)
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
data = list(zip(itemid, course_id, drop_result.numpy()))
data = group_and_merge(data)

temp ={}
for row in data:
    if row[0] not in temp:
        temp[row[0]] = [row[2]]
    else:
        temp[row[0]].append(row[2])

print("json结果保存")
for key, value in temp.items():
    output = {}
    output["label_list"] = value
    output["item_id"] = key
    filename = current_time + 'normalized_result.json'
    with open(filename,'a') as file_obj:
        #json.dump(output, file_obj)
        #json.dumps(output,cls=NpEncoder)
        file_obj.write(json.dumps(output,cls=NpEncoder))
        file_obj.write('\n')
print("保存完成！")
