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


def load_characteristic_vector(file_name):
    with open(file_name,"r",encoding = "utf-8") as file:
        reader = csv.reader(file)
        data = [row for row in reader]
        vector = np.array(data)
        vector = np.delete(vector, 0, axis=0)
        vector = np.delete(vector, 0, axis=1)
        vector = vector.astype(np.float)
    return vector

def group_and_merge(list):
    t = pd.DataFrame(list)
    data = t[2].groupby([t[0],t[1]]).max()
    result = pd.DataFrame(data).reset_index().values.tolist()
    return result


def load_label(file_name):
    with open(file_name, "r", encoding = "utf-8") as file:
        reader = csv.reader(file)
        data = [row[12] for row in reader]
        vector = np.array(data)
        vector = np.delete(vector, 0, axis = 0)
        vector = vector.astype(np.int)
        return vector


# 加载数据，将训练数据集3（340万条数据）作为训练集，训练数据4（37万条数据）作为测试集
print("开始加载数据")
train_data = load_characteristic_vector('./zhangyan/Processed_train_4.csv')
train_label = load_label('./zhangyan/user_video_act_train_4.csv')
test_data = load_characteristic_vector('./zhangyan/Processed_train_3.csv')
test_label = load_label('./zhangyan/user_video_act_train_3.csv')
pre_data = load_characteristic_vector('./zhangyan/Processed_pre.csv')
x_train = tf.convert_to_tensor(train_data, dtype=tf.float32)
y_train = tf.convert_to_tensor(train_label, dtype=tf.int32)
x_test = tf.convert_to_tensor(test_data, dtype=tf.float32)
y_test = tf.convert_to_tensor(test_label, dtype=tf.int32)
result_pre = tf.convert_to_tensor(pre_data, dtype=tf.float32)
print("数据加载成功！")


# 创建模型，目前设的是三层
def create_model():
    return Sequential([
        #layers.Dense(256, activation='relu'),
        #layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        #layers.Dense(16, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])

model = create_model()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

train_dataset = train_dataset.shuffle(60000).batch(128)
test_dataset = test_dataset.batch(128)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
#loss_object = tf.keras.losses.categorical_crossentropy()
optimizer = optimizers.Adam(lr = 0.001)

train_loss = metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = metrics.SparseCategoricalAccuracy('train_accuracy')
#train_accuracy = metrics.Accuracy('train_accuracy')
test_loss = metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = metrics.SparseCategoricalAccuracy('test_accuracy')
#test_accuracy = metrics.Accuracy('test_accuracy')

def train_step(x_train, y_train):
    with tf.GradientTape() as tape:
        predictions = model(x_train, training=True)
        loss = loss_object(y_train, predictions)
        loss = tf.reduce_sum(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        #train_accuracy(y_train, predictions)
        train_accuracy.update_state(y_train, predictions)


def test_step(x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    loss = tf.reduce_sum(loss)
    test_loss(loss)
    #test_accuracy(y_test, predictions)
    test_accuracy.update_state(y_test, predictions)

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

# 创建日志目录，以供模型训练完使用tensorboard进行查看
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# 设置训练epoch
EPOCHS = 10
# 设置是否训练的标志符，“1”表示进行训练，“0”表示不训练直接加载模型
TRAIN = 0
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
        print(template.format(epoch + 1,train_loss.result(),train_accuracy.result(),test_loss.result(),test_accuracy.result()))

        # 清空容器状态
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()
    print("训练完成，在命令行输入tensorboard --logdir logs/gradient_tape的存储路径")
    # 模型保存
    print("保存模型")
    model.save('model.h5')

else:
    # 模型加载
    print("开始加载模型")
    model = tf.keras.models.load_model('model.h5')

# 计算预测结果
print("开始计算结果")
result = model(result_pre)
drop_result = tf.argmax(result, axis=1)
file_name = './zhangyan/user_video_act_val_triple_withId_noLabel_1.csv'
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
    filename = 'result.json'
    with open(filename,'a') as file_obj:
        #json.dump(output, file_obj)
        #json.dumps(output,cls=NpEncoder)
        file_obj.write(json.dumps(output,cls=NpEncoder))
        file_obj.write('\n')
print("保存完成！")