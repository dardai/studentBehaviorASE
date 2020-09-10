# -*- coding: utf-8 -*-
import datetime
import os
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

def load_label(file_name):
    with open(file_name, "r", encoding = "utf-8") as file:
        reader = csv.reader(file)
        data = [row[12] for row in reader]
        vector = np.array(data)
        vector = np.delete(vector, 0, axis = 0)
        vector = vector.astype(np.int)
        return vector


# 加载数据，将训练数据集3（340万条数据）作为训练集，训练数据4（37万条数据）作为测试集
train_data = load_characteristic_vector('../zhangyan/Processed_train_3.csv')
print(train_data.shape)
train_label = load_label('../zhangyan/user_video_act_train_3.csv')
test_data = load_characteristic_vector('../zhangyan/Processed_train_4.csv')
test_label = load_label('../zhangyan/user_video_act_train_4.csv')
x_train = tf.convert_to_tensor(train_data, dtype=tf.float32)
y_train = tf.convert_to_tensor(train_label, dtype=tf.int32)
x_test = tf.convert_to_tensor(test_data, dtype=tf.float32)
y_test = tf.convert_to_tensor(test_label, dtype=tf.int32)
#y = y[0:2000]

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

train_dataset = train_dataset.shuffle(60000).batch(64)
test_dataset = test_dataset.batch(64)

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
        train_accuracy(y_train, predictions)

def test_step(x_test, y_test):
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    loss = tf.reduce_sum(loss)
    test_loss(loss)
    test_accuracy(y_test, predictions)

# 创建日志目录，以供模型训练完使用tensorboard进行查看
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

# 设置训练epoch
EPOCHS = 100

for epoch in range(EPOCHS):
    for (x_train, y_train) in train_dataset:
        train_step(x_train, y_train)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for (x_test, y_test) in test_dataset:
        test_step(x_test, y_test)
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

print("训练完成，在命令行输入tensorboard --lodir logs/gradient_tape的存储路径")