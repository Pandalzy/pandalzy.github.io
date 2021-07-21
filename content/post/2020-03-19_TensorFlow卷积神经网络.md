---
title: TensorFlow卷积神经网络
date: 2020-03-19 19:13:27
tags: [TensorFlow, 深度学习]
---

## 卷积神经网络简介

### 与传统多层神经网络对比

-   传统意义上的多层神经网络是只有输入层、隐藏层与输出层。其中隐藏层的层数根据需要而定，没有明确的理论推导来说明到底多少层合适
-   卷积神经网络 CNN，在原来多层神经网络的基础上，加入了更加有效的特征学习部分，具体操作就是在原来的全连接层前面加入了卷积层与池化层。**卷积神经网络出现，使得神经网络层数得以加深，“深度”学习由此而来**
<!-- more -->

```
输入层
隐藏层
	卷积层
	激活层
	池化层
	全连接层
输出层
```

> 通常所说的深度学习，一般指的是这些 CNN 等新的结构以及一些新的方法（比如新的激活函数 Relu 等），解决了传统多层神经网路的一些难以解决的问题

## 卷积神经网络原理

在隐藏层加入卷积层和池化层，激活层

### 结构

-   卷积层
    -   通过在原始图像上平移来提取特征
-   激活层
    -   增加非线性分割能力
-   池化层
    -   减少学习的参数，降低网络的复杂度（最大池化和平均池化）
-   全连接层
    -   为了能够达到分类效果

### 卷积层（Convolution Layer）

卷积神经网络中每层卷积层由若干卷积单元（卷积核）组成，每个卷积单元的参数都是通过反向传播算法最佳化得到的。

卷积运算的目的是特征提取，第一层卷积层可能只能提取一些低级的特征如，边缘、线条和角等层级，更多层的网络能从低级特征中迭代提取更复杂的特征

卷积核、filter、过滤器、模型参数、卷积单元，相同

#### 卷积核四大要素

-   个数
    -   不同的卷积核带的权重和偏置都不一样，即随机初始化的参数
-   大小
    -   `1*1 3*3 5*5`
-   步长
    -   跳几格
-   零填充大小

卷积核可以理解为一个观察的人，带着若干权重和一个偏置去观察，进行特征加运算

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200314195926.png)

> 上述要加上偏置

#### 输出大小计算公式

-   输出体积大小`H1*W1*D1`
    -   输入图像`32*32*1`
-   四个超参数
    -   filter 数量 K
    -   filter 大小 F
    -   步长 S
    -   零填充大小 P
-   输出体积大小`H2*W2*D2`
    -   `H2=(H1-F+2P)/S+1`
    -   `W2=(W1-F+2P)/S+1`
    -   `D2=K`

#### 计算案例

-   假设已知的条件：输入图像`32*32*1`，50 个 filter，大小为`5*5`，移动步长为 1，零填充为 1，请求出输出大小

    -   `H2=(H1-F+2P)/S+1=(32-5+2)/1+1=30`
    -   `W2=(H1-F+2P)/S+1=(32-5+2)/1+1=30`
    -   `D2=K=50`
    -   `[30, 30, 50]`

-   假设已知的条件：输入图像`32*32*1`，50 个 filter，大小为`3*3`，移动步长为 1，输出大小`32*32`，求零填充

#### 多通道图片观察

### 卷积网络 API

-   `tf.nn.conv2d(input, filter, strides, padding, name)`
    -   计算给定 4-D input 和 filter 张量的 2 维卷积
    -   input：给定的输入张量，具有`[batch, height, width, channel]`，类型为 float32, 64
        -   batch：批数量
    -   filter：指定过滤器的权重数量
        -   `[filter_height, filter_width, in_channels, out_channels]`
            -   [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
            -   第三维`in_channels`，就是参数 input 的第四维
        -   变量，`initial_value=random_normal(shape=[F, F, 3/1, K])`
    -   strides：`strides = [1, stride, stride, 1]`，步长
    -   padding：“SAME”，“VAKID”
-   零填充的两种方式
    -   SAME：越过边缘取样，取样的面积和输入图像的像素宽度一致
        -   公式：`ceil(H/S)`
            -   H 为输入图片的高或者宽，S 为步长
            -   无论过滤器的大小是多少，零填充的数量由 API 计算
    -   VALID：不越过边缘取样，取样的面积小于输入人的图像的像素宽度，不填充

> 在 Tensorflow 中，卷积 API 设置为“SAME”之后，如果步长为 1，输出高度与输入大小一样

### 激活函数

#### Relu

$$
relu=max(0,x)
$$

x 小于 0，值为 0，大于 0，为其本身

#### 为什么采用新的激活函数

-   Relu 优点
    -   有效解决梯度消失问题
    -   计算速度非常快，只需要判断输入是否大于 0，SGD（批梯度下降）的求解速度远快于 sigmoid 和 tanh
-   sigmoid 缺点
    -   计算量相对较大，在深层网络中，sigmoid 函数反向传播时，很容易就会出现梯度消失的情况

#### 激活函数 API

-   `tf.nn.relu(features, name=None)`
    -   features：卷积后加上偏置的结果
    -   return：结果

### 池化层

Pooling 层主要的作用是特征提取，通过去掉 Feature Map 中不重要的样本，进一步减少参数的数量。方法有很多，通常采用最大池化

-   max_polling：取池化窗口的最大值
-   avg_poling：取池化窗口的平均值

#### 池化层 API

-   `tf.nn.max_pool(value, ksize, strides, padding, name)`
    -   输入上执行最大池数
    -   value：4-D Tensor 形状`[batch, height, width, channel]`
    -   channel：不是原始图片的通道数，而是多少 filter 观察
    -   ksize：池化窗口大小，`[1, ksize, ksize, 1]`
    -   strides：步长大小，`[1, strides, strides, 1]`
    -   padding：使用填充算法类型，“SAME”，“VAKID”

> 卷积向下取整，池化向上取整

## 案例：CNN-Mnist 手写数字识别

### 网络设计

-   第一个卷积大层
    -   卷积层：32 个 filter、大小 5\*5、strides=1、padding='SAME'
        -   `tf.nn.conv2d(input, filter, strides, padding'SAME')`
        -   input
            -   `[None, 28, 28, 1]`
        -   filter
            -   `weights = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 1, 32]))`
            -   `bias = tf.Variable(initial_value=tf.random_normal(shape=[32]))`
        -   strides
            -   `[1, 1, 1, 1]`
        -   输出形状：`[None, 28, 28, 32]`
    -   激活层：Relu
        -   `tf.nn.relu(features)`
    -   池化层：大小`2*2`、strides=2
        -   `tf.nn.max_pool()`
        -   输入形状：`[None, 28, 28, 32]`
        -   输出形状：`[None, 14, 14, 32]`（公式计算）
-   第二个卷积大层
    -   卷积层：64 个 filter、大小 5\*5、strides=1、padding='SAME'
        -   `tf.nn.conv2d(input, filter, strides, padding'SAME')`
        -   input
            -   `[None, 14, 14, 32]`
        -   filter
            -   `weights = tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, 64]))`
            -   `bias = tf.Variable(initial_value=tf.random_normal(shape=[64]))`
        -   strides
            -   `[1, 1, 1, 1]`
        -   输出形状：`[None, 14, 14, 64]`
    -   激活层：Relu
        -   `tf.nn.relu(features)`
    -   池化层：大小`2*2`、strides=2
        -   输入形状：`[None, 14, 14, 64]`
        -   输出形状：`[None, 7, 7, 64]`（公式计算）
-   全连接层
    -   `tf.reshape()`
    -   `[None, 7, 7, 64]->[None, 7*7*64]`
    -   `[None, 7*7*64] * [7*7*64, 10] = [None, 10]`
    -   `y_predict = tf.matmul(pool2, weights) + bias`

### 调参：提高准确率

-   学习率，一般：0.01
-   随机初始化的权重、偏置的值
    -   `tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))`
    -   `stddev`
-   选择优化器
    -   `tf.train.AdamOptimizer(0.01).minimize(loss)`
-   调整网络
    -   使用`batch normalization`：批标准化
    -   `droupout`层：使一些神经元失效
    -   防止过拟合

### 改为高级 API

### 主要代码

```python
def create_model(x):
    """
    实现构建卷积神经网络
    :return:
    """
    y_predict = 0
    # 第一个卷积大层
    with tf.variable_scope('conv1'):
        # 卷积层
        input_x = tf.reshape(x, shape=[-1, 28, 28, 1])

        conv1_weights = create_variable([5, 5, 1, 32])
        conv1_bias = create_variable([32])
        conv1_x = tf.nn.conv2d(input_x, filter=conv1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv1_bias
        # 激活层
        relu1_x = tf.nn.relu(conv1_x)
        # 池化层
        pool1_x = tf.nn.max_pool(relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 第二个卷积大层
    with tf.variable_scope('conv2'):
        # 卷积层

        conv2_weights = create_variable([5, 5, 32, 64])
        conv2_bias = create_variable([64])
        conv2_x = tf.nn.conv2d(pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_bias

        # 激活层
        relu2_x = tf.nn.relu(conv2_x)

        # 池化层
        pool2_x = tf.nn.max_pool(relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 全连接层
    with tf.variable_scope("full_connection"):
        # [None, 7, 7, 64]->[None, 7 * 7 * 64]
        # [None, 7 * 7 * 64] * [7 * 7 * 64, 10] = [None, 10]
        x_fc = tf.reshape(pool2_x, shape=[-1, 7 * 7 * 64])
        weights_fc = create_variable(shape=[7 * 7 * 64, 10])
        bias_fc = create_variable(shape=[10])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict
```

## 实战：验证码识别

### 数据集

一个图片对应 4 个目标值

`NZPP -> [13, 25, 15, 15] -> [[one-hot], [], [], []]`

### 损失衡量

softmax 交叉熵，只适用于类别相互排斥的，一个样本对应一个目标值

`[4, 26] -> [4*26]`

使用 sigmoid 交叉熵

-   sigmoid 交叉熵损失函数

-   `tf.nn.sigmoid_cross_entropy_with_logits(labels=None, logits=None)`
    -   labels：真实值，为 one_hot 编码形式，和 logits 一样
    -   logits：logits 值，输出层的加权计算结果
-   对真实值进行 one_hot 编码
-   `tf.one_hot(indices, depth, axis=None, name=None)`
    -   indices：需要编码的张量
    -   depth：one_hot 编码的深度，这里 26 个字母，为 26
    -   axis：填充的维度，默认是-1

### 流程分析

1. 读取图片数据
    1. filename -> 标签值
2. 解析 csv 文件，将标签值转为`[字母序号, 字母序号, 字母序号, 字母序号]`
3. 将 filename 与标签值联系起来
4. 构建卷积神经网络
    1. 利用手写识别网络
    2. 产生 y_predict
5. 构造损失函数
6. 优化损失
7. 计算准确率
8. 开启会话、开启线程

### 代码

```python
import tensorflow as tf
import glob
import pandas as pd
import numpy as np


def read_pic():
    """
    读取图片数据
    :return:
    """
    # 1、文件队列
    file_names = glob.glob('./GenPics/*.jpg')
    file_queue = tf.train.string_input_producer(file_names)
    # 2、读取与解码
    reader = tf.WholeFileReader()
    file_name, image = reader.read(file_queue)
    decoded = tf.image.decode_jpeg(image)
    # 更新形状，将图片形状确定下来
    decoded.set_shape([20, 80, 3])

    # 修改图片类型
    image_cast = tf.cast(decoded, tf.float32)
    # 3、批处理
    filename_batch, image_batch = tf.train.batch([file_name, image_cast], batch_size=100, num_threads=1, capacity=200)
    return filename_batch, image_batch


def parse_csv():
    """
    解析csv文件，建立文件名与标签值表格
    :return:
    """
    csv_data = pd.read_csv("./GenPics/labels.csv", names=["file_num", "chars"], index_col="file_num")

    labels = []
    for label in csv_data["chars"]:
        tmp = []
        for letter in label:
            tmp.append(ord(letter) - ord("A"))
        labels.append(tmp)

    csv_data["labels"] = labels

    return csv_data


def filename2label(filenames, csv_data):
    """
    将样本特征值与目标值一一对应
    :param filenames:
    :param csv_data:
    :return:
    """
    labels = []

    # 将b'文件名中的数字提取出来并索引相应的标签值

    for file_name in filenames:
        digit_str = "".join(list(filter(str.isdigit, str(file_name))))
        label = csv_data.loc[int(digit_str), "labels"]
        labels.append(label)
    return np.array(labels)


def create_variable(shape):
    """
    创建变量
    :param shape:
    :return:
    """
    return tf.Variable(initial_value=tf.random_normal(shape=shape, stddev=0.01))


def create_model(x):
    """
    实现构建卷积神经网络
    :param x:[None, 20, 80, 3]
    :return:
    """
    # 第一个卷积大层
    with tf.variable_scope('conv1'):
        # 卷积层
        conv1_weights = create_variable([5, 5, 3, 32])
        conv1_bias = create_variable([32])
        conv1_x = tf.nn.conv2d(x, filter=conv1_weights, strides=[1, 1, 1, 1], padding='SAME') + conv1_bias
        # 激活层
        relu1_x = tf.nn.relu(conv1_x)
        # 池化层
        pool1_x = tf.nn.max_pool(relu1_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 第二个卷积大层
    with tf.variable_scope('conv2'):
        # [None, 20, 80, 3] --> [None, 10, 40, 32]
        # 卷积层
        conv2_weights = create_variable([5, 5, 32, 64])
        conv2_bias = create_variable([64])
        conv2_x = tf.nn.conv2d(pool1_x, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME') + conv2_bias
        # 激活层
        relu2_x = tf.nn.relu(conv2_x)
        # 池化层
        pool2_x = tf.nn.max_pool(relu2_x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 全连接层
    with tf.variable_scope("full_connection"):
        # [None, 10, 40, 32] -> [None, 5, 20, 64]
        # [None, 5, 20, 64] -> [None, 5 * 20 * 64]
        # [None, 5 * 20 * 64] * [5 * 20 * 64, 4 * 26] = [None, 4 * 26]
        x_fc = tf.reshape(pool2_x, shape=[-1, 5 * 20 * 64])
        weights_fc = create_variable(shape=[5 * 20 * 64, 4 * 26])
        bias_fc = create_variable(shape=[104])
        y_predict = tf.matmul(x_fc, weights_fc) + bias_fc

    return y_predict


if __name__ == '__main__':
    filename, image = read_pic()
    csv_data = parse_csv()

    # 1、准备数据
    x = tf.placeholder(dtype=tf.float32, shape=[None, 20, 80, 3])
    y_true = tf.placeholder(tf.float32, shape=[None, 4 * 26])

    # 2、构建模型
    y_predict = create_model(x)

    # 3、构造损失函数
    loss_list = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_predict)
    loss = tf.reduce_mean(loss_list)

    # 4、优化损失
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    # 5、计算准确率
    y_predict_argmax = tf.argmax(tf.reshape(y_predict, shape=[-1, 4, 26]), axis=2)
    y_true_argmax = tf.argmax(tf.reshape(y_true, shape=[-1, 4, 26]), axis=2)
    equal = tf.equal(y_predict_argmax, y_true_argmax)
    equal_list = tf.reduce_all(equal, axis=1)
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(1000):
            filename_value, image_value = sess.run([filename, image])
            labels = filename2label(filename_value, csv_data)
            # 标签值转换为one-hot
            labels_value = tf.reshape(tf.one_hot(labels, depth=26), shape=[-1, 4 * 26]).eval()
            _, error, accuracy_value = sess.run([optimizer, loss, accuracy],
                                                feed_dict={x: image_value, y_true: labels_value})
            print("第%d次训练后损失为%f，准确率为%f" % (i + 1, error, accuracy_value))

        coord.request_stop()
        coord.join(threads)

```
