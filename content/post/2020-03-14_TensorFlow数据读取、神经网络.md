---
title: TensorFlow数据读取、神经网络
date: 2020-03-14 18:03:00
tags: [TensorFlow, 深度学习]
---

## 文件读取流程

-   QueueRunner
    -   基于队列的输入管道从 TensorFLow 图形开头文件中读取数据
-   Feeding
    -   运行每一步时，Python 代码提供数据
-   预加载数据
    -   TensorFlow 图中的张量包含的数据（对于小数据集）

<!-- more -->

### 文件读取流程

-   第一阶段：构造文件名队列
-   第二阶段：读取与解码
-   第三阶段：批处理

#### 构造文件名队列

-   `tf.train.string_input_producer(string_tensor, shuffle=True)`
    -   string_tensor：含有文件名+路径的一阶张量
    -   num_epochs：过几遍数据，默认无限遍数据
    -   return：文件队列`file_queue`

#### 读取和解码

从队列当中读取文件内容，并进行解码操作

##### 读取文件内容

默认每次读取一个样本

> 文本文件默认一次读取一行，图片默认一次读取一张图片，二进制文件一次读取指定字节数（最好一个样本的字节数），TFRecords 默认一次读取一个 example

-   `tf.TextLineReader`
    -   阅读文本文件都好分隔符值（CSV）格式，默认按行读取
    -   return：读取器实例
-   `tf.WholeFileReader()`
    -   用于读取图片文件
    -   return：读取器实例
-   `tf.FixedLengthRecordReader(record_bytes)`
    -   二进制文件
    -   要读取每个记录是固定数量字节的二进制文件
    -   record_bytes：整数，指定每次读取（一个样本）的字节数
    -   return：读取器实例
-   `tf.TFRecordReader`
    -   读取 TFRecords 文件
    -   return：读取器实例

> 它们有共同的读取方法：`read(file_queue)`，并且都会返回一个 Tensor 元组
>
> `(key文件名字, value默认的内容（一个文本）)`
>
> 由于默认只会读取一个样本，所以如果想要进行批处理，需要使用`tf.train.batch`或`tf.train.shuffle_batch`进行批处理操作，便于之后指定每批次多个样本的训练

##### 内容解码

-   文本
    -   `tf.decode_csv()`
-   图片
    -   `tf.image.decode_jpeg(contents)`
        -   将 JPEG 编码的图像解码为 uint8 张量
        -   return：uint8 张量，3-D 形状[height, width, channels]
    -   `tf.image.decode_png(contents)`
-   二进制
    -   `tf.decode_raw(value, tf.uint8)`
        -   与`tf.FixedLengthRecordReader`搭配使用，二进制读取为 uint8

> 解码阶段，默认所有的内容都解码成`tf.uint8`类型，如果需要转换成指定类型，可以使用`tf.cast()`进行转换

#### 批处理

解码之后，可以直接获取默认的一个样本内容，但如果想要获取多个样本，需要加入到新的队列进行批处理

-   `tf.train.batch(tensors, batch_size, num_threads=1, capacity=32, name=None)`
    -   读取指定大小（个数）的张量
    -   tensors：可以是包含张量的列表，批处理的内容放到列表当中
    -   batch_size：从队列中读取的批处理大小
    -   num_threads：进入队列的线程数
    -   capacity：整数，队列中元素的最大数量
    -   return：tensors
-   `tf.train.shuffle_batch`

### 线程操作

以上的队列都是`tf.train.QueueRunner`对象

每个 QueueRunner 都负责一个阶段，会话中，`tf.train.start_queue_runners`函数会要求图中的每个 QueueRunner 启动它的运行队列操作的线程，（这些操作需要在会话中开启）

-   `tf.train.start_queue_runners(sess=None, coord=None)`
    -   收集图中所有的队列线程，默认同时启动线程
    -   sess：所在的会话
    -   coord：线程协调器
    -   return：所有线程
-   `tf.train.Coordinator()`
    -   线程调度员，对线程进行管理和协调
    -   `request_stop()`：请求停止
    -   `should_stop()`：询问是否结束
    -   `join(threads=None, stop_grace_period_secs=120)`：回收线程
    -   return：线程协调员实例

## 图片数据

### 图片基本知识

-   特征抽取
    -   文本——数值（二维数组`shape(n_samples, m_features)`）
    -   字典——数值（二维数组`shape(n_samples, m_features)`）
    -   图片——数值（二维数组`shape(n_samples, m_features)`）

#### 图片三要素

-   图片长度、图片宽度、图片通道数
-   灰度图
    -   [长, 宽, 1]
    -   每一个像素点一个[0, 255]数
-   彩色图
    -   [长, 宽, 3]
    -   每一个像素点三个[0, 255]数组成

#### 张量形状

`Tensor(指令名称, shape, dtype)`

-   一张图片
    -   `shape = (height, width, channels)`
-   多张图片
    -   `shape = (batch, height, width, channels)`
    -   batch：表示一个批次的张量数量

### 图片特征值处理

-   样本数据量大
-   样本大小形状不统一
-   解决：把图片缩小到统一大小
-   `tf.image.resize_images(imags, size)`
    -   缩小放大图片
    -   images：4-D 形状`[batch, height, width, channels]`或 3-D 形状的张量`[height, width, channels]`的图片数据
    -   size：1-D int32 张量：new_height, new_width, 图像的新尺寸
    -   返回 4-D 格式或者 3-D 格式图片

### 数据格式

-   存储：uint8（节约空间）
-   矩阵计算：float32（提高精度）

### 案例：狗图片读取

#### 构造文件名队列

#### 读取与解码

使样本的形状和类型统一

#### 批处理

```python
import os
import tensorflow as tf


def pic_read(file_list):
    """
    狗图片读取
    :param file_list:
    :return:
    """
    # 1、构造文件名队列
    file_queue = tf.train.string_input_producer(file_list)
    # 2、读取与解码
    reader = tf.WholeFileReader()
    key, value = reader.read(file_queue)
    # print(key, value)
    image = tf.image.decode_jpeg(value)
    # print(image)

    # 3、批处理
    # 图片统一
    resize_image = tf.image.resize_images(image, [200, 200])
    print(resize_image)
    # Tensor("resize/Squeeze:0", shape=(200, 200, ?), dtype=float32)

    # 通道数固定，shape形状确定
    # 也可用动态修改
    resize_image.set_shape(shape=[200, 200, 3])

    image_batch = tf.train.batch([resize_image], batch_size=100, num_threads=2, capacity=100)
    print(image_batch)
    # Tensor("batch:0", shape=(100, 200, 200, 3), dtype=float32)

    with tf.Session() as sess:
        # 开启线程
        # 创建线程协调员
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        key_new, value_new, image_new = sess.run([key, value, image])
        print(key_new)
        print(value_new)
        print(image_new)

        # 回收线程
        coord.request_stop()
        coord.join(threads=threads)


if __name__ == '__main__':
    # 构造路径+文件名列表
    filename = os.listdir(r'.\dog')
    # print(filename)
    file_list = [os.path.join(r'.\dog', file) for file in filename]
    # print(file_list)
    pic_read(file_list)

```

## 二进制文件

### CIFAR10 二进制数据集介绍

每 3073 个字节是一个样本

-   1 个目标值+3072 像素
    -   1024 字节，红色通道
    -   1024 字节，绿色通道
    -   1024，蓝色通道

### 流程

#### 构造文件名队列

#### 读取与解码

```python
reader = tf.FixedLengthRecordReader(3073)
key, value = reader.read(file_queue)
decode = tf.decode_raw(value, tf.uint8)
```

对 tensor 对象切片

原图片矩阵
[[32 * 32 的二维数组],
[32 * 32 的二维数组],
[32 * 32 的二维数组]]
--> [3, 32, 32]

需转换为 shape：[height, width, channel] -> [32, 32, 3]

```python
tf.transpose(data, [1, 2, 0])
# [0, 1, 2] --> [1, 2, 0]，三维数组位置转换，
# 原来1号位置放到0号，2号位置放到1号，0号放到3号
```

#### 批处理

```python
import os
import tensorflow as tf


class BinRead:
    def __init__(self):
        # 初始化
        self.height = 32
        self.width = 32
        self.channels = 3
        # 字节数
        self.image_bytes = self.height * self.width * self.channels
        self.label_bytes = 1
        self.all_bytes = self.image_bytes + self.label_bytes

    def bin_read(self, file_l):
        # 1、构造文件名队列
        file_queue = tf.train.string_input_producer(file_l)
        # 2、读取与解码
        reader = tf.FixedLengthRecordReader(self.all_bytes)
        key, value = reader.read(file_queue)
        # 解码
        value_decode = tf.decode_raw(value, tf.uint8)
        # 将目标值与特征值切片
        label = tf.slice(value_decode, [0], [self.label_bytes])
        image = tf.slice(value_decode, [1], [self.image_bytes])

        # 调整图片形状，动态调整
        image_reshape = tf.reshape(image, shape=[self.channels, self.height, self.width])
        # 图片转置
        image_transpose = tf.transpose(image_reshape, [1, 2, 0])
        # 调整图片类型
        # image_cast = tf.cast(image_transpose, tf.float32)
        # 3、批处理
        # image_batch = tf.train.batch([image_cast], batch_size=100, num_threads=2, capacity=100)
        image_batch = tf.train.batch([image_transpose], batch_size=100, num_threads=2, capacity=100)
        print(image_batch)
        # Tensor("batch:0", shape=(100, 32, 32, 3), dtype=float32)
        with tf.Session() as sess:
            # 开启线程，创建线程协调员
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            key_new, value_new, value_decode_new, image_batch_new = sess.run([key, value, value_decode, image_batch])
            print(key_new)
            print(value_new)
            print(value_decode_new)
            # [  1  35  27 ... 169 168 168]
            print(image_batch_new)

            # 回收线程
            coord.request_stop()
            coord.join(threads=threads)


if __name__ == '__main__':
    # 构造路径+文件名列表
    filename = os.listdir(r'.\cifar-10-batches-bin')
    # print(filename)
    file_list = [os.path.join(r'.\cifar-10-batches-bin', file) for file in filename if file[-3:] == 'bin']
    print(file_list)
    BinRead().bin_read(file_list)

```

## TFRecords 文件

它其实是一种二进制文件，能够更好的利用内存，更方便复制和移动，并且不需要单独的标签文件

使用步骤：

1. 获取数据
2. 将数据填入到 Example 协议内存块（protocol buffer）
3. 将协议内存块序列化为字符串，并且通过`tf.python_io.TFRecordWriter`写入到 TFRecords 文件

文件格式：\*.tfrecords

### 结构分析

```
features {
    feature {
        key: "age"
        value { float_list {
            value: 29.0
       }}
     }
    feature {
        key: "movie"
        value { bytes_list {
            value: "The Shawshank Redemption"
            value: "Fight Club"
       }}
     }
 }
```

-   `tf.train.Example`协议内存块（协议内存块包含了字段`Features`）
-   `Features`包含了一个`Feature`字段
-   `Feature`包含了要写入的数据、并指明数据类型
    -   这是一个样本的结构，批数据需要循环存入这样的结构

```python
example = tf.train.Example(features=tf.train.Features(feature={
    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
}))
example.SerializeToString()
```

-   `tf.train.Example(features=None)`
    -   写入 tfrecords 文件
    -   features：`tf.train.Features`类型的特征实例
    -   return：example 格式协议块
-   `tf.train.Features(feature=None)`
    -   构建每个样本的信息键值对
    -   feature：**字典数据**，key 为要保存的名字
    -   value：`tf.train.Feature`实例
    -   return：Features 类型
-   `tf.train.Feature(options)`
    -   options
        -   `bytes_list=tf.train.BytesList(value=[Bytes])`
        -   `int64_list=tf.train.Int64List(value=[Value])`
        -   等
    -   支持存入的类型如下
        -   `tf.train.BytesList(value=[Bytes])`
        -   `tf.train.Int64List(value=[Value])`
        -   `tf.train.FloatList(value=[Value])`

> 这种结构很好的实现了数据和标签（训练的类别标签）或者其他属性数据存储在同一文件中

### 案例：CIFAR10 数据存入 TFRecords

#### 分析

-   构造存储实例，`tf.python_io.TFRecordWriter(path)`
    -   写入 tfrecord 文件
    -   path：文件路径
    -   return：写方法
        -   `.write(record)`：写入一个 example
        -   `close()`
-   循环将数据填入到`Example`协议内存块

#### 代码

```python
@staticmethod
def write_to_tfr(image_batch, label_batch):
    """
    将样本特征值目标值一起写入文件
    :param image_batch:
    :param label_batch:
    :return:
    """
    with tf.python_io.TFRecordWriter('cifar.tfrecords') as writer:
        # 循环构造example对象，并序列化写入
        for i in range(100):
            image = image_batch[i].tostring()
            label = label_batch[i][0]
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            }))

            writer.write(example.SerializeToString())
```

### 案例：读取 TFRecords

需要有一个解析 Example 过程，可以使用`tf.TFRecordReader`的`tf.parse_single_example`解析器，可以将`Example`协议内存块解析为张量

```python
feature = tf.parse_single_example(value, features={
    'image': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
})
image = feature['image']
label = feature['label']
```

-   `tf.parse_single_example(serialized, features=None, name=None)`
    -   解析一个单一的 Example 原型
    -   serialized：标量字符串 Tensor，一个序列化的 Example
    -   features：dict'字典数据，键为读取的名字，值为 FixedLenFeature
    -   return：一个 键值对组成的字典，键为读取的名字
-   `tf.FixedLenFeature(shape,dtype)`
    -   shape：输入数据的形状，一般不指定，为空列表
    -   dtype：输入数据类型，与存储进文件的类型要一致
    -   类型只能是 float32，int64，string

#### 步骤

1. 构造文件名队列
2. 读取与解码
    1. 读取
    2. 解析 example
    3. 解码`tf.decode_raw()`
3. 构造批处理队列

```python
    @staticmethod
    def read_tfrecords():
        # 1、构造文件名队列
        file_queue = tf.train.string_input_producer(['cifar.tfrecords'])
        # 2、读取与解码
        # 读取
        reader = tf.TFRecordReader()
        key, value = reader.read(file_queue)
        # 解析example
        feature = tf.parse_single_example(value, features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
        image = feature['image']
        label = feature['label']
        # 解码
        image_decoded = tf.decode_raw(image, tf.uint8)
        # 形状调整
        image_reshaped = tf.reshape(image_decoded, [32, 32, 3])
        # 3、构造批处理队列

        # 开启会话
        with tf.Session() as sess:
            # 开启线程
            # 创建线程协调员
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            image_new, label_new = sess.run([image_reshaped, label])
            print(image_new)
            print(label_new)
            # 回收线程
            coord.request_stop()
            coord.join(threads=threads)

```

## 神经网络基础

### 神经网络

由三层构成：输入层，隐藏层，输出层

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200312214125.jpg)

> ​ 其中每层圆圈代表一个神经元，隐藏层和输出层的神经元把输入的数据计算后输出，输出层的神经元只是输出

-   神经网络特点
    -   每个连接都有权值
    -   同一层神经元之间没有连接
    -   最后的输出结果对应的层也称之为**全连接层**

#### 感知机

感知机就是模拟大脑神经网络处理数据的过程

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200313180033.png)

> 感知机是一种基础分类模型，类似于逻辑回归，不同的是，感知机的激活函数用的书`sign`，而逻辑回归用的是`sigmoid`，**感知机也具有连接的权重和偏置**
>
> $$
> u=\sum_i^n{w_ix_i}+b
> $$
>
> $$
> y=sign(u)=\left\{
> \begin{aligned}
> +1,& &u>0 \\
> -1,& &u\leq0
> \end{aligned}\right.
> $$

可解决问题：或、与、异或

### 神经网络基础

-   损失函数
    -   交叉熵损失
    -   总损失、平均损失
    -   最小二乘法——线性回归损失——均方误差
-   优化损失函数

神经网络主要用于分类，任意事件发生的概率在 0-1 之间，且总有某一个事件发生（概率和为 1）。如果将分类问题中“一个样例属于某个类别”，看成一个概率事件，那么训练数据的正确答案就符合一个概率分布。Softmax 回归就是一个常用的将神经网络前向传播得到的结果也变成概率分布的方法。

#### Softmax 回归

logits 加上 softmax 映射——多分类

将神经网络输出转换成概率结果

$$
softmax(y)_i=\frac{e^{y_i}}{\sum_{j=1}^n{e^{y_j}}}
$$

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200313184347.png)

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200314180553.png)

#### 交叉熵损失公式

$$
H_{y^\prime}=-\sum_i^ny_{i^\prime}log(y_i)
$$

为了能够衡量距离，目标值需要进行 one-hot 编码，能与概率值一一对应，如下图

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200313190058.png)

#### softmax、交叉熵损失 API

-   `tf.nn.softmax_cross_entropy_with_logits(labels=None, logits=None, name=None)`
    -   计算 logits 和 labels 之间的交叉损失熵
    -   labels：标签值（真实值）
    -   logits：样本加权之后的值
    -   return：损失值列表
-   `tf.reduce_mean(input_tensor)`
    -   计算张量的尺寸的元素平均值

### 案例：Mnist 手写数字识别

#### 数据集

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200313193736.png)

-   特征值
    -   28\*28=784
-   目标值
    -   分类：one-hot 编码
    -   可能值：0, 1, 2, ...,9
    -   编码：0, 1, 0, ...,0（值为 1）

#### 数据获取 API

-   `from tensorflow.examples.tutorials.mnist import input_data`
    -   `mnist = input_data.read_data_sets(path, one_hot=True)`
        -   `mnist.train.next_batch(100)`提供批量获取功能
        -   `mnist.train.images`
        -   `mnist.train.labels`
        -   `mnist.test.images`
        -   `mnist.test.labels`

#### 分析-代码

-   完善功能
    -   增加准确率
    -   增加变量 tensorboard 显示
    -   增加模型保存加载
    -   增加模型预测结果输出

采用只有一层，即最后一个输出层的神经网络，也称为全连接层神经网络

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200313193212.png)

##### 全连接

-   `y = w1x1 + w2x2 + ... + b`

```python
x[None, 784] * w[784, 10] + bias[10] = y_predict[None, 10]
r = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict, name=None)
error = tf.reduce_mean(r)

```

##### 准确率

```python
equal_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

```

##### 代码

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def full_connection():
    """
    用全连接对手写数字进行识别
    :return:
    """
    # 1、准备数据
    mnist = input_data.read_data_sets("./mnist_data", one_hot=True)

    # 用占位符定义真实数据
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y_true = tf.placeholder(dtype=tf.float32, shape=[None, 10])

    # 2、构造模型 - 全连接
    # [None, 784] * W[784, 10] + Bias = [None, 10]
    weights = tf.Variable(initial_value=tf.random_normal(shape=[784, 10], stddev=0.01))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[10], stddev=0.1))
    y_predict = tf.matmul(x, weights) + bias


    # 3、构造损失函数
    loss_list = tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_true)
    loss = tf.reduce_mean(loss_list)
    # 4、优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

    # 5、增加准确率计算
    equal_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))
    accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 初始化变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # 开始训练
        for i in range(50000):
            # 获取真实值
            image, label = mnist.train.next_batch(500)

            _, loss_value, accuracy_value = sess.run([optimizer, loss, accuracy], feed_dict={x: image, y_true: label})

            print("第%d次的损失为%f，准确率为%f" % (i + 1, loss_value, accuracy_value))




if __name__ == "__main__":
    full_connection()

```
