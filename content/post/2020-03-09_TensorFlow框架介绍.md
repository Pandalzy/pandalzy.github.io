---
title: TensorFlow框架介绍
date: 2020-03-09 21:47:11
tags: [TensorFlow, 深度学习]
---

## TF 数据流图

### 加法案例

```python
import tensorflow as tf

def tensorflow_demo():
    # 实现加法
    a_t = tf.constant(2)
    b_t = tf.constant(3)

    c_t = a_t + b_t
    print(c_t)

    # 开启会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print(c_t_value)


if __name__ == '__main__':
    tensorflow_demo()

# Tensor("add:0", shape=(), dtype=int32)
# 5
```

<!-- more -->

### TF 结构分析

-   一个构建图阶段
    -   流程图：定义数据（张量 Tensor）和操作（节点 op）
-   一个执行图阶段
    -   调用各方资源，讲定义好的数据和操作运行起来

#### 数据流图介绍

Tensor-张量-数据

Flow-流动

## 图与 TensorBoard

### 图结构

数据（Tensor）+操作（Operation）

### 相关操作

-   默认图

    -   调用方法`tf.get_default_graph()`

    -   查看属性`.graph`

```python
import tensorflow as tf


def graph_demo():
    """
    图的演示
    :return:
    """
    a_t = tf.constant(2)
    b_t = tf.constant(3)

    c_t = a_t + b_t
    print(c_t)

    # 查看默认图
    default_g = tf.get_default_graph()
    print('图方法', default_g)

    print('图属性', c_t.graph)
    # 开启会话
    with tf.Session() as sess:
        c_t_value = sess.run(c_t)
        print(c_t_value)
        print('图属性', sess.graph)


if __name__ == '__main__':
    graph_demo()

# Tensor("add:0", shape=(), dtype=int32)
# 图方法 <tensorflow.python.framework.ops.Graph object at 0x000001C855C96E88>
# 图属性 <tensorflow.python.framework.ops.Graph object at 0x000001C855C96E88>
# 5
# 图属性 <tensorflow.python.framework.ops.Graph object at 0x000001C855C96E88>
```

-   创建图

```python
new_g = tf.Graph()
with new_g.as_default():
    # 定义数据和操作
```

```python
new_g = tf.Graph()
with new_g.as_default():
    a_new = tf.constant(20)
    b_new = tf.constant(30)
    c_new = a_new + b_new
    print(c_new)

with tf.Session(graph=new_g) as sess:
    c_new_value = sess.run(c_new)
    print(c_new_value)
    print('创建图', sess.graph)

# Tensor("add:0", shape=(), dtype=int32)
# 50
# 创建图 <tensorflow.python.framework.ops.Graph object at 0x000001E7C282D5C8>
```

### 可视化 TensorBoard

-   数据序列化-event 文件
    -   `tf.summary.FileWriter(path, graph=sess.graph)`
-   启动 TensorBorad
    -   cmd 运行
    -   `tensorboard --logdir=./tmp/summary --host=127.0.0.1`
    -   ![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200305181919.png)

### OP

-   数据：Tensor 对象
-   操作：Operation 对象

## 会话

-   `tf.Session`：用于完整的程序当中
-   `tf.InteractiveSession`：用于交互式上下文中的 TensorFlow，例如 shell
-   使用`a.eval()`

*   会话资源需要回收

    -   直接使用`with`

*   初始化会话参数

    -   target：如果将此参数留空，会话将仅仅使用本地计算机中的设备。可以指定`grpc://`网址，以便指定 TensorFlow 服务器的地址，这使得会话可以访问该服务器控制的计算机上的所有设备。

    -   graph：默认情况下，新的`tf.Session`将绑定到当前的默认图

    -   config：此参数允许指定一个`tf.ConfigProto`以便控制会话的行为。例如，ConfigProto 协议用于打印设备使用信息

```python
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

# Device mapping: no known devices.
# add: (Add): /job:localhost/replica:0/task:0/device:CPU:0
# Const: (Const): /job:localhost/replica:0/task:0/device:CPU:0
# Const_1: (Const): /job:localhost/replica:0/task:0/device:CPU:0
```

-   `run(fetches, feed_dict=None)`

    -   通过使用`sess.run()`来运行 operation

    -   fetches：单一的 operation，或者列表、元组（其他不属于 tensorflow 的类型不行）

    -   feed_dict：参数允许调用者覆盖图中张量的值，运行时赋值

        -   与`tf.placeholder`搭配使用，则会检查值的形状是否与占位符兼容

```python
with tf.Session() as sess:
	# 查看c_t的值
    print(c_t.eval())
    print(sess.run(c_t))
    # 查看a，b，c的值
    print([a, b ,c])
```

-   feed_dict 操作

```python
def session_run_demo():
    # 定义占位符
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    sum_ab = tf.add(a, b)
    # 开启会话
    with tf.Session() as sess:
        print(sess.run(sum_ab, feed_fict={a:3.0, b:4.0}))
```

## 张量

### 数据类型

-   张量如何存储
    -   标量 一个数字 0 阶张量 `shape()`
    -   向量 一维数组 1 阶张量 `shape(n,)`n 为数组长度
    -   矩阵 二维数组 2 阶张量 `shape(n, m, )`n 行，m 列
    -   张量 n 维数组 n 阶张量
-   默认 tf.float32
    -   整型 tf.int32
    -   浮点型 tf.float32

### 创建张量指令

### 张量的变换

-   类型的修改

    -   `ndarry.astype(type)`
        -   `tf.cast(tensor, dtype)`
        -   不会改变原始的 tensor
        -   返回新的 tensor
    -   `ndarry.tostring()`

-   形状的修改

    -   `ndarray.reshape(shape)`行变成列，列变成行

    -   `ndarray.resize(shape)`

    -   静态形状：初始创建张量时形状

        -   `tensor.set_shape()`
        -   只有在形状没有完全固定下来的情况下，才能改变

```python
  a = tf.placeholder(dtype=tf.float32, shape=[None, None])
  b = tf.placeholder(dtype=tf.float32, shape=[None, 10])
  c = tf.placeholder(dtype=tf.float32, shape=[10, 20])
  # a Tensor("Placeholder:0", shape=(?, ?), dtype=float32)
  # b Tensor("Placeholder_1:0", shape=(?, 10), dtype=float32)
  # c Tensor("Placeholder_2:0", shape=(10, 20), dtype=float32)

  a.set_shape([2, 3]) # a可以设置不同行列
  b.set_shape([2, 10]) # b只可以设置行，列已固定
  # c不可设置，因为行列以固定

  # a Tensor("Placeholder:0", shape=(2, 3), dtype=float32)
  # b Tensor("Placeholder_1:0", shape=(2, 10)
```

-   动态形状：

    -   `tensor.reshape(tensor, shape)`

    -   不会改变原始 tensor，只会返回一个新的 tensor

    -   必须保持张量的元素数量前后一致

```python
a = tf.placeholder(dtype=tf.float32, shape=[None, None])
a.set_shape([2, 3])
print(tf.reshape(a, shape=[2, 3, 1]))
# 变化后的元素数量等于原来的，2*3*1=2*3
```

### 张量的数学公式

参考官方 api

-   基本运算符
-   基本数学函数
-   矩阵运算
-   reduce 操作
-   序列索引操作

## 变量 OP

TensorFlow 中的变量

-   存储持久化
-   可修改值
-   可指定被训练

### 创建变量

```python
tf.Variable(initial_value=None, trainable=True, collections=None,)
```

-   initial_value：初始化的值
-   trainable：是否被训练
-   collections：新变量将添加到列出的图的集合中 collections
-   **变量需要显式初始化，才能运行值**

```python
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
```

```python
import tensorflow as tf

def variable_demo():
    """
    变量演示
    :return:
    """
    a = tf.Variable(initial_value=50, )
    b = tf.Variable(initial_value=40, )
    c = tf.add(a, b)
    print(a)
    print(b)
    print(c)
    # 初始化变量
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        a_t, b_t, c_t = sess.run([a, b, c])
        print(a_t, b_t, c_t)

if __name__ == '__main__':
    variable_demo()

# <tf.Variable 'Variable:0' shape=() dtype=int32_ref>
# <tf.Variable 'Variable_1:0' shape=() dtype=int32_ref>
# Tensor("Add:0", shape=(), dtype=int32)
# 50 40 90
```

-   命名空间修改，使结构更加清晰

```python
with tf.variable_scope('test'):
    a = tf.Variable(initial_value=50, )
    b = tf.Variable(initial_value=40, )
    c = tf.add(a, b)

# <tf.Variable 'test/Variable:0' shape=() dtype=int32_ref>
# <tf.Variable 'test/Variable_1:0' shape=() dtype=int32_ref>
# Tensor("test/Add:0", shape=(), dtype=int32)
# 多出了一个test/
```

## 高级 API

### 基础 API

-   `tf.app`
    -   相当于为 TensorFlow 进行的脚本提供一个 main 函数入口，可以定义脚本运行的 flags。
-   `tf.image`
    -   图像处理操作，主要是颜色的变换、变形和图像的编码和解码
-   `tf.summary`
    -   用来生成 TensorBoard 可用的统计日志，目前 summary 主要提供了 4 种类型：audio、image、histogram、scalar
-   `tf.python_io`
    -   用来读写 TFRecords 文件
-   `tf.train`
    -   提供了一些训练器，与`tf.nn`组合起来，实现一些网络的优化计算
-   `tf.nn`
    -   提供了一些构建神经网络的底层函数。TensorFlow 构建网络的核心模块。其中包含了添加各种层的函数，比如添加卷积层、池化层等

### 高级 API

-   `tf.keras`
    -   在于快速构建模型
-   `tf.layers`
    -   以更高级的概念层来定义一个模型，类似`tf.keras`
-   `tf.contrib`
    -   提供计算图中的网络层、正则化、摘要操作，是构建计算图的高级操作，但是，包含不稳定和实验代码，可能以后 API 会变。
-   `tf.estimator`
    -   相当于 Model+Training+Evaluate 的合体，在模块中，已经实现了几种简单的分类器和回归器，包括：Baseline、Learning 和 DNN。这里的 DNN 的网络，只是全连接网络，没有提供卷积之类的。

### 高级 API 图示

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200308221528.jpg)

## 案例：实现线性回归

### API

#### 运算

-   矩阵运算
    -   `tf.matmul(x, w)`
-   平方
    -   `tf.square(error)`
-   均值
    -   `tf.reduce_mean(error)`

#### 梯度下降优化

-   `tf.train.GradientDescentOptimizer(learning_rate)`
    -   梯度下降优化
    -   learning_rate：学习率，一般为 0~1 之间比较小的值
    -   method
        -   `minimize(loss)`
    -   return：梯度下降 op

### 线性回归原理

-   构建模型
    -   `y = w1x1 + w2x2 + ... + wnxn + b`
-   构造损失函数
    -   均方误差
-   优化损失
    -   梯度下降

### 案例：线性回归的训练

-   准备真实数据
    -   100 个样本
    -   x：特征值 形状：（100, 1）
    -   y_true：目标值 目标值：（100, 1）
    -   `y_true = 0.8x + 0.7`
-   假定 x 和 y 之间的关系满足
    -   `y = kx + b`
    -   趋近于`k = 0.8 b = 0.7`
-   流程分析
    -   `(100, 1) * (1, 1) = (100, 1)`
    -   `y_predict = x * weights(1, 1) + bias(1, 1)`
        -   bias 可以是标量，也可以是向量
    -   构建模型
        -   `y_predict = tf.matmul(x, weights) + bias`
    -   构造损失函数
        -   `error = tf.reduce_mean(tf.square(y_predict - y_true))`
    -   优化损失
        -   `optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)`
        -   学习率会影响迭代训练次数

```python
import tensorflow as tf


def linear_regression():
    """
    实现线性回归
    :return:
    """
    # 1、准备数据
    x = tf.random_normal(shape=[100, 1])
    y_true = tf.matmul(x, [[0.8]]) + 0.7

    # 2、构造模型
    # 定义模型参数，用变量
    weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))
    y_predict = tf.matmul(x, weights) + bias

    # 3、构造损失函数
    error = tf.reduce_mean(tf.square(y_predict - y_true))

    # 4、优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 显式的初始化变量
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        # 查看初始化模型参数之后的值
        print('训练前模型参数：权重%f，偏置%f，损失%f' % (weights.eval(), bias.eval(), error.eval()))

        # 开始训练
        for i in range(1000):
            # 迭代1000次训练
            sess.run(optimizer)
        print('训练后模型参数：权重%f，偏置%f，损失%f' % (weights.eval(), bias.eval(), error.eval()))


if __name__ == '__main__':
    linear_regression()

# 训练前模型参数：权重-3.123351，偏置-0.332511，损失15.018922
# 训练后模型参数：权重0.799999，偏置0.699999，损失0.000000
```

### 增加其他功能

-   增加 TensorBoard 显示
-   增加命名空间
-   模型保存与加载
-   命令行参数设置

#### 增加变量显示

目的：在 TensorBoard 中观察模型的参数、损失值等变量值的变化

-   创建事件文件

-   收集变量
    -   `tf.summary.scalar(name='', tensor)`，收集对于损失函数和准确率等单值变量，name 为变量的名字，tensor 为值
    -   `tf.summary.histogram(name='', tensor)`，收集高纬度的变量参数
    -   `tf.summary.image(name='', tensor)`，收集输入的图片张量显示图片
-   合并变量写入事件文件
    -   `merged = tf.summary.merge_all()`
    -   运行合并：`summary = sess.sun(merged)`，每次迭代都需运行
    -   添加：`FileWriter.add_summary(summary, i)`，i 表示第几次的值

```python
import tensorflow as tf


def linear_regression():
    """
    实现线性回归
    :return:
    """
    # 1、准备数据
    x = tf.random_normal(shape=[100, 1])
    y_true = tf.matmul(x, [[0.8]]) + 0.7

    # 2、构造模型
    # 定义模型参数，用变量
    weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))
    bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]))
    y_predict = tf.matmul(x, weights) + bias

    # 3、构造损失函数
    error = tf.reduce_mean(tf.square(y_predict - y_true))

    # 4、优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

    # 1_收集变量
    tf.summary.scalar('error', error)
    tf.summary.histogram('weights', weights)
    tf.summary.histogram('bias', bias)
    # 2_合并变量
    merged = tf.summary.merge_all()

    # 显式的初始化变量
    init = tf.global_variables_initializer()
    # 开启会话
    with tf.Session() as sess:
        sess.run(init)
        # 查看初始化模型参数之后的值

        # 0_创建事件文件
        file_writer = tf.summary.FileWriter('./tmp/liner', graph=sess.graph)

        print('训练前模型参数：权重%f，偏置%f，损失%f' % (weights.eval(), bias.eval(), error.eval()))

        # 开始训练
        for i in range(1000):
            # 迭代1000次训练
            sess.run(optimizer)
            # 3_运行合并变量操作
            summary = sess.run(merged)
            # 4_将每次迭代后的变量写入事件文件
            file_writer.add_summary(summary, i)

        print('训练后模型参数：权重%f，偏置%f，损失%f' % (weights.eval(), bias.eval(), error.eval()))



if __name__ == '__main__':
    linear_regression()

```

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200309190938.png)

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200309191005.png)

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200309191030.png)

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200309191053.png)

#### 增加命名空间

-   `with tf.variable_scope('')`

```python
with tf.variable_scope('prepare_data'):
    # 1、准备数据
    x = tf.random_normal(shape=[100, 1], name='Feature')
    y_true = tf.matmul(x, [[0.8]]) + 0.7

with tf.variable_scope('create_mode'):
    # 2、构造模型
    # 定义模型参数，用变量
    weights = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name='Weights')
    bias = tf.Variable(initial_value=tf.random_normal(shape=[1, 1]), name='Bias')
    y_predict = tf.matmul(x, weights) + bias
with tf.variable_scope('loss_fun'):
    # 3、构造损失函数
    error = tf.reduce_mean(tf.square(y_predict - y_true))
with tf.variable_scope('optimizer'):
    # 4、优化损失
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)
```

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200309191853.png)

![](https://raw.githubusercontent.com/Pandalzy/cloud_img/master/img/blog20200309191942.png)

#### 模型的保存与加载

-   `tf.train.Saver(var_list=None, max_to_keep=5)`
    -   保存和加载模型（文件格式：checkpoint 文件）
    -   var_list：指定将要保存和还原的变量，它可以作为一个 dict 或一个列表传递
    -   max_to_keep：指示要保留的最近检查点文件的最大数量。创建新文件时，会删除较旧的文件。如果无或 0，则保留所有检查点文件。默认为 5（即保留最新的 5 个检查点文件）
-   步骤
    -   实例化 Saver，`saver = tf.train.Saver(var_list=None, max_to_keep=5)`
    -   保存，`saver.save(sess, path)`
        -   路径需存在
        -   例：`saver.save(sess, '/tmp/ckpt/test/myregression.ckpt')`
    -   加载，`saver.restore(sess, path)`
        -   例：`saver.restore(sess, '/tmp/ckpt/test/myregression.ckpt')`

#### 命令行参数使用

-   `tf.app.flags`
    -   `tf.app.flags.DEFINE_integer('max_step', 0, '训练模型的步数')`
        -   参数`(flag_name, default_value, docstring)`
    -   `tf.app.flags.DEFINE_string(flag_name, default_value, docstring)`
    -   `tf.app.flags.DEFINE_boolean(flag_name, default_value, docstring)`
    -   `tf.app.flags.DEFINE_float(flag_name, default_value, docstring)`
-   在 flags 有一个 FLAGS 标志，它在程序中可以调用到前面具体定义的 flag_name
    -   `FLAGS = tf.app.flags.FLAGS`
    -   通过`FLAGS.max_step`调用命令行中传过来的参数

```python
import tensorflow as tf

tf.app.flags.DEFINE_integer('max_step', 0, '训练模型的步数')
tf.app.flags.DEFINE_string('model_dir', 'Unknown', '模型路径+名字')
FLAGS = tf.app.flags.FLAGS


def command_demo():
    """
    命令行演示
    :return:
    """
    print(FLAGS.max_step)
    print(FLAGS.model_dir)


if __name__ == '__main__':
    command_demo()
```

```shell
>python day01_deeplearning.py
0
Unknown

>python day01_deeplearning.py --max_step=200 --model_dir='/tmp/t.txt'
200
'/tmp/t.txt'
```

-   通过`tf.app.run()`启动`main(argv)`函数

```python
import tensorflow as tf


def main(argv):
    print(argv)


if __name__ == '__main__':
    tf.app.run()

# ['C:/Users/yuan/PycharmProjects/tensor/day01_deeplearning.py']
# 当前py文件所在目录
```
