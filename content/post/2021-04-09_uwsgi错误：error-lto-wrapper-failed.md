---
title: 'uWSGI错误：error: lto-wrapper failed collect2'
date: 2021-04-09 10:31:25
tags: [uWSGI]
categories: [tips]
---

uWSGI错误：error: lto-wrapper failed collect2
红色错误警告，主要报错内容是：
error: lto-wrapper failed
collect2: error: ld returned 1 exit status
原因是gcc版本高

<!-- more -->

```shell
# 查看当前系统安装所有版本的gcc
ls /usr/bin/gcc* -l 
# 如果gcc有5以下的版本，则不用在安装
sudo apt-get install gcc-4.8
# 更改gcc系统默认版本
sudo rm /usr/bin/gcc # 删除已有软连接
sudo ln -s /usr/bin/gcc-4.8 /usr/bin/gcc # 创建指向gcc4.8的软连接
```

