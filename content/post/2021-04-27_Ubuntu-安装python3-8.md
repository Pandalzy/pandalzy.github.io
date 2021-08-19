---
title: ubuntu 安装python3.8
date: 2021-04-27 09:09:51
tags: [python]
categories: [tips]
---

下载源码

```sh
wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz
```

解压源码

```sh
tar -xvzf Python-3.8.0.tgz
```

<!-- more -->

进入目录

```sh
cd Python-3.8.0
```

配置安装路径

```sh
./configure --with-ssl --prefix=/usr/local/python3
```

安装python3.8依赖

```sh
# sudo apt-get update
# sudo apt-get upgrade
# sudo apt-get dist-upgrade
sudo apt-get install build-essential python-dev python-setuptools python-pip python-smbus libncursesw5-dev libgdbm-dev libc6-dev zlib1g-dev libsqlite3-dev tk-dev libssl-dev openssl libffi-dev
```

编译

```sh
make
```

安装

```sh
sudo make install
```

删除软链接

```shell
sudo rm -rf /usr/bin/python3
sudo rm -rf /usr/bin/pip3
```

新建软链接

```shell
sudo ln -s /usr/local/python3/bin/python3.8 /usr/bin/python3
sudo ln -s /usr/local/python3/bin/pip3.8 /usr/bin/pip3
```

检查是否安装成功

```sh
python3
```

