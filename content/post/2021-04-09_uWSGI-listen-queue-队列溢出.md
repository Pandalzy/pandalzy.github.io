---
title: uWSGI listen queue 队列溢出的问题
date: 2021-04-09 10:33:19
tags: [uWSGI]
categories: [tips]
---

如果没有设置 uwsgi 的 –listen，如果`sysctl -a | grep net.core.somaxconn`发现net.core.somaxconn=128。
那你使用uwsgi启动的服务，单机最大支持并发数为100*(启动的uwsgi进程数)。
如果启动进程为4个，则最大并发只能支持400，这样会在 uwsgi 的 log 日志中出现错误 uWSGI listen queue of socket 4 full。
同时，nginx对应也会出现错误 upstream time out。

<!-- more -->

方法一：

修改系统参数`vim /etc/sysctl.conf`

在文件最后添加一行记录`net.core.somaxconn=1024`

执行`sysctl -p`重新 load 参数设置，这样会立即生效，并且以后重新启动机器也会生效。

设置 uwsgi 启动的`--listen 1024`

这样 你的机器并发数就可以得到一个很大的提升。

方法二：

docker 部署，运行时添加 `--sysctl net.core.somaxconn=4096`

docker-compose 中

```
sysctls:
  net.core.somaxconn: 1024
```