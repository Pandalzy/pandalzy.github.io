---
title: 'Celery Cannot connect to amqp'
date: 2021-04-09 10:29:32
tags: [Celery]
categories: [tips]
---

Celery 提示[ERROR/MainProcess] consumer: Cannot connect to amqp://guest: @127.0.0.1:5672//: [Errno 61] Connection refused.
解决方法：需要启动rabbitmq-server
sudo rabbitmq-server -detached

<!-- more -->