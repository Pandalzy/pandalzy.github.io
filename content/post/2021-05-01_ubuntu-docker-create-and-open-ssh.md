---
title: ubuntu docker create and open ssh
date: 2021-05-01 09:18:29
tags: [ubuntu, docker]
categories: [tips]
---

## The preparatory work

Pull ubuntu image

```bash
docker pull ubuntu:20.04
```

Run ubuntu container back

```
docker run -itd -p 3022:22 -p 3021:8080 ubuntu:20.04 bash
```

> 3022 is ssh port, 3021 is web port

<!-- more -->

## Configuration

Into the container

```bash
docker exec -it 1c0 bash
```

> 1c0 is the first three letters of container id

Set the root password

```
passwd root
```

Update and upgrade

```bash
apt update && apt upgrade
```

### Install package

Install ssh client and vim

```bash
apt install ssh vim
```

Install ssh server

```bash
apt install openssh-server
```

### Verify SSHServer

```bash
ps -e | grep ssh
```

> if not start run `/etc/init.d/ssh start`

![](https://gitee.com/pandalzy/cloud_img/raw/master/imgs/20210501092843.png)

## More

then you can add other user and do more

