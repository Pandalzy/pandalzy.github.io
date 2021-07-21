---
title: Linux修改默认shell
date: 2021-04-27 09:21:12
tags: [linux]
categories: [tips]
---

输入`cat /etc/shells`查看可用shell

```sh
# /etc/shells: valid login shells
/bin/sh
/bin/bash
/bin/rbash
/bin/dash
```

设置默认shell

```sh
chsh -s /bin/bash
# 输入管理员密码就可以了
```

检查是否设置成功

```sh
grep 用户名 /etc/passwd
# 会有设置的shell
```

