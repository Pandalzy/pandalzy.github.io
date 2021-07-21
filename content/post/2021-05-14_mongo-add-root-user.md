---
title: mongo add root user
date: 2021-05-14 17:15:29
tags: [mongo]
categories: [tips]
---

## 添加用户

```shell
use admin
db.createUser({
    user: "admin",
    pwd: "123456",
    roles: [{
        role: "root",
        db: "admin"
    }]
})
```

<!-- more -->

## 修改配置

```
net:
	port: 27017
	# bindIp: 127.0.0.1
security:
	authorization: enabled
```

