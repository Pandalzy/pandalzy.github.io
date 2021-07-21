---
title: Django部署：uwsgi、nginx配置模板
date: 2020-08-11 17:49:36
tags: [Django]
categories: [后端]
---

## uwsgi配置

```
[uwsgi]
socket = 127.0.0.1:3001
stats = 127.0.0.1:9090
chdir = /www/xxxx/app
module = app.wsgi
master = true 

processes = 1
pidfile = uwsgi.pid
disable-logging = true
daemonize = uwsgi.log
```

<!-- more -->

## nginx配置

```nginx
server{
    listen 80;
    server_name localhost;
    root /www/xxx;

    # http自动跳转https
    # rewrite ^(.*)$ https://$host$1 permanent;
    
    location /media {
        alias /www/xxx/app/media;
    }
    
    location /static {
      alias /www/xxx/app/static;
    }
    
    location / {
        include uwsgi_params;
        uwsgi_pass 127.0.0.1:3001;  # 与uwsgi的scoket对应
    }
    
    # access_log  /www/wwwlogs/app.log;
    # error_log  /www/wwwlogs/app.error.log;
}
```

