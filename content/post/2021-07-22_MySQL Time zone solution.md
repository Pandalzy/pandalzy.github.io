---
title: "MySQL Time Zone Solution"
date: 2021-07-22T21:24:10+08:00
lastmod: 2021-07-22T21:24:10+08:00
draft: true
tags: [MySQL]
categories: [Tips]
---

## Viewing the Current Time Zone

```sql
> select curtime();   #or select now()
+-----------+
| curtime() |
+-----------+
| 15:18:10  |
+-----------+
 
> show variables like "%time_zone%";
+------------------+--------+
| Variable_name    | Value  |
+------------------+--------+
| system_time_zone | CST    |
| time_zone        | SYSTEM |
+------------------+--------+
2 rows in set (0.00 sec)
```

## Modify the time zone

```mysql
set global time_zone = '+8:00';  ## Change the MySQL global time zone to Beijing time
set time_zone = '+8:00';  ## Change the time zone of the current session
flush privileges;
```

## More

```bash
# vim /etc/my.cnf
default-time_zone = '+8:00'
 
# /etc/init.d/mysqld restart 
```

