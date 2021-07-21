---
title: mysql-server-5.7 is not configured yet
date: 2021-04-27 09:18:42
tags: [MySQL]
categories: [tips]
---

```sh
Do you want to continue? [Y/n] Y
Setting up mysql-server-5.7 (5.7.28-0ubuntu0.18.04.4) ...
/var/lib/dpkg/info/mysql-server-5.7.postinst: line 191: /usr/share/mysql-common/configure-symlinks: No such file or directory
dpkg: error processing package mysql-server-5.7 (--configure):
 installed mysql-server-5.7 package post-installation script subprocess returned error exit status 127
dpkg: dependency problems prevent configuration of mysql-server:
 mysql-server depends on mysql-server-5.7; however:
  Package mysql-server-5.7 is not configured yet.

dpkg: error processing package mysql-server (--configure):
 dependency problems - leaving unconfigured
No apport report written because the error message indicates its a followup error from a previous failure.
                                                                                                          Errors were encountered while processing:
 mysql-server-5.7
 mysql-server
E: Sub-process /usr/bin/dpkg returned an error code (1)
```

<!-- more -->

```shell
sudo apt purge mysql-client-5.7 mysql-client-core-5.7 mysql-common mysql-server-5.7 mysql-server-core-5.7 mysql-server
sudo apt update && sudo apt dist-upgrade && sudo apt autoremove && sudo apt -f install
sudo apt install mysql-server
```

