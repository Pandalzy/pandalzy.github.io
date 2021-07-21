---
title: Use of nohup and &
date: 2021-06-01 19:43:12
---

## 作用

nohup 是 no hungup 的缩写，以为“不挂断”，在使用 Xshell 等工具执行脚本时，有时候会由于网络问题，导致失去连接，终端断开，程序运行一半就意外结束了。这种时候，就可以用 nohup 指令来运行指令，使程序可以忽略挂起信号继续运行。

<!-- more -->

## 语法

```bash
nohup Command [ Arg ... ] [　& ]
```

nohup 命令运行由 Command 参数和任何相关的 Arg 参数指定的命令，忽略所有挂断（SIGHUP）信号。在注销后使用 nohup 命令运行后台中的程序。要运行后台中的 nohup 命令，添加 & （ 表示“and”的符号）到命令的尾部。

如果不将 nohup 命令的输出重定向，输出将附加到当前目录的 nohup.out 文件中。如果当前目录的 nohup.out 文件不可写，输出重定向到 `$HOME/nohup.out` 文件中。

## nohup 和 & 的区别

- &：是指在后台运行，当用户退出（挂起）的时候，命令自动跟着结束
- nohup：不挂断的运行，注意并没有后台运行的功能，就是指用nohup运行命令可以使命令永久的执行下去，和用户终端没有关系，例如我们断开SSH连接都不会影响他的运行，注意了nohup没有后台运行的意思；&才是后台运行

因此将nohup和&结合使用，就可以实现使命令永久地在后台执行的功能

## 举例

1. `sh test.sh &`

将sh test.sh任务放到后台 ，关闭xshell，对应的任务也跟着停止

2. `nohup sh test.sh`

将 `sh test.sh` 任务放到后台，关闭标准输入，**终端不再能够接收任何输入（标准输入）**，重定向标准输出和标准错误到当前目录下的 nohup.out 文件，即使关闭 xshell 退出当前session依然继续运行

3. `nohup sh test.sh &`

将sh test.sh任务放到后台，但是依然可以使用标准输入，**终端能够接收任何输入**，重定向标准输出和标准错误到当前目录下的nohup.out文件，即使关闭xshell退出当前session依然继续运行

## 输出重定向

作业在后台运行的时候，可以把输出重定向到某个文件中，相当于一个日志文件，记录运行过程中的输出。使用方法：

```bash
nohup command > out.file 2>&1 &
```

`command>out.file` 是将 command 的输出重定向到 out.file 文件，即输出内容不打印到屏幕上，而是输出到out.file文件中。

在上面，0 – stdin (standard input)，1 – stdout (standard output)，2 – stderr (standard error) ；

2>&1 是将标准错误（2）重定向到标准输出（&1），标准输出（&1）再被重定向输入到 out.file 文件中。

