---
title: mac 终端美化 iterm2+zsh
date: 2021-04-09 15:23:34
tags:
categories:
---

![](https://gitee.com/pandalzy/cloud_img/raw/master/imgs/%E6%88%AA%E5%B1%8F2021-04-09%20%E4%B8%8B%E5%8D%883.24.39.png)

<!-- more -->

## 下载 iTerm2

[iTerm2 - macOS Terminal Replacement](https://iterm2.com/)

快捷打开快捷键设置

![快捷打开快捷键设置](https://gitee.com/pandalzy/cloud_img/raw/master/imgs/%E6%88%AA%E5%B1%8F2021-04-09%20%E4%B8%8B%E5%8D%883.40.36.png)

## 配色方案

[Solarized (ethanschoonover.com)](https://ethanschoonover.com/solarized/)

下载解压，然后打开 iTerm2 下的偏好设置 preference，选择 solarized 文件下的 Solarized Dark.itermcolors

![](https://gitee.com/pandalzy/cloud_img/raw/master/imgs/%E6%88%AA%E5%B1%8F2021-04-09%20%E4%B8%8B%E5%8D%883.44.44.png)

## 安装 oh-my-zsh

[ohmyzsh](https://github.com/ohmyzsh/ohmyzsh)

## 配置主题

1. 用 vim 编辑隐藏文件 .zshrc， 终端输入`vim ~/.zshrc`
2. `ZSH_THEME="agnoster"` 将 zsh 主题修改为 agnoster

![](https://gitee.com/pandalzy/cloud_img/raw/master/imgs/%E6%88%AA%E5%B1%8F2021-04-09%20%E4%B8%8B%E5%8D%883.50.19.png)

1. 应用“agnoster”主题需要特殊的字体支持，否则会出现乱码情况，
2. [fonts/Meslo LG M Regular for Powerline.ttf at master · powerline/fonts (github.com)](https://github.com/powerline/fonts/blob/master/Meslo Slashed/Meslo LG M Regular for Powerline.ttf)

![](https://gitee.com/pandalzy/cloud_img/raw/master/imgs/%E6%88%AA%E5%B1%8F2021-04-09%20%E4%B8%8B%E5%8D%883.51.00.png)

## 代码补全

克隆仓库到本地 ~/.oh-my-zsh/custom/plugins 路径下

```sh
git clone git://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
```

用 vim 编辑 .zshrc 文件，找到插件设置命令，默认是`plugins=(git)`，我们把它修改为`plugins=(zsh-autosuggestions git)`

![](https://gitee.com/pandalzy/cloud_img/raw/master/imgs/%E6%88%AA%E5%B1%8F2021-04-09%20%E4%B8%8B%E5%8D%883.57.51.png)

PS：重新打开终端时可能看不到变化，可能字体颜色太淡了，把其改亮一些：
1. `cd ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions`
2. 用 vim 编辑 zsh-autosuggestions.zsh 文件，修改`ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=10'`

## 语法高亮

- 使用 homebrew 包管理工具安装 [zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting) 插件
  - `brew install zsh-syntax-highlighting`
- 配置 .zshrc 文件，插入一行
  - `source /usr/local/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh`

![](https://gitee.com/pandalzy/cloud_img/raw/master/imgs/%E6%88%AA%E5%B1%8F2021-04-09%20%E4%B8%8B%E5%8D%884.13.43.png)