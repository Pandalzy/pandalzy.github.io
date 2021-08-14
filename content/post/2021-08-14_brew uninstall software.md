---
title: "Brew Uninstall Software"
date: 2021-08-14T11:51:59+08:00
lastmod: 2021-08-14T11:51:59+08:00
draft: false
tags: [Brew]
categories: [Tips]
---



Using `uninstall` to uninstall the software will only uninstall the software itself, but will not uninstall its dependent packages at the same time. 

Use the following command to uninstall completely, and will not affect other software.

```bash
brew tap beeftornado/rmtree
```

This step may take a long time.

```bash
brew rmtree xxx
brew cleanup
```
