---
title: "Zathura"
date: 2025-05-23T08:47:59.134Z
draft: false
tags: []
---

first, tap this repo and install:
```shell
brew tap homebrew-zathura/zathura
brew install zathura
```
install some plugins:

```shell
brew install zathura-djvu zathura-pdf-mupdf zathura-ps
```
After you install all required plugins you need to put them in a directory where zathura can find them. To do this, run the following command. You have to run this command only after installing new plugins.
```shell
d=$(brew --prefix zathura)/lib/zathura ; mkdir -p $d ; for n in cb djvu pdf-mupdf pdf-poppler ps ; do p=$(brew --prefix zathura-$n)/lib$n.dylib ; [[ -f $p ]] && ln -s $p $d ; done
```

To use zathura as macOS application, run following command. You have to run this command each time you're installing new plugins to update bundle info.
```shell
curl https://raw.githubusercontent.com/homebrew-zathura/homebrew-zathura/refs/heads/master/convert-into-app.sh | sh
```

# Uninstall

[](https://github.com/homebrew-zathura/homebrew-zathura#uninstall)

Homebrew will throw errors unless you uninstall plugins before Zathura.

```shell
brew uninstall --force zathura-pdf-mupdf
brew uninstall --ignore-dependencies --force girara
brew uninstall zathura
```

Optionally untap the repo

```shell
brew untap $(brew tap | grep zathura)
```

#  How to use
## navigate
j/k
h/l
n/p: next page, last page
gg/G
:page number
+/-: zoom in/out
0: reset zoom
ctrl + scroll: zoom in/out
f: full screen
r: rotate
s: 120%?
b: bookmark
ctrl b: add bookmark
/: search
?: search
n/N: next match
q: quit
F1: help