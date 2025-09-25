---
title: "How to use docker easy connect"
date: 2025-09-25T09:58:21.892Z
draft: false
tags: []
---

- Install docker
- Use command
```
docker run --rm --device /dev/net/tun \
	--cap-add NET_ADMIN -ti -e PASSWORD=xxxx \
	-e DISABLE_PKG_VERSION_XML=1 -e URLWIN=1 \
	-v $HOME/.ecdata:/root -p 127.0.0.1:5901:5901 \
	-p 127.0.0.1:1080:1080 -p 127.0.0.1:8888:8888 \
	hagb/docker-easyconnect:latest
```
It maps the 127.0.0.1:1080 as socks proxy, and 127.0.0.1:8888 as http proxy. Mostly, you should use 127.0.0.1:1080 as your proxy.
If 1080 or 8888 is occupied by some other application, you can change to another port like 1081 or 8889.
- Optional: You can make it into your bashrc or zshrc. And you can use prompt "easyconnect" to launch your easyconnect docker
```
alias easyconnect='docker run --rm --device /dev/net/tun --cap-add NET_ADMIN -ti -e PASSWORD=xxxx -e DISABLE_PKG_VERSION_XML=1 -e URLWIN=1 -v $HOME/.ecdata:/root -p 127.0.0.1:5901:5901 -p 127.0.0.1:1081:1080 -p 127.0.0.1:8889:8888 hagb/docker-easyconnect:latest'
```
- Type "easyconnect" to launch easyconnect docker
- Use VNC software to connect to "127.0.0.1:5091". If you are using mac, you can use software "ScreenSharing". The password is "xxxx".
- Use screen sharing, type https://remote.hkust-gz.edu.cn to enter the domain, and type your account and password.
- Now you need to use the socks port "127.0.0.1:1080" as your proxy.