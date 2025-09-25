---
title: "How to use docker easy connect"
date: 2025-09-25T09:55:37.345Z
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
	-p 127.0.0.1:1081:1080 -p 127.0.0.1:8889:8888 \
	hagb/docker-easyconnect:latest
```
- Optional: You can make it into your bashrc or zshrc. And you can use prompt "easyconnect" to launch your easyconnect docker
```
alias easyconnect='docker run --rm --device /dev/net/tun --cap-add NET_ADMIN -ti -e PASSWORD=xxxx -e DISABLE_PKG_VERSION_XML=1 -e URLWIN=1 -v $HOME/.ecdata:/root -p 127.0.0.1:5901:5901 -p 127.0.0.1:1081:1080 -p 127.0.0.1:8889:8888 hagb/docker-easyconnect:latest'
```
- Type "easyconnect" to launch easyconnect docker
- Use VNC software to connect to "127.0.0.1:5091". If you are using mac, you can use software "ScreenSharing". The password is "xxxx".
- Use screen sharing, type https://remote.hkust-gz.edu.cn to enter the domain, and type your account and password.
- Now you need to use the socks port "127.0.0.1:1080" as your proxy.