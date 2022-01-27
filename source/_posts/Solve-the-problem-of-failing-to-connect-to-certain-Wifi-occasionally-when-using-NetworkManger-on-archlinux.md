title: >-
  Solve the problem of failing to connect to certain Wifi occasionally when
  using NetworkManger on archlinux
date: 2018-08-19 12:29:25
tags: [archlinux, network-manager, linux]
categories: Coding
thumbnail: /images/wifi_678x452.png
---

I use archlinux and network-manager, and every now and then, I failed to connect to a certain wifi, while the other wifi worked pretty well. I didn't figure out the reason, but I found a solution.

First check which wifi connections do you have.

```bash
sudo ls /etc/NetworkManager/system-connections
AndroidAP	OpenWrt15
```

Suppose we have a problem connecting to `OpenWrt15`, what we need to do is just run the following command to move `OpenWrt15` to another location(Warning: make sure you know the wifi password of `OpenWrt15` before moving the file, because you will need to re-input the password again later on. Usually you can find the password in the `OpenWrt15` file)

```bash
sudo mv /etc/NetworkManager/system-connections/OpenWrt15 /tmp
```

Then restart NetworkManager.

```bash
sudo systemctl restart NetworkManager.service
```

And try connecting to that wifi again. You will need to re-input the password. If you don't know the password, you can probably find it in the original `/tmp/OpenWrt15` file.

Happy Hacking!