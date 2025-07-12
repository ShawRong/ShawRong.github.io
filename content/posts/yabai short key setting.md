---
title: "yabai short key setting"
date: 2025-07-12T07:26:05.558Z
draft: false
tags: []
---

# Basic .skhdrc configuration for yabai
# Save this as ~/.skhdrc

# Navigation - focus windows
alt - h : yabai -m window --focus west
alt - j : yabai -m window --focus south
alt - k : yabai -m window --focus north
alt - l : yabai -m window --focus east

# Moving windows
shift + alt - h : yabai -m window --warp west
shift + alt - j : yabai -m window --warp south
shift + alt - k : yabai -m window --warp north
shift + alt - l : yabai -m window --warp east

# Resize windows
shift + alt - a : yabai -m window --resize left:-50:0; yabai -m window --resize right:-50:0
shift + alt - s : yabai -m window --resize bottom:0:50; yabai -m window --resize top:0:50
shift + alt - w : yabai -m window --resize top:0:-50; yabai -m window --resize bottom:0:-50
shift + alt - d : yabai -m window --resize right:50:0; yabai -m window --resize left:50:0

# Balance windows
shift + alt - 0 : yabai -m space --balance

# Move windows to spaces
shift + alt - 1 : yabai -m window --space 1; yabai -m space --focus 1
shift + alt - 2 : yabai -m window --space 2; yabai -m space --focus 2
shift + alt - 3 : yabai -m window --space 3; yabai -m space --focus 3
shift + alt - 4 : yabai -m window --space 4; yabai -m space --focus 4

# Switch spaces
alt - 1 : yabai -m space --focus 1
alt - 2 : yabai -m space --focus 2
alt - 3 : yabai -m space --focus 3
alt - 4 : yabai -m space --focus 4

# Layout controls
alt - r : yabai -m space --rotate 90
alt - y : yabai -m space --mirror y-axis
alt - x : yabai -m space --mirror x-axis
alt - t : yabai -m window --toggle float
alt - f : yabai -m window --toggle zoom-fullscreen
alt - space : yabai -m space --layout $(yabai -m query --spaces --space | jq -r 'if .type == "bsp" then "stack" else "bsp" end')

# Focus monitor
alt - tab : yabai -m display --focus recent
alt - 1 : yabai -m display --focus 1
alt - 2 : yabai -m display --focus 2

# Move window to monitor
shift + alt - tab : yabai -m window --display recent; yabai -m display --focus recent
shift + alt - 1 : yabai -m window --display 1; yabai -m display --focus 1
shift + alt - 2 : yabai -m window --display 2; yabai -m display --focus 2

# Restart yabai
shift + alt - r : yabai --restart-service

# Float/unfloat and center window
alt - c : yabai -m window --toggle float; yabai -m window --grid 4:4:1:1:2:2

# Make window zoom to parent node
alt - z : yabai -m window --toggle zoom-parent

# Toggle picture-in-picture
alt - p : yabai -m window --toggle pip



# start skhd
```c
skhd --start-service
```