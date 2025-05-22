---
title: "Lazyvim-Memo"
date: 2025-05-22T17:17:38.735Z
draft: false
tags: []
---

I come across an error when I tried to use nvim in iterm2(set to a background).
The down side of termnial start to wink when using nvim.
The error is fixed by changing the nvim background to transparent.

add a new file in ~/.config/nvim/lua/plugins/colorscheme.lua
```
return {
{
  "folke/tokyonight.nvim",
  opts = {
    transparent = true,
    styles = {
      sidebars = "transparent",
      floats = "transparent",
    },
  },
},
}
```

## Note:
Actually, it doesn't work at all.
Fixed by dragging the little bar down side to the bottom.


# Auto Completion Problem
A problem comes to me: How to toggle the auto completion of Rust language?
- First, Toggle the plugin: nvim-cmp
- Install rust analyzer using brew
- Toggle rust lang plugin in lazyvim (just get into the front page, and use 'x' to select)
- Done