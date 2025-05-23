---
title: "Lazyvim-Memo"
date: 2025-05-23T07:33:31.656Z
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

# Minimize the Animation
put following code to init.lua
```lua
-- 禁用动画和优化性能
vim.opt.winblend = 0               -- 禁用窗口透明混合
vim.opt.pumblend = 0               -- 禁用补全菜单透明
vim.opt.cursorline = false         -- 禁用光标行高亮（显著提升流畅度）
vim.opt.lazyredraw = true          -- 延迟重绘（减少宏/操作时的卡顿）
vim.opt.updatetime = 300           -- 更快的响应时间（默认4000ms）
vim.opt.timeoutlen = 300           -- 快捷键超时时间（降低延迟）

-- 如果仍有卡顿，可以尝试关闭语法高亮（临时测试）
-- vim.cmd("syntax off")
```