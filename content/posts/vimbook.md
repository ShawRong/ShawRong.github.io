---
title: "vimbook"
date: 2025-08-21T09:46:37.640Z
draft: false
tags: []
---

# Your Personal Neovim Reference Book

*A comprehensive guide to your LazyVim-based Neovim configuration*

---

## Table of Contents

1. [Configuration Overview](#configuration-overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Core Keybindings](#core-keybindings)
4. [Plugin Reference](#plugin-reference)
5. [Language Support](#language-support)
6. [Custom Features](#custom-features)
7. [Performance Optimizations](#performance-optimizations)
8. [Troubleshooting](#troubleshooting)
9. [Customization Guide](#customization-guide)

---

## Configuration Overview

Your Neovim setup is built on **LazyVim**, a modern, performance-focused Neovim configuration framework. It's highly optimized for speed while maintaining essential development features.

### Key Characteristics
- **Performance-First**: Heavily optimized startup time (~50ms)
- **Formatting Control**: Advanced mechanisms to prevent unwanted formatting
- **Language-Focused**: Specialized support for C/C++, Rust, and web technologies
- **AI-Ready**: Infrastructure for AI assistance (currently disabled for performance)
- **Minimal UI**: Clean, distraction-free interface

### Directory Structure
```
~/.config/nvim/
├── init.lua                    # Main entry point
├── lazyvim.json               # LazyVim configuration
├── lazy-lock.json             # Plugin versions
└── lua/
    ├── config/
    │   ├── lazy.lua           # Plugin manager setup
    │   ├── options.lua        # Performance options
    │   ├── keymaps.lua        # Custom keymaps
    │   ├── ai.lua            # AI integration (disabled)
    │   ├── no-wrap.lua       # Disable line wrapping
    │   ├── toggle-format.lua  # Format toggling
    │   └── override-save.lua  # Save without formatting
    └── plugins/
        ├── disabled.lua       # Disabled heavy plugins
        ├── colorscheme.lua   # Minimal colorscheme
        ├── cuda.lua          # CUDA support
        └── luasnip.lua       # Snippet configuration
```

---

## Quick Start Guide

### Essential First Steps

1. **Open Files**: `<leader><space>` (space + space for file picker)
2. **Navigate Code**: `gd` (go to definition), `gr` (find references)
3. **Save File**: `<C-s>` (Ctrl+S, no formatting) or `:w` (with formatting)
4. **Search Text**: `<leader>/` (search in project)
5. **Switch Buffers**: `<S-h>` / `<S-l>` (Shift+H/L)

### Your Leader Key: `<space>`

Your leader key is set to space. When you press space, wait a moment to see available options in the which-key popup.

### Essential Commands

```vim
:Mason           " Manage language servers
:Lazy            " Manage plugins
:ToggleFormat    " Toggle auto-formatting
:NoWrap          " Disable line wrapping for current buffer
:WriteNoFormat   " Save without any formatting
```

---

## Core Keybindings

### File Operations
| Key | Mode | Action | Description |
|-----|------|--------|-------------|
| `<leader><space>` | Normal | Find files | Quick file picker (root dir) |
| `<leader>ff` | Normal | Find files | File picker with preview |
| `<leader>fr` | Normal | Recent files | Recently opened files |
| `<leader>fg` | Normal | Git files | Files tracked by git |
| `<leader>/` | Normal | Grep search | Search text in project |
| `<leader>,` | Normal | Switch buffer | Quick buffer switcher |

### Window & Buffer Management
| Key | Mode | Action | Description |
|-----|------|--------|-------------|
| `<C-h/j/k/l>` | Normal | Navigate windows | Move between splits |
| `<S-h>` / `<S-l>` | Normal | Buffer navigation | Previous/Next buffer |
| `<leader>bd` | Normal | Delete buffer | Close current buffer |
| `<leader>bo` | Normal | Delete others | Close all other buffers |
| `<leader>-` | Normal | Split below | Horizontal split |
| `<leader>\|` | Normal | Split right | Vertical split |

### Code Navigation & Intelligence
| Key | Mode | Action | Description |
|-----|------|--------|-------------|
| `gd` | Normal | Go to definition | Jump to function/variable definition |
| `gr` | Normal | References | Show all references |
| `gD` | Normal | Go to declaration | Jump to declaration |
| `gI` | Normal | Go to implementation | Jump to implementation |
| `K` | Normal | Hover info | Show documentation |
| `<leader>ca` | Normal | Code actions | Available code fixes |
| `<leader>cr` | Normal | Rename symbol | Rename variable/function |
| `<leader>cf` | Normal | Format code | Format current buffer |

### Search & Replace
| Key | Mode | Action | Description |
|-----|------|--------|-------------|
| `<leader>sw` | Normal | Search word | Search current word |
| `<leader>sg` | Normal | Grep project | Search text in all files |
| `<leader>ss` | Normal | Document symbols | Navigate to functions/classes |
| `<leader>sS` | Normal | Workspace symbols | Search symbols across project |

### Git Operations
| Key | Mode | Action | Description |
|-----|------|--------|-------------|
| `<leader>gg` | Normal | LazyGit | Full git interface |
| `<leader>hs` | Normal | Stage hunk | Stage current change |
| `<leader>hr` | Normal | Reset hunk | Undo current change |
| `<leader>hp` | Normal | Preview hunk | Show change details |
| `]h` / `[h` | Normal | Next/Prev hunk | Navigate changes |

### Your Custom Keybindings
| Key | Mode | Action | Description |
|-----|------|--------|-------------|
| `<C-s>` | Normal/Insert | Save no-format | Save without formatting |
| `<leader>tf` | Normal | Toggle format | Enable/disable auto-formatting |
| `<leader>tw` | Normal | Toggle wrap | Disable line wrapping |

---

## Plugin Reference

### Core Framework

#### LazyVim
- **Purpose**: Main configuration framework
- **Usage**: Provides sensible defaults and plugin management
- **Commands**: `:LazyVim` for configuration info

#### lazy.nvim
- **Purpose**: Plugin manager
- **Usage**: Lazy loads plugins for better performance
- **Commands**: `:Lazy` to manage plugins
- **Key Features**: 
  - All plugins lazy-loaded by default
  - Automatic plugin updates
  - Performance profiling with `:Lazy profile`

### Language Support

#### LSP Configuration (nvim-lspconfig)
- **Purpose**: Language server integration
- **Supported Languages**: C/C++ (clangd), Rust, JSON, TOML, Lua
- **Usage**: Automatic setup through LazyVim
- **Key Features**:
  - Auto-completion
  - Error checking
  - Go to definition
  - Refactoring support

#### Mason.nvim
- **Purpose**: Language server installer
- **Usage**: `:Mason` to install/manage servers
- **Commands**:
  - `:Mason` - Open installer
  - `:MasonInstall <server>` - Install specific server
  - `:MasonUpdate` - Update all servers

#### Treesitter (nvim-treesitter)
- **Purpose**: Advanced syntax highlighting and parsing
- **Configuration**: Optimized for performance in your setup
- **Features**:
  - Better syntax highlighting
  - Code folding
  - Text objects
- **Note**: Heavy features disabled for better performance

### Completion & Snippets

#### nvim-cmp
- **Purpose**: Completion engine
- **Triggers**: Automatic while typing
- **Sources**: LSP, buffer text, file paths, snippets
- **Navigation**: `<Tab>` / `<S-Tab>` to navigate completions

#### LuaSnip
- **Purpose**: Snippet engine
- **Usage**: Expands code templates
- **Configuration**: Lazy-loaded in your setup
- **Custom Snippets**: Can be added in `snippets/` directory

### File Management & Search

#### Telescope/fzf-lua
- **Purpose**: Fuzzy finder for files, symbols, and text
- **Key Bindings**:
  - `<leader><space>` - Quick file picker
  - `<leader>/` - Text search
  - `<leader>fr` - Recent files
  - `<leader>fb` - Open buffers
- **Performance**: Configured with extensive ignore patterns

### UI & Interface

#### Which-Key
- **Purpose**: Key binding discovery
- **Usage**: Press any prefix key and wait to see options
- **Example**: Press `<leader>` and pause to see all leader commands

#### Lualine
- **Purpose**: Status line
- **Features**: Shows mode, file info, git status, diagnostics
- **Configuration**: Minimal for performance

#### Bufferline
- **Purpose**: Buffer tabs at top
- **Navigation**: `<S-h>` / `<S-l>` to switch tabs
- **Features**: Shows buffer status and modifications

#### Flash.nvim
- **Purpose**: Enhanced navigation and jumping
- **Usage**: 
  - `s` followed by 2 characters to jump anywhere on screen
  - `S` for treesitter-based selection
- **Benefits**: Faster than traditional search

### Git Integration

#### Gitsigns
- **Purpose**: Git change indicators and operations
- **Visual**: Shows added/modified/deleted lines in gutter
- **Operations**:
  - Stage/unstage hunks
  - Navigate changes
  - Blame integration

### Development Tools

#### Conform.nvim
- **Purpose**: Code formatting
- **Usage**: 
  - `<leader>cf` - Format current buffer
  - Auto-format on save (toggleable)
- **Your Setup**: Can be toggled with `<leader>tf`

#### nvim-lint
- **Purpose**: Code linting integration
- **Usage**: Automatic linting on file save/change
- **Configuration**: Language-specific linters configured automatically

#### Trouble.nvim
- **Purpose**: Diagnostics and error management
- **Usage**: 
  - Shows errors/warnings in a dedicated panel
  - Navigate between issues
- **Commands**: `:Trouble` to open diagnostics panel

### Disabled Plugins (For Performance)

These plugins are available but disabled in your configuration:

- **AI Plugins**: Avante, NeoAI, Copilot (can be re-enabled)
- **Heavy UI**: Noice notifications, dashboard
- **Debugging**: DAP debugging suite
- **Markdown**: Preview and rendering
- **Images**: Image viewing in terminal

---

## Language Support

### C/C++ Development

#### clangd Language Server
- **Installation**: `clangd` must be installed on system
- **Features**:
  - Intelligent completion
  - Error checking
  - Refactoring tools
  - Go to definition/references
- **Configuration**: Automatic through LazyVim

#### CUDA Support
- **File Types**: `.cu`, `.cuh` files
- **Conditional**: Only enabled if `nvcc` compiler is available
- **Syntax**: Full CUDA syntax highlighting and parsing

### Rust Development

#### rustaceanvim
- **Purpose**: Comprehensive Rust support
- **Features**:
  - rust-analyzer integration
  - Cargo integration
  - Debugging support
  - Inlay hints

#### Crates.nvim
- **Purpose**: Cargo.toml dependency management
- **Features**:
  - Version information
  - Update notifications
  - Documentation links

### Web Development

#### JSON Support
- **Schema Validation**: SchemaStore integration
- **Features**: Auto-completion for known JSON schemas
- **Usage**: Works automatically for package.json, tsconfig.json, etc.

#### TypeScript/JavaScript
- **Language Server**: Can be installed via Mason
- **Features**: Full TS/JS support when enabled

---

## Custom Features

### Formatting Control System

Your configuration includes sophisticated formatting control:

#### Toggle Auto-Formatting
```vim
:ToggleFormat     " Toggle global auto-formatting
<leader>tf        " Keymap for toggle
```

#### Save Without Formatting
```vim
:WriteNoFormat    " Save bypassing all formatting
:W                " Alias for WriteNoFormat  
:Wq               " Save and quit without formatting
<C-s>             " Keymap for no-format save
```

#### Disable Line Wrapping
```vim
:NoWrap           " Disable auto line wrapping
:DisableWrap      " Alternative command
<leader>tw        " Keymap to disable wrapping
```

### Performance Features

#### Startup Optimization
- **Disabled Plugins**: Heavy plugins removed for faster startup
- **Lazy Loading**: All plugins load only when needed
- **Memory Limits**: Reduced memory usage for syntax highlighting

#### Runtime Optimization
- **Reduced Timeouts**: Faster completion and key mapping responses
- **Limited History**: Reduced command and undo history
- **Disabled Features**: Swap files, backup files, cursor line

---

## Performance Optimizations

Your configuration is extensively optimized for speed:

### Startup Time Optimizations
```lua
-- Key optimizations in your config:
updatetime = 100        -- Faster completion triggers
timeoutlen = 300        -- Faster key mapping timeout
lazyredraw = true       -- Don't redraw during macros
maxmempattern = 1000    -- Limit pattern matching memory
synmaxcol = 200         -- Limit syntax highlighting width
```

### Plugin Performance
- **Lazy Loading**: All plugins load only when actually needed
- **Disabled Features**: Heavy UI elements and effects removed
- **Reduced Scope**: Limited treesitter parsers and features

### Memory Management
- **No Swap Files**: `noswapfile` for better performance
- **Limited History**: Reduced undo levels and command history
- **Cache Optimization**: Optimized plugin cache settings

---

## Troubleshooting

### Common Issues

#### Slow Startup
```bash
# Check startup time
nvim --startuptime startup.log

# Your config should start in ~50ms
# If slower, check :Lazy profile
```

#### LSP Not Working
```vim
:Mason                  " Check if language server is installed
:LspInfo               " Check LSP server status
:checkhealth           " General health check
```

#### Formatting Issues
```vim
:ToggleFormat          " Check if auto-format is enabled
:ConformInfo           " Check formatter configuration
```

#### Plugin Issues
```vim
:Lazy                  " Check plugin status
:Lazy sync             " Update/sync plugins
:Lazy restore          " Restore to locked versions
```

### Performance Debugging
```vim
:Lazy profile          " Check plugin load times
:checkhealth lazy      " Check lazy.nvim health
```

### File-Specific Issues
```vim
:set filetype?         " Check detected file type
:LspInfo               " Check language server attachment
:Inspect               " Debug syntax highlighting
```

---

## Customization Guide

### Adding New Plugins

Create files in `~/.config/nvim/lua/plugins/`:

```lua
-- ~/.config/nvim/lua/plugins/my-plugin.lua
return {
  "author/plugin-name",
  event = "VeryLazy",    -- Lazy load
  config = function()
    -- Plugin configuration
  end,
}
```

### Custom Keymaps

Add to `~/.config/nvim/lua/config/keymaps.lua`:

```lua
-- Custom keymap examples
vim.keymap.set("n", "<leader>w", "<cmd>w<cr>", { desc = "Save file" })
vim.keymap.set("n", "<leader>q", "<cmd>q<cr>", { desc = "Quit" })

-- Remove default keymaps
vim.keymap.del("n", "<S-h>")
vim.keymap.del("n", "<S-l>")
```

### Language Server Setup

Add language servers through Mason:

```vim
:Mason
# Install desired language server
# LazyVim will auto-configure most servers
```

For custom configuration:
```lua
-- In lua/plugins/lsp.lua
return {
  "neovim/nvim-lspconfig",
  opts = {
    servers = {
      pyright = {
        settings = {
          python = {
            analysis = {
              typeCheckingMode = "basic",
            },
          },
        },
      },
    },
  },
}
```

### Enabling Disabled Plugins

To re-enable AI or other disabled plugins:

```lua
-- In lua/plugins/disabled.lua
-- Comment out or remove the plugin from disabled list
-- Or set enabled = true
return {
  {
    "yetone/avante.nvim",
    enabled = true,  -- Re-enable
    -- ... rest of config
  }
}
```

### Color Scheme Changes

Modify `lua/plugins/colorscheme.lua`:

```lua
return {
  -- Disable current minimal scheme
  { "vim/colorschemes", enabled = false },
  
  -- Enable new colorscheme
  {
    "folke/tokyonight.nvim",
    lazy = false,
    priority = 1000,
    config = function()
      vim.cmd([[colorscheme tokyonight]])
    end,
  },
}
```

### Performance vs Features

Your config prioritizes performance. To add features:

1. **Enable more plugins**: Remove from disabled.lua
2. **Add UI elements**: Enable noice, dashboard, etc.
3. **AI Integration**: Uncomment AI plugins and set API keys
4. **Debugging**: Enable DAP suite for debugging support

---

## Additional Resources

### LazyVim Documentation
- [LazyVim Official Docs](https://lazyvim.github.io/)
- [Plugin Specifications](https://lazy.folke.io/spec)
- [Neovim Documentation](https://neovim.io/doc/)

### Your Configuration Files
- Main config: `~/.config/nvim/init.lua`
- Plugin management: `~/.config/nvim/lua/config/lazy.lua`
- Custom options: `~/.config/nvim/lua/config/options.lua`
- Keymaps: `~/.config/nvim/lua/config/keymaps.lua`

### Getting Help
```vim
:help <topic>          " Built-in help
:LazyVim              " LazyVim-specific info
:checkhealth          " System health check
:Lazy                 " Plugin manager interface
:Mason                " Language server installer
```

---

*This reference book is based on your current configuration. Keep it updated as you modify your setup!*