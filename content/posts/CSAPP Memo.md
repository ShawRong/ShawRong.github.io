---
title: "CSAPP Memo"
date: 2025-06-07T13:13:47.477Z
draft: false
tags: []
---

# Ch1
## Hardware of system
### Bus
It's a lot of different things, but we can describe them as a whole abstract thing, and call it bus.
There are several features of it.
- It passes message base on a uniform width called WORD (4 bytes or 8 bytes).
- It can be implemented by different kinds of hardware(mostly PCIe). 
- It basically connect between cpu and other components like I/O devices and Memory (in older architecture, they called southern bridge and northern bridge).
- People classify buses base on what they do. There are data bus, address bus and control bus.

Some key idea: 
- bus width, a word (4 or 8 bytes)
- bus speed, measured in MHz/GHz
- bandwidth (throughput), total data transferred per second.

modern bus: 
- modern bus(PCIe and USB) use high-speed serial lanes instead of wide parallel wires.
- PCIe lanes straight to CPU for I/O devices (some of them).
- There are no so call northern bridge in modern architecture.  (It's built-in memory controller now and some GPU controller now.)
- There are still southern bridge chip-set now. Because modern cpu can only handle limited number of PCIe directly connected to it. 
- There devices connected to cpu through southern bridge chip-set are slower. (Mostly some disk dirver, usb, network things)

### I/O devices 
It contains a lot of things, like keyboard, mouse or monitor.

I/O devices connected to I/O bus through **adaptor**
or **controller**.
**Difference**:
- Adaptor is a device plugin in mother board through some slot
- Controller is some built-in chip set in the mother board.
### Main Memory
DRAM: Dynamic random access memory
### CPU
**register file** : the group of registers

### DMA, GPU etc.
There is a trend to make extra device to help cpu to do something it can not do very well. We know cpu is a computing device for general purpose. And we need some new computing device for specific purpose. It's a good idea to introduce some device like DMA, GPU or DPU to help CPU. We can regard DMA, DPU and GPU just like a different kind of CPU reside in the bus.

**DMA**(direct memory access): If we want load blocks to main memory from disks or ssd without DMA, we need first load things to CPU register file, and put things from register file to main memory. This is trivial.  So we use DMA to do this dirty work.

**GPU**: 


### Cache
Usually there are cache in L1 and L2 level, there exists L3 cache in some architectures.

Cache is using hardware call **SRAM** (static random access memory).

## OS
Operating system provide three important thing to the user: 
- file
- virtual memory
- process
file is abstraction of I/O devices.
virtual memory is abstraction of main memory and I/O device.
process is abstraction of I/O device, main memory and processor. 

The application built based on OS interacts with OS instead of hardware directly. 

### process
**context**: It includes pc, values in the register file and content in the memory (virtual memory of the process).
**context switch**: save the context of old process. load the context of new process.

**system call**: If application needs operating system to do something, it will trigger a command system call. kernel is not a process, it's a collection of code and data structure for OS to manage all the process.

**thread**: It uses the context of the same process, and share the same code and global data. Advantage: It's easier to share data between thread comparing with process.

### virtual memory
Every process gets its own virtual address space .

from 0 to maximum address, we get:
- code and data(.BSS .DATA), some global variable and constant (loaded from executable file)
- heap, dynamic alloced memory
- skip...
- shared library. It's for standard c lib and math lib. recall dynamic lib linking
- stack. For function call
- kernel space. system call

### file
including every I/O devices, disk, keyboard, monitor and network.
## Sundry
### Compile
**Compile**: There are several key points when we talk about compiling
There is a clear pipeline if we consider the process of compiling a **source file** into a **executable program**. 

```
hello.c -pre-processor(cpp)-> hello.i -compiler(ccl)-> hello.s -assembler-> hello.o -linker(ld)-> hello
```

- the pre processor (cpp) will translate the original source file into a full file, like unfold the # include, or other things start with #. 
- assembly. It will translate a .i file into .s file, it's a plain text file containing program in assemble language form. 
- The (as) will translate .s into real machine language command, i.e .o file. These command are packaged as a file in a format called **relocatable object program**.
- link (ld). We know we need to link several different .o file (and dynamic library) to get a real executable file. This is what linker do. It can merge printf.o into our hello.o file, to get executable hello program (It can be loaded into memory and executed by operating system).