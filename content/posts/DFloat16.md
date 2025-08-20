---
title: "DFloat16"
date: 2025-08-20T11:36:03.072Z
draft: false
tags: []
---

#lossless #compression  #mlsystem #float #gpu #LLM #llm
**Title**: 70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float

Draft Note
========
This is a draft of notation of important ideas

Abstract
-------
note:
- Low entropy in the b-float16 weight representation of LLMs, which reveals significant inefficiency in existing storage format.
- main idea of the code design: by applying entropy coding, this system assigns dynamic-length encodings to weights based on frequency. (without any loss of precision).
- to facilitate the efficient inference based on this design with this encoding, 3 designs are facilitated: 
	- decomposition of LUTs into compact LUTs, which fit in GPU SRAM.
	- two-phase kernel for coordinating thread read/write positions using lightweight auxiliary variables
	- transformer-block-level decompression to minimize latency. 
- d float16 achieves 1.9-38.8 higher throughput, enables 5.3-13.17 longer context lengths.
- 70%size, lossless compression


questions: 
- [x]  what's entropy coding?
	- It says according to the paper Deep Compression by Han Song(2015), b float16 exponents carry less information than their allocated bit width. 
	- We know fro brain float16, 1bit for sign, 8 bits for the exponent, 7 bits for the mantissa
	- with an entropy of approximately 2.6 bits compared to the assigned 8 bits. (This is from the old paper published in 2015.)
	- The entropy coding here, is Huffman coding. Huffman coding is a lossless data compression algorithm that use less bits for frequent symbols .
	 - The compression itself needs kernel to facilitate the gpu inference
- [x] what's the dynamic-length encoding, what's dynamic, what's get coded. what's the length referring to.
	- This encoding is a data format called Dynamic-Length Float. It's a data format, which comes directly from Huffman encoding.
	- Why it's dynamic, cause the length of code for exponent are changing according to Huffman code
	- The exponent component 8bits are get coded. The distribution of the raw data representation of the exponent is distributed around 100-150, which means most of the data are in a range around 1e-5 to 1e5.
	- The length refers to the bits usage for a single float point. 
- [x] Are they all got coded or open exponent part?
	- According to previous study and statistics about the LLM weights, the authors found the entropy of the exponent part are low compared with the assigned full 8 bits. The other 2 component named sign and mantissa are uniform, representing high entropy.
	- So the main work focus on minimize the bits allocated to the exponent part, which leads to 30% improvement in memory perspective.
- [x] what's the meaning of saying 'when L = 32, a direct lookup table would require 2^32 billion entries'? why we need that much entries, that's too much thinking of my previous thought.
	- we need to be aware of the previous statement 'Each entry in LUT maps an L-bit sequence to the decoded symbol whose Huffman code.'
	- So actually, it takes a 32bit long sequence, and map it to the corresponding symbol without erase the duplicate sequence with prefix.
	- It's a dumb shit, I think.
- [x] It's weight get encoded or what?
	- only weight are encoded. 
- [x] They are using LUTs for decoding, what's stored in LUTs? They said LUTs are 'decomposed'? What's the method to do decomposition? Tree or block?
	- It's symbol, i.e. the corresponding exponents stored in LUTs, and the entries are sequences of Huffman code. It demands a 32 bit long sequence to identify the corresponding symbol using the LUT. But 32 bit sequence leads to 2^32 entries, which is too large to fit in the SRAM. It's divided into 4 different LUTs, using 1 byte to map to the symbol, and the the failure of the current byte look up will leads to the next LUT look up using the next byte. The whole LUT is decomposed into 4 different one.
- [x] It mentions read and write operations, It's easy to think of read weight while inferencing. Can writing towards weight happen while inference?
	- It needs write because of the stage of decompressing. We need to recover our original weight matrix, which demands to reserve some memory to write things.
- [x] It says identification for w/r positions in encoded/decoded weight. what's the meaning of decoded weight? what's the positions? Why it's necessary?
	- You can check the answer of the previous question
- [x] It says 'encoded weights are tightly packed, and it's hard to read/write'. Are they stored in a compact way, like a continuous bit stream without any gap? 
	- Yes, the entries for LUTs are stored in a compact way like a bit stream. The system handles 32bits every operation.
- [x] It involves write while decoding, write decoded weights to where?
	- To the SRAM.
- [x] It says decompress small scaled weights are trivial, so we need to decompress a transformer-block level matrix. what's the weight matrix decompressed? 
	- It make the matrix to decompress a batch, to make full use of the block of GPU.
- [x] how does these 3 design facilitate the efficient inference?
	- It facilitate LUT look up. 
	- It is compensation instead of improvement. It's the necessary cost of adaption of entropy code.
- [x] what's LUTs(lookup tables), why it's needed? how is it being compressed? decomposition? how? can we make a better design?
	- It's needed because we want to make use of Huffman code instead of the sparse, low entropy original exponent. 
	- It's compressed as a entires of the Huffman tree to store in the memory instead of the full exponent. We need to transfer it from entries to the full exponent by looking up the LUTs
	- As a result, the compression is basically make up LUTs, and store entires along with the signal bit and mantissa bits.
	- The decompression is just look up, and recover the original B Float16 format.
- [x]  This is a question related to the Two-Phase Kernel. What does it do?
	- We don't know to starting position for each thread to begin decoding
	- Each thread are not clear about the real index of the elements being decoded, leading to incapability of figuring out the correct output/write location for storing the result
	- Authors designate a array called Gaps to store the valid offset of the assigned starting byte of each thread, to tell the starting position for each thread under the context that every thread is assigned starting point identified by a byte. 
	- A naive approach to handle the 2nd issue is assign an array to different thread to write thing, which is impractical due to storage overhead. The authors designed a pretty dummy algorithm to perform 2 phase kernel which I would like to call it Ekko inspired by the LOL character Ekko who can reverse the time (🎵反方向的钟). It read the sequence, and count how many elements it's going to decode. And after that, It read the sequence again(Ekko hit R), writing the decoded result to memory this time.
- [x] why we need it to be fit in GPU SRAM? what's SRAM? is it shared memory? How can we avoid the latency of loading composition of LUTs
	- It's shared memory. We fit it into a small size to load into SRAM to facilitate the look up, without the need of loading many times.
- [x] what's two-phase kernel for coordinating thread read/write? what are these specifically phases?
	- First, we know it's a kernel, we perform these element-wise function simultaneously.
	- at phase1, we do position appointment. And after the position to be w/r are determined, we perform phase2, which do the operation w/r.
	- Benefits: Massive Parallelism, Memory coalescing: Threads can be arranged to access memory efficiently, Avoiding Racing Condition
- [x] why we need transformer-block-level decompression?
	- because we want to pack many matrix up, to form a large data to decompression to improve the throughput.
- [x] did they stored the so-called gap offsets and block-level output positions on the gpu? 
	- Yes, they form them once, and store them in the memory for future usage.
- [x] In the experiment part, the figure 3 illustrate a comparison between throughout and latency tested on original model running with GPU and CPU offloading and lossless and compressed model with running on pure GPU. Is it reasonable?
	- Slight reasonable, if we constrain the context to a limited memory senario.

summary:
- Entropy coding is a core technique in lossless data compression that leverages statistical redundancy to reduce data size. It includes Huffman coding, arithmetic coding and Asymmetric Numeral Systems. A wide-known feature of Huffman code is that no code is a prefix of any other.  The entropy coding here is Huffman coding, which is lossless.
- About the entropy coding, the cited paper claims are too old to validate. The claim 'bits of the exponent with an entropy 2.6 bits compared to 8 bits' may not holds.
- bit by bit traversal of a Huffman tree is not suitable for GPU parallel architecture.
- so we need kernels to handle it. (it's a problem pretty old)
- It's observed that the distribution of exponent value isn't uniform, representing low entropy. The sign and mantissa component are right about the distribution.
- LUT maps sequence to a decoded symbol(exponent) according to the prefix of the sequence.
- It uses a Huffman tree in depth of maximum 32. It's processed that is larger than 32.
- We retrieve L bits from the bit stream, figuring out the corresponding symbol according to the codebook. And after this, for the reason we have no idea about the length of prefix, just knowing the symbol, we read another structure called 'Code Lengths' to lookup the length of this prefix, which is a lookup table who maps each symbol to its corresponding prefix length.
- The designed LUTs maps 32bit sequence to the symbol without diminish duplicate ones, leading to 2^32 entries, which is skeptical. By the way, the author realize their inability of handling such a big LUTs, and divide it into 4 LUTs with 2^8 entries each, according to the fact that there are 4 bytes with the constraint of 32bits. All together, the entires occupy about 1280 bytes totally. The search procedure are in a one by one way looking up, querying the 1st table the the next. If it hits a valid result, this symbol, i.e. Huffman code, will be returned, otherwise the search will not end, switching to the next table. It's factual that 1280 bytes fits in GPU SRAM and enables fast access.
- There is problem that the look up without prefix bytes which had been used in the previous search procedure can lead to conflict and ambiguity. For example, we get two 32 bits sequence to look up, sharing same lower 2 bytes yet distinct higher 2 bytes leading to invalid result causing further look up using lower bytes, can lead to wrong decoded symbol. It's avoidable by performing frequencies adjustment. 
- We don't know the starting position for each thread to begin decoding. And each thread is not clear about the real index of the elements being decoded, leading to incapability of figuring out the correct writing location for storing the result. Authors designate a array called Gaps to store the valid offset of the assigned starting byte of each thread, to tell the starting position for each thread under the context that every thread is assigned starting point identified by a byte.  A naive approach to handle the writing issue is assign an array to different thread to write things, which is impractical due to storage overhead. The authors designed a pretty dummy algorithm to perform 2 phase kernel which I would like to call it Ekko inspired by the LOL character Ekko who can reverse the time (🎵反方向的钟). It read the sequences, and count how many symbols it's going to decode. And after that, It read the sequence again(Ekko hit R), writing the decoded result to memory this time.
- They stored the gap offsets and block-level output position as auxiliary data to help read and write.
- The compressed weights are decoded while inference and decoded matrix is discarded after usage.
- Apparently, it's possible to improve the throughput by enlarging the size of data being decoded. And The authors decide to decode a block of transformer before executing the computation in a transformer block. So before executing any computation in a transformer block, they first decompress all its associated weights.
- In the experiment part, it shows compression ratio about 70% compared wth the original model, The perplexity and accuracy seems achieve an absolutely no loss result.
- In the appendix, it shows a figure with comparison between original model running on 2 GPU and their model running on a single gpu. It shows a significant pretty bad performance with regard to the throughput and latency, considering the fact that A8000 gets no nvlink.
- Figure 5 shows a comparison to show that the additional latency due to decompression get to be less important as the token batch being processed increases, which does not make any sense, of course. If I can spent a day 60 hours, the time I used to sleep is less significant compared with the time I spent to do other things, too.
- Basically, it sacrifices to achieve a 30% smaller model size.

idea:

TODO: 
- [ ]  Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding.  We still need to read the old paper "deep compression" by Han Song.
- [ ]  There are pretty much entropy coding-based compression methods. they are paper about CNNs, Deep Compression by Han Song, and three papers named "Neuzip", "Zipnn" and "Huff-llm".  Need to read later on. I want to know is it benefit for vector quantization?
- [ ] Can we use the difference to do this lossless compression. You can see the details from url: [GridFour](I think the main element get encoded are weights.)




checkpoint:
- Read https://gwlucastrig.github.io/GridfourDocs/notes/LosslessCompressionForFloatingPointData.html
- read 70%size from 3.2 However, when L = 32...