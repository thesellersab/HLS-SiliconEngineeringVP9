# Comprehensive Interview Questions for High-Level Synthesis Role at Google - Video Compression IP Design

## 1) Foundational (Simple)

### HLS Scheduling \& QoR

1. What is the difference between latency and initiation interval (II) in HLS pipelining, and how do they impact throughput calculation?
2. Explain the concept of ASAP (As Soon As Possible) scheduling versus ALAP (As Late As Possible) scheduling in HLS synthesis.
3. What does it mean when an HLS tool reports "II achieved = 3" for a pipelined loop?
4. How do you calculate the theoretical maximum throughput (pixels/second) of a video processing pipeline given a clock frequency and initiation interval?
5. What is resource binding in HLS, and why might the tool choose to share a single multiplier between multiple operations?
6. Describe the difference between function-level pipelining and loop-level pipelining in HLS.
7. What factors typically prevent achieving an II of 1 in HLS loop pipelining?
8. Explain how HLS scheduling handles resource constraints when multiple operations compete for the same hardware resource.
9. What is the significance of the critical path in HLS timing analysis?
10. How does loop-carried dependency affect the minimum achievable initiation interval?

### Memory \& Bandwidth

11. What are the key differences between BRAM, UltraRAM (URAM), and distributed RAM in FPGA implementations?
12. Calculate the memory bandwidth requirement for processing 4K@60fps 4:2:0 8-bit video (assume sequential access patterns).
13. Explain the concept of memory banking and why it's important for achieving high throughput in video processing.
14. What is the difference between single-port and dual-port memory interfaces in HLS?
15. How do you estimate the number of memory ports required for a 2D convolution kernel processing?
16. What is burst access in AXI4 interfaces, and why is it preferred over single-beat transfers?
17. Describe the memory hierarchy considerations when processing video frames that don't fit in on-chip memory.
18. What is the typical access pattern for a line buffer in video processing, and how does it affect memory design?
19. Explain the concept of memory interleaving and how it can improve bandwidth utilization.
20. What are the implications of different memory access patterns (sequential vs. random) on HLS memory interface optimization?

### Codec-Specific Blocks

21. Name the main functional blocks in an HEVC encoder pipeline and their typical processing order.
22. What is the purpose of motion estimation in video compression, and what are the typical search ranges used?
23. Explain the difference between integer and fractional motion estimation in video codecs.
24. What is the role of intra prediction in video compression, and how many directional modes does HEVC support?
25. Describe the basic operation of Discrete Cosine Transform (DCT) in video compression.
26. What is quantization in video coding, and how does it affect compression efficiency and quality?
27. Explain the purpose of deblocking filter and Sample Adaptive Offset (SAO) in HEVC.
28. What are Coding Tree Units (CTUs) and how do they relate to processing parallelization?
29. Describe the difference between luma and chroma processing in 4:2:0 video formats.
30. What is entropy coding and name two entropy coding methods used in modern video codecs?

### Verification \& Signoff

31. What is C/RTL co-simulation in HLS, and why is it essential for verification?
32. Explain the three phases of C/RTL co-simulation in Vitis HLS.
33. What does it mean when C/RTL co-simulation reports "FAIL" and how do you debug it?
34. What are the requirements for writing a self-checking testbench in HLS?
35. How do you verify bit-accuracy between C-simulation and RTL implementation?
36. What information can you extract from HLS synthesis reports to evaluate design quality?
37. Explain the importance of waveform analysis in debugging RTL co-simulation mismatches.
38. What is the purpose of using different input datasets during C/RTL co-simulation?
39. How do you validate that your HLS design meets timing constraints for a target clock frequency?
40. What are the key metrics to monitor during HLS synthesis to ensure design quality?

### Tool-Specific (Vitis/Catapult)

41. Name five commonly used HLS pragmas in Vitis HLS for optimization.
42. What is the `hls::stream` data type in Vitis HLS, and when should you use it?
43. Explain the difference between `ap_int` and native C++ `int` types in HLS.
44. What does the `#pragma HLS PIPELINE` directive do, and where do you place it?
45. In Catapult HLS, what is the equivalent of Vitis HLS `hls::stream`?
46. What is the purpose of the `#pragma HLS ARRAY_PARTITION` directive?
47. How do you specify interface protocols in Vitis HLS using pragmas?
48. What is the difference between SystemC and C++ input languages in Catapult HLS?
49. Explain the concept of channels in Catapult HLS and their hardware implementation.
50. What is the `ap_fixed` data type, and when is it preferred over floating-point in video processing?

## 2) Intermediate (Medium)

### HLS Scheduling \& QoR

1. Design a strategy to resolve a scheduling conflict where three multiply operations compete for two DSP48 blocks in a single clock cycle.
2. You have a loop with II=4 but need II=1 for real-time performance - outline your optimization approach without changing the algorithm.
3. Explain how to analyze and optimize a design where the critical path is dominated by a long chain of additions in a DCT computation.
4. A motion estimation kernel shows resource utilization at 95% LUTs but only 30% DSP48s - how do you rebalance the design?
5. Your HLS synthesis reports achieving latency=100 cycles but C/RTL co-simulation shows actual latency=150 cycles - explain possible causes.
6. How do you handle a scenario where loop pipelining is prevented by a variable loop bound in motion estimation search?
7. Describe techniques to minimize the impact of control logic overhead in a highly pipelined video processing design.
8. You need to pipeline a loop containing a division operation - what are your optimization options given that division has high latency?
9. Explain how to achieve deterministic latency in a design with data-dependent control flow for real-time video applications.
10. Your design meets throughput requirements at 200MHz but fails timing at 300MHz - outline your optimization strategy.

### Memory \& Bandwidth

11. Design a line buffer architecture for a 7x7 convolution kernel processing 1920-pixel wide video lines.
12. Calculate and compare the memory bandwidth requirements for processing 4K@60fps for H.264 (16x16 blocks) versus HEVC (64x64 CTUs).
13. You have 4K@60fps input but only 50% of required memory bandwidth - design a data reuse strategy for a motion estimation engine.
14. Implement a ping-pong buffer scheme for continuous video processing with frame-based algorithms.
15. Design a memory interface that supports simultaneous read/write for overlapped processing of consecutive video frames.
16. Your design requires random access to reference frames but DRAM provides optimal sequential access - propose a solution.
17. How do you optimize memory access patterns for a 2D separable filter to minimize bandwidth requirements?
18. Design a cache-friendly data layout for storing reference frames in a multi-reference motion estimation system.
19. Calculate the optimal FIFO depth for a dataflow region where producer generates 64 samples/cycle and consumer processes 32 samples/cycle.
20. You're hitting memory bandwidth limits with concurrent ME and MC operations - how do you prioritize and schedule memory access?

### Codec-Specific Blocks

21. Design an HLS-based integer motion estimation engine that supports variable block sizes from 8x8 to 64x64.
22. Implement an efficient SAD (Sum of Absolute Differences) computation unit that processes 16 pixels in parallel.
23. How do you implement fractional interpolation filters for quarter-pixel motion compensation with minimal memory overhead?
24. Design a configurable transform unit that supports 4x4, 8x8, 16x16, and 32x32 DCT/IDCT operations.
25. Implement an intra prediction unit that can generate all 35 directional modes for HEVC with shared hardware resources.
26. How do you design a quantization unit that supports both uniform and non-uniform quantization matrices?
27. Implement a Context-Adaptive Binary Arithmetic Coding (CABAC) encoding engine suitable for HLS implementation.
28. Design a deblocking filter that processes CTU boundaries with minimal external memory access.
29. How do you implement Sample Adaptive Offset (SAO) filtering with overlapped processing to maintain throughput?
30. Design a rate control module that can adjust quantization parameters based on buffer fullness and complexity metrics.

### Verification \& Signoff

31. Your C-simulation passes but RTL co-simulation fails with data mismatches - outline your debugging methodology.
32. Design a comprehensive testbench strategy for a motion estimation IP that covers edge cases and corner conditions.
33. How do you verify timing closure for a multi-clock domain video processing system?
34. Implement a golden reference model validation strategy for a complex transform coding block.
35. Your design passes individual block tests but fails system integration - how do you isolate the issue?
36. Design a coverage-driven verification plan for an entropy coding module with multiple configuration modes.
37. How do you validate the bit-exact compatibility of your HLS-generated IP with a software reference codec?
38. Implement an automated regression testing framework for HLS video IP development.
39. Your design shows timing violations in post-place-and-route but passed HLS timing analysis - explain and resolve.
40. How do you verify the functional correctness of a pipelined design with multiple concurrent operations?

### Tool-Specific (Vitis/Catapult)

41. Debug a Vitis HLS design where `#pragma HLS DATAFLOW` causes deadlock during C/RTL co-simulation.
42. Your Catapult HLS design compiles but generates suboptimal QoR compared to equivalent Vitis implementation - analyze potential causes.
43. Implement efficient inter-task communication using `hls::stream` with proper backpressure handling.
44. How do you migrate a SystemC-based Catapult design to Vitis HLS while maintaining functionality?
45. Resolve a situation where `#pragma HLS ARRAY_PARTITION` improves latency but exceeds BRAM resources.
46. Design a strategy for mixed-precision arithmetic using both `ap_fixed` and floating-point in the same design.
47. Your Vitis HLS reports successful synthesis but generates inefficient AXI interfaces - how do you optimize?
48. Debug a scenario where Catapult HLS channels show different behavior compared to expected SystemC simulation.
49. Implement efficient resource sharing between multiple processing units in a single HLS design.
50. How do you handle design portability between different HLS tool versions when pragma syntax changes?

### System Integration

51. Design a streaming interface between motion estimation and motion compensation blocks with proper flow control.
52. Implement a multi-rate pipeline where different processing stages operate at different throughput rates.
53. How do you design handshaking protocols between HLS-generated IP and external DMA controllers?
54. Integrate an HLS-based encoder block into a larger SoC with ARM processors and hardware accelerators.
55. Design a configuration interface that allows runtime reconfiguration of video codec parameters.

## 3) Expert (Very Difficult)

### HLS Scheduling \& QoR

1. Architect a complete motion estimation engine in HLS with bounded II=1 for 64x64 search range, supporting variable block sizes and early termination.
2. Design a hierarchical scheduling approach for a full HEVC encoder where encoding complexity varies significantly across CTUs.
3. Implement advanced loop transformations (fusion, interchange, skewing) in HLS to optimize a 2D separable filter with data-dependent coefficients.
4. Develop a resource-aware scheduling algorithm for dynamically balancing computational load across multiple parallel processing engines.
5. Design a temporal scheduling strategy for processing multiple video streams simultaneously with shared hardware resources.
6. Implement a self-adaptive pipeline that can dynamically adjust II based on input complexity and available resources.
7. Architect a speculative execution framework for motion estimation that predicts and pre-computes likely motion vectors.
8. Design a multi-constraint optimization approach that simultaneously minimizes area, power, and latency for video codec blocks.
9. Develop a cross-module scheduling optimization that considers global resource allocation across an entire encoder pipeline.
10. Implement advanced retiming techniques in HLS to achieve timing closure for high-frequency video processing designs.

### Memory \& Bandwidth

11. Design a hierarchical memory architecture for reference frame management that minimizes DRAM traffic for 8K@60fps 10-bit encoding.
12. Architect a cache-coherent memory system for multi-reference motion estimation with optimal replacement policies.
13. Implement a bandwidth-adaptive memory controller that dynamically adjusts access patterns based on real-time congestion.
14. Design a distributed memory architecture that supports concurrent multi-directional access patterns for advanced intra prediction.
15. Develop a predictive prefetching strategy for motion compensation that anticipates future reference data requirements.
16. Architect a memory-centric design that minimizes power consumption while maintaining 4K@120fps throughput.
17. Implement a virtualized memory interface that abstracts different memory technologies (HBM, DDR4, GDDR6) for portable designs.
18. Design an adaptive memory partitioning scheme that dynamically allocates resources between different video processing stages.
19. Develop a memory bandwidth model that accurately predicts performance for complex video processing workloads.
20. Implement a distributed memory coherency protocol for multi-core video processing systems.

### Codec-Specific Blocks

21. Architect a unified transform unit that efficiently supports HEVC, AV1, and VVC transform sets with shared hardware resources.
22. Design an adaptive loop filter for AV1 that supports all filter types (Wiener, self-guided, CNN-based) in a single HLS implementation.
23. Implement a Rate-Distortion Optimization (RDO) engine in HLS with support for multiple cost functions and early termination.
24. Architect a multi-format entropy coding engine that supports CABAC (HEVC), multi-symbol arithmetic coding (AV1), and VLC.
25. Design a wavefront parallel processing architecture for HEVC that maintains dependencies while maximizing throughput.
26. Implement an advanced temporal motion vector prediction system with support for bi-directional prediction and weighted prediction.
27. Architect a Content-Adaptive Loop Filter (CALF) for next-generation codecs with machine learning-based parameter estimation.
28. Design a unified intra prediction engine that supports HEVC (35 modes), AV1 (56 modes), and VVC (95+ modes).
29. Implement a sophisticated rate control algorithm that adapts to content complexity and network conditions in real-time.
30. Architect a multi-pass encoding system that optimizes encoding decisions across multiple frames simultaneously.

### Verification \& Signoff

31. Develop a formal verification framework for proving functional equivalence between HLS-generated RTL and reference software codec.
32. Design a comprehensive coverage model that ensures exhaustive testing of all codec configurations and corner cases.
33. Implement a machine learning-based test generation system that automatically creates challenging test scenarios for video codecs.
34. Architect a hardware-software co-verification environment that validates entire video processing pipelines.
35. Develop a performance regression detection system that automatically identifies QoR degradations across HLS tool versions.
36. Design a statistical analysis framework for characterizing codec performance across diverse video content types.
37. Implement a security-focused verification methodology that validates against potential side-channel attacks in video processing.
38. Architect a cloud-based verification infrastructure for continuous integration/continuous deployment of video IP.
39. Develop a formal timing analysis framework that proves worst-case timing behavior for safety-critical video applications.
40. Design a comprehensive error injection and fault tolerance testing methodology for video processing systems.

### Tool-Specific (Vitis/Catapult)

41. Analyze and resolve significant QoR divergence between Vitis HLS and Catapult HLS for the same video processing algorithm.
42. Design a tool-agnostic HLS coding methodology that maximizes portability while achieving optimal results on both platforms.
43. Implement advanced optimization strategies that exploit tool-specific features (Vitis dataflow vs. Catapult channels) for maximum performance.
44. Develop a methodology for migrating large-scale video codec designs between HLS tools while maintaining verification coverage.
45. Architect a hybrid verification approach that leverages strengths of both Vitis and Catapult simulation capabilities.
46. Implement tool-specific memory optimization strategies that achieve optimal bandwidth utilization for each platform.
47. Design a performance benchmarking framework that fairly compares video codec implementations across different HLS tools.
48. Develop advanced debugging techniques for tracking down tool-specific optimization interactions that cause unexpected behavior.
49. Implement a cross-tool compatibility layer that enables reuse of IP blocks between Vitis and Catapult environments.
50. Architect a methodology for selecting optimal HLS tools and configurations for different video codec implementation requirements.

### System Integration

51. Design a complete multi-codec transcoding engine that dynamically switches between HEVC, AV1, and VP9 based on client capabilities.
52. Architect a video processing SoC with real-time scheduling that balances encode/decode/transcode workloads across available resources.
53. Implement a network-adaptive video streaming system that optimizes codec parameters based on real-time bandwidth measurements.
54. Design a distributed video processing architecture that spans multiple FPGAs with latency-optimized interconnects.
55. Architect a software-defined video processing platform that can reconfigure hardware resources for different codec standards.
56. Implement a unified video analytics and compression pipeline that performs real-time content analysis while encoding.
57. Design a fault-tolerant video processing system that maintains service availability during hardware failures.
58. Architect a power-aware video codec that dynamically trades off quality for power consumption based on thermal conditions.
59. Implement a machine learning-assisted codec that adapts encoding parameters based on learned content characteristics.
60. Design a next-generation video codec research platform that supports experimental algorithms and emerging standards simultaneously.
<span style="display:none">[^1][^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^2][^20][^21][^22][^23][^24][^25][^26][^27][^28][^29][^3][^30][^31][^32][^33][^34][^35][^36][^37][^38][^39][^4][^40][^41][^42][^43][^44][^45][^46][^47][^5][^6][^7][^8][^9]</span>

# Updated Interview Questions List - Additional Intermediate Questions

Here are the additional 55 questions (numbered 61-115) for the **Intermediate (Medium)** section:

### HLS Scheduling \& QoR

61. For a separable 5×5 2D filter at 4K60, how would you pipeline row and column passes to achieve II=1, and where would you place line/window buffers? (aim: reason about separable filtering and storage reuse)
62. Given an inner loop with a loop-carried dependency (accumulator), outline two HLS refactors to reach II=1 without changing numerical results. (aim: tree reduction vs. partial sums)
63. A loop reports II=2 "due to resource constraint (1 multiplier)". What specific HLS directives would you try, and how would you quantify the area/throughput trade-off? (aim: RESOURCE/ALLOCATION vs. UNROLL)
64. When would you pipeline the outer loop instead of the inner loop in an 8×8 transform, and what are the implications on array partitioning? (aim: schedule shape and memory port pressure)
65. Show how loop tiling (blocking) on CTU rows can reduce on-chip buffer size while preserving II=1 in the innermost kernels. (aim: tile-level reuse vs. pipeline fill)
66. A Vitis HLS schedule shows a "read-after-write on RAM" preventing II=1. Propose two code/data layout changes to remove the hazard. (aim: banking/partitioning vs. access reordering)
67. Under what conditions does PIPELINE silently get ignored, and how do you detect it early from reports? (aim: dependence analysis, variable tripcount, function-level pipe)
68. Explain how function inlining interacts with DATAFLOW and can either help or hinder stage-level concurrency. (aim: producer/consumer visibility)

### Memory \& Bandwidth

69. Compute the raw luma pixel rate for 3840×2160@60 and derive minimum pixels/clk at 300 MHz for real-time. (aim: quick throughput math)
70. For 4:2:0 10-bit, estimate total bytes/frame and sustainable GB/s for 4K60 if every pixel is read once and written once. (aim: bandwidth budgeting)
71. Design a burst DMA plan to fetch 64×64 tiles with 512-bit AXI: what burst length and alignment would you choose and why? (aim: coalescing and alignment)
72. How would you interleave luma/chroma buffers across multiple memory banks to sustain half-rate chroma reads without stalling luma? (aim: banking strategy)
73. Propose a ping-pong (double buffer) scheme for reference-block prefetch that overlaps compute and DRAM, including when to switch buffers. (aim: latency hiding)
74. Given a single-port BRAM inferred for a coefficient table used twice per cycle, list two ways to avoid a port conflict without duplicating the table N times. (aim: ROM_2P, LUTROM, time-mux with retiming)
75. Show how ARRAY_PARTITION (cyclic vs. complete) and ARRAY_RESHAPE change the effective memory ports for an 8×8 block and when each is preferable. (aim: port creation vs. register fanout)

### Codec-Specific Blocks

76. Sketch an HLS micro-architecture for half-pel 8-tap interpolation (H.264/HEVC) using separable filters; how do you schedule horizontal vs. vertical passes? (aim: reuse of partial results)
77. For intra prediction (planar/angular), what neighborhood pixels do you cache per PU, and how do you update the cache as PUs scan across a CTU? (aim: neighbor reuse and update)
78. In a deblocking filter, how do you stream edge pixels so that vertical and horizontal filtering both achieve II=1 without rereading CTU data? (aim: line/edge buffers)
79. Provide a fixed-point bit-width plan for 8×8 DCT→quant (10-bit input): intermediate bit growth, rounding, and final clipping to codec spec. (aim: guard bits and quantization)
80. For CDEF/SAO, how do you select window size and FIFO depths to maintain continuous streaming across tile boundaries? (aim: boundary/padding policy)
81. Outline a simple VC-1/HEVC-style inverse transform HLS schedule that avoids transposition RAM by streaming rows/cols cleverly. (aim: transpose-free pipelines)
82. For motion compensation of quarter-pel positions, compare two interpolation scheduling strategies and their memory footprints. (aim: compute vs. storage trade)

### Dataflow \& Deadlock Avoidance

83. Given three dataflow stages A→B→C with variable latency in B, determine FIFO depths to avoid bubbles at 300 MHz assuming A/C are II=1; show a sizing method. (aim: throughput matching)
84. Spot the bug: two dataflow processes both perform blocking reads on each other's streams in opposite order. What happens and how to fix? (aim: circular wait)
85. Show how to split a monolithic kernel into 4 dataflow tasks (ME→MC→DCT→Quant) and place FIFOs to isolate variable-latency stages. (aim: decoupling and backpressure)
86. A stream occasionally underflows at frame start. List initialization/reset sequencing mistakes that could cause this and your fixes. (aim: reset protocol)
87. How do you handle multi-rate Y:UV (2:1) in a single downstream consumer expecting interleaved YUV? Provide a scheduling or time-division approach. (aim: rate matching)

### Loop-Dependency \& Directive Side-Effects

88. Given code with a dependence distance of 2, explain why II=1 is still achievable and how the tool enforces it. (aim: modulo scheduling concept)
89. What adverse effects can UNROLL factor=8 on a memory-bound loop have, and how would you detect them from reports? (aim: port/bandwidth explosion)
90. Explain how aggressive INLINE might cause unintended resource duplication and how to bound it. (aim: ALLOCATION/instance limits)
91. Describe a safe refactor to convert a loop with if branches on pixel class into a predicated, pipeline-friendly form. (aim: control→data transformation)
92. Why can complete partition on a large array blow up LUT usage, and what alternative would you use? (aim: cyclic partition or reshape)

### Burst I/O \& AXI Integration

93. Write the high-level steps to implement strided 2D frame reads via AXI4 master in Vitis HLS with burst coalescing. (aim: addr generator, pragmas)
94. For AXI4-Stream input, how do you preserve frame metadata (resolution, tile IDs) without stalling pixel throughput? (aim: sideband TUSER scheme)
95. What are the failure modes if downstream deasserts TREADY for long periods, and how do you dimension buffers to guarantee no data loss? (aim: backpressure budgeting)
96. Describe an approach to pack two 10-bit pixels into a 20-bit stream while maintaining byte alignment on AXI; what packing/unpacking modules are needed? (aim: bit-packing and strobe)

### Numerics \& Fixed-Point

97. Compare round-to-nearest-even vs. truncation for quantization noise in transform coefficients; when would codec conformance force one choice? (aim: rounding policy)
98. Show how you would validate no overflow in a cascade of MACs: analytical bound vs. simulation with worst-case vectors. (aim: guard-bit proof)
99. In Catapult ac_fixed, how do you select quantization/overflow modes to match a bit-true C golden model? (aim: mode settings)

### Verification \& Signoff

100. Your RTL cosim mismatches begin at the first CTU boundary. List three typical off-by-one/flush bugs that cause edge artifacts. (aim: boundary handling)
101. Propose a unit test set for MC that catches fractional interpolation errors and edge padding mistakes. (aim: directed vectors)
102. What coverage targets would you set at C-sim and RTL for a deblock filter, and how would you measure them? (aim: functional coverage)
103. How would you instrument the HLS RTL to collect per-stage cycle counts and FIFO occupancies in gate-level sim without impacting timing? (aim: lightweight counters)
104. Explain a regression strategy that checks bit-true equivalence vs. a reference codec across profiles and bit depths while keeping sim time reasonable. (aim: sampling, checksum)

### Tool-Specific (Vitis/Catapult)

105. In Vitis, contrast ap_ctrl_none vs. ap_ctrl_hs for a streaming kernel; when is each appropriate in a video pipe? (aim: control protocol)
106. Show how to express the same streaming pipeline using Vitis hls::stream vs. Catapult ac_channel/sc_fifo, noting any synthesis-relevant differences. (aim: portability)
107. Catapult schedules a slower clock than Vitis for the same C++; what TCL/directive categories would you inspect first? (aim: schedule/binding/memory map)
108. How would you replace ap_uint/ap_fixed with portable bit-accurate types to keep code dual-tool compatible? (aim: typedef abstraction)
109. In Vitis, m_axi port reports "no burst coalescing". What code patterns typically block burst formation, and how do you refactor? (aim: pointer arithmetic, access stride)
110. Catapult infers single-port SRAM for a dual-read use; which pragma/memory map change enables true dual-port, and what ASIC memory constraints apply? (aim: memory binding)

### System Integration

111. Given a 300 MHz fabric clock and a 600 MHz CABAC sub-block, how do you safely interface them and manage backpressure? (aim: CDC + elastic buffers)
112. Propose a tiling/WPP schedule for 4K that maximizes concurrency without violating top-left dependency rules; how many CTU rows are in flight? (aim: dependency-aware parallelism)
113. For a mixed FPGA/ASIC prototyping setup, what differences in RAM/DSP resources would change your HLS directives between targets? (aim: portability constraints)
114. Outline a bring-up checklist to integrate four HLS IPs on an AXI interconnect with shared DDR: arbitration, QoS, and starvation testing. (aim: system validation)
115. Describe how you would add clock gating/enable conditions at HLS level to reduce dynamic power during blanking intervals without breaking timing. (aim: power-aware coding)

These additional questions expand the **Intermediate (Medium)** section with 55 highly technical questions that cover advanced HLS optimization techniques, dataflow design patterns, tool-specific implementation details, and system-level integration challenges specific to video compression IP development. The questions are designed to test deeper understanding of HLS methodology and practical problem-solving skills required for high-performance video codec implementation.




# Updated Interview Questions List - Additional Expert Questions

Here are the additional 50 questions (numbered 116-165) for the **Expert (Very Difficult)** section:

### HLS Architecture \& Pipeline Balancing

116. For a full encoder pipe (Intra/Inter → Transform → Quant → Entropy), propose a stage-by-stage II/latency budget at 4K60 and show how you would re-balance stages to remove a single slow stage bottleneck without increasing total area. (aim: pipeline balancing under fixed throughput)
117. Given three stages with II={1,1,4} and deep FIFOs between them, describe two HLS refactors to bring the overall pipeline to II=1 while capping extra DSPs to +25%. (aim: parallel duplication vs. partial unroll with selective resource binding)
118. Show how to convert a control-heavy RDO decision tree into a dataflow-friendly structure using predication and speculative execution in HLS, and explain how you will verify no QoR loss. (aim: control→data transformation at scale)
119. Design an elastic pipeline wrapper around an HLS kernel to tolerate ±10% burstiness at the input AXI4-Stream while guaranteeing no underflow at the output; derive minimal FIFO depths. (aim: jitter absorption)
120. For a multi-pass algorithm (e.g., lookahead + main encode), detail how you would compose two separately synthesized HLS IPs into a super-pipeline with protocol conversion while retaining end-to-end backpressure. (aim: cross-kernel composition)
121. Present a strategy to allow dynamic kernel clock overdrive (CABAC at 600 MHz vs. rest at 300 MHz) with CDC-safe pause/resume semantics in HLS. (aim: multi-clock elastic interface)

### Hierarchical Memory \& Bandwidth

122. Compute worst-case external bandwidth for 8K×4320 10-bit 4:2:0 at 60 fps for ME with ±64 search window if naïvely fetching reference every SAD tap; then outline a 3-level on-chip cache (CTU tile → line → window) to reduce DRAM traffic by >10×. (aim: bandwidth modeling + locality)
123. Propose a banked memory map for luma/chroma that enables four concurrent reference block reads and one writeback at 512-bit AXI; include bank conflict analysis. (aim: banked parallelism)
124. Show how to restructure address generation to transform strided chroma accesses into burst-coalesced AXI reads with minimal padding and without changing codec semantics. (aim: access linearization)
125. Given a single-ported SRAM inferred for a coefficient LUT read twice per cycle, compare: (a) ROM duplication, (b) time-multiplexing at 2× clock, (c) banking with interleaving; pick one under a 300 MHz timing limit and justify. (aim: port arbitration vs. timing)
126. Design a prefetcher in HLS that tracks motion vector statistics to preload likely reference blocks ahead of time; define its finite-state machine and miss-handling policy. (aim: predictive prefetch)
127. For HBM-based FPGA, partition the frame store across channels; specify channel assignment and outstanding transaction depths needed to sustain 8K60 worst-case ME traffic. (aim: HBM utilization planning)

### Motion Estimation \& Compensation

128. Architect a full ME engine (diamond search + hierarchical pyramid) in HLS to meet II=1 at the SAD MAC array; define PE array size, reuse buffers, and search schedule across pyramid levels. (aim: scalable ME micro-architecture)
129. Given quarter-pel interpolation using separable 8-tap filters, derive the minimal on-chip storage to reuse horizontal partials across vertical taps at 4K CTU scan. (aim: partial result reuse)
130. Present two designs for sub-pel interpolation: (A) compute-on-demand per candidate, (B) precompute a sub-pel grid cache; quantify compute vs. memory trade-offs for 8K60. (aim: compute/memory balance)
131. Spot the bug: an HLS MC block sometimes reads outside frame bounds at tile edges when TREADY deasserts downstream; identify the likely handshake/latency mismatch and propose a robust border-extend scheme. (aim: edge + backpressure interaction)
132. Show how to pipeline cost aggregation for multiple candidate MVs while allowing early termination (SATD threshold) without breaking II=1. (aim: conditional early-out in pipeline)
133. Define a deterministic arbitration policy to merge MV candidates from WPP neighbor dependencies in hardware without deadlock or starvation under HLS dataflow. (aim: dependency-safe merge)

### Transforms, Quant, Precision

134. For a 32×32 integer transform (VVC), derive internal bit growth and choose ap_fixed/ac_fixed widths that provably avoid overflow with round-to-nearest-even; provide a short proof. (aim: numeric proof)
135. Implement a transpose-free 2D transform pipeline in HLS; show how you will schedule row/column passes and align memory ports to avoid bank conflicts. (aim: transpose elimination)
136. Compare two quantization architectures: (i) per-coefficient hardware divider, (ii) multiply-by-reciprocal with shift; bind operations to DSPs and estimate latency/area at 300 MHz. (aim: division removal)
137. Propose a mixed-precision pipeline (e.g., 12-bit internal for early stages, 16-bit for accumulators) and define formal assertions that ensure output matches 10-bit spec after clipping for all legal inputs. (aim: precision staging + formal guard)
138. Spot the bug: RTL matches C model for most blocks, but DC coefficient is off by ±1 sporadically; list HLS patterns that cause tie-breaking differences and how to force stable rounding. (aim: rounding determinism)

### Entropy Coding \& Bitstream

139. Sketch a dual-lane CABAC micro-architecture to process two bins per cycle using lookahead; specify context memory partitioning and renormalization sharing. (aim: parallel entropy)
140. Provide an HLS-friendly design for rANS encoder with byte-aligned output; detail state updates, table layout, and how you handle under/overflow without stalling the stream. (aim: rANS hardwareization)
141. Describe how to bridge variable-rate entropy output to a fixed-rate AXI master without data loss; include credit-based flow control and watermark IRQs for software RC. (aim: rate matching)
142. Derive the maximum FIFO depth needed to absorb worst-case VLC bursts from a CTU while downstream DDR stalls for T cycles; present the formula and example values. (aim: burst buffering math)
143. Explain how to maintain deterministic bin ordering and context updates across WPP tiles when tiles are processed out-of-order in hardware for throughput. (aim: ordering correctness)

### In-Loop Filters \& Quality

144. For deblock + SAO + CDEF cascade, propose a fusion strategy that eliminates intermediate writes to external memory; show buffer ownership and lifetime across stages. (aim: in-loop fusion)
145. The SAO classifier uses histograms per CTU; design a streaming histogrammer in HLS that avoids read-modify-write conflicts at II=1. (aim: conflict-free counters)
146. Provide a design to switch filter strengths dynamically per edge class without stalls, with parameters coming over AXI4-Lite mid-frame. (aim: runtime reconfiguration)
147. Devise a boundary padding scheme that supports tiles and WPP simultaneously and prove that it cannot deadlock under TREADY backpressure. (aim: padding + flow control proof)

### System Integration: Tiles, WPP, CDC

148. Define a WPP scheduler that maximizes CTU row concurrency under top-left dependency; compute maximum in-flight CTUs and FIFO depth per row at 4K. (aim: dependency-aware scheduling)
149. For a tile-based pipeline with N parallel tile engines, design an interconnect for shared entropy engine access; analyze fairness and head-of-line blocking. (aim: shared resource arbitration)
150. Create a CDC plan for three clocks (video pixel, fabric, entropy fast) using async FIFOs; specify metastability protections and reset sequencing. (aim: multi-clock correctness)
151. Propose a synchronization mechanism for reference frame versioning so that ME/MC never reads a partially updated frame when encoder reuses buffers. (aim: memory coherency)
152. Architect a DMA descriptor ring for CTU-based transfers with reorder capability; explain how you'll maintain bitstream order at the output. (aim: DMA reordering)

### Advanced Verification, Formal \& Signoff

153. Outline a sequential equivalence checking (SEC) plan between C model and HLS RTL that tolerates different latencies but proves bit-true outputs; list key cut-points. (aim: SEC methodology)
154. Define property checks to guarantee no FIFO overflow/underflow across the entire dataflow graph in presence of arbitrary TREADY stalls. (aim: liveness/safety properties)
155. Build a reference differential test for MC that isolates sub-pel errors by injecting impulses at known positions; describe pass/fail metrics. (aim: targeted differential tests)
156. Propose a scalable regression matrix (content, QP, motion, tile/WPP settings) that reaches high functional coverage within a nightly time budget; include prioritization heuristics. (aim: pragmatic coverage)
157. Describe how you would capture cycle-accurate traces from HLS RTL and correlate with C-level transaction logs to localize a 1-in-10k-frame artifact. (aim: cross-domain debug)

### Tool-Specific (Vitis vs. Catapult) \& QoR Divergence

158. Vitis meets 350 MHz while Catapult meets 280 MHz for the same kernel; design a systematic experiment to isolate differences in (a) memory binding, (b) operator binding, (c) loop pipelining, and propose directive sets to align them. (aim: cross-tool forensics)
159. In Vitis, BIND_STORAGE/RESOURCE created distributed RAM causing LUT blow-up; propose a directive+code change to force BRAM/URAM with preserved II=1. (aim: storage binding control)
160. Catapult inferred a single-ported SRAM for a dual-read MC buffer; specify the TCL/directives to bind to a dual-port macro and adjust schedule to avoid multi-cycle stalls. (aim: memory macro binding)
161. Port a kernel using hls::stream<ap_uint<128>> to Catapult ac_channel<ac_int<128,false>>; list all code changes, including reset/empty semantics and blocking behavior. (aim: portability drill)

### Timing Closure, Retiming \& Physical-Aware HLS

162. Given a critical path through a 16-input adder tree in the DCT, propose a retimed pipeline using carry-save or balanced trees; show how you'll express it in HLS so the tool won't collapse stages. (aim: stable retiming)
163. Floorplanning indicates long routes between two RAMs and a DSP cluster; propose an HLS refactor (e.g., compute duplication or tiling) that reduces wirelength while preserving function. (aim: physical-aware restructuring)
164. The tool reports II=1 but post-route Fmax is 20% short; provide a checklist of HLS-level mitigations (operator balance, latency pragmas, breaking fanout, register slicing) and how you'll validate improvements early. (aim: timing convergence plan)
165. Show how to introduce clean clock gating enables in HLS (without gated clocks) around idle regions (blanking intervals, no-motion CTUs) and prove no functional change using assertions. (aim: power-aware timing-safe gating)

These additional questions expand the **Expert (Very Difficult)** section with 50 extremely challenging questions that test mastery of advanced HLS concepts, system-level architecture design, formal verification methodologies, and cutting-edge optimization techniques. The questions require deep understanding of video codec implementation, cross-tool expertise, and the ability to solve complex engineering trade-offs at the intersection of algorithms, hardware architecture, and physical implementation constraints.

**Final Summary:**

- **Foundational (Simple):** 50 questions (1-50)
- **Intermediate (Medium):** 115 questions (1-115)
- **Expert (Very Difficult):** 165 questions (1-165)

This comprehensive set provides a rigorous assessment framework for evaluating candidates across the full spectrum of HLS expertise required for high-performance video compression IP development at Google.




<div style="text-align: center">⁂</div>

[^1]: https://semiconductorclub.com/synthesis-interview-questions-synthesis-faqs/

[^2]: https://www.xilinx.com/support/documents/sw_manuals/xilinx2022_2/ug1399-vitis-hls.pdf

[^3]: https://www.indeed.com/cmp/Catapult-Learning/interviews

[^4]: https://runtimerec.com/high-level-synthesis/

[^5]: https://imperix.com/doc/help/xilinx-vitis-hls

[^6]: https://www.indeed.com/cmp/Katapult-Network/interviews

[^7]: https://semiconductorclub.com/what-is-high-level-synthesis/

[^8]: https://www.youtube.com/watch?v=iAGmzTY3toM

[^9]: https://hls.harvard.edu/jdadmissions/apply-to-harvard-law-school/jdapplicants/application-toolkit/jdadmissions-interview/our-favorite-interview-questions/

[^10]: https://www.vlsi-expert.com/2021/05/high-level-synthesis-intro.html

[^11]: https://www.youtube.com/watch?v=RLCpw7RyhZM

[^12]: https://www.reddit.com/r/lawschooladmissions/comments/1958ae7/interview_prep_advice_hls_but_also_general/

[^13]: https://www.esp.cs.columbia.edu/docs/mentor_cpp_acc/mentor_cpp_acc-guide/

[^14]: https://www.youtube.com/watch?v=v6qvrIY5Tgs

[^15]: https://research.sabanciuniv.edu/34738/1/FirasAbdulGhani_10162831.pdf

[^16]: https://discussion.mcebuddy2x.com/t/mp4-hevc-vs-av1-with-hardware-encoding-on-off/5313

[^17]: https://hellointern.in/blog/video-compression-interview-questions-and-answers-24232

[^18]: https://trepo.tuni.fi/bitstream/10024/161487/2/SmedbergJesse.pdf

[^19]: https://www.youtube.com/watch?v=elZH8iXGTPk

[^20]: https://www.designgurus.io/blog/netflix-system-design-interview-questions-guide

[^21]: https://www.ewadirect.com/proceedings/ace/article/view/10781

[^22]: https://www.youtube.com/watch?v=TBdRHPhZ6bc

[^23]: https://bytebytego.com/courses/system-design-interview/design-youtube

[^24]: https://news.ycombinator.com/item?id=16796127

[^25]: https://www.youtube.com/watch?v=Zo5SvR3JPYA

[^26]: https://xilinx.github.io/Vitis_Accel_Examples/2019.2/html/array_partition.html

[^27]: https://soldierchen.github.io/assets/pdf/trets22.pdf

[^28]: https://www.ijeat.org/wp-content/uploads/papers/v8i4/D6051048419.pdf

[^29]: https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-array_partition

[^30]: https://videosdk.live/developer-hub/hls/hls-video-player

[^31]: https://dl.acm.org/doi/fullHtml/10.1145/3491215

[^32]: https://users.ece.utexas.edu/~gerstl/ee382v_f14/soc/vivado_hls/VivadoHLS_Improving_Performance.pdf

[^33]: https://docs.amd.com/r/en-US/ug1399-vitis-hls/Optimizing-Techniques-and-Troubleshooting-Tips?contentId=hOvtKfFtqmwkePs8Paxf0Q

[^34]: https://core.ac.uk/download/pdf/196558078.pdf

[^35]: https://docs.amd.com/r/en-US/ug1399-vitis-hls/pragma-HLS-dataflow

[^36]: https://stackoverflow.com/questions/46027238/latency-and-initiation-interval-in-hls

[^37]: http://m.akhomesold.com/html_docs/xilinx2021_1/vitis_doc/cosimulationinvitishls.html

[^38]: https://docs.amd.com/r/en-US/ug1399-vitis-hls/Configuring-Dataflow-Memory-Channels

[^39]: https://hls.academy/topics/pipelining/

[^40]: https://docs.amd.com/r/en-US/ug1399-vitis-hls/Troubleshooting-C/RTL-Co-Simulation

[^41]: https://docs.amd.com/r/en-US/wp554-high-performance-design/FIFO-and-hls-stream

[^42]: https://www.cs.cornell.edu/courses/cs6120/2020fa/blog/pipeline-ii-analysis/

[^43]: https://www.reddit.com/r/FPGA/comments/16hr9be/vitis_hls_cosimulation/

[^44]: https://eda.sw.siemens.com/en-US/ic/catapult-high-level-synthesis/

[^45]: https://microchiptech.github.io/fpga-hls-docs/2023.1/optimizationguide.html

[^46]: https://stefanabikaram.com/writing/vitis-hls-debug-pretty-print/

[^47]: https://www.reddit.com/r/FPGA/comments/1525xkp/anyone_using_hls_professionally/


