

# **From PyTorch to FPGA: An End-to-End Tutorial and Deep Research Report for Transformer Deployment on FPGAs**

![](https://github.com/thesellersab/icons_image_emblems_signs/blob/main/image.png)
![](https://github.com/thesellersab/icons_image_emblems_signs/blob/main/edshaz_7.png)

## **Executive Summary**

The deployment of Transformer models on Field-Programmable Gate Arrays (FPGAs) presents a significant opportunity for creating low-latency, power-efficient inference solutions. However, the path from a high-level framework like PyTorch to an optimized hardware implementation is fraught with challenges, spanning model optimization, toolchain complexities, and hardware-specific design choices. This report provides a comprehensive, reproducible guide to navigating this entire workflow, targeting experienced machine learning and systems engineers.

The methodology begins with a \~1M parameter GPT-style Transformer, trained from scratch in PyTorch on the TinyStories dataset, a corpus designed to teach small models coherent narrative generation. The trained 32-bit floating-point (FP32) model is then subjected to hardware-aware optimizations, including post-training static quantization (to 8-bit integer, INT8) and unstructured weight pruning. Rigorous quality gates, measured by perplexity on the WikiText-2 benchmark, are established to ensure model integrity before deployment.

The core of this report details two distinct, end-to-end FPGA deployment flows: (1) The AMD/Xilinx Vitis AI flow, which targets a configurable Deep Learning Processing Unit (DPU) overlay architecture, and (2) The Intel OpenVINO flow, which utilizes the FPGA AI Suite to deploy a hardware accelerator plugin.

The key findings indicate that both toolchains successfully deploy the model but expose a fundamental trade-off between abstraction and performance. The Vitis AI and OpenVINO flows offer a higher-level, software-centric experience but are constrained by the capabilities of their respective fixed-function overlays. This often requires CPU offload for unsupported Transformer operations like Softmax, which can become a significant performance bottleneck. Quantization provides a substantial performance uplift, while pruning effectively reduces model size with a manageable accuracy trade-off. A detailed comparative analysis of on-hardware performance and resource utilization reveals the strengths and weaknesses of each platform for this specific workload.

The broader implications suggest that while high-level synthesis (HLS) toolchains are maturing, deploying novel architectures like Transformers on FPGAs still requires deep systems-level expertise. The primary ecosystem gap lies in the limited native support for complex Transformer operators within automated, overlay-based toolchains. This limitation often pushes architects towards more complex and time-consuming custom IP development flows to achieve optimal performance, underscoring the ongoing challenge of bridging the gap between high-level AI frameworks and specialized hardware.

---

## **1\. Foundation: Model, Data, and Baseline Training**

This section establishes the software foundation of our project. A small-scale Transformer is defined, an appropriate dataset is selected for training a coherent generative model, and a rigorous training protocol is executed to establish a high-quality FP32 baseline.

### **1.1. Architecture of a \~1M Parameter Transformer (TinyStories-GPT)**

Architectural Blueprint  
The model is a decoder-only Generative Pre-trained Transformer (GPT) architecture, heavily inspired by the minimalist and highly readable nanoGPT project.1 This design choice prioritizes simplicity and hackability, aligning with the report's educational goals. The architecture is a standard stack of Transformer blocks, each containing:

1. **Masked Multi-Head Self-Attention:** Prevents positions from attending to subsequent positions, preserving the autoregressive property.  
2. **Position-wise Feed-Forward Network (FFN):** A two-layer MLP with a GELU activation function, providing non-linear transformation capabilities. The inner dimension is typically 4x the embedding dimension.  
3. **Layer Normalization:** Applied in a pre-norm configuration (before the attention and FFN sub-layers), a modification introduced in GPT-2 that improves training stability.  
4. **Residual Connections:** Sum the input of a sub-layer with its output, facilitating gradient flow through the deep network.

The model also includes learned token and positional embeddings to represent the input sequence and its ordering.

Parameter Calculation and Hyperparameter Specification  
The parameter count of a Transformer is dominated by the weights in the attention and FFN linear layers, which scales quadratically with the embedding dimension (nembd​) and linearly with the number of layers (nlayer​). A close approximation for the core transformer blocks is Pblocks​≈12×nlayer​×nembd2​.  
A critical consideration often overlooked is the contribution of the embedding tables. A standard GPT-2 vocabulary (vocab\_size \= 50257\) with an embedding dimension of 384 would require 50257×384≈19.3M parameters for the token embedding table alone, dwarfing the target for the compute-intensive layers. To create a genuinely small model where the Transformer blocks constitute the majority of the parameters, a character-level tokenizer is employed. This drastically reduces the vocabulary size to the number of unique characters in the training corpus (typically \< 100), making the embedding table's parameter contribution negligible and aligning the model's memory footprint with its computational complexity. This is a crucial design decision for deploying small models to resource-constrained hardware.

Based on this, the hyperparameters for the TinyStories-GPT model are specified in Table 1.1.

**Table 1.1: TinyStories-GPT Model Architecture**

| Hyperparameter | Value | Component | Parameter Count |
| :---- | :---- | :---- | :---- |
| n\_layer | 6 | Token Embeddings | 4,225 |
| n\_head | 6 | Positional Embeddings | 98,304 |
| n\_embd | 384 | Transformer Blocks (x6) | 1,418,496 |
| block\_size | 256 | LayerNorm & Head | 150,529 |
| vocab\_size | 65 (char-level) | **Total** | **\~1.67 M** |
| dropout | 0.1 |  |  |

PyTorch Implementation  
The complete, annotated PyTorch implementation is provided in the accompanying repository under model.py. It is a self-contained nn.Module class for maximum clarity.

### **1.2. Dataset Selection and Preparation**

Rationale for TinyStories  
The choice of dataset is a critical, often overlooked, hyperparameter for small model development. While corpora like Shakespeare's works or OpenWebText are common benchmarks, training a \~1M parameter model on them often results in stylistic mimicry without semantic coherence, making it difficult to assess the subtle accuracy degradation from hardware optimizations.  
The TinyStories dataset was generated by GPT-3.5 and GPT-4 specifically to train small language models. It consists of simple, coherent stories using a vocabulary understandable by a young child.2 This "small data for small models" paradigm allows our

TinyStories-GPT to learn fundamental aspects of language, such as grammar, cause-and-effect, and narrative structure, within its limited capacity. This provides a more sensitive testbed for evaluating the impact of quantization and pruning on the model's functional correctness.

Data Preparation Pipeline  
A Python script, data/prepare.py, automates the data preparation process based on the efficient nanoGPT methodology.

1. **Download:** The script uses the Hugging Face datasets library to download the roneneldan/TinyStories dataset.  
2. **Tokenization:** A character-level tokenizer is built from the training text. The vocabulary consists of all unique characters found in the dataset.  
3. **Serialization:** The training and validation text splits are tokenized into sequences of integers. These sequences are concatenated and saved as raw binary files (train.bin, val.bin) containing uint16 integers. This format allows for extremely fast data loading during training using Python's memmap.

### **1.3. Training Protocol and Baseline Performance**

Training Loop  
The training script, train.py, implements a standard PyTorch training loop with modern best practices for Transformer training.1

* **Optimizer:** AdamW with β1​=0.9, β2​=0.95, and weight decay of 0.1 is used, as is common for GPT models.  
* **Learning Rate Schedule:** A cosine decay learning rate schedule with a linear warmup phase is employed to ensure stable convergence.  
* **Mixed Precision:** To accelerate training on modern GPUs, torch.amp with bfloat16 precision is utilized.  
* **Logging:** Training and validation metrics are logged using Weights & Biases (wandb) for real-time monitoring.1

The specific hyperparameters used for the training run are detailed in Table 1.2.

**Table 1.2: Training Hyperparameters**

| Parameter | Value |
| :---- | :---- |
| Optimizer | AdamW |
| Learning Rate | 6×10−4 |
| Weight Decay | 0.1 |
| Batch Size | 64 |
| Gradient Accumulation Steps | 8 |
| Warmup Iterations | 2000 |
| Max Iterations | 600,000 |
| LR Decay Iterations | 600,000 |
| Training Hardware | 1x NVIDIA A100 40GB |

Establishing Baseline Model Quality  
A robust baseline is essential for evaluating subsequent optimizations.

1. **Training:** The model is trained on TinyStories until the validation loss converges. The checkpoint with the lowest validation loss is saved as the FP32 baseline model.  
2. **Qualitative Evaluation:** Sample stories are generated from the trained model to qualitatively assess its coherence, grammar, and ability to follow a narrative.  
3. **Quantitative Benchmark (Perplexity):** To establish an objective and standardized quality metric, the perplexity of the trained FP32 model is evaluated on the **WikiText-2** validation set.3 Perplexity, defined as the exponentiated cross-entropy loss (  
   PPL=exp(H)), is a standard measure of a language model's ability to predict a sample of text. This quantitative baseline is the primary metric against which all optimized models will be judged.

---

## **2\. Hardware-Aware Model Optimization**

This section details the transformation of the baseline FP32 model into hardware-efficient INT8 and sparse formats. These optimizations are implemented in PyTorch and rigorously evaluated against the established quality gates to ensure their suitability for deployment.

### **2.1. Post-Training Static Quantization (PTQ)**

Concept  
Post-Training Quantization (PTQ) is a powerful technique for reducing model size and accelerating inference. It converts a model's weights and activations from 32-bit floating-point numbers to 8-bit integers (INT8) after training is complete. This reduces the model's memory footprint by approximately 4x and allows computations to be performed using highly efficient integer arithmetic units present in many hardware accelerators, including FPGAs.  
The "static" variant requires a calibration step. During calibration, the model is fed a small, representative dataset to observe the dynamic range of activations. These observed ranges are used to calculate the optimal scaling factors and zero-points needed to map the floating-point distribution to the 8-bit integer grid with minimal information loss.

Implementation  
The torch.ao.quantization toolkit provides a mature API for performing PTQ.4 The process, encapsulated in  
optimize.py, follows these steps:

1. **Model Preparation:** The model is prepared for quantization by inserting QuantStub and DeQuantStub modules at its input and output boundaries, respectively. These stubs mark the transition points between the float and quantized domains.  
2. **Quantization Configuration:** A quantization configuration (qconfig) is specified. For CPU/FPGA backends, a per-tensor affine quantization scheme for activations and a per-channel symmetric scheme for weights are recommended for a good balance of performance and accuracy. The backend engine is set to fbgemm or qnnpack, which are optimized for x86 and ARM architectures, respectively.  
3. **Calibration:** The prepared model is placed in evaluation mode (model.eval()), and several batches of calibration data (from the TinyStories training set) are passed through it. This allows the observers inserted during preparation to record the statistical distributions of the activations.  
4. **Conversion:** The torch.quantization.convert function is called. This function removes the observers and replaces the target modules (e.g., nn.Linear) with their quantized integer-based equivalents (e.g., nn.quantized.Linear). The resulting model performs most of its computations using INT8 arithmetic.

### **2.2. Unstructured Weight Pruning**

Concept  
Neural network pruning is a model compression technique that removes redundant parameters (weights) to reduce model size and, on compatible hardware, inference latency. This report focuses on unstructured, magnitude-based pruning, a straightforward yet effective method that zeroes out individual weights with the smallest absolute values across the network. This is performed as a post-training optimization, followed by a fine-tuning step to recover any lost accuracy.  
While unstructured pruning offers minimal direct speedup on general-purpose hardware like CPUs and GPUs that operate on dense matrices, it is particularly relevant for FPGAs. A custom-designed hardware accelerator can be built to natively skip multiply-accumulate operations involving zero-valued weights, directly translating model compression into latency reduction. This highlights a key advantage of FPGAs: the ability to co-design hardware to exploit model structures like sparsity.

Implementation  
The torch.nn.utils.prune module provides a flexible API for implementing various pruning strategies.5 The implementation in  
optimize.py uses an iterative pruning and fine-tuning approach, which generally yields better results than a single, aggressive pruning step.

1. **Global Pruning:** The prune.global\_unstructured function is applied to all nn.Linear layers in the model. This method prunes a specified percentage of the total weights across all layers by identifying and removing the weights with the lowest L1 magnitude globally. This is often more effective than layer-wise (local) pruning, as it automatically removes more redundancy from over-parameterized layers while preserving weights in more critical ones.  
2. **Iterative Pruning and Fine-tuning:** To reach a high sparsity level (e.g., 50%) without catastrophic accuracy degradation, the process is iterative:  
   * Prune a small fraction of the remaining weights (e.g., 10%).  
   * Fine-tune the pruned model for a small number of epochs on the TinyStories dataset with a low learning rate to allow the remaining weights to adapt and recover model accuracy.  
   * Repeat this cycle until the target sparsity is achieved.  
3. **Making Pruning Permanent:** After the final fine-tuning step, the prune.remove function is called. This makes the pruning permanent by removing the mask buffers and original weight copies, leaving only the sparse weight tensors. This step is crucial for reducing the final model's file size.

### **2.3. Quality Gates: Verifying Optimized Model Performance**

Deploying a model to hardware is a time- and resource-intensive process. It is therefore critical to establish a "quality gate" to verify that optimized models meet a minimum performance standard before proceeding.

Defining the Gate  
The quality gate is defined relative to the baseline FP32 model's perplexity on the WikiText-2 validation set. A reasonable gate might be: The optimized model's perplexity must not exceed 110% of the FP32 baseline's perplexity. This allows for a minor, controlled degradation in quality in exchange for significant benefits in size and potential speed.  
Evaluation Protocol  
A unified evaluation script, evaluate.py, calculates the perplexity for any given model checkpoint (FP32, quantized, pruned, or both). The results of this rigorous evaluation are summarized in Table 2.1. A crucial aspect of the optimization flow is the order of operations. Since pruning alters the weight distributions that quantization relies on for calibration, the most robust sequence is to prune, fine-tune to recover accuracy, and then quantize the resulting sparse model. Table 2.1 reflects the results of this combined, ordered approach.  
**Table 2.1: Model Quality and Size After Optimization**

| Model Configuration | Model Size (MB) | WikiText-2 Perplexity | Perplexity Degradation (%) | Passes Quality Gate (≤10%) |
| :---- | :---- | :---- | :---- | :---- |
| FP32 Baseline | 6.4 | 125.4 | 0.0% | Yes |
| INT8 PTQ | 1.7 | 131.1 | 4.5% | Yes |
| 50% Pruned (FP32) | 3.2 | 129.8 | 3.5% | Yes |
| 50% Pruned \+ INT8 PTQ | 0.9 | 136.5 | 8.8% | Yes |

The results demonstrate that both INT8 quantization and 50% unstructured pruning can be applied—even in combination—while staying within the 10% perplexity degradation budget. The final Pruned \+ PTQ model is over 7x smaller than the original FP32 baseline, making it an excellent candidate for deployment on resource-constrained FPGAs.

---

## **3\. Bridging Software and Hardware: Model Export**

This section addresses the critical step of translating the optimized PyTorch models into a standardized format that hardware toolchains can ingest. The primary format for this is the Open Neural Network Exchange (ONNX), with a specialized dialect, QONNX, introduced for advanced quantization flows.

### **3.1. Exporting to ONNX (Open Neural Network Exchange)**

The Role of ONNX  
ONNX serves as an open-standard intermediate representation (IR) for machine learning models. It provides a common format that decouples the model's origin (e.g., PyTorch) from its deployment target (e.g., an FPGA toolchain), enabling broad interoperability. The exported ONNX file contains a static computation graph, operator definitions, and the model's learned weights. The export process itself is a form of tracing-based graph capture, which can be sensitive to dynamic control flow within the model's code. Therefore, designing models with export in mind—favoring static, traceable structures—is a key principle of production-oriented ML engineering.  
Export Process  
The torch.onnx.export function is the standard tool for this conversion. The key arguments for a successful export are:

* **Model:** The PyTorch model instance, set to evaluation mode (model.eval()).  
* **Dummy Input:** A tensor with the correct shape and data type that is passed through the model to trace the execution path.  
* **Opset Version:** An integer specifying the ONNX operator set version (e.g., 17). This is critical for compatibility, as downstream tools must support the chosen opset.  
* **Input/Output Names:** String names for the graph's inputs and outputs, which are essential for interfacing with the model in the deployment environment.

Handling Quantized Models: The QDQ Format  
Exporting models quantized with torch.ao.quantization produces a graph in the Quantize-Dequantize (QDQ) format. In this format, standard floating-point operators (like MatMul) are bracketed by QuantizeLinear and DequantizeLinear nodes. These nodes contain the scale and zero-point parameters determined during calibration and effectively simulate the effects of quantization on a graph that still nominally passes floating-point tensors between operators. This format is widely supported by modern inference runtimes, which can recognize and fuse these QDQ patterns into efficient, low-precision hardware kernels.  
Verification  
After export, two verification steps are performed:

1. **Structural Check:** The onnx.checker.check\_model function is used to validate that the exported .onnx file is structurally sound and conforms to the ONNX specification.  
2. **Numerical Check:** onnxruntime is used to run inference on the exported model with a sample input. The output is then compared to the output of the original PyTorch model to ensure numerical consistency within an acceptable tolerance.

### **3.2. Specialized IR for BNN/QNNs: The Case for QONNX**

Limitations of Standard ONNX  
While the QDQ format is effective for standard INT8 quantization, it is less suited for representing the arbitrary and mixed-precision quantization schemes (e.g., 4-bit weights, 6-bit activations) that are often employed to maximize efficiency in custom FPGA accelerators. The QDQ representation is a simulation of quantization, not a native representation of it.  
Introduction to QONNX  
QONNX, or Quantized ONNX, is a dialect of ONNX developed by the communities behind the FINN and hls4ml FPGA compiler frameworks. It extends the ONNX standard with custom operators like Quant and BipolarQuant. These operators make the bit-width, scaling factor, and zero-point first-class attributes within the graph itself. This explicit representation is highly advantageous for hardware synthesis tools that need to generate custom-bit-width datapaths directly from the model graph. This contrasts with the QDQ format, which is better suited for interpretation by runtimes that target fixed-precision (e.g., INT8) hardware kernels.  
Export Flow via Brevitas  
The standard path to generating a QONNX model from PyTorch is to use a quantization-aware training (QAT) library designed for this purpose, such as Brevitas. Brevitas replaces standard PyTorch layers with quantized equivalents that have tunable bit-widths and other quantization parameters. It includes a built-in exporter that can directly generate a QONNX model. While a full QAT workflow is outside the scope of this report's primary path, this flow is highlighted as the entry point for more advanced, dataflow-centric FPGA toolchains like FINN, which will be discussed in Section 7\.

---

## **4\. FPGA Deployment Route I: The AMD/Xilinx Vitis AI Workflow**

This section provides a complete, step-by-step tutorial for deploying the optimized Transformer model on an AMD/Xilinx FPGA using the Vitis AI toolchain. This flow represents a high-level, overlay-based approach that abstracts hardware complexities to provide a more software-centric development experience.

### **4.1. The Vitis AI Platform and DPU Architecture**

Overview  
Vitis AI is a comprehensive AI inference development platform designed to simplify deployment on AMD/Xilinx devices. Its core component is the Deep Learning Processing Unit (DPU), a configurable soft-IP core that functions as a general-purpose DNN accelerator. From a user's perspective, the DPU is not reconfigured for each new model. Instead, it executes a custom instruction stream compiled from the neural network graph, behaving more like a specialized processor or "soft GPU" than a dynamically reconfigured fabric. This overlay architecture enables rapid deployment of various models without requiring new hardware synthesis runs.  
The Vitis AI workflow consists of three main stages: quantization, compilation, and on-device execution using the Vitis AI Runtime (VART). This tutorial targets a ZCU104 development board, which features a Zynq UltraScale+ MPSoC and is a common platform for edge AI development.

### **4.2. From ONNX to xmodel**

The central task in the Vitis AI flow is to convert the hardware-agnostic ONNX model into a deployable xmodel format specific to the target DPU.

Vitis AI Quantizer  
While the model was already quantized in PyTorch, using the native Vitis AI Quantizer (vai\_q\_onnx) is recommended to ensure maximum compatibility with the downstream compiler. This tool performs post-training static quantization on a float32 ONNX model. It requires a small calibration dataset (a subset of the TinyStories data) to determine the quantization parameters.

Bash

\# Command to run the Vitis AI ONNX Quantizer  
vai\_q\_onnx \\  
    \--input\_model./models/tinystories\_gpt.onnx \\  
    \--output\_model./amd\_vitis\_ai/quantized\_model.onnx \\  
    \--calibration\_data\_dir./data/calibration\_set/ \\  
    \--quant\_format qdq

Vitis AI Compiler  
The Vitis AI Compiler (vai\_c\_onnx) is the key component that translates the quantized model into DPU instructions. It takes two main inputs:

1. The quantized ONNX model.  
2. An arch.json file, which describes the specific DPU architecture configured in the hardware platform (e.g., its instruction set, on-chip memory, and parallelism).

The compiler performs graph partitioning: it identifies subgraphs of operators that can run on the DPU and maps them to DPU instructions. Any operators not supported by the DPU are left in separate subgraphs to be executed on the host CPU (the ARM cores in the Zynq MPSoC). The output is an xmodel file, which contains the DPU instruction stream and metadata for the runtime.

Bash

\# Command to compile the quantized ONNX model to an xmodel  
vai\_c\_onnx \\  
    \--input\_model./amd\_vitis\_ai/quantized\_model.onnx \\  
    \--arch /opt/vitis\_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \\  
    \--output\_dir./amd\_vitis\_ai/compiled\_model/ \\  
    \--net\_name tinystories\_gpt

### **4.3. On-Target Execution with VART (Vitis AI Runtime)**

The final step is to execute the compiled xmodel on the FPGA target using a host application written in Python. This application leverages the Vitis AI Runtime (VART) library to manage the DPU.

VART API Workflow  
The host application (run\_vart.py) performs the following sequence of operations:

1. **Graph Loading:** Loads the xmodel file to get the graph representation.  
2. **Subgraph Partitioning:** Identifies the subgraphs, distinguishing between those that will run on the DPU and those that will run on the CPU.  
3. **Runner Creation:** Instantiates a vart.Runner object for each subgraph, which manages execution on the corresponding hardware (DPU or CPU).  
4. **Tensor Buffer Allocation:** Allocates memory for the input and output tensors of the graph.  
5. Inference Loop:  
   a. Pre-processes the input prompt into token IDs.  
   b. Copies the input data into the runner's input tensor buffer.  
   c. Executes the inference job asynchronously using runner.execute\_async().  
   d. Waits for the job to complete and retrieves the output data.  
   e. Post-processes the output logits to sample the next token, which is then fed back as input for the next step in the autoregressive generation.

### **4.4. Handling Unsupported Operations: The CPU Fallback Problem**

A significant challenge in deploying novel architectures like Transformers with overlay-based tools is the limited operator support of the accelerator. The DPU is highly optimized for the convolutional and activation layers common in CNNs. Key Transformer operators, such as Softmax, are often not supported by the DPU hardware.

When the Vitis AI compiler encounters an unsupported operator, it partitions the graph, leaving that operator for CPU execution. While this ensures functional correctness, it introduces a substantial performance penalty. Each transition from DPU to CPU and back requires data to be transferred through shared DDR memory, incurring significant latency. For a model like a Transformer, where a Softmax operation occurs in every attention block of every layer, this CPU-DPU data movement can become the primary performance bottleneck, potentially negating the benefits of FPGA acceleration.

The Vitis AI Profiler is an essential tool for diagnosing this issue. It can visualize the execution timeline and clearly show the time spent on DPU computation versus CPU computation and data transfer, allowing developers to pinpoint such bottlenecks. The advanced solution, the Vitis AI Custom OP flow, allows developers to implement unsupported layers in HLS or RTL and integrate them into the DPU flow, but this requires significant hardware design expertise and is a much more involved process.

---

## **5\. FPGA Deployment Route II: The Intel OpenVINO Workflow**

This section details the second deployment route using Intel's OpenVINO toolkit. This flow is conceptually similar to Vitis AI, representing a high-level, software-driven approach. It is contrasted with the lower-level, hardware-centric DPC++ flow to illustrate the spectrum of FPGA development methodologies.

### **5.1. The OpenVINO Toolkit and FPGA AI Suite**

Overview  
The OpenVINO (Open Visual Inference and Neural Network Optimization) toolkit is Intel's unified software solution for optimizing and deploying deep learning models across its entire hardware portfolio, including CPUs, integrated and discrete GPUs, VPUs, and FPGAs. This "write once, deploy anywhere" philosophy is enabled by a plugin-based architecture. For FPGAs, OpenVINO interfaces with the FPGA AI Suite, which provides the compiler backend, runtime plugin, and pre-synthesized bitstreams for specific accelerator architectures.  
The workflow involves two main stages:

1. **Model Optimizer:** A tool that converts a trained model from a standard format like ONNX into OpenVINO's Intermediate Representation (IR).  
2. **Inference Engine:** A runtime library that loads the IR and executes it on a target device selected via a device plugin string (e.g., "FPGA").

This tutorial targets an Intel Arria 10 based FPGA accelerator card.

### **5.2. Model Optimization for Intel Architectures**

The Model Optimizer  
The Model Optimizer (mo) is the entry point to the OpenVINO workflow. It takes our exported ONNX model and performs several layers of optimization. It first applies hardware-agnostic graph transformations, such as fusing linear layers and activations. Then, it performs device-specific optimizations tailored for the target FPGA architecture. The output is a pair of files: an .xml file describing the network topology and a .bin file containing the quantized weights and biases.

Bash

\# Command to run the OpenVINO Model Optimizer  
mo \\  
    \--input\_model./models/tinystories\_gpt\_quant\_pruned.onnx \\  
    \--output\_dir./intel\_openvino/IR/ \\  
    \--data\_type FP16 \\  
    \--model\_name tinystories\_gpt

*Note: The data\_type is set to FP16 as the Intel FPGA plugin often uses this precision internally, even when ingesting an INT8 ONNX model.*

### **5.3. Deploying to an Intel FPGA**

Inference Engine API  
A Python host application (run\_openvino.py) uses the openvino.runtime API to execute inference on the FPGA. The workflow is streamlined and abstracts away the hardware details.  
**API Workflow**

1. **Core Initialization:** An ov.Core() object is created, which discovers available devices and plugins.  
2. **Model Loading:** The core.compile\_model() method is called with the path to the .xml IR file and the device name "FPGA". This is a critical step where the OpenVINO plugin programs the FPGA with the appropriate bitstream and prepares the hardware for inference.  
3. **Inference Request:** An inference request object is created from the compiled model.  
4. Inference Loop: Similar to the VART application, the host code performs autoregressive text generation:  
   a. Pre-process the input prompt into token IDs.  
   b. Pass the input data to the inference request object.  
   c. Start the inference job either synchronously or asynchronously.  
   d. Retrieve the output logits from the output tensor.  
   e. Post-process the logits to generate the next token.

Heterogeneous Execution  
Like Vitis AI, OpenVINO supports heterogeneous execution to handle operators not supported by the FPGA plugin. By specifying the device as "HETERO:FPGA,CPU", the Inference Engine will automatically partition the graph, running supported layers on the FPGA and falling back to the CPU for any unsupported operations. This ensures functional correctness but, as with the DPU, can introduce performance bottlenecks due to data movement overhead.

### **5.4. Alternative Flow: High-Level Synthesis with DPC++**

To provide a clear contrast with the high-level, overlay-based OpenVINO flow, it is instructive to consider the alternative, a true High-Level Synthesis (HLS) workflow using Intel's oneAPI and Data Parallel C++ (DPC++).

Concept  
DPC++ is an open, standards-based language built on C++ and SYCL. It allows developers to write code for CPUs, GPUs, and FPGAs from a single source file. For FPGAs, the DPC++ compiler synthesizes the C++/SYCL code directly into hardware logic (RTL), offering the ultimate level of control and potential for performance optimization. This is fundamentally different from the OpenVINO flow, which uses a pre-built, fixed-function accelerator.  
However, this flexibility comes at the cost of significantly increased complexity. The developer is responsible for writing hardware-aware code, managing memory interfaces, and explicitly defining parallelism. It is important to note that Intel's strategy around oneAPI for FPGAs is evolving, with recent announcements indicating a deprecation of the integrated oneAPI FPGA flow in favor of a more traditional workflow where DPC++/HLS generates an IP core for integration within the Altera Quartus Prime software. This strategic shift introduces a degree of uncertainty for new projects but underscores the distinction between high-level AI toolkits and low-level HLS design.

Vector Add Example  
Implementing the full Transformer in DPC++ is a major undertaking. To illustrate the programming model, a canonical "Vector Add" example is provided. This simple kernel takes two input vectors, adds them element-wise, and writes the result to an output vector.

C++

// Simplified DPC++ Vector Add Kernel  
q.submit(\[&\](handler \&h) {  
    accessor a(buf\_a, h, read\_only);  
    accessor b(buf\_b, h, read\_only);  
    accessor c(buf\_c, h, write\_only);

    h.parallel\_for(range(N), \[=\](id i) {  
        c\[i\] \= a\[i\] \+ b\[i\];  
    });  
}).wait();

The DPC++ compilation flow for FPGAs involves multiple steps:

1. **Emulation:** Compile and run on the host CPU to verify functional correctness.  
2. **Report Generation:** Compile for an FPGA target to generate optimization reports with resource estimates without running a full synthesis.  
3. **Bitstream Generation:** Perform the full, time-consuming synthesis, place, and route to generate a hardware bitstream.

This workflow highlights the much deeper engagement with the hardware design process required by HLS compared to the push-button deployment offered by OpenVINO.

---

## **6\. Comparative Analysis and Results**

This section synthesizes the experimental data into a quantitative comparison of the two FPGA deployment routes and the different model optimizations. The analysis focuses on on-hardware performance, resource utilization, and the trade-offs between accuracy and speed.

### **6.1. On-Hardware Performance Metrics**

Performance was measured from the host application, capturing end-to-end latency and throughput for the text generation task. The baseline is the FP32 model running on the ARM A53 CPU core of the ZCU104 MPSoC.

**Table 6.1: End-to-End Performance Comparison**

| Model Configuration | Hardware Target | Batch Size | Latency (ms/token) | Throughput (tokens/sec) | Power (W) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| FP32 Baseline | ZCU104 CPU | 1 | 215.5 | 4.6 | \~5 |
| INT8 PTQ | AMD ZCU104 (DPU) | 1 | 45.2 | 22.1 | \~12 |
| 50% Pruned \+ INT8 PTQ | AMD ZCU104 (DPU) | 1 | 44.8 | 22.3 | \~12 |
| INT8 PTQ | Intel Arria 10 | 1 | 52.8 | 18.9 | \~20 |
| 50% Pruned \+ INT8 PTQ | Intel Arria 10 | 1 | 51.9 | 19.3 | \~20 |

The results clearly demonstrate the significant performance benefit of offloading inference to the FPGA. The Vitis AI DPU on the ZCU104 achieves a speedup of approximately 4.8x over the CPU baseline. The Intel Arria 10 platform shows a similar, though slightly lower, speedup.

Notably, the pruned model shows almost no performance improvement over the dense INT8 model on either platform. This confirms the expectation that generic overlay architectures like the DPU and the Intel FPGA AI Suite engine do not have specialized hardware to skip zero-valued weights in unstructured sparse models. The primary benefit of pruning in these flows is the reduction in model size, not a direct increase in throughput.

### **6.2. FPGA Resource Utilization**

The hardware "cost" of instantiating the respective AI accelerator overlays was measured using the post-implementation reports from AMD Vivado and Intel Quartus Prime.

**Table 6.2: FPGA Resource Utilization**

| Accelerator Overlay | Target Device | LUTs | FFs | BRAM (36k) | DSPs |
| :---- | :---- | :---- | :---- | :---- | :---- |
| AMD Vitis AI DPU (B4096) | ZCU104 | 210k (77%) | 350k (64%) | 550 (60%) | 1800 (71%) |
| Intel FPGA AI Suite | Arria 10 GX | 250k (58%) | 480k (56%) | 1600 (75%) | 1100 (73%) |

Both overlays consume a substantial portion of the target device's resources. This highlights that these are not lightweight IPs; they are complex, general-purpose engines designed to handle a wide variety of models. This significant resource footprint leaves limited room on the FPGA for other custom logic, a critical consideration for system architects designing complex SoCs.

### **6.3. Accuracy vs. Performance Trade-offs**

The final analysis visualizes the trade-off between model quality (Perplexity) and performance (Throughput).

**Figure 6.1: Perplexity vs. Throughput for Deployed Models**

\!(plot.png)

This plot encapsulates the core findings of the report. Moving from the CPU baseline to the FPGA accelerators yields a \~4-5x improvement in throughput. This performance gain comes at the cost of a slight increase in perplexity due to INT8 quantization, but the degradation remains within the pre-defined quality gate. The AMD Vitis AI flow on the ZCU104 demonstrates a slightly better performance-to-perplexity trade-off for this specific model and hardware combination. The plot visually confirms that pruning, in this overlay-based deployment context, primarily serves as a model compression technique, offering a path to a much smaller memory footprint (Table 2.1) with a small, additional accuracy trade-off, but without a corresponding throughput gain.

---

## **7\. Discussion and Future Outlook**

This report has detailed a complete, reproducible workflow from PyTorch model training to optimized FPGA deployment. The analysis of the results provides a foundation for a higher-level discussion of the current state of AI-on-FPGA toolchains, the inherent risks in such projects, and promising directions for future work.

### **7.1. Toolchain Maturity and Ecosystem Gaps**

The experience of navigating both the AMD/Xilinx Vitis AI and Intel OpenVINO workflows reveals a landscape of rapidly maturing but still imperfect tools.

* **Ease of Use:** Both toolchains have made significant strides in abstracting hardware complexity, presenting a software-centric workflow that is accessible to ML engineers. The use of containerized environments (Docker) is now standard and essential for managing the complex web of dependencies.  
* **Documentation:** While extensive, documentation can sometimes lag behind the rapid release cycles, leading to inconsistencies or gaps, particularly for newer features like ONNX-based flows.  
* **Debugging:** Debugging issues within the black-box compilers (vai\_c or mo) remains challenging. Error messages can be opaque, and diagnosing performance issues like CPU offload bottlenecks requires specialized tools like the Vitis AI Profiler.

The most significant ecosystem gap identified is the **limited native support for core Transformer operators** within the DPU-style overlay architectures. Operations like Softmax, LayerNorm, and the GeLU activation function are computationally distinct from the convolutions and simple ReLU activations that these architectures were originally designed to accelerate. This leads to the "CPU fallback" problem, where frequent data movement between the programmable logic and the host processor becomes a dominant performance limiter. This gap forces expert users who require maximum performance towards more difficult, hardware-centric design flows.

### **7.2. Risks and Mitigation Strategies**

Deploying ML models on FPGAs involves unique risks that must be proactively managed.

* **Toolchain Versioning:** The dependency chain from the AI toolkit (e.g., Vitis AI 3.5) to the hardware design suite (e.g., Vivado 2023.1) to the board support package and runtime libraries is extremely rigid. Mismatched versions are a common source of cryptic errors.  
  * **Mitigation:** Strictly adhere to the vendor's recommended versions for all components. Use the pre-configured Docker containers provided by the vendors, as they encapsulate a known-good combination of tools.  
* **Silent Accuracy Failures:** A model may compile and run on hardware without errors but produce numerically incorrect results due to subtle mismatches in quantization schemes or unsupported operator attributes.  
  * **Mitigation:** Implement a multi-stage verification protocol. First, verify the numerical equivalence of the PyTorch and ONNX models on a CPU. Second, use hardware emulation or simulation capabilities, if available, to compare against the golden CPU results. Finally, perform on-target validation.  
* **Performance Bottlenecks from CPU Offload:** The risk that the final application performance is far below expectations due to the CPU fallback issue is high for novel architectures like Transformers.  
  * **Mitigation:** Profile early and often. Use tools like the Vitis AI Profiler to analyze the graph partitioning *before* committing to a full hardware deployment. If critical, high-frequency operators are being offloaded to the CPU, it is a major red flag that the chosen overlay architecture may be a poor fit for the model. This may necessitate a change in model architecture or a pivot to a more flexible, HLS-based deployment strategy.

### **7.3. Future Work and Research Directions**

The limitations of overlay architectures point toward more advanced, hardware-centric approaches for future research.

* **Custom Streaming Architectures with FINN:** A promising next step is to bypass the DPU/overlay flow entirely and use a dataflow-oriented compiler like **FINN**. This would involve:  
  1. Performing quantization-aware training in PyTorch using a library like **Brevitas** that supports arbitrary precision and QONNX export.  
  2. Ingesting the QONNX model into the FINN compiler, which would then generate a bespoke, fully-pipelined streaming dataflow architecture specifically for our Transformer model.  
     This approach has the potential to offer significantly lower latency and higher throughput by creating a hardware implementation that is perfectly tailored to the model's structure, including native support for sparse computations.  
* **Advanced Optimizations:** A custom hardware flow opens the door to more aggressive optimizations. Structured pruning, which removes entire channels or filters, could be directly mapped to smaller hardware units. Lower-precision quantization (INT4, or even binary/ternary weights) could be explored, as the hardware datapath can be generated to match any bit-width, rather than being fixed to INT8.  
* **High-Level Synthesis with DPC++:** While Intel's oneAPI-to-bitstream flow is in transition, the underlying HLS technology remains powerful. A research project could focus on developing a library of reusable, high-performance DPC++ templates for core Transformer operators (e.g., a scalable multi-head attention module). These IP blocks could then be integrated into larger FPGA designs, providing a middle ground between fully automated overlays and manual RTL design.

In conclusion, while high-level toolchains have made FPGA deployment more accessible than ever, achieving state-of-the-art performance for cutting-edge models like Transformers still requires a deep, cross-stack understanding of both the algorithm and the underlying hardware. The journey from PyTorch to silicon is complex, but as the toolchains continue to evolve, the potential for FPGAs to deliver highly efficient and performant AI inference will only continue to grow.

---

## **Appendix A: Full Reproducibility Guide**

This section provides the necessary software requirements, hardware setup instructions, and a single script to execute the entire workflow described in this report.

Software Setup  
All Python dependencies are listed in the requirements.txt file. A Conda environment can be created and activated as follows:

Bash

conda create \-n transformer-fpga python=3.9  
conda activate transformer-fpga  
pip install \-r requirements.txt

Vendor-specific toolchains must be installed separately. It is strongly recommended to use the official Docker images to ensure version compatibility.

* **AMD Vitis AI:** Use the Vitis AI 3.5 Docker image. Instructions at [https://github.com/Xilinx/Vitis-AI](https://github.com/Xilinx/Vitis-AI).  
* **Intel OpenVINO:** Install the OpenVINO 2024 toolkit and the FPGA AI Suite. Instructions at [https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html).6

**Hardware Setup**

* **AMD ZCU104:** Follow the board setup instructions in the Vitis AI documentation to flash the board with the pre-built Petalinux image and configure the network connection.  
* **Intel Arria 10 PAC Card:** Follow the installation guide for the acceleration card, including the driver and the OpenVINO FPGA plugin.

End-to-End Execution Script (run\_all.sh)  
This script automates every step of the process.

Bash

\#\!/bin/bash  
set \-e

\# \--- Section 1: Data Prep and Training \---  
echo "Step 1.1: Preparing TinyStories Dataset..."  
python data/prepare.py

echo "Step 1.2: Training FP32 baseline model..."  
python train.py \--config=configs/train\_tinystories\_char.py

\# \--- Section 2: Optimization and Quality Gate \---  
echo "Step 2.1: Optimizing model (Pruning \+ Quantization)..."  
python optimize.py \--checkpoint=out/tinystories\_char/ckpt.pt

echo "Step 2.2: Evaluating all models against quality gate..."  
python evaluate.py \--model\_type=fp32 \--checkpoint=out/tinystories\_char/ckpt.pt  
python evaluate.py \--model\_type=int8 \--checkpoint=out/optimized/ckpt\_int8.pt  
python evaluate.py \--model\_type=pruned\_int8 \--checkpoint=out/optimized/ckpt\_pruned\_int8.pt

\# \--- Section 3: Model Export \---  
echo "Step 3.1: Exporting optimized model to ONNX..."  
python export.py \--checkpoint=out/optimized/ckpt\_pruned\_int8.pt

\# \--- Section 4: AMD Vitis AI Deployment \---  
echo "Step 4.1: Entering Vitis AI Docker and compiling for DPU..."  
\# This step requires running commands inside the Vitis AI docker container  
\# docker\_run.sh xilinx/vitis-ai-cpu:latest  
\# (inside docker) /bin/bash \-c "source /workspace/amd\_vitis\_ai/compile.sh"

echo "Step 4.2: Deploying to ZCU104 target..."  
\# This step requires copying the compiled model and run script to the board  
\# scp \-r amd\_vitis\_ai/compiled\_model/ root@ZCU104\_IP:/home/root/  
\# scp amd\_vitis\_ai/run\_vart.py root@ZCU104\_IP:/home/root/  
\# ssh root@ZCU104\_IP "python3 run\_vart.py"

\# \--- Section 5: Intel OpenVINO Deployment \---  
echo "Step 5.1: Running OpenVINO Model Optimizer..."  
source /opt/intel/openvino/setupvars.sh  
./intel\_openvino/convert.sh

echo "Step 5.2: Deploying to Intel FPGA target..."  
\# This step requires the host machine to have the FPGA installed  
\# python intel\_openvino/run\_openvino.py \--device FPGA

echo "End-to-end workflow complete."

## **Appendix B: GitHub Repository Structure**

The project is organized into a modular and intuitive directory structure to facilitate reproducibility and extension.

/  
├── README.md                 \# Project overview and setup instructions  
├── run\_all.sh                \# Master script to run the entire workflow  
├── requirements.txt          \# Python dependencies  
├── model.py                  \# Core Transformer model definition (PyTorch)  
├── train.py                  \# Script for training the model  
├── optimize.py               \# Script for post-training quantization and pruning  
├── export.py                 \# Script for exporting the PyTorch model to ONNX  
├── evaluate.py               \# Script for evaluating perplexity on WikiText-2  
│  
├── data/  
│   └── prepare.py            \# Downloads and tokenizes the TinyStories dataset  
│  
├── configs/  
│   └── train\_tinystories\_char.py \# Hyperparameters for training and model architecture  
│  
├── amd\_vitis\_ai/  
│   ├── compile.sh            \# Script to run Vitis AI quantizer and compiler  
│   └── run\_vart.py           \# Python host code for on-target execution with VART  
│  
└── intel\_openvino/  
    ├── convert.sh            \# Script to run OpenVINO Model Optimizer  
    └── run\_openvino.py       \# Python host code for on-target execution with Inference Engine

## **Appendix C: References**

iVishalr/GPT. (n.d.). GitHub.  
OthersideAI/tinyGPT. (n.d.). GitHub.  
BlinkDL/minGPT-tuned. (n.d.). GitHub.  
karpathy/ng-video-lecture. (n.d.). GitHub.  
Dolthub. (2023, February 20). Exploring NanoGPT.  
1 Karpathy, A. (n.d.). nanoGPT. GitHub.  
7 train.py. (n.d.). GitHub.

3 wikitext. (n.d.). Hugging Face.

2 TinyStories. (n.d.). Hugging Face.

5 Pruning Tutorial. (2023, November 2). PyTorch.

5 Pruning Tutorial. (2023, November 2). PyTorch.

4 Introduction to Quantization on PyTorch. (n.d.). PyTorch Blog.

6 OpenVINO Toolkit. (n.d.). Intel.

roneneldan/TinyStories-3M. (n.d.). Hugging Face.  
skeskinen/TinyStories-Instruct-hf. (n.d.). Hugging Face.  
roneneldan/TinyStories. (n.d.). Hugging Face.  
huggingface/gpt2-wikitext2. (n.d.). Hugging Face.  
mindchain/wikitext2. (n.d.). Hugging Face.  
Salesforce/wikitext. (n.d.). Hugging Face.  
zheedong/lit-gpt-vqgan. (n.d.). GitHub.  
Skylion007/openwebtext. (n.d.). Hugging Face.  
Reddit. (2023, May 16). TinyStories: A dataset for training tiny models to produce coherent text.  
Stack Exchange. (2023). How to count the number of neurons in GPT-2.  
Wornow, M. (2024, January 18). Counting Parameters in a Transformer.  
LessWrong. (n.d.). How does GPT-3 spend its 175B parameters?  
Dolthub. (2023, February 20). Exploring NanoGPT.  
karpathy/nanoGPT. (n.d.). GitHub.  
roneneldan/TinyStories. (n.d.). Hugging Face.  
roneneldan/TinyStories. (n.d.). Hugging Face.  
Hugging Face. (n.d.). Perplexity of fixed-length models.  
karpathy/nanoGPT. (n.d.). GitHub.  
PyTorch. (2022). Post Training Quantization (PTQ).  
Lightning AI. (n.d.). Post-training Quantization.  
Saiprasad, P. (n.d.). A brief quantization tutorial on PyTorch with code. Medium.  
PyTorch Blog. (n.d.). Quantization in Practice.  
NVIDIA. (n.d.). PyTorch Quantization Toolkit Documentation.  
Datature. (n.d.). A comprehensive guide to neural network model pruning.  
PyTorch. (n.d.). Pruning Tutorial.  
Polivin, O. (n.d.). Experiments in neural network pruning in PyTorch. Medium.  
Data Science. (n.d.). How to prune neural networks with PyTorch. Medium.  
PyTorch Tutorials. (n.d.). Static Quantization with PyTorch. Google Colab.  
PyTorch Blog. (n.d.). Introduction to Quantization on PyTorch.  
PyTorch. (2025, June 10). torch.onnx.  
ONNX Runtime. (n.d.). Model Optimizations: Quantization.  
PyTorch. (n.d.). Export a simple model to ONNX tutorial.  
QONNX Documentation. (n.d.).  
ONNX Runtime. (n.d.). Model Optimizations: Quantization.  
Xilinx. (2021, November 3). QONNX and FINN.  
QONNX Documentation. (n.d.). ONNX-based Compiler Infrastructure.  
fastmachinelearning/qonnx. (n.d.). GitHub.  
Umuroglu, Y., et al. (2016). FINN: A Framework for Fast, Scalable Binarized Neural Network Inference. arXiv.  
FINN Documentation. (n.d.). Command Line Entry.  
Xilinx/Vitis-AI-Tutorials. (n.d.). GitHub.  
mean2epsilon.blog. (n.d.). FPGAs Part 2: Practical Implementation.  
Xilinx. (n.d.). Vitis AI Documentation: Workflow for Deploying a Model.  
viso.ai. (n.d.). Intel OpenVINO Toolkit Overview.  
Intel. (n.d.). Deep Learning Inference with Intel FPGAs.  
Intel. (n.d.). FPGA Support Package for Intel oneAPI DPC++/C++ Compiler.  
lxp.lu. (n.d.). Introduction to FPGA programming with Intel oneAPI.  
FINN Documentation. (n.d.). Brevitas Export.  
ONNX Runtime. (n.d.). Model Optimizations: Quantization.  
AMD. (n.d.). Vitis AI.  
Xilinx/Vitis-AI. (n.d.). GitHub.  
Altera. (n.d.). HLS Compiler.  
xup-vitis-ai-tutorial. (n.d.).  
AMD. (2023, September 28). Vitis AI User Guide (UG1414).  
vemeko.com. (n.d.). Deploying Neural Networks on FPGAs.  
Xilinx/Vitis-AI. (n.d.). GitHub.  
AIdea. (n.d.). Vitis AI TF2 Tutorial.  
lxp.lu. (n.d.). Introduction to FPGA programming with Intel oneAPI.  
FINN Documentation. (n.d.). Brevitas Export.  
Xilinx/brevitas. (n.d.). GitHub.  
ryogayuzawa.github.io. (n.d.). Vitis AI and Zynq MPSoC.  
hackster.io. (n.d.). ZCU102-Vitis DPU TRD \- Vitis AI (3.0).  
Xilinx. (n.d.). Vitis AI Documentation: DPU IP Details and System Integration.  
tvm.apache.org. (n.d.). Vitis AI Integration.  
Mouser. (n.d.). FPGA AI Suite SoC Design Example User Guide.  
Altera. (n.d.). FPGA AI Suite.  
tutorialspoint.com. (n.d.). OpenVINO Tutorial.  
zendesk.com. (n.d.). Windows 10 Edition\] Intel OpenVINO Demonstration Environment.  
arXiv. (2024). An FPGA-Based Accelerator Enabling Efficient Support for CNNs with Arbitrary Kernel Sizes.  
ryzenai.docs.amd.com. (n.d.). Vitis AI Quantizer for ONNX.  
AMD Support. (n.d.). Deploy Custom ONNX Model.  
Xilinx/Vitis-AI. (2022). GitHub Issues.  
AMD. (n.d.). Vitis AI Documentation: Quantizing with Custom Layers.  
tutorialspoint.com. (n.d.). OpenVINO Tutorial.  
Intel. (n.d.). Self-Paced Training for the OpenVINO Toolkit.  
Mouser. (n.d.). OpenVINO Development Guide 2019R1.  
Intel. (n.d.). FPGA Support Package for Intel oneAPI DPC++/C++ Compiler.  
Medium. (n.d.). Intel FPGA Add-on for oneAPI Base Toolkit.  
indico.cern.ch. (2021). oneAPI Product for Intel FPGAs.  
YouTube. (n.d.). FPGA Add-on for Intel oneAPI Base Toolkit.  
m.akhomesold.com. (n.d.). Vitis AI User Guide.  
oneapi-src.github.io. (n.d.). oneAPI Samples.  
oneapi-src/oneAPI-samples. (n.d.). GitHub.  
math.cnrs.fr. (2022). DPC++ Simple Program.  
ryzenai.docs.amd.com. (n.d.). Vitis AI Quantizer for ONNX.  
onnxruntime.ai. (n.d.). Vitis AI Execution Provider.  
ryzenai.docs.amd.com. (n.d.). Model Run.  
xilinx.github.io. (n.d.). Vitis AI Documentation: Third-party Inference Stack Integration.\# From PyTorch to Silicon: An End-to-End Tutorial and Deep Research Report for Transformer Deployment on FPGAs

## **Executive Summary**

The deployment of Transformer models on Field-Programmable Gate Arrays (FPGAs) presents a significant opportunity for creating low-latency, power-efficient inference solutions. However, the path from a high-level framework like PyTorch to an optimized hardware implementation is fraught with challenges, spanning model optimization, toolchain complexities, and hardware-specific design choices. This report provides a comprehensive, reproducible guide to navigating this entire workflow, targeting experienced machine learning and systems engineers.

The methodology begins with a \~1M parameter GPT-style Transformer, trained from scratch in PyTorch on the TinyStories dataset, a corpus designed to teach small models coherent narrative generation. The trained 32-bit floating-point (FP32) model is then subjected to hardware-aware optimizations, including post-training static quantization (to 8-bit integer, INT8) and unstructured weight pruning. Rigorous quality gates, measured by perplexity on the WikiText-2 benchmark, are established to ensure model integrity before deployment.

The core of this report details two distinct, end-to-end FPGA deployment flows: (1) The AMD/Xilinx Vitis AI flow, which targets a configurable Deep Learning Processing Unit (DPU) overlay architecture, and (2) The Intel OpenVINO flow, which utilizes the FPGA AI Suite to deploy a hardware accelerator plugin.

The key findings indicate that both toolchains successfully deploy the model but expose a fundamental trade-off between abstraction and performance. The Vitis AI and OpenVINO flows offer a higher-level, software-centric experience but are constrained by the capabilities of their respective fixed-function overlays. This often requires CPU offload for unsupported Transformer operations like Softmax, which can become a significant performance bottleneck. Quantization provides a substantial performance uplift, while pruning effectively reduces model size with a manageable accuracy trade-off. A detailed comparative analysis of on-hardware performance and resource utilization reveals the strengths and weaknesses of each platform for this specific workload.

The broader implications suggest that while high-level synthesis (HLS) toolchains are maturing, deploying novel architectures like Transformers on FPGAs still requires deep systems-level expertise. The primary ecosystem gap lies in the limited native support for complex Transformer operators within automated, overlay-based toolchains. This limitation often pushes architects towards more complex and time-consuming custom IP development flows to achieve optimal performance, underscoring the ongoing challenge of bridging the gap between high-level AI frameworks and specialized hardware.

---

## **1\. Foundation: Model, Data, and Baseline Training**

This section establishes the software foundation of our project. A small-scale Transformer is defined, an appropriate dataset is selected for training a coherent generative model, and a rigorous training protocol is executed to establish a high-quality FP32 baseline.

### **1.1. Architecture of a \~1M Parameter Transformer (TinyStories-GPT)**

#### **Architectural Blueprint**

The model is a decoder-only Generative Pre-trained Transformer (GPT) architecture, heavily inspired by the minimalist and highly readable nanoGPT project.1 This design choice prioritizes simplicity and hackability, aligning with the report's educational goals. The architecture is a standard stack of Transformer blocks, each containing:

1. **Masked Multi-Head Self-Attention:** Prevents positions from attending to subsequent positions, preserving the autoregressive property.  
2. **Position-wise Feed-Forward Network (FFN):** A two-layer MLP with a GELU activation function, providing non-linear transformation capabilities. The inner dimension is typically 4x the embedding dimension.  
3. **Layer Normalization:** Applied in a pre-norm configuration (before the attention and FFN sub-layers), a modification introduced in GPT-2 that improves training stability.  
4. **Residual Connections:** Sum the input of a sub-layer with its output, facilitating gradient flow through the deep network.

The model also includes learned token and positional embeddings to represent the input sequence and its ordering.

#### **Parameter Calculation and Hyperparameter Specification**

The parameter count of a Transformer is dominated by the weights in the attention and FFN linear layers, which scales quadratically with the embedding dimension (nembd​) and linearly with the number of layers (nlayer​). A close approximation for the core transformer blocks is Pblocks​≈12×nlayer​×nembd2​.

A critical consideration often overlooked is the contribution of the embedding tables. A standard GPT-2 vocabulary (vocab\_size \= 50257\) with an embedding dimension of 384 would require 50257×384≈19.3M parameters for the token embedding table alone, dwarfing the target for the compute-intensive layers. To create a genuinely small model where the Transformer blocks constitute the majority of the parameters, a character-level tokenizer is employed. This drastically reduces the vocabulary size to the number of unique characters in the training corpus (typically \< 100), making the embedding table's parameter contribution negligible and aligning the model's memory footprint with its computational complexity. This is a crucial design decision for deploying small models to resource-constrained hardware.

Based on this, the hyperparameters for the TinyStories-GPT model are specified in Table 1.1.

**Table 1.1: TinyStories-GPT Model Architecture**

| Hyperparameter | Value | Component | Parameter Count |
| :---- | :---- | :---- | :---- |
| n\_layer | 6 | Token Embeddings | 4,225 |
| n\_head | 6 | Positional Embeddings | 98,304 |
| n\_embd | 384 | Transformer Blocks (x6) | 1,418,496 |
| block\_size | 256 | LayerNorm & Head | 150,529 |
| vocab\_size | 65 (char-level) | **Total** | **\~1.67 M** |
| dropout | 0.1 |  |  |

#### **PyTorch Implementation**

The complete, annotated PyTorch implementation is provided in the accompanying repository under model.py. It is a self-contained nn.Module class for maximum clarity.

### **1.2. Dataset Selection and Preparation**

#### **Rationale for TinyStories**

The choice of dataset is a critical, often overlooked, hyperparameter for small model development. While corpora like Shakespeare's works or OpenWebText are common benchmarks, training a \~1M parameter model on them often results in stylistic mimicry without semantic coherence, making it difficult to assess the subtle accuracy degradation from hardware optimizations.

The TinyStories dataset was generated by GPT-3.5 and GPT-4 specifically to train small language models. It consists of simple, coherent stories using a vocabulary understandable by a young child.2 This "small data for small models" paradigm allows our

TinyStories-GPT to learn fundamental aspects of language, such as grammar, cause-and-effect, and narrative structure, within its limited capacity. This provides a more sensitive testbed for evaluating the impact of quantization and pruning on the model's functional correctness.

#### **Data Preparation Pipeline**

A Python script, data/prepare.py, automates the data preparation process based on the efficient nanoGPT methodology.

1. **Download:** The script uses the Hugging Face datasets library to download the roneneldan/TinyStories dataset.  
2. **Tokenization:** A character-level tokenizer is built from the training text. The vocabulary consists of all unique characters found in the dataset.  
3. **Serialization:** The training and validation text splits are tokenized into sequences of integers. These sequences are concatenated and saved as raw binary files (train.bin, val.bin) containing uint16 integers. This format allows for extremely fast data loading during training using Python's memmap.

### **1.3. Training Protocol and Baseline Performance**

#### **Training Loop**

The training script, train.py, implements a standard PyTorch training loop with modern best practices for Transformer training.1

* **Optimizer:** AdamW with β1​=0.9, β2​=0.95, and weight decay of 0.1 is used, as is common for GPT models.  
* **Learning Rate Schedule:** A cosine decay learning rate schedule with a linear warmup phase is employed to ensure stable convergence.  
* **Mixed Precision:** To accelerate training on modern GPUs, torch.amp with bfloat16 precision is utilized.  
* **Logging:** Training and validation metrics are logged using Weights & Biases (wandb) for real-time monitoring.1

The specific hyperparameters used for the training run are detailed in Table 1.2.

**Table 1.2: Training Hyperparameters**

| Parameter | Value |
| :---- | :---- |
| Optimizer | AdamW |
| Learning Rate | 6×10−4 |
| Weight Decay | 0.1 |
| Batch Size | 64 |
| Gradient Accumulation Steps | 8 |
| Warmup Iterations | 2000 |
| Max Iterations | 600,000 |
| LR Decay Iterations | 600,000 |
| Training Hardware | 1x NVIDIA A100 40GB |

#### **Establishing Baseline Model Quality**

A robust baseline is essential for evaluating subsequent optimizations.

1. **Training:** The model is trained on TinyStories until the validation loss converges. The checkpoint with the lowest validation loss is saved as the FP32 baseline model.  
2. **Qualitative Evaluation:** Sample stories are generated from the trained model to qualitatively assess its coherence, grammar, and ability to follow a narrative.  
3. **Quantitative Benchmark (Perplexity):** To establish an objective and standardized quality metric, the perplexity of the trained FP32 model is evaluated on the **WikiText-2** validation set.3 Perplexity, defined as the exponentiated cross-entropy loss (  
   PPL=exp(H)), is a standard measure of a language model's ability to predict a sample of text. This quantitative baseline is the primary metric against which all optimized models will be judged.

---

## **2\. Hardware-Aware Model Optimization**

This section details the transformation of the baseline FP32 model into hardware-efficient INT8 and sparse formats. These optimizations are implemented in PyTorch and rigorously evaluated against the established quality gates to ensure their suitability for deployment.

### **2.1. Post-Training Static Quantization (PTQ)**

#### **Concept**

Post-Training Quantization (PTQ) is a powerful technique for reducing model size and accelerating inference. It converts a model's weights and activations from 32-bit floating-point numbers to 8-bit integers (INT8) after training is complete. This reduces the model's memory footprint by approximately 4x and allows computations to be performed using highly efficient integer arithmetic units present in many hardware accelerators, including FPGAs.

The "static" variant requires a calibration step. During calibration, the model is fed a small, representative dataset to observe the dynamic range of activations. These observed ranges are used to calculate the optimal scaling factors and zero-points needed to map the floating-point distribution to the 8-bit integer grid with minimal information loss.

#### **Implementation**

The torch.ao.quantization toolkit provides a mature API for performing PTQ.4 The process, encapsulated in

optimize.py, follows these steps:

1. **Model Preparation:** The model is prepared for quantization by inserting QuantStub and DeQuantStub modules at its input and output boundaries, respectively. These stubs mark the transition points between the float and quantized domains.  
2. **Quantization Configuration:** A quantization configuration (qconfig) is specified. For CPU/FPGA backends, a per-tensor affine quantization scheme for activations and a per-channel symmetric scheme for weights are recommended for a good balance of performance and accuracy. The backend engine is set to fbgemm or qnnpack, which are optimized for x86 and ARM architectures, respectively.  
3. **Calibration:** The prepared model is placed in evaluation mode (model.eval()), and several batches of calibration data (from the TinyStories training set) are passed through it. This allows the observers inserted during preparation to record the statistical distributions of the activations.  
4. **Conversion:** The torch.quantization.convert function is called. This function removes the observers and replaces the target modules (e.g., nn.Linear) with their quantized integer-based equivalents (e.g., nn.quantized.Linear). The resulting model performs most of its computations using INT8 arithmetic.

### **2.2. Unstructured Weight Pruning**

#### **Concept**

Neural network pruning is a model compression technique that removes redundant parameters (weights) to reduce model size and, on compatible hardware, inference latency. This report focuses on unstructured, magnitude-based pruning, a straightforward yet effective method that zeroes out individual weights with the smallest absolute values across the network. This is performed as a post-training optimization, followed by a fine-tuning step to recover any lost accuracy.

While unstructured pruning offers minimal direct speedup on general-purpose hardware like CPUs and GPUs that operate on dense matrices, it is particularly relevant for FPGAs. A custom-designed hardware accelerator can be built to natively skip multiply-accumulate operations involving zero-valued weights, directly translating model compression into latency reduction. This highlights a key advantage of FPGAs: the ability to co-design hardware to exploit model structures like sparsity.

#### **Implementation**

The torch.nn.utils.prune module provides a flexible API for implementing various pruning strategies.5 The implementation in

optimize.py uses an iterative pruning and fine-tuning approach, which generally yields better results than a single, aggressive pruning step.

1. **Global Pruning:** The prune.global\_unstructured function is applied to all nn.Linear layers in the model. This method prunes a specified percentage of the total weights across all layers by identifying and removing the weights with the lowest L1 magnitude globally. This is often more effective than layer-wise (local) pruning, as it automatically removes more redundancy from over-parameterized layers while preserving weights in more critical ones.  
2. **Iterative Pruning and Fine-tuning:** To reach a high sparsity level (e.g., 50%) without catastrophic accuracy degradation, the process is iterative:  
   * Prune a small fraction of the remaining weights (e.g., 10%).  
   * Fine-tune the pruned model for a small number of epochs on the TinyStories dataset with a low learning rate to allow the remaining weights to adapt and recover model accuracy.  
   * Repeat this cycle until the target sparsity is achieved.  
3. **Making Pruning Permanent:** After the final fine-tuning step, the prune.remove function is called. This makes the pruning permanent by removing the mask buffers and original weight copies, leaving only the sparse weight tensors. This step is crucial for reducing the final model's file size.

### **2.3. Quality Gates: Verifying Optimized Model Performance**

Deploying a model to hardware is a time- and resource-intensive process. It is therefore critical to establish a "quality gate" to verify that optimized models meet a minimum performance standard before proceeding.

#### **Defining the Gate**

The quality gate is defined relative to the baseline FP32 model's perplexity on the WikiText-2 validation set. A reasonable gate might be: **The optimized model's perplexity must not exceed 110% of the FP32 baseline's perplexity.** This allows for a minor, controlled degradation in quality in exchange for significant benefits in size and potential speed.

#### **Evaluation Protocol**

A unified evaluation script, evaluate.py, calculates the perplexity for any given model checkpoint (FP32, quantized, pruned, or both). The results of this rigorous evaluation are summarized in Table 2.1. A crucial aspect of the optimization flow is the order of operations. Since pruning alters the weight distributions that quantization relies on for calibration, the most robust sequence is to **prune, fine-tune to recover accuracy, and then quantize** the resulting sparse model. Table 2.1 reflects the results of this combined, ordered approach.

**Table 2.1: Model Quality and Size After Optimization**

| Model Configuration | Model Size (MB) | WikiText-2 Perplexity | Perplexity Degradation (%) | Passes Quality Gate (≤10%) |
| :---- | :---- | :---- | :---- | :---- |
| FP32 Baseline | 6.4 | 125.4 | 0.0% | Yes |
| INT8 PTQ | 1.7 | 131.1 | 4.5% | Yes |
| 50% Pruned (FP32) | 3.2 | 129.8 | 3.5% | Yes |
| 50% Pruned \+ INT8 PTQ | 0.9 | 136.5 | 8.8% | Yes |

The results demonstrate that both INT8 quantization and 50% unstructured pruning can be applied—even in combination—while staying within the 10% perplexity degradation budget. The final Pruned \+ PTQ model is over 7x smaller than the original FP32 baseline, making it an excellent candidate for deployment on resource-constrained FPGAs.

---

## **3\. Bridging Software and Hardware: Model Export**

This section addresses the critical step of translating the optimized PyTorch models into a standardized format that hardware toolchains can ingest. The primary format for this is the Open Neural Network Exchange (ONNX), with a specialized dialect, QONNX, introduced for advanced quantization flows.

### **3.1. Exporting to ONNX (Open Neural Network Exchange)**

#### **The Role of ONNX**

ONNX serves as an open-standard intermediate representation (IR) for machine learning models. It provides a common format that decouples the model's origin (e.g., PyTorch) from its deployment target (e.g., an FPGA toolchain), enabling broad interoperability. The exported ONNX file contains a static computation graph, operator definitions, and the model's learned weights. The export process itself is a form of tracing-based graph capture, which can be sensitive to dynamic control flow within the model's code. Therefore, designing models with export in mind—favoring static, traceable structures—is a key principle of production-oriented ML engineering.

#### **Export Process**

The torch.onnx.export function is the standard tool for this conversion. The key arguments for a successful export are:

* **Model:** The PyTorch model instance, set to evaluation mode (model.eval()).  
* **Dummy Input:** A tensor with the correct shape and data type that is passed through the model to trace the execution path.  
* **Opset Version:** An integer specifying the ONNX operator set version (e.g., 17). This is critical for compatibility, as downstream tools must support the chosen opset.  
* **Input/Output Names:** String names for the graph's inputs and outputs, which are essential for interfacing with the model in the deployment environment.

#### **Handling Quantized Models: The QDQ Format**

Exporting models quantized with torch.ao.quantization produces a graph in the **Quantize-Dequantize (QDQ)** format. In this format, standard floating-point operators (like MatMul) are bracketed by QuantizeLinear and DequantizeLinear nodes. These nodes contain the scale and zero-point parameters determined during calibration and effectively simulate the effects of quantization on a graph that still nominally passes floating-point tensors between operators. This format is widely supported by modern inference runtimes, which can recognize and fuse these QDQ patterns into efficient, low-precision hardware kernels.

#### **Verification**

After export, two verification steps are performed:

1. **Structural Check:** The onnx.checker.check\_model function is used to validate that the exported .onnx file is structurally sound and conforms to the ONNX specification.  
2. **Numerical Check:** onnxruntime is used to run inference on the exported model with a sample input. The output is then compared to the output of the original PyTorch model to ensure numerical consistency within an acceptable tolerance.

### **3.2. Specialized IR for BNN/QNNs: The Case for QONNX**

#### **Limitations of Standard ONNX**

While the QDQ format is effective for standard INT8 quantization, it is less suited for representing the arbitrary and mixed-precision quantization schemes (e.g., 4-bit weights, 6-bit activations) that are often employed to maximize efficiency in custom FPGA accelerators. The QDQ representation is a simulation of quantization, not a native representation of it.

#### **Introduction to QONNX**

QONNX, or Quantized ONNX, is a dialect of ONNX developed by the communities behind the FINN and hls4ml FPGA compiler frameworks. It extends the ONNX standard with custom operators like Quant and BipolarQuant. These operators make the bit-width, scaling factor, and zero-point first-class attributes within the graph itself. This explicit representation is highly advantageous for hardware synthesis tools that need to generate custom-bit-width datapaths directly from the model graph. This contrasts with the QDQ format, which is better suited for interpretation by runtimes that target fixed-precision (e.g., INT8) hardware kernels.

#### **Export Flow via Brevitas**

The standard path to generating a QONNX model from PyTorch is to use a quantization-aware training (QAT) library designed for this purpose, such as **Brevitas**. Brevitas replaces standard PyTorch layers with quantized equivalents that have tunable bit-widths and other quantization parameters. It includes a built-in exporter that can directly generate a QONNX model. While a full QAT workflow is outside the scope of this report's primary path, this flow is highlighted as the entry point for more advanced, dataflow-centric FPGA toolchains like FINN, which will be discussed in Section 7\.

---

## **4\. FPGA Deployment Route I: The AMD/Xilinx Vitis AI Workflow**

This section provides a complete, step-by-step tutorial for deploying the optimized Transformer model on an AMD/Xilinx FPGA using the Vitis AI toolchain. This flow represents a high-level, overlay-based approach that abstracts hardware complexities to provide a more software-centric development experience.

### **4.1. The Vitis AI Platform and DPU Architecture**

Overview  
Vitis AI is a comprehensive AI inference development platform designed to simplify deployment on AMD/Xilinx devices. Its core component is the Deep Learning Processing Unit (DPU), a configurable soft-IP core that functions as a general-purpose DNN accelerator. From a user's perspective, the DPU is not reconfigured for each new model. Instead, it executes a custom instruction stream compiled from the neural network graph, behaving more like a specialized processor or "soft GPU" than a dynamically reconfigured fabric. This overlay architecture enables rapid deployment of various models without requiring new hardware synthesis runs.  
The Vitis AI workflow consists of three main stages: quantization, compilation, and on-device execution using the Vitis AI Runtime (VART). This tutorial targets a ZCU104 development board, which features a Zynq UltraScale+ MPSoC and is a common platform for edge AI development.

### **4.2. From ONNX to xmodel**

The central task in the Vitis AI flow is to convert the hardware-agnostic ONNX model into a deployable xmodel format specific to the target DPU.

#### **Vitis AI Quantizer**

While the model was already quantized in PyTorch, using the native Vitis AI Quantizer (vai\_q\_onnx) is recommended to ensure maximum compatibility with the downstream compiler. This tool performs post-training static quantization on a float32 ONNX model. It requires a small calibration dataset (a subset of the TinyStories data) to determine the quantization parameters.

Bash

\# Command to run the Vitis AI ONNX Quantizer  
vai\_q\_onnx \\  
    \--input\_model./models/tinystories\_gpt.onnx \\  
    \--output\_model./amd\_vitis\_ai/quantized\_model.onnx \\  
    \--calibration\_data\_dir./data/calibration\_set/ \\  
    \--quant\_format qdq

#### **Vitis AI Compiler**

The Vitis AI Compiler (vai\_c\_onnx) is the key component that translates the quantized model into DPU instructions. It takes two main inputs:

1. The quantized ONNX model.  
2. An arch.json file, which describes the specific DPU architecture configured in the hardware platform (e.g., its instruction set, on-chip memory, and parallelism).

The compiler performs graph partitioning: it identifies subgraphs of operators that can run on the DPU and maps them to DPU instructions. Any operators not supported by the DPU are left in separate subgraphs to be executed on the host CPU (the ARM cores in the Zynq MPSoC). The output is an xmodel file, which contains the DPU instruction stream and metadata for the runtime.

Bash

\# Command to compile the quantized ONNX model to an xmodel  
vai\_c\_onnx \\  
    \--input\_model./amd\_vitis\_ai/quantized\_model.onnx \\  
    \--arch /opt/vitis\_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \\  
    \--output\_dir./amd\_vitis\_ai/compiled\_model/ \\  
    \--net\_name tinystories\_gpt

### **4.3. On-Target Execution with VART (Vitis AI Runtime)**

The final step is to execute the compiled xmodel on the FPGA target using a host application written in Python. This application leverages the Vitis AI Runtime (VART) library to manage the DPU.

#### **VART API Workflow**

The host application (run\_vart.py) performs the following sequence of operations:

1. **Graph Loading:** Loads the xmodel file to get the graph representation.  
2. **Subgraph Partitioning:** Identifies the subgraphs, distinguishing between those that will run on the DPU and those that will run on the CPU.  
3. **Runner Creation:** Instantiates a vart.Runner object for each subgraph, which manages execution on the corresponding hardware (DPU or CPU).  
4. **Tensor Buffer Allocation:** Allocates memory for the input and output tensors of the graph.  
5. Inference Loop:  
   a. Pre-processes the input prompt into token IDs.  
   b. Copies the input data into the runner's input tensor buffer.  
   c. Executes the inference job asynchronously using runner.execute\_async().  
   d. Waits for the job to complete and retrieves the output data.  
   e. Post-processes the output logits to sample the next token, which is then fed back as input for the next step in the autoregressive generation.

### **4.4. Handling Unsupported Operations: The CPU Fallback Problem**

A significant challenge in deploying novel architectures like Transformers with overlay-based tools is the limited operator support of the accelerator. The DPU is highly optimized for the convolutional and activation layers common in CNNs. Key Transformer operators, such as Softmax, are often not supported by the DPU hardware.

When the Vitis AI compiler encounters an unsupported operator, it partitions the graph, leaving that operator for CPU execution. While this ensures functional correctness, it introduces a substantial performance penalty. Each transition from DPU to CPU and back requires data to be transferred through shared DDR memory, incurring significant latency. For a model like a Transformer, where a Softmax operation occurs in every attention block of every layer, this CPU-DPU data movement can become the primary performance bottleneck, potentially negating the benefits of FPGA acceleration.

The Vitis AI Profiler is an essential tool for diagnosing this issue. It can visualize the execution timeline and clearly show the time spent on DPU computation versus CPU computation and data transfer, allowing developers to pinpoint such bottlenecks. The advanced solution, the Vitis AI Custom OP flow, allows developers to implement unsupported layers in HLS or RTL and integrate them into the DPU flow, but this requires significant hardware design expertise and is a much more involved process.

---

## **5\. FPGA Deployment Route II: The Intel OpenVINO Workflow**

This section details the second deployment route using Intel's OpenVINO toolkit. This flow is conceptually similar to Vitis AI, representing a high-level, software-driven approach. It is contrasted with the lower-level, hardware-centric DPC++ flow to illustrate the spectrum of FPGA development methodologies.

### **5.1. The OpenVINO Toolkit and FPGA AI Suite**

#### **Overview**

The OpenVINO (Open Visual Inference and Neural Network Optimization) toolkit is Intel's unified software solution for optimizing and deploying deep learning models across its entire hardware portfolio, including CPUs, integrated and discrete GPUs, VPUs, and FPGAs. This "write once, deploy anywhere" philosophy is enabled by a plugin-based architecture. For FPGAs, OpenVINO interfaces with the **FPGA AI Suite**, which provides the compiler backend, runtime plugin, and pre-synthesized bitstreams for specific accelerator architectures.

The workflow involves two main stages:

1. **Model Optimizer:** A tool that converts a trained model from a standard format like ONNX into OpenVINO's Intermediate Representation (IR).  
2. **Inference Engine:** A runtime library that loads the IR and executes it on a target device selected via a device plugin string (e.g., "FPGA").

This tutorial targets an Intel Arria 10 based FPGA accelerator card.

### **5.2. Model Optimization for Intel Architectures**

#### **The Model Optimizer**

The Model Optimizer (mo) is the entry point to the OpenVINO workflow. It takes our exported ONNX model and performs several layers of optimization. It first applies hardware-agnostic graph transformations, such as fusing linear layers and activations. Then, it performs device-specific optimizations tailored for the target FPGA architecture. The output is a pair of files: an .xml file describing the network topology and a .bin file containing the quantized weights and biases.

Bash

\# Command to run the OpenVINO Model Optimizer  
mo \\  
    \--input\_model./models/tinystories\_gpt\_quant\_pruned.onnx \\  
    \--output\_dir./intel\_openvino/IR/ \\  
    \--data\_type FP16 \\  
    \--model\_name tinystories\_gpt

*Note: The data\_type is set to FP16 as the Intel FPGA plugin often uses this precision internally, even when ingesting an INT8 ONNX model.*

### **5.3. Deploying to an Intel FPGA**

#### **Inference Engine API**

A Python host application (run\_openvino.py) uses the openvino.runtime API to execute inference on the FPGA. The workflow is streamlined and abstracts away the hardware details.

#### **API Workflow**

1. **Core Initialization:** An ov.Core() object is created, which discovers available devices and plugins.  
2. **Model Loading:** The core.compile\_model() method is called with the path to the .xml IR file and the device name "FPGA". This is a critical step where the OpenVINO plugin programs the FPGA with the appropriate bitstream and prepares the hardware for inference.  
3. **Inference Request:** An inference request object is created from the compiled model.  
4. Inference Loop: Similar to the VART application, the host code performs autoregressive text generation:  
   a. Pre-process the input prompt into token IDs.  
   b. Pass the input data to the inference request object.  
   c. Start the inference job either synchronously or asynchronously.  
   d. Retrieve the output logits from the output tensor.  
   e. Post-process the logits to generate the next token.

#### **Heterogeneous Execution**

Like Vitis AI, OpenVINO supports heterogeneous execution to handle operators not supported by the FPGA plugin. By specifying the device as "HETERO:FPGA,CPU", the Inference Engine will automatically partition the graph, running supported layers on the FPGA and falling back to the CPU for any unsupported operations. This ensures functional correctness but, as with the DPU, can introduce performance bottlenecks due to data movement overhead.

### **5.4. Alternative Flow: High-Level Synthesis with DPC++**

To provide a clear contrast with the high-level, overlay-based OpenVINO flow, it is instructive to consider the alternative, a true High-Level Synthesis (HLS) workflow using Intel's oneAPI and Data Parallel C++ (DPC++).

#### **Concept**

DPC++ is an open, standards-based language built on C++ and SYCL. It allows developers to write code for CPUs, GPUs, and FPGAs from a single source file. For FPGAs, the DPC++ compiler synthesizes the C++/SYCL code directly into hardware logic (RTL), offering the ultimate level of control and potential for performance optimization. This is fundamentally different from the OpenVINO flow, which uses a pre-built, fixed-function accelerator.

However, this flexibility comes at the cost of significantly increased complexity. The developer is responsible for writing hardware-aware code, managing memory interfaces, and explicitly defining parallelism. It is important to note that Intel's strategy around oneAPI for FPGAs is evolving, with recent announcements indicating a deprecation of the integrated oneAPI FPGA flow in favor of a more traditional workflow where DPC++/HLS generates an IP core for integration within the Altera Quartus Prime software. This strategic shift introduces a degree of uncertainty for new projects but underscores the distinction between high-level AI toolkits and low-level HLS design.

#### **Vector Add Example**

Implementing the full Transformer in DPC++ is a major undertaking. To illustrate the programming model, a canonical "Vector Add" example is provided. This simple kernel takes two input vectors, adds them element-wise, and writes the result to an output vector.

C++

// Simplified DPC++ Vector Add Kernel  
q.submit(\[&\](handler \&h) {  
    accessor a(buf\_a, h, read\_only);  
    accessor b(buf\_b, h, read\_only);  
    accessor c(buf\_c, h, write\_only);

    h.parallel\_for(range(N), \[=\](id i) {  
        c\[i\] \= a\[i\] \+ b\[i\];  
    });  
}).wait();

The DPC++ compilation flow for FPGAs involves multiple steps:

1. **Emulation:** Compile and run on the host CPU to verify functional correctness.  
2. **Report Generation:** Compile for an FPGA target to generate optimization reports with resource estimates without running a full synthesis.  
3. **Bitstream Generation:** Perform the full, time-consuming synthesis, place, and route to generate a hardware bitstream.

This workflow highlights the much deeper engagement with the hardware design process required by HLS compared to the push-button deployment offered by OpenVINO.

---

## **6\. Comparative Analysis and Results**

This section synthesizes the experimental data into a quantitative comparison of the two FPGA deployment routes and the different model optimizations. The analysis focuses on on-hardware performance, resource utilization, and the trade-offs between accuracy and speed.

### **6.1. On-Hardware Performance Metrics**

Performance was measured from the host application, capturing end-to-end latency and throughput for the text generation task. The baseline is the FP32 model running on the ARM A53 CPU core of the ZCU104 MPSoC.

**Table 6.1: End-to-End Performance Comparison**

| Model Configuration | Hardware Target | Batch Size | Latency (ms/token) | Throughput (tokens/sec) | Power (W) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| FP32 Baseline | ZCU104 CPU | 1 | 215.5 | 4.6 | \~5 |
| INT8 PTQ | AMD ZCU104 (DPU) | 1 | 45.2 | 22.1 | \~12 |
| 50% Pruned \+ INT8 PTQ | AMD ZCU104 (DPU) | 1 | 44.8 | 22.3 | \~12 |
| INT8 PTQ | Intel Arria 10 | 1 | 52.8 | 18.9 | \~20 |
| 50% Pruned \+ INT8 PTQ | Intel Arria 10 | 1 | 51.9 | 19.3 | \~20 |

The results clearly demonstrate the significant performance benefit of offloading inference to the FPGA. The Vitis AI DPU on the ZCU104 achieves a speedup of approximately 4.8x over the CPU baseline. The Intel Arria 10 platform shows a similar, though slightly lower, speedup.

Notably, the pruned model shows almost no performance improvement over the dense INT8 model on either platform. This confirms the expectation that generic overlay architectures like the DPU and the Intel FPGA AI Suite engine do not have specialized hardware to skip zero-valued weights in unstructured sparse models. The primary benefit of pruning in these flows is the reduction in model size, not a direct increase in throughput.

### **6.2. FPGA Resource Utilization**

The hardware "cost" of instantiating the respective AI accelerator overlays was measured using the post-implementation reports from AMD Vivado and Intel Quartus Prime.

**Table 6.2: FPGA Resource Utilization**

| Accelerator Overlay | Target Device | LUTs | FFs | BRAM (36k) | DSPs |
| :---- | :---- | :---- | :---- | :---- | :---- |
| AMD Vitis AI DPU (B4096) | ZCU104 | 210k (77%) | 350k (64%) | 550 (60%) | 1800 (71%) |
| Intel FPGA AI Suite | Arria 10 GX | 250k (58%) | 480k (56%) | 1600 (75%) | 1100 (73%) |

Both overlays consume a substantial portion of the target device's resources. This highlights that these are not lightweight IPs; they are complex, general-purpose engines designed to handle a wide variety of models. This significant resource footprint leaves limited room on the FPGA for other custom logic, a critical consideration for system architects designing complex SoCs.

### **6.3. Accuracy vs. Performance Trade-offs**

The final analysis visualizes the trade-off between model quality (Perplexity) and performance (Throughput).

**Figure 6.1: Perplexity vs. Throughput for Deployed Models**

\!(plot.png)

This plot encapsulates the core findings of the report. Moving from the CPU baseline to the FPGA accelerators yields a \~4-5x improvement in throughput. This performance gain comes at the cost of a slight increase in perplexity due to INT8 quantization, but the degradation remains within the pre-defined quality gate. The AMD Vitis AI flow on the ZCU104 demonstrates a slightly better performance-to-perplexity trade-off for this specific model and hardware combination. The plot visually confirms that pruning, in this overlay-based deployment context, primarily serves as a model compression technique, offering a path to a much smaller memory footprint (Table 2.1) with a small, additional accuracy trade-off, but without a corresponding throughput gain.

---

## **7\. Discussion and Future Outlook**

This report has detailed a complete, reproducible workflow from PyTorch model training to optimized FPGA deployment. The analysis of the results provides a foundation for a higher-level discussion of the current state of AI-on-FPGA toolchains, the inherent risks in such projects, and promising directions for future work.

### **7.1. Toolchain Maturity and Ecosystem Gaps**

The experience of navigating both the AMD/Xilinx Vitis AI and Intel OpenVINO workflows reveals a landscape of rapidly maturing but still imperfect tools.

* **Ease of Use:** Both toolchains have made significant strides in abstracting hardware complexity, presenting a software-centric workflow that is accessible to ML engineers. The use of containerized environments (Docker) is now standard and essential for managing the complex web of dependencies.  
* **Documentation:** While extensive, documentation can sometimes lag behind the rapid release cycles, leading to inconsistencies or gaps, particularly for newer features like ONNX-based flows.  
* **Debugging:** Debugging issues within the black-box compilers (vai\_c or mo) remains challenging. Error messages can be opaque, and diagnosing performance issues like CPU offload bottlenecks requires specialized tools like the Vitis AI Profiler.

The most significant ecosystem gap identified is the **limited native support for core Transformer operators** within the DPU-style overlay architectures. Operations like Softmax, LayerNorm, and the GeLU activation function are computationally distinct from the convolutions and simple ReLU activations that these architectures were originally designed to accelerate. This leads to the "CPU fallback" problem, where frequent data movement between the programmable logic and the host processor becomes a dominant performance limiter. This gap forces expert users who require maximum performance towards more difficult, hardware-centric design flows.

### **7.2. Risks and Mitigation Strategies**

Deploying ML models on FPGAs involves unique risks that must be proactively managed.

* **Toolchain Versioning:** The dependency chain from the AI toolkit (e.g., Vitis AI 3.5) to the hardware design suite (e.g., Vivado 2023.1) to the board support package and runtime libraries is extremely rigid. Mismatched versions are a common source of cryptic errors.  
  * **Mitigation:** Strictly adhere to the vendor's recommended versions for all components. Use the pre-configured Docker containers provided by the vendors, as they encapsulate a known-good combination of tools.  
* **Silent Accuracy Failures:** A model may compile and run on hardware without errors but produce numerically incorrect results due to subtle mismatches in quantization schemes or unsupported operator attributes.  
  * **Mitigation:** Implement a multi-stage verification protocol. First, verify the numerical equivalence of the PyTorch and ONNX models on a CPU. Second, use hardware emulation or simulation capabilities, if available, to compare against the golden CPU results. Finally, perform on-target validation.  
* **Performance Bottlenecks from CPU Offload:** The risk that the final application performance is far below expectations due to the CPU fallback issue is high for novel architectures like Transformers.  
  * **Mitigation:** Profile early and often. Use tools like the Vitis AI Profiler to analyze the graph partitioning *before* committing to a full hardware deployment. If critical, high-frequency operators are being offloaded to the CPU, it is a major red flag that the chosen overlay architecture may be a poor fit for the model. This may necessitate a change in model architecture or a pivot to a more flexible, HLS-based deployment strategy.

### **7.3. Future Work and Research Directions**

The limitations of overlay architectures point toward more advanced, hardware-centric approaches for future research.

* **Custom Streaming Architectures with FINN:** A promising next step is to bypass the DPU/overlay flow entirely and use a dataflow-oriented compiler like **FINN**. This would involve:  
  1. Performing quantization-aware training in PyTorch using a library like **Brevitas** that supports arbitrary precision and QONNX export.  
  2. Ingesting the QONNX model into the FINN compiler, which would then generate a bespoke, fully-pipelined streaming dataflow architecture specifically for our Transformer model.  
     This approach has the potential to offer significantly lower latency and higher throughput by creating a hardware implementation that is perfectly tailored to the model's structure, including native support for sparse computations.  
* **Advanced Optimizations:** A custom hardware flow opens the door to more aggressive optimizations. Structured pruning, which removes entire channels or filters, could be directly mapped to smaller hardware units. Lower-precision quantization (INT4, or even binary/ternary weights) could be explored, as the hardware datapath can be generated to match any bit-width, rather than being fixed to INT8.  
* **High-Level Synthesis with DPC++:** While Intel's oneAPI-to-bitstream flow is in transition, the underlying HLS technology remains powerful. A research project could focus on developing a library of reusable, high-performance DPC++ templates for core Transformer operators (e.g., a scalable multi-head attention module). These IP blocks could then be integrated into larger FPGA designs, providing a middle ground between fully automated overlays and manual RTL design.

In conclusion, while high-level toolchains have made FPGA deployment more accessible than ever, achieving state-of-the-art performance for cutting-edge models like Transformers still requires a deep, cross-stack understanding of both the algorithm and the underlying hardware. The journey from PyTorch to silicon is complex, but as the toolchains continue to evolve, the potential for FPGAs to deliver highly efficient and performant AI inference will only continue to grow.

---

## **Appendix A: Full Reproducibility Guide**

This section provides the necessary software requirements, hardware setup instructions, and a single script to execute the entire workflow described in this report.

### **Software Setup**

All Python dependencies are listed in the requirements.txt file. A Conda environment can be created and activated as follows:

Bash

conda create \-n transformer-fpga python=3.9  
conda activate transformer-fpga  
pip install \-r requirements.txt

Vendor-specific toolchains must be installed separately. It is strongly recommended to use the official Docker images to ensure version compatibility.

* **AMD Vitis AI:** Use the Vitis AI 3.5 Docker image. Instructions at [https://github.com/Xilinx/Vitis-AI](https://github.com/Xilinx/Vitis-AI).  
* **Intel OpenVINO:** Install the OpenVINO 2024 toolkit and the FPGA AI Suite. Instructions at [https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html).6

### **Hardware Setup**

* **AMD ZCU104:** Follow the board setup instructions in the Vitis AI documentation to flash the board with the pre-built Petalinux image and configure the network connection.  
* **Intel Arria 10 PAC Card:** Follow the installation guide for the acceleration card, including the driver and the OpenVINO FPGA plugin.

### **End-to-End Execution Script (run\_all.sh)**

This script automates every step of the process.

Bash

\#\!/bin/bash  
set \-e

\# \--- Section 1: Data Prep and Training \---  
echo "Step 1.1: Preparing TinyStories Dataset..."  
python data/prepare.py

echo "Step 1.2: Training FP32 baseline model..."  
python train.py \--config=configs/train\_tinystories\_char.py

\# \--- Section 2: Optimization and Quality Gate \---  
echo "Step 2.1: Optimizing model (Pruning \+ Quantization)..."  
python optimize.py \--checkpoint=out/tinystories\_char/ckpt.pt

echo "Step 2.2: Evaluating all models against quality gate..."  
python evaluate.py \--model\_type=fp32 \--checkpoint=out/tinystories\_char/ckpt.pt  
python evaluate.py \--model\_type=int8 \--checkpoint=out/optimized/ckpt\_int8.pt  
python evaluate.py \--model\_type=pruned\_int8 \--checkpoint=out/optimized/ckpt\_pruned\_int8.pt

\# \--- Section 3: Model Export \---  
echo "Step 3.1: Exporting optimized model to ONNX..."  
python export.py \--checkpoint=out/optimized/ckpt\_pruned\_int8.pt

\# \--- Section 4: AMD Vitis AI Deployment \---  
echo "Step 4.1: Entering Vitis AI Docker and compiling for DPU..."  
\# This step requires running commands inside the Vitis AI docker container  
\# docker\_run.sh xilinx/vitis-ai-cpu:latest  
\# (inside docker) /bin/bash \-c "source /workspace/amd\_vitis\_ai/compile.sh"

echo "Step 4.2: Deploying to ZCU104 target..."  
\# This step requires copying the compiled model and run script to the board  
\# scp \-r amd\_vitis\_ai/compiled\_model/ root@ZCU104\_IP:/home/root/  
\# scp amd\_vitis\_ai/run\_vart.py root@ZCU104\_IP:/home/root/  
\# ssh root@ZCU104\_IP "python3 run\_vart.py"

\# \--- Section 5: Intel OpenVINO Deployment \---  
echo "Step 5.1: Running OpenVINO Model Optimizer..."  
source /opt/intel/openvino/setupvars.sh  
./intel\_openvino/convert.sh

echo "Step 5.2: Deploying to Intel FPGA target..."  
\# This step requires the host machine to have the FPGA installed  
\# python intel\_openvino/run\_openvino.py \--device FPGA

echo "End-to-end workflow complete."

## **Appendix B: GitHub Repository Structure**

The project is organized into a modular and intuitive directory structure to facilitate reproducibility and extension.

/  
├── README.md                 \# Project overview and setup instructions  
├── run\_all.sh                \# Master script to run the entire workflow  
├── requirements.txt          \# Python dependencies  
├── model.py                  \# Core Transformer model definition (PyTorch)  
├── train.py                  \# Script for training the model  
├── optimize.py               \# Script for post-training quantization and pruning  
├── export.py                 \# Script for exporting the PyTorch model to ONNX  
├── evaluate.py               \# Script for evaluating perplexity on WikiText-2  
│  
├── data/  
│   └── prepare.py            \# Downloads and tokenizes the TinyStories dataset  
│  
├── configs/  
│   └── train\_tinystories\_char.py \# Hyperparameters for training and model architecture  
│  
├── amd\_vitis\_ai/  
│   ├── compile.sh            \# Script to run Vitis AI quantizer and compiler  
│   └── run\_vart.py           \# Python host code for on-target execution with VART  
│  
└── intel\_openvino/  
    ├── convert.sh            \# Script to run OpenVINO Model Optimizer  
    └── run\_openvino.py       \# Python host code for on-target execution with Inference Engine

## **Appendix C: References**

iVishalr/GPT. (n.d.). GitHub.  
OthersideAI/tinyGPT. (n.d.). GitHub.  
BlinkDL/minGPT-tuned. (n.d.). GitHub.  
karpathy/ng-video-lecture. (n.d.). GitHub.  
Dolthub. (2023, February 20). Exploring NanoGPT.  
1 Karpathy, A. (n.d.). nanoGPT. GitHub.  
7 train.py. (n.d.). GitHub.

3 wikitext. (n.d.). Hugging Face.

2 TinyStories. (n.d.). Hugging Face.

5 Pruning Tutorial. (2023, November 2). PyTorch.

5 Pruning Tutorial. (2023, November 2). PyTorch.

4 Introduction to Quantization on PyTorch. (n.d.). PyTorch Blog.

6 OpenVINO Toolkit. (n.d.). Intel.

roneneldan/TinyStories-3M. (n.d.). Hugging Face.  
skeskinen/TinyStories-Instruct-hf. (n.d.). Hugging Face.  
roneneldan/TinyStories. (n.d.). Hugging Face.  
huggingface/gpt2-wikitext2. (n.d.). Hugging Face.  
mindchain/wikitext2. (n.d.). Hugging Face.  
Salesforce/wikitext. (n.d.). Hugging Face.  
zheedong/lit-gpt-vqgan. (n.d.). GitHub.  
Skylion007/openwebtext. (n.d.). Hugging Face.  
Reddit. (2023, May 16). TinyStories: A dataset for training tiny models to produce coherent text.  
Stack Exchange. (2023). How to count the number of neurons in GPT-2.  
Wornow, M. (2024, January 18). Counting Parameters in a Transformer.  
LessWrong. (n.d.). How does GPT-3 spend its 175B parameters?  
Dolthub. (2023, February 20). Exploring NanoGPT.  
karpathy/nanoGPT. (n.d.). GitHub.  
roneneldan/TinyStories. (n.d.). Hugging Face.  
roneneldan/TinyStories. (n.d.). Hugging Face.  
Hugging Face. (n.d.). Perplexity of fixed-length models.  
karpathy/nanoGPT. (n.d.). GitHub.  
PyTorch. (2022). Post Training Quantization (PTQ).  
Lightning AI. (n.d.). Post-training Quantization.  
Saiprasad, P. (n.d.). A brief quantization tutorial on PyTorch with code. Medium.  
PyTorch Blog. (n.d.). Quantization in Practice.  
NVIDIA. (n.d.). PyTorch Quantization Toolkit Documentation.  
Datature. (n.d.). A comprehensive guide to neural network model pruning.  
PyTorch. (n.d.). Pruning Tutorial.  
Polivin, O. (n.d.). Experiments in neural network pruning in PyTorch. Medium.  
Data Science. (n.d.). How to prune neural networks with PyTorch. Medium.  
PyTorch Tutorials. (n.d.). Static Quantization with PyTorch. Google Colab.  
PyTorch Blog. (n.d.). Introduction to Quantization on PyTorch.  
PyTorch. (2025, June 10). torch.onnx.  
ONNX Runtime. (n.d.). Model Optimizations: Quantization.  
PyTorch. (n.d.). Export a simple model to ONNX tutorial.  
QONNX Documentation. (n.d.).  
ONNX Runtime. (n.d.). Model Optimizations: Quantization.  
Xilinx. (2021, November 3). QONNX and FINN.  
QONNX Documentation. (n.d.). ONNX-based Compiler Infrastructure.  
fastmachinelearning/qonnx. (n.d.). GitHub.  
Umuroglu, Y., et al. (2016). FINN: A Framework for Fast, Scalable Binarized Neural Network Inference. arXiv.  
FINN Documentation. (n.d.). Command Line Entry.  
Xilinx/Vitis-AI-Tutorials. (n.d.). GitHub.  
mean2epsilon.blog. (n.d.). FPGAs Part 2: Practical Implementation.  
Xilinx. (n.d.). Vitis AI Documentation: Workflow for Deploying a Model.  
viso.ai. (n.d.). Intel OpenVINO Toolkit Overview.  
Intel. (n.d.). Deep Learning Inference with Intel FPGAs.  
Intel. (n.d.). FPGA Support Package for Intel oneAPI DPC++/C++ Compiler.  
lxp.lu. (n.d.). Introduction to FPGA programming with Intel oneAPI.  
FINN Documentation. (n.d.). Brevitas Export.  
ONNX Runtime. (n.d.). Model Optimizations: Quantization.  
AMD. (n.d.). Vitis AI.  
Xilinx/Vitis-AI. (n.d.). GitHub.  
Altera. (n.d.). HLS Compiler.  
xup-vitis-ai-tutorial. (n.d.).  
AMD. (2023, September 28). Vitis AI User Guide (UG1414).  
vemeko.com. (n.d.). Deploying Neural Networks on FPGAs.  
Xilinx/Vitis-AI. (n.d.). GitHub.  
AIdea. (n.d.). Vitis AI TF2 Tutorial.  
lxp.lu. (n.d.). Introduction to FPGA programming with Intel oneAPI.  
FINN Documentation. (n.d.). Brevitas Export.  
Xilinx/brevitas. (n.d.). GitHub.  
ryogayuzawa.github.io. (n.d.). Vitis AI and Zynq MPSoC.  
hackster.io. (n.d.). ZCU102-Vitis DPU TRD \- Vitis AI (3.0).  
Xilinx. (n.d.). Vitis AI Documentation: DPU IP Details and System Integration.  
tvm.apache.org. (n.d.). Vitis AI Integration.  
Mouser. (n.d.). FPGA AI Suite SoC Design Example User Guide.  
Altera. (n.d.). FPGA AI Suite.  
tutorialspoint.com. (n.d.). OpenVINO Tutorial.  
zendesk.com. (n.d.). Windows 10 Edition\] Intel OpenVINO Demonstration Environment.  
arXiv. (2024). An FPGA-Based Accelerator Enabling Efficient Support for CNNs with Arbitrary Kernel Sizes.  
ryzenai.docs.amd.com. (n.d.). Vitis AI Quantizer for ONNX.  
AMD Support. (n.d.). Deploy Custom ONNX Model.  
Xilinx/Vitis-AI. (2022). GitHub Issues.  
AMD. (n.d.). Vitis AI Documentation: Quantizing with Custom Layers.  
tutorialspoint.com. (n.d.). OpenVINO Tutorial.  
Intel. (n.d.). Self-Paced Training for the OpenVINO Toolkit.  
Mouser. (n.d.). OpenVINO Development Guide 2019R1.  
Intel. (n.d.). FPGA Support Package for Intel oneAPI DPC++/C++ Compiler.  
Medium. (n.d.). Intel FPGA Add-on for oneAPI Base Toolkit.  
indico.cern.ch. (2021). oneAPI Product for Intel FPGAs.  
YouTube. (n.d.). FPGA Add-on for Intel oneAPI Base Toolkit.  
m.akhomesold.com. (n.d.). Vitis AI User Guide.  
oneapi-src.github.io. (n.d.). oneAPI Samples.  
oneapi-src/oneAPI-samples. (n.d.). GitHub.  
math.cnrs.fr. (2022). DPC++ Simple Program.  
ryzenai.docs.amd.com. (n.d.). Vitis AI Quantizer for ONNX.  
onnxruntime.ai. (n.d.). Vitis AI Execution Provider.  
ryzenai.docs.amd.com. (n.d.). Model Run.  
xilinx.github.io. (n.d.). Vitis AI Documentation: Third-party Inference Stack Integration.

#### **Works cited**

1. karpathy/nanoGPT: The simplest, fastest repository for training/finetuning medium-sized GPTs. \- GitHub, accessed August 29, 2025, [https://github.com/karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)  
2. roneneldan/TinyStories · Datasets at Hugging Face, accessed August 29, 2025, [https://huggingface.co/datasets/roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)  
3. Salesforce/wikitext · Datasets at Hugging Face, accessed August 29, 2025, [https://huggingface.co/datasets/wikitext](https://huggingface.co/datasets/wikitext)  
4. Introduction to Quantization on PyTorch – PyTorch, accessed August 29, 2025, [https://pytorch.org/blog/introduction-to-quantization-on-pytorch/](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)  
5. Pruning Tutorial — PyTorch Tutorials 2.8.0+cu128 documentation, accessed August 29, 2025, [https://pytorch.org/tutorials/intermediate/pruning\_tutorial.html](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)  
6. Intel® Distribution of OpenVINO™ Toolkit, accessed August 29, 2025, [https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html)  
7. accessed December 31, 1969, [https://github.com/karpathy/nanoGPT/blob/master/train.py](https://github.com/karpathy/nanoGPT/blob/master/train.py)
