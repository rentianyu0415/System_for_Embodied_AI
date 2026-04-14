# System for Embodied AI

这是一个课程项目仓库，内容分成 3 个独立任务。现在三个任务都已经完成。仓库按任务拆开组织，每个任务都有自己单独的目录，这样看代码、看说明、找实验材料都会更直接。

## 项目总览

本项目包含下面三个部分：

- 任务 1：用 PyTorch 从零实现 Transformer
- 任务 2：阅读 `llama.cpp` 文档，并在本地部署基于 LoRA 的 Llama2 7B 模型
- 任务 3：实现模型加速，并对不同量化结果做对比

## 仓库结构

```text
System_for_Embodied_AI/
├── README.md
├── .gitignore
├── task1_transformer/
│   └── transformer.py
├── task2_llama/
│   ├── docs/
│   ├── flagalpha_base_cfg/
│   ├── llama-b8771-bin-win-cuda-12.4-x64/
│   └── llama.cpp/
├── task3_acceleration/
│   ├── llama.cpp/
│   ├── models/
│   └── scripts/
└── task3_results_visualization.png
```

## 每个目录是做什么的

### 根目录

- `README.md`
  - 仓库总说明，也就是您现在看到的这个文件。
- `task3_results_visualization.png`
  - 任务 3 的总览结果图，用来展示不同模型版本的显存占用和生成速度。

### task1_transformer

```text
task1_transformer/
└── transformer.py
```

这个目录对应任务 1，内容最集中，核心代码都在 `transformer.py` 里。

文件说明：

- `transformer.py`
  - 用 PyTorch 从零实现 Transformer 的主要模块。
  - 包含位置编码、多头注意力、前馈网络、编码器、解码器和完整的 Transformer。
  - 文件底部带了一个最小运行示例，可以直接用来检查输出形状是否正确。

如果您想先看模型代码，建议从这里开始。

### task2_llama

```text
task2_llama/
├── docs/
├── flagalpha_base_cfg/
├── llama-b8771-bin-win-cuda-12.4-x64/
└── llama.cpp/
```

这个目录对应任务 2，目标是理解 `llama.cpp` 的本地推理流程，并完成 LoRA 版 Llama2 7B 的本地部署。

子目录说明：

- `docs/`
  - 预留给任务 2 的部署说明、环境说明和实验记录。
- `flagalpha_base_cfg/`
  - 基础模型配置文件目录。
  - 里面放了 `config.json`、`generation_config.json`、`tokenizer.model` 等文件。
  - 这些文件用来补足模型配置和分词器信息。
- `llama-b8771-bin-win-cuda-12.4-x64/`
  - Windows + CUDA 12.4 环境下使用的 `llama.cpp` 二进制程序目录。
  - 里面有 `llama-cli.exe`、`llama-bench.exe`、`llama-quantize.exe` 等工具。
  - `llama2_custom.jinja` 也放在这里，用来定义聊天模板。
  - 这个目录本质上是本地运行环境。
- `llama.cpp/`
  - `llama.cpp` 源码目录。
  - 用来阅读文档、查看工具实现和对照本地推理流程。

从结构上看，任务 2 可以分成三部分：

- 一部分是源码和文档，用来理解工具怎么工作。
- 一部分是本地运行程序，用来真正执行推理命令。
- 一部分是模型配置文件，用来保证模型和 LoRA 能正常加载。

### task3_acceleration

```text
task3_acceleration/
├── llama.cpp/
├── models/
└── scripts/
```

这个目录对应任务 3，目标是在任务 2 的基础上做加速实验，并对比不同量化方案的效果。

子目录说明：

- `llama.cpp/`
  - 任务 3 使用的 `llama.cpp` 源码目录。
  - 主要用来查看量化、推理和工具相关代码。
- `models/`
  - 存放任务 3 使用的模型文件。
  - 当前包括全精度版本和多个量化版本，例如 `llama2-7B-f16.gguf`、`llama2-7B-Q8_0.gguf`、`llama2-7B-Q4_K_M.gguf`、`llama2-7B-Q2_K.gguf`。
  - 这个目录对应实验输入。
- `scripts/`
  - 存放任务 3 的结果处理脚本和图片。
  - `plot_model_comparison.py` 用来生成模型对比图。
  - `model_comparison.png` 是脚本生成的图像结果。

从结构上看，任务 3 也可以分成三部分：

- 一部分是工具源码，用来支撑量化和推理。
- 一部分是模型文件，用来跑不同方案。
- 一部分是分析脚本和结果图，用来展示实验结果。

## 三个任务之间的关系

这三个任务是按学习和实验顺序安排的：

1. 先在任务 1 里手写 Transformer，理解模型基础结构。
2. 再在任务 2 里使用 `llama.cpp` 做本地部署，理解大模型推理流程和 LoRA 加载方法。
3. 最后在任务 3 里继续做量化实验，对比加速前后的效果。

所以这个仓库既有模型基础实现，也有本地部署实验，还有加速对比结果。

## 运行环境

- Windows
- Python
- PyTorch
- `numpy`
- `matplotlib`
- CUDA 12.4
- `llama.cpp`
