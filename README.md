# AIMO3 Eval 框架 🚀

> 适配新版代码的统一评测与运行说明（中文/English）

A lightweight evaluation framework for the Kaggle AIMO3 competition, supporting both **local** and **remote** model inference modes.

---

## 📋 Table of Contents

- [English](#english)
- [中文](#中文)

---

## English

### 🎯 Features

- ✅ **Dual Mode Support**: Run with local vLLM server or remote APIs (OpenAI, DeepSeek, etc.)
- ✅ **Flexible Data Loading**: Support CSV files or custom problem lists
- ✅ **Multiple Evaluation Metrics**: Pass@1, Majority@k, and more
- ✅ **Parallel Processing**: Concurrent problem solving with configurable workers
- ✅ **Mathematical Grading**: Intelligent answer extraction and equivalence checking using SymPy
- ✅ **Result Recording**: Automatic parquet export with detailed attempt logs
- ✅ **Easy Integration**: Simple, clean API with sensible defaults

### 🏗️ Architecture

```
aimo3_eval/
├── config.py           # 🔧 Configuration management
├── engine/
│   ├── runner.py      # 🏃 Main evaluation orchestrator
│   ├── solver.py      # 🧠 Problem solver implementation
│   ├── sandbox.py     # 📦 Safe code execution environment
│   └── vllm_server.py # 🖥️ Local vLLM server manager
├── data/
│   └── loader.py      # 📥 Data loading utilities
└── metrics/
    ├── evaluator.py   # 📊 Metrics calculation
    ├── extractor.py   # 🔍 Answer extraction from responses
    └── math_utils.py  # 🧮 Mathematical operations
```

### 🚀 Quick Start

#### 1️⃣ Installation

```bash
# Clone the repository
git clone <repo-url>
cd aimo3_eval

# Install dependencies (requires Python 3.11+)
uv pip install -e .
# or
pip install -e .
```

#### 2️⃣ Configuration

Set up your environment:

```bash
# For remote mode (OpenAI/DeepSeek API)
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.deepseek.com"  # or your provider's URL

# For local mode, set model path (optional)
export MODEL_PATH="/path/to/model"
```

#### 3️⃣ Basic Usage

```python
from aimo3_eval import CFG, CoTSolver, TIRSolver, DataLoader, EvalRunner

# Configure
cfg = CFG(
    mode='remote',
    remote_model_name="deepseek-reasoner",
    remote_base_url="https://api.deepseek.com",
    remote_api_key="your-key",
    attempts=8,  # Number of tries per problem
    workers=4    # Parallel workers
)

# Load data (map Kaggle's 'answer' to 'ground_truth')
df = DataLoader.load_csv(
    "data/reference.csv",
    id_col="id",
    problem_col="problem",
    ground_truth_col="answer",
)
# or create custom dataset:
# df = DataLoader.load_custom_data(
#     problems=["What is 2+2?"],
#     ids=["demo_1"],
#     ground_truths=["4"]
# )

# Solve and evaluate (choose one solver)
solver = TIRSolver(cfg)
# solver = CoTSolver(cfg)
runner = EvalRunner(cfg, solver)
results = runner.load_data(df).run()
```

### 🔧 Configuration Options

The `CFG` class provides fine-grained control:

```python
cfg = CFG(
    # Mode: 'local' or 'remote'
    mode='remote',
  
    # Remote API settings
    remote_base_url="https://api.deepseek.com",
    remote_api_key="your-key",
    remote_model_name="deepseek-reasoner",
  
    # Local model settings (for mode='local')
    model_path="/path/to/model",
    gpu_memory_utilization=0.90,
    tensor_parallel_size=1,
  
    # Inference parameters
    temperature=0.7,
    top_p=0.95,
    max_tokens=16384,
  
    # Evaluation settings
    attempts=8,           # Number of attempts per problem (for Maj@k)
    workers=8,           # Concurrent workers
    timeout_per_problem=300,
  
    # Output
    output_dir="./outputs"
)
```

### 📊 Output Format

Results are saved in `outputs/{timestamp}/`:

- **attempts.parquet**: Detailed log of all attempts with answers
- **times.parquet**: Timing statistics
- **metrics.json**: Final metrics (Pass@1, Maj@k, etc.)

### 🎓 Metrics Explained

- **Pass@1**: Percentage of correct answers on first attempt
- **Maj@k**: Majority voting accuracy across k attempts
- **Attempt Statistics**: Min/max/average solving time per problem

### 💡 Advanced Usage

#### Local Mode with Custom Model

```python
cfg = CFG(
    mode='local',
    model_path="/path/to/your/model",
    served_model_name="my-model",
    tensor_parallel_size=2,  # For multi-GPU
)
server = VLLMServer(cfg)
server.start()
```

#### Custom Prompts

```python
cfg.system_prompt = "Your custom system prompt..."
cfg.tool_prompt = "Your tool usage instructions..."
```

### ⚙️ Dependencies

- `python>=3.11`
- `openai>=2.14.0` - API client
- `polars>=1.36.1` - Data processing
- `sympy>=1.14.0` - Mathematical operations
- `jupyter>=1.1.1` - Interactive notebooks

### 📝 Notes

- 🔒 The framework uses a sandboxed environment for code execution
- 🧮 Mathematical equivalence is checked using SymPy (e.g., `sqrt(4)` == `2`)
- ⏱️ Each problem has a configurable timeout to prevent hanging
- 🎯 System prompt is optimized for IMO-style problems

### 🤝 Contributing

Contributions welcome! Please ensure code follows the existing structure and includes proper documentation.

---

## 中文

### 🎯 功能特性

- ✅ **双模式支持**：支持本地 vLLM 服务或远程 API（OpenAI、DeepSeek 等）
- ✅ **灵活的数据加载**：支持 CSV 文件或自定义问题列表
- ✅ **多种评估指标**：Pass@1、Majority@k 等
- ✅ **并行处理**：可配置的并发 Worker 数
- ✅ **数学评分**：使用 SymPy 进行智能答案提取和等价性检查
- ✅ **结果记录**：自动导出 parquet 格式的详细日志
- ✅ **易于集成**：简洁的 API 和合理的默认配置

### 🏗️ 项目结构

```
aimo3_eval/
├── config.py           # 🔧 配置管理
├── engine/
│   ├── runner.py      # 🏃 主评估协调器
│   ├── solver.py      # 🧠 问题求解器
│   ├── sandbox.py     # 📦 安全代码执行环境
│   └── vllm_server.py # 🖥️ 本地 vLLM 服务器管理
├── data/
│   └── loader.py      # 📥 数据加载工具
└── metrics/
    ├── evaluator.py   # 📊 指标计算
    ├── extractor.py   # 🔍 答案提取
    └── math_utils.py  # 🧮 数学运算
```

### 🚀 快速开始

#### 1️⃣ 安装

```bash
# 克隆仓库
git clone <repo-url>
cd aimo3_eval

# 安装依赖（需要 Python 3.11+）
uv pip install -e .
# 或
pip install -e .
```

#### 2️⃣ 配置环境

```powershell
# 远程模式（OpenAI/DeepSeek API）
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_API_BASE = "https://api.deepseek.com"

# 本地模式（可选）
$env:MODEL_PATH = "C:\\path\\to\\model"
```

#### 3️⃣ 基础用法

```python
from aimo3_eval import CFG, CoTSolver, TIRSolver, DataLoader, EvalRunner

# 配置
cfg = CFG(
    mode='remote',
    remote_model_name="deepseek-reasoner",
    remote_base_url="https://api.deepseek.com",
    remote_api_key="your-key",
    attempts=8,    # 每个问题的尝试次数
    workers=4      # 并行 Worker 数
)

# 加载数据（若原列为 answer，会自动映射为 ground_truth）
df = DataLoader.load_csv("data/reference.csv", id_col="id", problem_col="problem", ground_truth_col="answer")
# 或自定义数据集：
# df = DataLoader.load_custom_data(
#     problems=["2+2等于多少？"],
#     ids=["demo_1"],
#     ground_truths=["4"]
# )

# 求解并评估（二选一）
solver = TIRSolver(cfg)
# solver = CoTSolver(cfg)
runner = EvalRunner(cfg, solver)
results = runner.load_data(df).run()
```

### 🔧 配置说明

`CFG` 类提供精细的控制选项：

```python
cfg = CFG(
    # 模式：'local' 或 'remote'
    mode='remote',
  
    # 远程 API 设置
    remote_base_url="https://api.deepseek.com",
    remote_api_key="your-key",
    remote_model_name="deepseek-reasoner",
  
    # 本地模型设置（mode='local' 时使用）
    model_path="/path/to/model",
    gpu_memory_utilization=0.90,
    tensor_parallel_size=1,
  
    # 推理参数
    temperature=0.7,
    top_p=0.95,
    max_tokens=16384,
  
    # 评估设置
    attempts=8,              # 每个问题的尝试次数（用于 Maj@k）
    workers=8,              # 并发 Worker 数
    timeout_per_problem=300,
  
    # 输出路径
    output_dir="./outputs"
)
```

### 📊 输出格式

结果保存在 `outputs/{时间戳}/` 目录下：

- **attempts.parquet**: 所有尝试的详细日志
- **times.parquet**: 时间统计数据
- **metrics.json**: 最终指标（Pass@1、Maj@k 等）

### 🎓 指标说明

- **Pass@1**: 第一次尝试成功的百分比
- **Maj@k**: 在 k 次尝试中投票正确的百分比
- **Attempt Statistics**: 每个问题的求解时间统计

### 💡 高级用法

#### 本地模式与自定义模型

```python
cfg = CFG(
    mode='local',
    model_path="/path/to/your/model",
    served_model_name="my-model",
    tensor_parallel_size=2,  # 多 GPU 并行
)
server = VLLMServer(cfg)
server.start()
```

#### 自定义提示词

```python
cfg.system_prompt = "你的自定义系统提示词..."
cfg.tool_prompt = "你的工具使用说明..."
```

### ⚙️ 依赖项

- `python>=3.11`
- `openai>=2.14.0` - API 客户端
- `polars>=1.36.1` - 数据处理
- `sympy>=1.14.0` - 数学运算
- `jupyter>=1.1.1` - 交互式笔记本

### 📝 注意事项

- 🔒 框架使用沙箱环境执行代码，确保安全性
- 🧮 数学等价性通过 SymPy 检查（例如 `sqrt(4)` == `2`）
- ⏱️ 每个问题都有可配置的超时时间以防止卡顿
- 🎯 系统提示词已针对 IMO 风格的问题进行优化

### 🤝 参与贡献

欢迎贡献代码！请确保代码遵循现有结构并包含适当的文档说明。

---

**Generated by GitHub Copilot**

