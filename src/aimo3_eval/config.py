import os
from dataclasses import dataclass

@dataclass
class CFG:
    # --- Mode Selection ---
    # 选项: 'local' (启动本地vLLM) 或 'remote' (连接外部API)
    mode: str = "remote"

    # --- Remote API Config (仅当 mode='remote' 时生效) ---
    remote_base_url: str = "https://api.deepseek.com" # 例如 DeepSeek 或 OpenAI
    remote_api_key: str = "xxx"
    remote_model_name: str = "deepseek-reasoner" # 远程模型的名字

    # --- Local Model Config (仅当 mode='local' 时生效) ---
    model_path: str = os.getenv('MODEL_PATH', '/kaggle/input/gpt-oss-120b/transformers/default/1')
    served_model_name: str = 'gpt-oss' # 本地服务的名字
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1  # 注意：大模型本地跑可能需要调整
    
    # --- Inference Config ---
    temperature: float = 0.7  # 稍微降低一点，提高 pass@1 稳定性，maj@k 时可调高
    top_p: float = 0.95
    max_tokens: int = 16384    # 单次生成的最大长度
    max_turns: int = 50
    
    # --- Server Config ---
    port: int = 8000
    server_timeout: int = 300 # 等待 vLLM 启动的秒数
    
    # --- Tree Search / Solver Config ---
    attempts: int = 8       # 每个问题尝试多少次 (Maj@k 的 k)
    workers: int = 8       # 并发 Worker 数
    timeout_per_problem: int = 300
    
    # --- Prompts ---
    system_prompt: str = (
        'You are a world-class International Mathematical Olympiad (IMO) competitor. '
        'The final answer must be a non-negative integer between 0 and 99999. '
        'You must place the final integer answer inside \\boxed{}.'
    )
    tool_prompt: str = (
        'Use this tool to execute Python code. '
        'You must use print() to output results.'
    )
    
    # --- Paths ---
    output_dir: str = "./outputs"