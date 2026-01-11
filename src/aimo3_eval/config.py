import os
import json
from dataclasses import dataclass, fields

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
    max_model_len: int | None = None
    
    # --- Inference Config ---
    temperature: float = 0.7  # 稍微降低一点，提高 pass@1 稳定性，maj@k 时可调高
    top_p: float = 0.95
    max_tokens: int | None = None    # 单次生成的最大长度，None 表示使用模型默认值
    max_turns: int = 50
    
    # --- Server Config ---
    port: int = 8000
    server_timeout: int = 300 # 等待 vLLM 启动的秒数
    
    # --- Tree Search / Solver Config ---
    attempts: int = 8       # 每个问题尝试多少次 (Maj@k 的 k)
    workers: int = 16       # 并发 Worker 数
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
    
    # --- Extra Kwargs ---
    vllm_server_kwargs: dict | None = None  # 传递给 vLLM 启动的额外参数
    inference_kwargs: dict | None = None    # 传递给推理 API 的额外参数
    
    @classmethod
    def from_json(cls, path: str, api_key: str | None = None) -> "CFG":
        """
        从 JSON 文件加载配置。
        
        Args:
            path: config.json 文件路径
            api_key: API Key（remote 模式必须提供，因为保存的配置中 API Key 已被隐藏）
            
        Returns:
            CFG 实例
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 分离 CFG 预定义字段和额外的 kwargs
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        extra_kwargs = {k: v for k, v in data.items() if k not in valid_fields}
        
        # 如果 JSON 中没有显式定义 vllm_server_kwargs 和 inference_kwargs，
        # 将所有额外参数作为 inference_kwargs
        if 'vllm_server_kwargs' not in filtered_data and 'inference_kwargs' not in filtered_data:
            if extra_kwargs:
                filtered_data['inference_kwargs'] = extra_kwargs
        
        # 处理 remote 模式下的 API Key
        if filtered_data.get('mode') == 'remote':
            if api_key:
                filtered_data['remote_api_key'] = api_key
            elif filtered_data.get('remote_api_key') in ('***', '', None):
                raise ValueError(
                    "Remote 模式需要提供 API Key。"
                    "请使用 CFG.from_json(path, api_key='your-api-key') 传入 API Key。"
                )
        
        return cls(**filtered_data)