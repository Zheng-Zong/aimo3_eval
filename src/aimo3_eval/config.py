import os
import json
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Literal


@dataclass
class RemoteConfig:
    """远程 API 配置（仅在 mode=remote 时使用）"""
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-reasoner"


@dataclass
class LocalConfig:
    """本地 vLLM 配置（仅在 mode=local 时使用）"""
    model_path: str = os.getenv("MODEL_PATH", "/kaggle/input/gpt-oss-120b/transformers/default/1")
    served_model_name: str = "gpt-oss"
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    port: int = 8000
    server_timeout: int = 300
    vllm_server_kwargs: dict | None = None


@dataclass
class InferenceConfig:
    """推理相关配置"""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int | None = None
    max_turns: int = 50
    extra: dict | None = None


@dataclass
class SolverConfig:
    """求解器并发与超时配置"""
    attempts: int = 8
    workers: int = 16
    timeout_per_problem: int = 300
    seed: int = 42
    solver_type: str | None = None


@dataclass
class PromptConfig:
    """提示词配置"""
    system_prompt: str = (
        "You are a world-class International Mathematical Olympiad (IMO) competitor. "
        "The final answer must be a non-negative integer between 0 and 99999. "
        "You must place the final integer answer inside \\boxed{}."
    )
    tool_prompt: str = (
        "Use this tool to execute Python code. "
        "You must use print() to output results."
    )


@dataclass
class HarmonyConfig:
    """Harmony 专用配置（仅 HarmonyTIRSolver 使用）"""
    # encoding: str = "HARMONY_GPT_OSS"
    context_tokens: int = 65536
    search_tokens: int = 1024
    buffer_tokens: int = 512
    min_p: float = 0.02
    stream_interval: int = 200


@dataclass
class CFG:
    """统一配置入口（已分组）"""
    mode: Literal["remote", "local"] = "remote"
    output_dir: str = "./outputs"
    mask_api_key: bool = True
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    harmony: HarmonyConfig = field(default_factory=HarmonyConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.mode not in ("remote", "local"):
            raise ValueError("mode 必须是 'remote' 或 'local'")
        if self.mode == "remote":
            if not self.remote.api_key:
                raise ValueError("remote.api_key 不能为空")
            if not self.remote.model:
                raise ValueError("remote.model 不能为空")
            if not self.remote.base_url:
                raise ValueError("remote.base_url 不能为空")
        if self.mode == "local":
            if not self.local.model_path:
                raise ValueError("local.model_path 不能为空")
            if not self.local.served_model_name:
                raise ValueError("local.served_model_name 不能为空")

    def _drop_empty(self, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned = {}
            for k, v in value.items():
                v = self._drop_empty(v)
                if v is None:
                    continue
                if v == {} or v == []:
                    continue
                cleaned[k] = v
            return cleaned
        if isinstance(value, list):
            cleaned = [self._drop_empty(v) for v in value]
            return [v for v in cleaned if v is not None]
        return value

    def to_dict(self, mask_sensitive: bool | None = None) -> dict:
        """只导出当前模式需要的参数，并清理空值。"""
        if mask_sensitive is None:
            mask_sensitive = self.mask_api_key
        data: dict[str, Any] = {
            "mode": self.mode,
            "output_dir": self.output_dir,
            "mask_api_key": self.mask_api_key,
            "inference": asdict(self.inference),
            "solver": asdict(self.solver),
            "prompts": asdict(self.prompts),
        }

        if self.mode == "remote":
            data["remote"] = asdict(self.remote)
        else:
            data["local"] = asdict(self.local)

        if self.solver.solver_type and "Harmony" in self.solver.solver_type:
            data["harmony"] = asdict(self.harmony)

        if mask_sensitive and "remote" in data:
            if "api_key" in data["remote"]:
                data["remote"]["api_key"] = ""

        return self._drop_empty(data)

    @classmethod
    def from_json(cls, path: str, api_key: str | None = None) -> "CFG":
        """从 JSON 加载（新格式）。支持脱敏 api_key。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = cls()

        def update_dataclass(obj: Any, patch: dict) -> None:
            for key, value in patch.items():
                if not hasattr(obj, key):
                    continue
                current = getattr(obj, key)
                if is_dataclass(current) and isinstance(value, dict):
                    update_dataclass(current, value)
                else:
                    setattr(obj, key, value)

        if isinstance(data, dict):
            update_dataclass(cfg, data)

        if cfg.mode == "remote":
            if api_key:
                cfg.remote.api_key = api_key
            elif cfg.remote.api_key in ("", None):
                env_key = os.getenv("OPENAI_API_KEY", "")
                if env_key:
                    cfg.remote.api_key = env_key
                    print("🔑 Loaded remote.api_key from environment variable OPENAI_API_KEY.")

        cfg.validate()
        return cfg