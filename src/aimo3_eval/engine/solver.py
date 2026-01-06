import json
import time
import re
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from openai import OpenAI
from aimo3_eval.engine.sandbox import AIMO3Sandbox

# 为了类型提示，防止运行时循环引用
if TYPE_CHECKING:
    from aimo3_eval.config import CFG

# --- 定义工具 Schema ---
PYTHON_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Execute Python code in a stateful Jupyter notebook environment. Use print() to see outputs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute. Variables are preserved between calls."
                    }
                },
                "required": ["code"]
            }
        }
    }
]

class AIMO3Solver:
    def __init__(self, cfg: "CFG"):
        self.cfg = cfg
        
        # 根据模式选择连接地址
        if self.cfg.mode == 'remote':
            print(f"🌐 Connecting to Remote API: {self.cfg.remote_model_name}")
            api_key = self.cfg.remote_api_key
            base_url = self.cfg.remote_base_url
            self.target_model = self.cfg.remote_model_name
        else:
            print(f"🏠 Connecting to Local vLLM: {self.cfg.served_model_name}")
            api_key = "sk-local"
            base_url = f"http://localhost:{self.cfg.port}/v1"
            self.target_model = self.cfg.served_model_name

        # 初始化客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=cfg.timeout_per_problem
        )
        
        # 初始化 Sandbox Pool
        self.sandbox_pool = queue.Queue()
        self._init_sandboxes()

    def _init_sandboxes(self):
        print(f"🔧 Initializing {self.cfg.workers} sandboxes...")
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as exe:
            # 传递 timeout 参数
            futures = [exe.submit(AIMO3Sandbox, timeout=10.0) for _ in range(self.cfg.workers)]
            for f in as_completed(futures):
                self.sandbox_pool.put(f.result())
        print("✅ Sandboxes ready.")

    def solve(self, problem: str, problem_id: str) -> Dict[str, Any]:
        """
        Orchestrator: 并发执行多次尝试 (Maj@k)
        """
        start_time = time.time()
        attempts_data = []
        
        # 并行执行 k 次采样
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as executor:
            futures = []
            for i in range(self.cfg.attempts):
                futures.append(executor.submit(self._run_single_attempt, problem, i))
                
            for future in as_completed(futures):
                attempts_data.append(future.result())

        # 简单的答案聚合 (Extract Final Answer)
        valid_answers = [a['final_answer'] for a in attempts_data if a['final_answer'] is not None]
        
        # 众数投票 (Majority Vote)
        if valid_answers:
            from collections import Counter
            final_consensus = Counter(valid_answers).most_common(1)[0][0]
        else:
            final_consensus = None

        return {
            "id": problem_id,
            "problem": problem,
            "final_answer": final_consensus,
            "attempts": attempts_data,  # 保存完整轨迹供复盘
            "time_taken": time.time() - start_time
        }

    def _run_single_attempt(self, problem: str, attempt_idx: int) -> Dict[str, Any]:
        """
        Core Logic: 单次 TIR (Tool-Integrated Reasoning) 循环
        """
        sandbox = self.sandbox_pool.get() # 从池中获取沙箱
        
        messages = [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user", "content": problem}
        ]
        
        final_answer = None
        turn_count = 0
        
        try:
            # --- The Main Loop ---
            # 使用 getattr 获取 max_turns，如果没有配置则默认 7
            max_turns = getattr(self.cfg, 'max_turns', 7)
            
            while turn_count < max_turns:
                turn_count += 1
                
                # 1. 调用 LLM
                try:
                    response = self.client.chat.completions.create(
                        model=self.target_model,
                        messages=messages,
                        tools=PYTHON_TOOL,
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens,
                        # stop=["<|im_end|>"] # 如果需要可以取消注释
                    )
                    message = response.choices[0].message
                except Exception as e:
                    messages.append({"role": "system", "content": f"Error: {str(e)}"})
                    break

                # 2. 将模型的回复加入历史 (注意：这里加入的是 OpenAI 对象)
                messages.append(message)

                # 3. 检查是否有工具调用 (Tool Calls)
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "python_interpreter":
                            # A. 解析代码
                            try:
                                arguments = json.loads(tool_call.function.arguments)
                                code = arguments.get("code", "")
                            except json.JSONDecodeError:
                                code = ""
                                output = "Error: Invalid JSON format in tool arguments."

                            # B. 沙箱执行
                            if code:
                                try:
                                    output = sandbox.execute(code)
                                    if len(output) > 2000:
                                        output = output[:2000] + "\n...[Output Truncated]"
                                except Exception as e:
                                    output = f"Execution Error: {str(e)}"
                            else:
                                output = "Error: No code provided."

                            # C. 将结果追加回消息列表 (Role: Tool)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": output
                            })
                
                # 4. 如果没有工具调用，检查是否已经得出答案
                else:
                    content = message.content or ""
                    if "\\boxed{" in content:
                        extracted = self._extract_boxed_content(content)
                        if extracted:
                            final_answer = extracted
                            break
                    
        except Exception as e:
            print(f"Critical Error in attempt {attempt_idx}: {e}")
        finally:
            sandbox.reset()
            self.sandbox_pool.put(sandbox)

        # === 关键修复：清洗 messages 对象，防止 Polars 报错 ===
        clean_messages = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                # OpenAI v1.x+ (Pydantic V2)
                clean_messages.append(msg.model_dump())
            elif hasattr(msg, "to_dict"):
                # 旧版本或某些兼容库
                clean_messages.append(msg.to_dict())
            elif isinstance(msg, dict):
                clean_messages.append(msg)
            else:
                # 兜底转换
                try:
                    clean_messages.append(dict(msg))
                except:
                    # 最后的防线：转字符串
                    clean_messages.append({"role": "unknown", "content": str(msg)})

        return {
            "attempt_id": attempt_idx,
            "final_answer": final_answer,
            "messages": clean_messages  # <--- 返回清洗后的字典列表
        }

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """
        简单的正则提取，Phase 2 已有更强的 Extractor，这里只是 Loop 内的快速检查
        """
        matches = re.findall(r'\\boxed\s*\{(.*?)\}', text)
        if matches:
            return matches[-1]
        return None

    def cleanup(self):
        """Shut down sandboxes"""
        while not self.sandbox_pool.empty():
            try:
                sb = self.sandbox_pool.get_nowait()
                sb.close()
            except:
                pass