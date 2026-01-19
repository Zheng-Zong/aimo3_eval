import json
import time
import re
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from collections import Counter

from openai import OpenAI
from aimo3_eval.engine.sandbox import AIMO3Sandbox

# Harmony 模板支持（可选导入）
try:
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        SystemContent,
        ReasoningEffort,
        ToolNamespaceConfig,
        Author,
        Message,
        Role,
        TextContent,
        Conversation
    )
    HARMONY_AVAILABLE = True
except ImportError:
    HARMONY_AVAILABLE = False

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

class TIRSolver:
    def __init__(self, cfg: "CFG"):
        self.cfg = cfg
        
        # 根据模式选择连接地址
        if self.cfg.mode == 'remote':
            print(f"🌐 Connecting to Remote API: {self.cfg.remote.model}")
            api_key = self.cfg.remote.api_key
            base_url = self.cfg.remote.base_url
            self.target_model = self.cfg.remote.model
        else:
            print(f"🏠 Connecting to Local vLLM: {self.cfg.local.served_model_name}")
            api_key = "sk-local"
            base_url = f"http://localhost:{self.cfg.local.port}/v1"
            self.target_model = self.cfg.local.served_model_name

        # 初始化客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=cfg.solver.timeout_per_problem
        )
        
        # 初始化 Sandbox Pool
        self.sandbox_pool = queue.Queue()
        self._init_sandboxes()

    def _init_sandboxes(self):
        print(f"🔧 Initializing {self.cfg.solver.workers} sandboxes...")
        with ThreadPoolExecutor(max_workers=self.cfg.solver.workers) as exe:
            # 传递 timeout 参数
            futures = [exe.submit(AIMO3Sandbox, timeout=30) for _ in range(self.cfg.solver.workers)]
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
        with ThreadPoolExecutor(max_workers=self.cfg.solver.workers) as executor:
            futures = []
            for i in range(self.cfg.solver.attempts):
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

        # 计算所有 attempts 的时间统计
        attempt_times = [a['time_taken'] for a in attempts_data]
        min_time = min(attempt_times) if attempt_times else 0
        max_time = max(attempt_times) if attempt_times else 0
        avg_time = sum(attempt_times) / len(attempt_times) if attempt_times else 0

        return {
            "id": problem_id,
            "problem": problem,
            "final_answer": final_consensus,
            "attempts": attempts_data,  # 保存完整轨迹供复盘
            "min_attempt_time": min_time,  # 最短 attempt 时间
            "max_attempt_time": max_time,  # 最长 attempt 时间
            "avg_attempt_time": avg_time   # 平均 attempt 时间
        }

    def _run_single_attempt(self, problem: str, attempt_idx: int) -> Dict[str, Any]:
        """
        Core Logic: 单次 TIR (Tool-Integrated Reasoning) 循环
        """
        attempt_start_time = time.time()  # 记录 attempt 开始时间
        sandbox = self.sandbox_pool.get() # 从池中获取沙箱
        
        messages = [
            {"role": "system", "content": self.cfg.prompts.system_prompt},
            {"role": "user", "content": problem}
        ]
        
        final_answer = None
        turn_count = 0
        
        # 统计信息
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        
        try:
            # --- The Main Loop ---
            # 使用 getattr 获取 max_turns，如果没有配置则默认 7
            max_turns = self.cfg.inference.max_turns
            
            while turn_count < max_turns:
                turn_count += 1
                
                # 1. 调用 LLM
                try:
                    completion_kwargs = {
                        'model': self.target_model,
                        'messages': messages,
                        'tools': PYTHON_TOOL,
                        'temperature': self.cfg.inference.temperature,
                        'top_p': self.cfg.inference.top_p,
                    }
                    if self.cfg.inference.max_tokens is not None:
                        completion_kwargs['max_tokens'] = self.cfg.inference.max_tokens
                    
                    # 添加额外的推理参数
                    if self.cfg.inference.extra:
                        completion_kwargs.update(self.cfg.inference.extra)
                    
                    response = self.client.chat.completions.create(**completion_kwargs)
                    message = response.choices[0].message
                    
                    # 累加 tokens
                    if hasattr(response, 'usage') and response.usage:
                        total_tokens += response.usage.total_tokens
                        
                except Exception as e:
                    messages.append({"role": "system", "content": f"Error: {str(e)}"})
                    break

                # 2. 将模型的回复加入历史，转为纯字典，避免下次请求携带 SDK 对象
                messages.append(self._normalize_message(message))

                # 3. 检查是否有工具调用 (Tool Calls)
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "python_interpreter":
                            python_calls += 1  # 记录 Python 调用次数
                            
                            # A. 解析代码
                            try:
                                arguments = json.loads(tool_call.function.arguments)
                                code = arguments.get("code", "")
                            except json.JSONDecodeError:
                                code = ""
                                output = "Error: Invalid JSON format in tool arguments."
                                python_errors += 1

                            # B. 沙箱执行
                            if code:
                                try:
                                    output = sandbox.execute(code)
                                    if len(output) > 2000:
                                        output = output[:2000] + "\n...[Output Truncated]"
                                except Exception as e:
                                    output = f"Execution Error: {str(e)}"
                                    python_errors += 1  # 记录执行错误
                            else:
                                output = "Error: No code provided."
                                python_errors += 1

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
            "messages": clean_messages,  # <--- 返回清洗后的字典列表
            "time_taken": time.time() - attempt_start_time,  # 记录该 attempt 的耗时
            "python_calls": python_calls,  # Python 调用次数
            "python_errors": python_errors,  # Python 错误次数
            "total_tokens": total_tokens  # 总 token 使用量
        }

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """
        简单的正则提取，Phase 2 已有更强的 Extractor，这里只是 Loop 内的快速检查
        """
        matches = re.findall(r'\\boxed\s*\{(.*?)\}', text)
        if matches:
            return matches[-1]
        return None

    def _normalize_message(self, msg: Any) -> Dict[str, Any]:
        """确保消息为纯 dict，避免 ChatCompletionMessage 等 SDK 对象在下一轮调用出错"""
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        if hasattr(msg, "to_dict"):
            return msg.to_dict()
        if isinstance(msg, dict):
            return msg
        try:
            return dict(msg)
        except Exception:
            return {"role": "unknown", "content": str(msg)}

    def cleanup(self):
        """Shut down sandboxes"""
        while not self.sandbox_pool.empty():
            try:
                sb = self.sandbox_pool.get_nowait()
                sb.close()
            except:
                pass


class CoTSolver:
    """
    纯 Chain-of-Thought Solver，不使用任何工具。
    模型直接通过推理得出答案。
    """
    
    def __init__(self, cfg: "CFG"):
        self.cfg = cfg
        
        # 根据模式选择连接地址
        if self.cfg.mode == 'remote':
            print(f"🌐 [CoT] Connecting to Remote API: {self.cfg.remote.model}")
            api_key = self.cfg.remote.api_key
            base_url = self.cfg.remote.base_url
            self.target_model = self.cfg.remote.model
        else:
            print(f"🏠 [CoT] Connecting to Local vLLM: {self.cfg.local.served_model_name}")
            api_key = "sk-local"
            base_url = f"http://localhost:{self.cfg.local.port}/v1"
            self.target_model = self.cfg.local.served_model_name

        # 初始化客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=cfg.solver.timeout_per_problem
        )

    def solve(self, problem: str, problem_id: str) -> Dict[str, Any]:
        """
        Orchestrator: 并发执行多次尝试 (Maj@k)
        """
        start_time = time.time()
        attempts_data = []
        
        # 并行执行 k 次采样
        with ThreadPoolExecutor(max_workers=self.cfg.solver.workers) as executor:
            futures = []
            for i in range(self.cfg.solver.attempts):
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

        # 计算所有 attempts 的时间统计
        attempt_times = [a['time_taken'] for a in attempts_data]
        min_time = min(attempt_times) if attempt_times else 0
        max_time = max(attempt_times) if attempt_times else 0
        avg_time = sum(attempt_times) / len(attempt_times) if attempt_times else 0

        return {
            "id": problem_id,
            "problem": problem,
            "final_answer": final_consensus,
            "attempts": attempts_data,
            "min_attempt_time": min_time,
            "max_attempt_time": max_time,
            "avg_attempt_time": avg_time
        }

    def _run_single_attempt(self, problem: str, attempt_idx: int) -> Dict[str, Any]:
        """
        Core Logic: 单次纯 CoT 推理，不使用工具
        """
        attempt_start_time = time.time()
        
        messages = [
            {"role": "system", "content": self.cfg.prompts.system_prompt},
            {"role": "user", "content": problem}
        ]
        
        final_answer = None
        total_tokens = 0
        reasoning_content = ""
        response_content = ""
        
        try:
            # 单次 LLM 调用，不使用工具
            completion_kwargs = {
                'model': self.target_model,
                'messages': messages,
                'temperature': self.cfg.inference.temperature,
                'top_p': self.cfg.inference.top_p,
            }
            if self.cfg.inference.max_tokens is not None:
                completion_kwargs['max_tokens'] = self.cfg.inference.max_tokens

            # 追加额外推理参数（与 TIRSolver 行为一致）
            if self.cfg.inference.extra:
                completion_kwargs.update(self.cfg.inference.extra)
            
            response = self.client.chat.completions.create(**completion_kwargs)
            message = response.choices[0].message
            
            # 累加 tokens
            if hasattr(response, 'usage') and response.usage:
                total_tokens = response.usage.total_tokens
            
            # 获取推理内容（如果模型支持，如 DeepSeek-R1）
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning_content = message.reasoning_content
            
            # 获取回复内容
            response_content = message.content or ""
            
            # 将模型回复加入历史
            messages.append(self._normalize_message(message))
            
            # 提取答案
            if "\\boxed{" in response_content:
                extracted = self._extract_boxed_content(response_content)
                if extracted:
                    final_answer = extracted
                        
        except Exception as e:
            print(f"Critical Error in CoT attempt {attempt_idx}: {e}")
            messages.append({"role": "system", "content": f"Error: {str(e)}"})

        # 清洗 messages 对象
        clean_messages = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                clean_messages.append(msg.model_dump())
            elif hasattr(msg, "to_dict"):
                clean_messages.append(msg.to_dict())
            elif isinstance(msg, dict):
                clean_messages.append(msg)
            else:
                try:
                    clean_messages.append(dict(msg))
                except:
                    clean_messages.append({"role": "unknown", "content": str(msg)})

        return {
            "attempt_id": attempt_idx,
            "final_answer": final_answer,
            "messages": clean_messages,
            "time_taken": time.time() - attempt_start_time,
            "total_tokens": total_tokens
        }

    def _extract_boxed_content(self, text: str) -> Optional[str]:
        """简单的正则提取 \\boxed{} 内容"""
        matches = re.findall(r'\\boxed\s*\{(.*?)\}', text)
        if matches:
            return matches[-1]
        return None

    def _normalize_message(self, msg: Any) -> Dict[str, Any]:
        """确保消息为纯 dict"""
        if hasattr(msg, "model_dump"):
            return msg.model_dump()
        if hasattr(msg, "to_dict"):
            return msg.to_dict()
        if isinstance(msg, dict):
            return msg
        try:
            return dict(msg)
        except Exception:
            return {"role": "unknown", "content": str(msg)}

    def cleanup(self):
        """CoT Solver 不需要清理资源，但保持接口一致"""
        pass


# ============================================================================
# Harmony TIR Solver - 使用 Harmony 模板 + Completion 端口
# ============================================================================

class HarmonyTemplate:
    """Harmony 模板处理器"""
    
    def get_system_content(self, system_prompt: str, tool_config: "ToolNamespaceConfig") -> "SystemContent":
        return (
            SystemContent.new()
            .with_model_identity(system_prompt)
            .with_reasoning_effort(reasoning_effort=ReasoningEffort.HIGH)
            .with_tools(tool_config)
        )

    def apply_chat_template(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_config: "ToolNamespaceConfig"
    ) -> List["Message"]:
        system_content = self.get_system_content(system_prompt, tool_config)
        system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        user_message = Message.from_role_and_content(Role.USER, user_prompt)
        return [system_message, user_message]


class HarmonyTool:
    """Harmony 工具处理器 - 处理 Python 代码执行"""
    
    def __init__(self, tool_prompt: str, sandbox: AIMO3Sandbox, jupyter_timeout: float = 30.0):
        self._tool_prompt = tool_prompt
        self._jupyter_session = sandbox
        self._jupyter_timeout = jupyter_timeout
        self._execution_lock = threading.Lock()

    def _ensure_last_print(self, code: str) -> str:
        """确保最后一行有 print 输出"""
        lines = code.strip().split('\n')
        if not lines:
            return code
        
        last_line = lines[-1].strip()
        
        # 不需要处理的情况
        if not last_line or 'print' in last_line or 'import' in last_line or last_line.startswith('#'):
            return code
        
        lines[-1] = f'print({last_line})'
        return '\n'.join(lines)

    @property
    def instruction(self) -> str:
        return self._tool_prompt

    @property
    def tool_config(self) -> "ToolNamespaceConfig":
        return ToolNamespaceConfig(
            name='python',
            description=self.instruction,
            tools=[]
        )

    def _make_response(self, output: str, channel: Optional[str] = None) -> "Message":
        """创建工具响应消息"""
        content = TextContent(text=output)
        author = Author(role=Role.TOOL, name='python')
        message = Message(author=author, content=[content]).with_recipient('assistant')
        if channel:
            message = message.with_channel(channel)
        return message

    def process_sync(self, message: "Message") -> List["Message"]:
        """同步处理工具调用"""
        raw_script = message.content[0].text
        final_script = self._ensure_last_print(raw_script)
        
        with self._execution_lock:
            try:
                output = self._jupyter_session.execute(final_script)
                if len(output) > 2000:
                    output = output[:2000] + "\n...[Output Truncated]"
            except Exception as e:
                output = f'[ERROR] {str(e)}'
        
        return [self._make_response(output, channel=message.channel)]


class HarmonyTIRSolver:
    """
    使用 Harmony 模板的 TIR Solver。
    通过 completion 端口与 vLLM 通信，手动处理消息编码。
    专为 GPT-OSS 等需要 Harmony 模板的模型设计。
    """
    
    def __init__(self, cfg: "CFG"):
        if not HARMONY_AVAILABLE:
            raise ImportError(
                "openai_harmony 未安装。请运行: pip install openai-harmony"
            )
        
        self.cfg = cfg
        
        # 根据模式选择连接地址
        if self.cfg.mode == 'remote':
            print(f"🌐 [Harmony] Connecting to Remote API: {self.cfg.remote.model}")
            api_key = self.cfg.remote.api_key
            base_url = self.cfg.remote.base_url
            self.target_model = self.cfg.remote.model
        else:
            print(f"🏠 [Harmony] Connecting to Local vLLM: {self.cfg.local.served_model_name}")
            api_key = "sk-local"
            base_url = f"http://localhost:{self.cfg.local.port}/v1"
            self.target_model = self.cfg.local.served_model_name

        # 初始化客户端
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=cfg.solver.timeout_per_problem
        )
        
        # 初始化 Harmony 编码和模板
        encoding_name = "HARMONY_GPT_OSS"
        self.encoding = load_harmony_encoding(encoding_name)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.template = HarmonyTemplate()
        
        # Harmony 特定配置
        self.context_tokens = self.cfg.harmony.context_tokens
        self.search_tokens = self.cfg.harmony.search_tokens
        self.buffer_tokens = self.cfg.harmony.buffer_tokens
        self.min_p = self.cfg.harmony.min_p
        self.stream_interval = self.cfg.harmony.stream_interval
        
        # 初始化 Sandbox Pool
        self.sandbox_pool = queue.Queue()
        self._init_sandboxes()

    def _init_sandboxes(self):
        """初始化沙箱池"""
        print(f"🔧 [Harmony] Initializing {self.cfg.solver.workers} sandboxes...")
        with ThreadPoolExecutor(max_workers=self.cfg.solver.workers) as exe:
            futures = [exe.submit(AIMO3Sandbox, timeout=30) for _ in range(self.cfg.solver.workers)]
            for f in as_completed(futures):
                self.sandbox_pool.put(f.result())
        print("✅ [Harmony] Sandboxes ready.")

    def solve(self, problem: str, problem_id: str) -> Dict[str, Any]:
        """
        Orchestrator: 并发执行多次尝试 (Maj@k)
        """
        start_time = time.time()
        attempts_data = []
        
        # 并行执行 k 次采样
        with ThreadPoolExecutor(max_workers=self.cfg.solver.workers) as executor:
            futures = []
            for i in range(self.cfg.solver.attempts):
                futures.append(executor.submit(self._run_single_attempt, problem, i))
                
            for future in as_completed(futures):
                attempts_data.append(future.result())

        # 答案聚合
        valid_answers = [a['final_answer'] for a in attempts_data if a['final_answer'] is not None]
        
        # 众数投票 (Majority Vote)
        if valid_answers:
            final_consensus = Counter(valid_answers).most_common(1)[0][0]
        else:
            final_consensus = None

        # 计算时间统计
        attempt_times = [a['time_taken'] for a in attempts_data]
        min_time = min(attempt_times) if attempt_times else 0
        max_time = max(attempt_times) if attempt_times else 0
        avg_time = sum(attempt_times) / len(attempt_times) if attempt_times else 0

        return {
            "id": problem_id,
            "problem": problem,
            "final_answer": final_consensus,
            "attempts": attempts_data,
            "min_attempt_time": min_time,
            "max_attempt_time": max_time,
            "avg_attempt_time": avg_time
        }

    def _run_single_attempt(self, problem: str, attempt_idx: int) -> Dict[str, Any]:
        """
        Core Logic: 单次 Harmony TIR 循环
        使用 completion 端口和流式处理
        """
        attempt_start_time = time.time()
        sandbox = self.sandbox_pool.get()
        
        # 初始化工具
        local_tool = HarmonyTool(
            tool_prompt=self.cfg.prompts.tool_prompt,
            sandbox=sandbox,
            jupyter_timeout=30.0
        )
        
        final_answer = None
        python_calls = 0
        python_errors = 0
        total_tokens = 0
        
        # 计算 attempt 的种子
        attempt_seed = int(self.cfg.solver.seed + attempt_idx) ** 2
        
        try:
            # 构建初始消息
            messages = self.template.apply_chat_template(
                self.cfg.prompts.system_prompt,
                problem,
                local_tool.tool_config
            )
            conversation = Conversation.from_messages(messages)
            
            max_turns = self.cfg.inference.max_turns
            
            for turn in range(max_turns):
                # 将会话渲染为 token IDs
                prompt_ids = self.encoding.render_conversation_for_completion(
                    conversation, Role.ASSISTANT
                )
                max_tokens = self.context_tokens - len(prompt_ids)
                
                if max_tokens < self.buffer_tokens:
                    print(f"⚠️ [Harmony] Attempt {attempt_idx}: Context exhausted")
                    break
                
                # 调用 completion 端口（流式）
                try:
                    stream = self.client.completions.create(
                        model=self.target_model,
                        temperature=self.cfg.inference.temperature,
                        max_tokens=max_tokens,
                        prompt=prompt_ids,
                        seed=attempt_seed,
                        stream=True,
                        extra_body={
                            'min_p': self.min_p,
                            'stop_token_ids': self.stop_token_ids,
                            'return_token_ids': True
                        }
                    )
                except Exception as e:
                    print(f"⚠️ [Harmony] Stream creation failed: {e}")
                    break
                
                # 处理流式响应
                token_buffer = []
                text_chunks = []
                
                try:
                    for chunk in stream:
                        choice = chunk.choices[0]
                        new_tokens = getattr(choice, 'token_ids', None)
                        new_text = choice.text
                        
                        if new_tokens:
                            token_buffer.extend(new_tokens)
                            total_tokens += len(new_tokens)
                            text_chunks.append(new_text)
                        
                        # 检查是否有 boxed 答案
                        if '}' in new_text:
                            search_text = ''.join(text_chunks[-self.search_tokens:])
                            answer = self._scan_for_answer(search_text)
                            if answer is not None:
                                final_answer = answer
                                break
                finally:
                    stream.close()
                
                if final_answer is not None:
                    break
                
                if not token_buffer:
                    break
                
                # 解析新消息
                new_messages = self.encoding.parse_messages_from_completion_tokens(
                    token_buffer, Role.ASSISTANT
                )
                conversation.messages.extend(new_messages)
                last_message = new_messages[-1]
                
                # 检查是否是最终答案
                if last_message.channel == 'final':
                    answer_text = last_message.content[0].text
                    final_answer = self._scan_for_answer(answer_text)
                    break
                
                # 检查是否需要调用工具
                if last_message.recipient == 'python':
                    python_calls += 1
                    tool_responses = local_tool.process_sync(last_message)
                    
                    response_text = tool_responses[0].content[0].text
                    if response_text.startswith('[ERROR]') or 'Traceback' in response_text or 'Error:' in response_text:
                        python_errors += 1
                    
                    conversation.messages.extend(tool_responses)
                    
        except Exception as e:
            print(f"Critical Error in Harmony attempt {attempt_idx}: {e}")
            python_errors += 1
        finally:
            sandbox.reset()
            self.sandbox_pool.put(sandbox)

        # 清洗消息用于返回
        clean_messages = self._clean_conversation(conversation)

        return {
            "attempt_id": attempt_idx,
            "final_answer": final_answer,
            "messages": clean_messages,
            "time_taken": time.time() - attempt_start_time,
            "python_calls": python_calls,
            "python_errors": python_errors,
            "total_tokens": total_tokens
        }

    def _scan_for_answer(self, text: str) -> Optional[int]:
        """扫描文本中的 \\boxed{} 答案"""
        pattern = r'\\boxed\s*\{\s*([0-9,]+)\s*\}'
        matches = re.findall(pattern, text)
        
        if matches:
            try:
                clean_value = matches[-1].replace(',', '')
                value = int(clean_value)
                if 0 <= value <= 99999:
                    return value
            except ValueError:
                pass
        return None

    def _clean_conversation(self, conversation: "Conversation") -> List[Dict[str, Any]]:
        """将 Conversation 对象转换为可序列化的字典列表"""
        clean_messages = []
        for msg in conversation.messages:
            try:
                if hasattr(msg, 'model_dump'):
                    clean_messages.append(msg.model_dump())
                elif hasattr(msg, 'to_dict'):
                    clean_messages.append(msg.to_dict())
                else:
                    # 手动构建字典
                    content_text = ""
                    if hasattr(msg, 'content') and msg.content:
                        if hasattr(msg.content[0], 'text'):
                            content_text = msg.content[0].text
                        else:
                            content_text = str(msg.content)
                    
                    role = "unknown"
                    if hasattr(msg, 'author') and hasattr(msg.author, 'role'):
                        role = str(msg.author.role.value) if hasattr(msg.author.role, 'value') else str(msg.author.role)
                    
                    clean_messages.append({
                        "role": role,
                        "content": content_text
                    })
            except Exception:
                clean_messages.append({"role": "unknown", "content": str(msg)})
        return clean_messages

    def cleanup(self):
        """关闭沙箱"""
        while not self.sandbox_pool.empty():
            try:
                sb = self.sandbox_pool.get_nowait()
                sb.close()
            except:
                pass