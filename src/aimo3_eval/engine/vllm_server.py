import subprocess
import sys
import time
import os
import requests
from aimo3_eval.config import CFG

class VLLMServer:
    def __init__(self, cfg: CFG):
        self.cfg = cfg
        self.process = None
        self.log_file = None

    def start(self):
        """启动 vLLM 服务"""
        print(f"🚀 Starting vLLM server on port {self.cfg.port}...")
        
        cmd = [
            sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', self.cfg.model_path,
            '--served-model-name', self.cfg.served_model_name,
            '--tensor-parallel-size', str(self.cfg.tensor_parallel_size),
            '--gpu-memory-utilization', str(self.cfg.gpu_memory_utilization),
            '--port', str(self.cfg.port),
            '--trust-remote-code',
            '--disable-log-requests' # 减少日志干扰
        ]
        
        if self.cfg.max_model_len is not None:
            cmd.extend(['--max-model-len', str(self.cfg.max_model_len)])
        
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.log_file = open(os.path.join(self.cfg.output_dir, 'vllm_server.log'), 'w')
        
        self.process = subprocess.Popen(
            cmd,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True # 确保作为独立进程组运行
        )
        
        self._wait_for_ready()

    def _wait_for_ready(self):
        base_url = f"http://localhost:{self.cfg.port}/v1/models"
        start_time = time.time()
        
        while True:
            if time.time() - start_time > self.cfg.server_timeout:
                self.stop()
                raise TimeoutError("vLLM Server start timeout")
                
            if self.process.poll() is not None:
                raise RuntimeError("vLLM Server process exited unexpectedly")
                
            try:
                resp = requests.get(base_url)
                if resp.status_code == 200:
                    print(f"✅ vLLM Server ready in {time.time() - start_time:.1f}s")
                    return
            except requests.exceptions.ConnectionError:
                pass
            
            time.sleep(2)

    def stop(self):
        """安全关闭服务"""
        if self.process:
            print("🛑 Stopping vLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutError:
                self.process.kill()
        
        if self.log_file:
            self.log_file.close()