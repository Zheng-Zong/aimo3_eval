"""
统一的评估运行器，封装整个评估流程。
"""
import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Protocol, runtime_checkable

import polars as pl

from aimo3_eval.config import CFG
from aimo3_eval.metrics.math_utils import MathGrader
from aimo3_eval.metrics.evaluator import evaluate_attempts_parquet


@runtime_checkable
class SolverProtocol(Protocol):
    """Solver 协议，所有 Solver 必须实现这个接口"""
    def solve(self, problem: str, problem_id: str) -> Dict[str, Any]:
        """
        求解单个问题。
        
        Returns:
            包含以下字段的字典:
            - final_answer: 最终答案
            - attempts: 所有尝试的列表
            - min_attempt_time, max_attempt_time, avg_attempt_time: 时间统计
        """
        ...
    
    def cleanup(self) -> None:
        """清理资源"""
        ...


class ResultRecorder:
    """结果记录器，负责保存和管理评估结果"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.attempts_path = os.path.join(output_dir, "attempts.parquet")
        self.times_path = os.path.join(output_dir, "times.parquet")
        
        self.all_attempts: List[Dict[str, Any]] = []
        self.all_times: List[Dict[str, Any]] = []
    
    def add_result(
        self,
        problem_id: str,
        problem: str,
        ground_truth: str,
        solve_result: Dict[str, Any]
    ) -> bool:
        """
        添加一个问题的求解结果。
        
        Args:
            problem_id: 问题 ID
            problem: 问题内容
            ground_truth: 正确答案
            solve_result: Solver 返回的结果
            
        Returns:
            最终答案是否正确
        """
        final_answer = str(solve_result['final_answer']) if solve_result['final_answer'] is not None else ''
        is_correct = MathGrader.is_equiv(final_answer, ground_truth) if ground_truth else False
        
        # 处理每个 attempt
        for attempt in solve_result['attempts']:
            attempt_answer = str(attempt.get('final_answer', ''))
            attempt_is_correct = MathGrader.is_equiv(attempt_answer, ground_truth) if ground_truth else False
            
            attempt_record = {
                "attempt_id": attempt['attempt_id'],
                "problem_id": problem_id,
                "problem": problem,
                "solution": str(attempt.get('messages', [])),
                "answer": attempt_answer,
                "ground_truth": ground_truth,
                "isCorrect": attempt_is_correct,
                "time": attempt.get('time_taken', 0.0),
                "python_calls": attempt.get('python_calls', 0),
                "python_errors": attempt.get('python_errors', 0)
            }
            self.all_attempts.append(attempt_record)
        
        # 添加时间汇总
        time_record = {
            "problem_id": problem_id,
            "problem": problem,
            "min_attempt_time": solve_result['min_attempt_time'],
            "max_attempt_time": solve_result['max_attempt_time'],
            "avg_attempt_time": solve_result['avg_attempt_time']
        }
        self.all_times.append(time_record)
        
        return is_correct
    
    def save(self) -> tuple[str, str]:
        """保存所有结果到 parquet 文件"""
        if self.all_attempts:
            attempts_df = pl.DataFrame(self.all_attempts)
            attempts_df.write_parquet(self.attempts_path)
        
        if self.all_times:
            times_df = pl.DataFrame(self.all_times)
            times_df.write_parquet(self.times_path)
        
        return self.attempts_path, self.times_path


class EvalRunner:
    """
    统一的评估运行器。
    
    Example:
        ```python
        cfg = CFG(mode='remote', remote_api_key='sk-xxx')
        solver = AIMO3Solver(cfg)
        
        runner = EvalRunner(cfg, solver)
        runner.load_data(df)  # 或 runner.load_csv(path)
        results = runner.run()
        ```
    """
    
    def __init__(
        self,
        cfg: CFG,
        solver: SolverProtocol,
        server: Optional[Any] = None,
        run_name: Optional[str] = None
    ):
        """
        初始化运行器。
        
        Args:
            cfg: 配置对象
            solver: 实现了 SolverProtocol 的求解器
            server: 可选的服务器实例（如 VLLMServer），用于本地模式
            run_name: 运行名称，用于输出目录。如果不提供，自动生成
        """
        self.cfg = cfg
        self.solver = solver
        self.server = server
        self.df: Optional[pl.DataFrame] = None
        
        # 构建输出目录
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = cfg.remote_model_name if cfg.mode == 'remote' else cfg.served_model_name
            model_name_clean = model_name.replace('/', '_').replace(':', '_')
            run_name = f"{cfg.mode}_{model_name_clean}_{timestamp}"
        
        self.output_dir = os.path.join(cfg.output_dir, run_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.recorder = ResultRecorder(self.output_dir)
    
    def load_data(self, df: pl.DataFrame) -> "EvalRunner":
        """
        加载数据。
        
        Args:
            df: 包含 id, problem, ground_truth 列的 DataFrame
            
        Returns:
            self (支持链式调用)
        """
        self.df = df
        return self
    
    def load_csv(
        self,
        path: str,
        id_col: str = "id",
        problem_col: str = "problem",
        ground_truth_col: str = "answer"
    ) -> "EvalRunner":
        """
        从 CSV 加载数据。
        
        Returns:
            self (支持链式调用)
        """
        from aimo3_eval.data.loader import DataLoader
        self.df = DataLoader.load_csv(path, id_col, problem_col, ground_truth_col)
        return self
    
    def run(
        self,
        save_interval: int = 1,
        evaluate_after: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行评估流程。
        
        Args:
            save_interval: 每处理多少个问题保存一次结果
            evaluate_after: 是否在完成后自动评估
            verbose: 是否打印详细信息
            
        Returns:
            评估结果字典（如果 evaluate_after=True）
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() or load_csv() first.")
        
        if verbose:
            print(f"📂 Output directory: {self.output_dir}")
            print(f"📚 Loaded {len(self.df)} problems.")
            print("🚀 Start Inference...")
        
        try:
            for idx, row in enumerate(self.df.iter_rows(named=True), 1):
                problem_id = row['id']
                problem = row['problem']
                ground_truth = str(row.get('ground_truth', ''))
                
                if verbose:
                    print(f"[{idx}/{len(self.df)}] Processing {problem_id}...")
                
                # 求解
                result = self.solver.solve(problem, problem_id)
                
                # 记录结果
                is_correct = self.recorder.add_result(
                    problem_id, problem, ground_truth, result
                )
                
                # 打印结果
                if verbose:
                    final_answer = result['final_answer']
                    print(f" -> Answer: {final_answer} | Ground Truth: {ground_truth} | Correct: {is_correct}")
                    print(f" -> Time - Min: {result['min_attempt_time']:.2f}s | "
                          f"Avg: {result['avg_attempt_time']:.2f}s | "
                          f"Max: {result['max_attempt_time']:.2f}s")
                
                # 定期保存
                if idx % save_interval == 0:
                    self.recorder.save()
                    if verbose:
                        print(f" ✅ Saved!")
            
            # 最终保存
            attempts_path, times_path = self.recorder.save()
            
            if verbose:
                print(f"\n✅ Final results saved to:")
                print(f"   - {attempts_path}")
                print(f"   - {times_path}")
            
            # 评估
            if evaluate_after:
                metrics = evaluate_attempts_parquet(
                    attempts_path,
                    use_math_equiv=True,
                    verbose=verbose
                )
                
                # 保存 metrics.json（仅包含最终指标）
                metrics_json = {
                    "Pass@1": metrics['acc_pass@1'],
                    "Pass@k": metrics['acc_pass@k'],
                    "Maj@k": metrics['acc_maj@k']
                }
                metrics_path = os.path.join(self.output_dir, "metrics.json")
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_json, f, indent=2, ensure_ascii=False)
                
                if verbose:
                    print(f"   - {metrics_path}")
                
                return metrics
            
            return {"attempts_path": attempts_path, "times_path": times_path}
            
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """清理资源"""
        if hasattr(self.solver, 'cleanup'):
            self.solver.cleanup()
        if self.server is not None and hasattr(self.server, 'stop'):
            self.server.stop()
