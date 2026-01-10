import os
from datetime import datetime
import polars as pl
from aimo3_eval.config import CFG
from aimo3_eval.data.loader import DataLoader
from aimo3_eval.engine.vllm_server import VLLMServer
from aimo3_eval.engine.solver import AIMO3Solver
from aimo3_eval.metrics.evaluator import evaluate_attempts_parquet
from aimo3_eval.metrics.math_utils import MathGrader


def save_results(run_output_dir: str, all_attempts: list, all_times: list):
    """
    保存结果到两个 parquet 文件：
    - attempts.parquet: 每个 attempt 的详细信息
    - times.parquet: 每个问题的时间汇总
    """
    attempts_path = os.path.join(run_output_dir, "attempts.parquet")
    times_path = os.path.join(run_output_dir, "times.parquet")
    
    # 保存 attempts
    if all_attempts:
        attempts_df = pl.DataFrame(all_attempts)
        attempts_df.write_parquet(attempts_path)
    
    # 保存 times
    if all_times:
        times_df = pl.DataFrame(all_times)
        times_df.write_parquet(times_path)
    
    return attempts_path, times_path


def main():
    cfg = CFG(
        mode='remote',
        remote_api_key='sk-xxxx'
    )
    
    # 构建带时间戳的输出目录
    start_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = cfg.remote_model_name if cfg.mode == 'remote' else cfg.served_model_name
    # 清理模型名称中的特殊字符
    model_name_clean = model_name.replace('/', '_').replace(':', '_')
    run_output_dir = os.path.join(cfg.output_dir, f"{cfg.mode}_{model_name_clean}_{start_datetime}")
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"📂 Output directory: {run_output_dir}")
    
    # 只有在 local 模式下才启动本地 vLLM
    server = None
    if cfg.mode == 'local':
        server = VLLMServer(cfg)
        server.start()
    else:
        print("⏩ Skipping local server start (Remote Mode)")

    # Solver 会根据 cfg 自动连到正确的地方
    solver = AIMO3Solver(cfg)
    
    df = DataLoader.load_csv(
        "D:\\work_source\\deep_learning\\kaggle\\AIMO3\\aimo3_eval\\data\\reference.csv",
        id_col="id",
        problem_col="problem",
        ground_truth_col="answer"
    )
    df = DataLoader.load_custom_data(
        problems=["What is 2+2?", "Calculate sum of 1 to 100."],
        ids=["demo_1", "demo_2"],
        ground_truths=["4", "5050"]
    )
    print(f"📚 Loaded {len(df)} problems.")
    
    # 准备输出文件路径
    attempts_path = os.path.join(run_output_dir, "attempts.parquet")
    times_path = os.path.join(run_output_dir, "times.parquet")
    
    all_attempts = []  # 存储所有 attempt 数据
    all_times = []   # 存储所有问题的汇总结果
    
    try:
        # 4. 主循环 - 实时增量写入
        print("🚀 Start Inference...")
        for idx, row in enumerate(df.iter_rows(named=True), 1):
            problem_id = row['id']
            problem = row['problem']
            ground_truth = str(row.get('ground_truth', ''))
            
            print(f"[{idx}/{len(df)}] Processing {problem_id}...")
            res = solver.solve(problem, problem_id)
            
            # 提取结果
            final_answer = str(res['final_answer']) if res['final_answer'] is not None else ''
            
            # 判断答案是否正确
            is_correct = MathGrader.is_equiv(final_answer, ground_truth) if ground_truth else False
            
            # 打印结果和时间信息
            print(f" -> Answer: {final_answer} | Ground Truth: {ground_truth} | Correct: {is_correct}")
            print(f" -> Time - Min: {res['min_attempt_time']:.2f}s | Avg: {res['avg_attempt_time']:.2f}s | Max: {res['max_attempt_time']:.2f}s")
            
            # 处理每个 attempt，添加到 all_attempts，并计算各指标
            for attempt in res['attempts']:
                attempt_answer = str(attempt.get('final_answer', ''))
                attempt_is_correct = MathGrader.is_equiv(attempt_answer, ground_truth) if ground_truth else False
                
                attempt_record = {
                    "attempt_id": attempt['attempt_id'],
                    "problem_id": problem_id,
                    "problem": problem,
                    "solution": str(attempt.get('messages', [])),  # 将 messages 转为字符串作为 solution
                    "answer": attempt_answer,
                    "ground_truth": ground_truth,
                    "isCorrect": attempt_is_correct,
                    "time": attempt.get('time_taken', 0.0),
                    "python_calls": attempt.get('python_calls', 0),
                    "python_errors": attempt.get('python_errors', 0)
                }
                all_attempts.append(attempt_record)
            
            # 添加问题汇总结果到 all_times
            result_record = {
                "problem_id": problem_id,
                "problem": problem,
                "min_attempt_time": res['min_attempt_time'],
                "max_attempt_time": res['max_attempt_time'],
                "avg_attempt_time": res['avg_attempt_time']
            }
            all_times.append(result_record)
            
            # 实时保存
            save_results(run_output_dir, all_attempts, all_times)
            print(f" ✅ Saved! ")
            
    finally:
        # 5. 清理
        solver.cleanup()
        if server:
            server.stop()
        
    results = evaluate_attempts_parquet(
        attempts_path,
        use_math_equiv=True,
        verbose=True
    )

    print(f"\n✅ Final results saved to:")
    print(f"   - {attempts_path}")
    print(f"   - {times_path}")

if __name__ == "__main__":
    main()