import os
import polars as pl
from aimo3_eval.config import CFG
from aimo3_eval.data.loader import DataLoader
from aimo3_eval.engine.vllm_server import VLLMServer
from aimo3_eval.engine.solver import AIMO3Solver
from aimo3_eval.metrics.evaluator import evaluate_dataframe

def main():
    cfg = CFG()
    
    # 只有在 local 模式下才启动本地 vLLM
    server = None
    if cfg.mode == 'local':
        server = VLLMServer(cfg)
        server.start()
    else:
        print("⏩ Skipping local server start (Remote Mode)")

    # Solver 会根据 cfg 自动连到正确的地方
    solver = AIMO3Solver(cfg)
    
    hard_problem = '''Let $n \geq 6$ be a positive integer. We call a positive integer $n$-Norwegian if it has three distinct positive divisors whose sum is equal to $n$. Let $f(n)$ denote the smallest $n$-Norwegian positive integer. Let $M=3^{2025!}$ and for a non-negative integer $c$ define 
\begin{equation*}
    g(c)=\frac{1}{2025!}\left\lfloor \frac{2025! f(M+c)}{M}\right\rfloor.
\end{equation*}
We can write 
\begin{equation*}
    g(0)+g(4M)+g(1848374)+g(10162574)+g(265710644)+g(44636594)=\frac{p}{q}
\end{equation*}
where $p$ and $q$ are coprime positive integers. What is the remainder when $p+q$ is divided by $99991$?'''
    # df = DataLoader.load_kaggle_csv("path/to/train.csv")
    df = DataLoader.load_custom_data(
        problems=["What is 2+2?", "Calculate sum of 1 to 100.", hard_problem],
        ids=["demo_1", "demo_2", "demo_3"],
        ground_truths=["4", "5050", "8687"]

    )
    print(f"📚 Loaded {len(df)} problems.")
    
    results = []
    
    try:
        # 4. 主循环 (Polars 这里的 iter_rows 比较方便调试)
        # 生产环境可以用 map_elements 但要注意 Solver 内部已有 ThreadPool
        print("🚀 Start Inference...")
        for row in df.iter_rows(named=True):
            print(f"Processing {row['id']}...")
            res = solver.solve(row['problem'], row['id'])
            res["ground_truth"] = row.get('ground_truth', None)
            
            # 简单打印
            print(f" -> Answer: {res['final_answer']}")
            results.append(res)
            
    finally:
        # 5. 清理
        solver.cleanup()
        if server:
            server.stop()
        
    # 6. 保存结果 (初步)
    # Phase 2 我们会把这个换成实时 Parquet Append
    result_df = pl.DataFrame(results)
    print(result_df)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)
    result_df.write_json(f"{cfg.output_dir}/results.json")

    # 4. 结果处理
    result_df = pl.DataFrame(results)

    # 如果数据集中包含 ground_truth，则进行评测
    if "ground_truth" in result_df.columns and result_df["ground_truth"].null_count() < result_df.height:
        print("\n📊 Running Evaluation...")
        metrics = evaluate_dataframe(result_df)
        
        print("-" * 30)
        print(f"Pass@1: {metrics['acc_pass@1']:.2%}")
        print(f"Pass@k: {metrics['acc_pass@k']:.2%}")
        print(f"Maj@k:  {metrics['acc_maj@k']:.2%}")
        print("-" * 30)
    else:
        print("\n⚠️ No ground truth found, skipping evaluation.")

    # 保存结果
    result_df.write_json(f"{cfg.output_dir}/results_with_metrics.json")

if __name__ == "__main__":
    main()