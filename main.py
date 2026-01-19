import os
from aimo3_eval import (
    CFG,
    RemoteConfig,
    InferenceConfig,
    CoTSolver,
    TIRSolver,
    VLLMServer,
    DataLoader,
    EvalRunner,
)


def main():
    # 1. 配置
    cfg = CFG(
        mode="remote",
        remote=RemoteConfig(
            model="deepseek-reasoner",
            base_url=os.getenv("OPENAI_API_BASE") or "https://api.deepseek.com",
            api_key=os.getenv("OPENAI_API_KEY", "")
        ),
        inference=InferenceConfig(max_tokens=32768)
    )
    # cfg = CFG.from_json("config.json")
    
    # 2. 初始化服务器（仅 local 模式需要）
    server = None
    if cfg.mode == 'local':
        server = VLLMServer(cfg)
        server.start()
    
    # 3. 初始化 Solver
    solver = CoTSolver(cfg)
    
    # 4. 加载数据
    df = DataLoader.load_custom_data(
        problems=["What is 2+2?", "Calculate sum of 1 to 100."],
        ids=["demo_1", "demo_2"],
        ground_truths=["4", "5050"]
    )
    # 或者从 CSV 加载:
    # df = DataLoader.load_csv(
    #     "data/reference.csv", 
    #     ground_truth_col="answer", 
    #     id_col="id", 
    #     problem_col="problem" 
    # )
    
    # 5. 运行评估
    runner = EvalRunner(cfg, solver, server=server)
    results = runner.load_data(df).run()
    
if __name__ == "__main__":
    main()