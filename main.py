import os
from aimo3_eval import CFG, AIMO3Solver, VLLMServer, DataLoader, EvalRunner


def main():
    # 1. 配置
    cfg = CFG(
        mode='remote',
        remote_model_name="deepseek-reasoner",
        remote_base_url=os.getenv('OPENAI_API_BASE'),
        remote_api_key=os.getenv('OPENAI_API_KEY'),
    )
    
    # 2. 初始化服务器（仅 local 模式需要）
    server = None
    if cfg.mode == 'local':
        server = VLLMServer(cfg)
        server.start()
    
    # 3. 初始化 Solver
    solver = AIMO3Solver(cfg)
    
    # 4. 加载数据
    df = DataLoader.load_custom_data(
        problems=["What is 2+2?", "Calculate sum of 1 to 100."],
        ids=["demo_1", "demo_2"],
        ground_truths=["4", "5050"]
    )
    # 或者从 CSV 加载:
    # df = DataLoader.load_csv("data/reference.csv")
    
    # 5. 运行评估
    runner = EvalRunner(cfg, solver, server=server)
    results = runner.load_data(df).run()
    
if __name__ == "__main__":
    main()