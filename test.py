from aimo3_eval import evaluate_attempts_parquet

results = evaluate_attempts_parquet(
    "D:\\work_source\\deep_learning\\kaggle\\AIMO3\\aimo3_eval\\outputs\\remote_deepseek-reasoner_20260109_201411\\attempts.parquet",
    use_math_equiv=True,
    verbose=True
)