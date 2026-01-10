import polars as pl
from collections import Counter
from typing import Optional
from .extractor import AnswerExtractor
from .math_utils import MathGrader


def evaluate_attempts_parquet(
    parquet_path: str,
    use_math_equiv: bool = True,
    verbose: bool = True
) -> dict:
    """
    从 attempts.parquet 文件计算 pass@1, pass@k, maj@k 指标。
    
    Args:
        parquet_path: parquet 文件路径，需包含以下列:
            - attempt_id: int (尝试编号，用于排序确定第一次尝试)
            - problem_id: str (题目ID)
            - answer: str (该次尝试的答案)
            - ground_truth: str (标准答案)
            - isCorrect: bool (可选，如存在则直接使用)
            - time: float (可选，用于时间统计)
        use_math_equiv: 是否使用 MathGrader.is_equiv 进行数学等价判断
                       如果为 False，则使用简单的字符串比较
        verbose: 是否打印详细信息
    
    Returns:
        包含以下指标的字典:
        - acc_pass@1: 第一次尝试的准确率
        - acc_pass@k: k次尝试中至少有一次正确的比例
        - acc_maj@k: 众数投票的准确率
        - total_problems: 总题目数
        - total_attempts: 总尝试次数
        - k: 每道题的尝试次数
        - per_problem: 每道题的详细指标 (可选)
    """
    # 读取 parquet 文件
    df = pl.read_parquet(parquet_path)
    
    if verbose:
        print(f"📊 Loaded {len(df)} attempts from {parquet_path}")
        print(f"   Columns: {df.columns}")
    
    # 验证必要的列
    required_cols = ["attempt_id", "problem_id", "answer", "ground_truth"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # 按 problem_id 分组
    pass_1_list = []
    pass_k_list = []
    maj_k_list = []
    per_problem_results = []
    
    # 获取所有唯一的 problem_id
    problem_ids = df["problem_id"].unique().to_list()
    
    for problem_id in problem_ids:
        # 获取该题目的所有 attempts
        problem_df = df.filter(pl.col("problem_id") == problem_id).sort("attempt_id")
        
        ground_truth = problem_df["ground_truth"][0]
        answers = problem_df["answer"].to_list()
        
        # 过滤无效答案
        valid_answers = [a for a in answers if a and str(a) != 'None' and str(a) != '']
        
        if not valid_answers:
            pass_1_list.append(0)
            pass_k_list.append(0)
            maj_k_list.append(0)
            per_problem_results.append({
                "problem_id": problem_id,
                "pass@1": False,
                "pass@k": False,
                "maj@k": False,
                "first_answer": None,
                "majority_answer": None
            })
            continue
        
        # 判断函数
        def is_correct(pred: str, truth: str) -> bool:
            if use_math_equiv:
                return MathGrader.is_equiv(str(pred), str(truth))
            else:
                return str(pred).strip() == str(truth).strip()
        
        # --- pass@1: 第一次尝试是否正确 ---
        first_answer = valid_answers[0]
        is_pass_1 = is_correct(first_answer, ground_truth)
        pass_1_list.append(1 if is_pass_1 else 0)
        
        # --- pass@k: 任意一次正确即可 ---
        is_pass_k = any(is_correct(a, ground_truth) for a in valid_answers)
        pass_k_list.append(1 if is_pass_k else 0)
        
        # --- maj@k: 众数投票 ---
        counts = Counter(valid_answers)
        majority_answer = counts.most_common(1)[0][0]
        is_maj_k = is_correct(majority_answer, ground_truth)
        maj_k_list.append(1 if is_maj_k else 0)
        
        per_problem_results.append({
            "problem_id": problem_id,
            "pass@1": is_pass_1,
            "pass@k": is_pass_k,
            "maj@k": is_maj_k,
            "first_answer": first_answer,
            "majority_answer": majority_answer,
            "ground_truth": ground_truth,
            "num_attempts": len(valid_answers)
        })
    
    # 计算汇总指标
    total_problems = len(problem_ids)
    metrics = {
        "acc_pass@1": sum(pass_1_list) / total_problems if total_problems else 0,
        "acc_pass@k": sum(pass_k_list) / total_problems if total_problems else 0,
        "acc_maj@k": sum(maj_k_list) / total_problems if total_problems else 0,
        "total_problems": total_problems,
        "total_attempts": len(df),
        "k": len(df) // total_problems if total_problems else 0,
        "correct_pass@1": sum(pass_1_list),
        "correct_pass@k": sum(pass_k_list),
        "correct_maj@k": sum(maj_k_list),
    }
    
    # 添加时间统计（如果存在 time 列）
    if "time" in df.columns:
        metrics["avg_attempt_time"] = df["time"].mean()
        metrics["min_attempt_time"] = df["time"].min()
        metrics["max_attempt_time"] = df["time"].max()
        metrics["total_time"] = df["time"].sum()
    
    # 添加每道题的详细结果
    metrics["per_problem"] = per_problem_results
    
    if verbose:
        print(f"\n📈 Evaluation Results (k={metrics['k']}):")
        print(f"   pass@1: {metrics['acc_pass@1']:.4f} ({metrics['correct_pass@1']}/{total_problems})")
        print(f"   pass@k: {metrics['acc_pass@k']:.4f} ({metrics['correct_pass@k']}/{total_problems})")
        print(f"   maj@k:  {metrics['acc_maj@k']:.4f} ({metrics['correct_maj@k']}/{total_problems})")
        if "total_time" in metrics:
            print(f"   Total time: {metrics['total_time']:.2f}s")
    
    return metrics


def evaluate_dataframe(df: pl.DataFrame) -> dict:
    """
    输入 DataFrame 必须包含:
    - 'ground_truth': str/int
    - 'attempts': List[dict] (来自 Solver 的输出)
      其中每个 attempt dict 需包含 'final_answer' 字段
      
    返回: 包含 pass@k, maj@k 等指标的字典
    """
    
    # 1. 数据预处理：确保 ground_truth 是字符串
    df = df.with_columns(pl.col("ground_truth").cast(pl.Utf8))
    
    # 2. 定义处理单行的逻辑 (Python Native)
    # 由于 SymPy 无法向量化，我们使用 map_elements 或 iter_rows
    # 对于测试集规模 (几百条)，Python Loop 性能完全足够且更易调试
    
    pass_1_list = [] # 第一次尝试是否正确
    pass_k_list = [] # k次尝试中是否至少有一次正确
    maj_k_list = []  # 众数是否正确
    
    # 遍历每一道题
    rows = df.to_dicts()
    for row in rows:
        truth = row['ground_truth']
        attempts = row['attempts']
        
        if not attempts:
            pass_1_list.append(0)
            pass_k_list.append(0)
            maj_k_list.append(0)
            continue

        # 收集该题目的所有预测答案 (已由 Solver 提取过，这里再做一次清洗确保万无一失)
        # 注意：Phase 1 的 Solver 返回的 attempt['final_answer'] 已经是提取过的了
        # 但为了鲁棒，我们可以把 attempt['messages'] 拿出来重跑 extractor (可选)
        # 这里假设 Solver 的 output 已经是清洗过的字符串
        preds = [str(a.get('final_answer', '')) for a in attempts]
        preds = [p for p in preds if p and p != 'None'] # 过滤无效值

        if not preds:
            pass_1_list.append(0)
            pass_k_list.append(0)
            maj_k_list.append(0)
            continue

        # --- Metric: Pass@1 ---
        # 取第一次尝试
        is_pass_1 = MathGrader.is_equiv(preds[0], truth)
        pass_1_list.append(1 if is_pass_1 else 0)

        # --- Metric: Pass@k ---
        # 只要有一个对
        is_pass_k = any(MathGrader.is_equiv(p, truth) for p in preds)
        pass_k_list.append(1 if is_pass_k else 0)

        # --- Metric: Maj@k ---
        # 众数投票
        if preds:
            from collections import Counter
            # 统计每个答案出现的次数
            counts = Counter(preds)
            # 找到票数最多的答案 (可能有多个并列，取第一个)
            major_pred = counts.most_common(1)[0][0]
            is_maj_k = MathGrader.is_equiv(major_pred, truth)
            maj_k_list.append(1 if is_maj_k else 0)
        else:
            maj_k_list.append(0)

    # 3. 汇总计算
    metrics = {
        "acc_pass@1": sum(pass_1_list) / len(pass_1_list) if pass_1_list else 0,
        "acc_pass@k": sum(pass_k_list) / len(pass_k_list) if pass_k_list else 0,
        "acc_maj@k": sum(maj_k_list) / len(maj_k_list) if maj_k_list else 0,
        "total_samples": len(rows)
    }
    
    # 4. 添加时间相关统计（如果存在这些字段）
    if "avg_attempt_time" in df.columns:
        metrics["overall_avg_attempt_time"] = df["avg_attempt_time"].mean()
    
    if "min_attempt_time" in df.columns:
        metrics["overall_min_attempt_time"] = df["min_attempt_time"].mean()
    
    if "max_attempt_time" in df.columns:
        metrics["overall_max_attempt_time"] = df["max_attempt_time"].mean()
    
    # 5. 统计全局最快和最慢的单次 attempt
    if "min_attempt_time" in df.columns:
        metrics["fastest_single_attempt"] = df["min_attempt_time"].min()
    
    if "max_attempt_time" in df.columns:
        metrics["slowest_single_attempt"] = df["max_attempt_time"].max()
    
    return metrics