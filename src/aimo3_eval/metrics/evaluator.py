import polars as pl
from .extractor import AnswerExtractor
from .math_utils import MathGrader

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
    
    return metrics