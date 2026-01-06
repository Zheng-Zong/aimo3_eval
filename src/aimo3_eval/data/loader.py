import polars as pl
import os

class DataLoader:
    @staticmethod
    def load_kaggle_csv(file_path: str) -> pl.DataFrame:
        """
        读取 Kaggle 格式 CSV，并统一列名
        Expected: [id, problem, answer(optional)]
        Output: [id, problem, ground_truth(optional)]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
            
        df = pl.read_csv(file_path)
        
        # 统一列名映射
        rename_map = {}
        if "answer" in df.columns:
            rename_map["answer"] = "ground_truth"
        elif "solution" in df.columns:
            rename_map["solution"] = "ground_truth"
            
        df = df.rename(rename_map)
        
        # 确保关键列存在
        if "problem" not in df.columns:
            raise ValueError("Dataset must contain a 'problem' column")
            
        return df

    @staticmethod
    def load_custom_data(problems: list[str], ids: list[str] = None, ground_truths: list[str] = None) -> pl.DataFrame:
        """用于手动测试几条特定数据"""
        if ids is None:
            ids = [str(i) for i in range(len(problems))]
        if ground_truths is None:
            ground_truths = [None] * len(problems)
            
        return pl.DataFrame({
            "id": ids,
            "problem": problems,
            "ground_truth": ground_truths
        })