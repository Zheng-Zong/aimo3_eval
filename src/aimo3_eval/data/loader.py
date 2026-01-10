import polars as pl
import os

class DataLoader:
    @staticmethod
    def load_csv(
        file_path: str,
        id_col: str = "id",
        problem_col: str = "problem",
        ground_truth_col: str = "ground_truth"
    ) -> pl.DataFrame:
        """
        通用 CSV 读取函数，允许自定义列名
        Expected: [id_col, problem_col, ground_truth_col(optional)]
        Output: [id, problem, ground_truth(optional)]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
            
        df = pl.read_csv(file_path)
        
        # 统一列名映射
        rename_map = {}
        if id_col != "id":
            rename_map[id_col] = "id"
        if problem_col != "problem":
            rename_map[problem_col] = "problem"
        if ground_truth_col in df.columns and ground_truth_col != "ground_truth":
            rename_map[ground_truth_col] = "ground_truth"
            
        df = df.rename(rename_map)
        
        # 确保关键列存在
        if "problem" not in df.columns:
            raise ValueError("Dataset must contain a 'problem' column")
        if "id" not in df.columns:
            # 新增 id 列
            df = df.with_column(pl.arange(0, df.height).cast(pl.Utf8).alias("id"))
            # print warning
            print("⚠️ Warning: 'id' column not found. Generated sequential IDs.")
            
        return df
    
    @staticmethod
    def load_parquet(
        file_path: str,
        id_col: str = "id",
        problem_col: str = "problem",
        ground_truth_col: str = "ground_truth",
        is_lazy: bool = False
    ) -> pl.DataFrame:
        """
        通用 Parquet 读取函数，允许自定义列名
        Expected: [id_col, problem_col, ground_truth_col(optional)]
        Output: [id, problem, ground_truth(optional)]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset not found: {file_path}")
            
        if is_lazy:
            df = pl.read_parquet(file_path, use_pyarrow=True).lazy()
        else:
            df = pl.read_parquet(file_path)
        
        # 统一列名映射
        rename_map = {}
        if id_col != "id":
            rename_map[id_col] = "id"
        if problem_col != "problem":
            rename_map[problem_col] = "problem"
        if ground_truth_col in df.columns and ground_truth_col != "ground_truth":
            rename_map[ground_truth_col] = "ground_truth"
            
        df = df.rename(rename_map)
        
        # 确保关键列存在
        if "problem" not in df.columns:
            raise ValueError("Dataset must contain a 'problem' column")
        if "id" not in df.columns:
            # 新增 id 列
            df = df.with_column(pl.arange(0, df.height).cast(pl.Utf8).alias("id"))
            # print warning
            print("⚠️ Warning: 'id' column not found. Generated sequential IDs.")
            
        return df

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
        if "id" not in df.columns:
            # 新增 id 列
            df = df.with_column(pl.arange(0, df.height).cast(pl.Utf8).alias("id"))
            # print warning
            print("⚠️ Warning: 'id' column not found. Generated sequential IDs.")
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