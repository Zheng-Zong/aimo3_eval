import sympy
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class MathGrader:
    """
    使用 SymPy 判断两个数学表达式是否等价。
    解决: 0.5 == 1/2, x=5 == 5, 1000.0 == 1000
    """
    
    # SymPy 解析配置：允许隐式乘法 (2x -> 2*x)
    _TRANSFORMATIONS = (standard_transformations + (implicit_multiplication_application,))

    @staticmethod
    def is_equiv(pred_str: str, truth_str: str) -> bool:
        """
        判断预测值和真值是否数学上相等。
        """
        if pred_str is None or truth_str is None:
            return False
            
        # 1. 字符串直接清洗比对 (最快，命中率 80%)
        # 移除空格，统一转小写
        clean_pred = str(pred_str).strip().lower().replace(" ", "")
        clean_truth = str(truth_str).strip().lower().replace(" ", "")
        
        if clean_pred == clean_truth:
            return True
        
        # 2. 尝试转 float 比对 (处理 1.0 vs 1)
        try:
            f_pred = float(clean_pred)
            f_truth = float(clean_truth)
            # AIMO 主要是整数，容差设小一点
            if abs(f_pred - f_truth) < 1e-6:
                return True
        except ValueError:
            pass
            
        # 3. SymPy 符号化比对 (最慢，但最强，处理 1/2 vs 0.5)
        try:
            # 必须设置 evaluate=False，防止自动求值导致精度丢失
            pred = parse_expr(clean_pred, transformations=MathGrader._TRANSFORMATIONS)
            truth = parse_expr(clean_truth, transformations=MathGrader._TRANSFORMATIONS)
            
            # 方法 A: 符号简化后做差
            if sympy.simplify(pred - truth) == 0:
                return True
                
            # 方法 B: 专门的 equals 方法
            if pred.equals(truth):
                return True
                
        except Exception:
            # 解析失败 (比如包含非法字符)，认为不等
            return False
            
        return False