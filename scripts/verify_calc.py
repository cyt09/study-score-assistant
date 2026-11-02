"""数值验证：用"与工具实现无关"的纯 Python 闭式解，核对趋势工具的输出。

为什么不用 np.polyfit 核对：polyfit 与工具同属 NumPy 生态，可能共享同一处实现缺陷；
这里用标准库手写公式 + 手算已知的精确答案做双重核对。

运行：python scripts/verify_calc.py
全部通过退出码为 0，任何不一致退出码为 1。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tools.trend import TrendError, analyze_trend  # noqa: E402


def pure_python_ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """标准库实现的最小二乘闭式解（与 app/tools/trend.py 的 NumPy 路径无关）。"""
    n = len(xs)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return slope, y_mean - slope * x_mean


CASES = [
    # (说明, x, y, predict_x, 手算期望)
    ("四次成绩 78-88 分（手算 slope=3.3, intercept=75, pred=91.5）",
     [1, 2, 3, 4], [78, 82, 85, 88], 5, {"slope": 3.3, "intercept": 75.0, "prediction": 91.5}),
    ("完全水平的成绩（手算 slope=0, intercept=80）",
     [1, 2, 3], [80, 80, 80], None, {"slope": 0.0, "intercept": 80.0, "prediction": None}),
    ("含小数成绩（与纯 Python 实现交叉核对）",
     [1, 2, 3, 4, 5], [76.5, 81.0, 79.5, 84.0, 86.5], 6, None),
    ("间隔序号（第 2/4/6/8 次，斜率不变，预测位置按真实序号）",
     [2, 4, 6, 8], [70, 74, 78, 82], 10, {"slope": 2.0, "intercept": 66.0, "prediction": 86.0}),
]

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str) -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}  {detail}")
    if not condition:
        FAILURES.append(f"{name}: {detail}")


def main() -> int:
    print("== 数值验证：趋势工具 vs 独立实现 / 手算答案 ==")
    for description, xs, ys, predict_x, expected in CASES:
        print(f"- {description}")
        result = analyze_trend(xs, ys, predict_x)
        slope_ref, intercept_ref = pure_python_ols(xs, ys)

        check("slope 与独立实现一致",
              abs(result["slope"] - slope_ref) < 1e-9,
              f"tool={result['slope']} ref={round(slope_ref, 10)}")
        check("intercept 与独立实现一致",
              abs(result["intercept"] - intercept_ref) < 1e-9,
              f"tool={result['intercept']} ref={round(intercept_ref, 10)}")
        check("polyfit 交叉校验差异为 0",
              result["cross_check"]["slope_diff"] == 0 and result["cross_check"]["intercept_diff"] == 0,
              json.dumps(result["cross_check"], ensure_ascii=False))

        if expected:
            if "slope" in expected:
                check("slope 与手算答案一致", abs(result["slope"] - expected["slope"]) < 1e-9,
                      f"tool={result['slope']} expect={expected['slope']}")
            if "intercept" in expected:
                check("intercept 与手算答案一致", abs(result["intercept"] - expected["intercept"]) < 1e-9,
                      f"tool={result['intercept']} expect={expected['intercept']}")
            if "prediction" in expected and expected["prediction"] is not None:
                check("prediction 与手算答案一致", abs(result["prediction"] - expected["prediction"]) < 1e-6,
                      f"tool={result['prediction']} expect={expected['prediction']}")

    print("\n== 异常输入验证：工具必须拒绝并说明原因 ==")
    bad_inputs = [
        ("只有一个数据点", lambda: analyze_trend([1], [80]), "INSUFFICIENT_DATA"),
        ("成绩含文字", lambda: analyze_trend([1, 2], [78, "八十二"]), "NON_NUMERIC"),
        ("x 与 y 数量不一致", lambda: analyze_trend([1, 2, 3], [80, 90]), "LENGTH_MISMATCH"),
        ("考试序号全部相同", lambda: analyze_trend([1, 1], [80, 90]), "DEGENERATE_X"),
        ("考试序号部分重复", lambda: analyze_trend([1, 1, 3], [80, 90, 92]), "DUPLICATE_X"),
        ("成绩里有非数字", lambda: analyze_trend([1, 2, 3], [80, "缺考", 92]), "NON_NUMERIC"),
        ("序号为0被拒绝", lambda: analyze_trend([0, 0], [80, 90]), "INVALID_X"),
        ("负数成绩", lambda: analyze_trend([1, 2, 3, 4], [78, 82, -20, 88]), "INVALID_SCORE"),
        ("成绩超满分上限", lambda: analyze_trend([1, 2, 3], [120, 180, 90]), "INVALID_SCORE"),
        ("考试序号为负", lambda: analyze_trend([1, -2, 3], [70, 80, 90]), "INVALID_X"),
        ("预测位置为负", lambda: analyze_trend([1, 2, 3, 4], [70, 80, 85, 90], -3), "BAD_PREDICT_X"),
    ]
    for name, fn, expected_code in bad_inputs:
        try:
            fn()
            check(name, False, "未抛出异常（不应发生）")
        except TrendError as exc:
            check(name, exc.code == expected_code,
                  f"code={exc.code} message={exc.message}")

    print()
    if FAILURES:
        print(f"验证失败 {len(FAILURES)} 项：")
        for item in FAILURES:
            print(" -", item)
        return 1
    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
