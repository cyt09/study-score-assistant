"""成绩趋势分析工具：一元线性最小二乘（NumPy 手写闭式解，不调用现成回归模型）。

设计要点
1. 只用 NumPy 做数组运算，斜率与截距按最小二乘闭式解逐个公式实现；
2. 同时用 np.polyfit 做内部交叉校验，二者结果必须一致（差异写进返回值，便于审核）；
3. 输入非法、样本不足、x 全部相同等情况抛出 TrendError，绝不输出无效预测；
4. 图表用 matplotlib 生成 PNG，保存到 data/charts 下，由 FastAPI 静态服务返回。
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # 无界面渲染，避免需要 GUI 环境
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from config import settings

_CJK_FONT_READY: bool | None = None

# 成绩有效上限：同时覆盖百分制（100）与 150 分制（如高考语数外），负数一律无效
_MAX_SCORE = 150.0

# t 分布双侧 95% 临界值（自由度 1—30），自由度再大就收敛到 1.96。
# 不引入 scipy：常用的几个值查表即可，也便于逐行核对。
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
    30: 2.042,
}


class TrendError(Exception):
    """工具可预期的失败：带上错误码、给用户看的说明，以及"改哪里"的结构化细节。

    data 里的字段供程序自动生成修改卡片使用，例如：
      {"field": "y", "invalid_indexes": [2], "values": ["八十二"]}
    """

    def __init__(self, code: str, message: str, hint: str = "", data: dict | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.data = data or {}

    def to_dict(self) -> dict:
        payload = {"ok": False, "error_code": self.code, "message": self.message, "hint": self.hint}
        if self.data:
            payload["detail"] = self.data
        return payload


def _setup_chinese_font() -> bool:
    """尽量使用系统中文字体，缺失时图表退回英文标签（不影响数据）。"""
    global _CJK_FONT_READY
    if _CJK_FONT_READY is not None:
        return _CJK_FONT_READY

    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            _CJK_FONT_READY = True
            return True
    _CJK_FONT_READY = False
    return False


def _as_float_list(values: Any, field: str, key: str) -> list[float]:
    """把输入转成浮点列表；一次列出所有非数字位置，方便用户一次改完。"""
    if values is None:
        raise TrendError(
            "EMPTY_INPUT",
            f"{field} 为空，没有可分析的数据",
            "请把每次考试的成绩填进下面的输入框",
            {"field": key},
        )
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TrendError(
            "BAD_INPUT_TYPE",
            f"{field} 必须是数字列表，收到的是 {type(values).__name__}",
            "例如 78、82、85、88",
            {"field": key},
        )

    numbers: list[float] = []
    broken: list[tuple[int, Any]] = []  # 根本不是数字
    infinite: list[int] = []  # inf / nan

    for index, item in enumerate(values, start=1):
        if isinstance(item, bool):
            broken.append((index, item))
            numbers.append(float("nan"))
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            broken.append((index, item))
            numbers.append(float("nan"))
            continue
        if not math.isfinite(number):
            infinite.append(index)
            numbers.append(float("nan"))
            continue
        numbers.append(number)

    if broken:
        detail = "、".join(f"第 {i} 个「{v}」" for i, v in broken[:4])
        raise TrendError(
            "NON_NUMERIC",
            f"{field} 里 {detail} 不是数字，无法参与计算",
            "请改成纯阿拉伯数字（不要带“分”字、汉字数字或百分号）",
            {"field": key, "invalid_indexes": [i for i, _ in broken], "values": [str(v) for _, v in broken]},
        )
    if infinite:
        detail = "、".join(f"第 {i} 个" for i in infinite[:4])
        raise TrendError(
            "NON_FINITE",
            f"{field} 里 {detail} 不是有限的数字",
            "请填写真实的成绩数值",
            {"field": key, "invalid_indexes": infinite},
        )
    return numbers


def _validate_pair(x: list[float], y: list[float], predict_x: float | None) -> float | None:
    if len(x) != len(y):
        raise TrendError(
            "LENGTH_MISMATCH",
            f"考试序号有 {len(x)} 个，成绩有 {len(y)} 个，数量对不上",
            "每一次考试对应一个成绩，请补齐或删掉多余的值",
            {"field": "x", "x_len": len(x), "y_len": len(y)},
        )
    if len(x) < 2:
        raise TrendError(
            "INSUFFICIENT_DATA",
            f"只有 {len(x)} 个数据点，无法拟合趋势（至少需要 2 次考试）",
            "请再补充一次成绩",
            {"field": "y", "count": len(x)},
        )

    # 成绩必须是真实考试里可能出现的数值：负数、超过 150 分（百分制/150 分制上限）都视为无效
    bad_scores = [(i, v) for i, v in enumerate(y, start=1) if v < 0 or v > _MAX_SCORE]
    if bad_scores:
        detail = "、".join(f"第 {i} 个 {v:g} 分" for i, v in bad_scores[:4])
        raise TrendError(
            "INVALID_SCORE",
            f"成绩 {detail} 超出有效范围（0—{_MAX_SCORE:g} 分）",
            "负数成绩或超过满分上限的成绩都无法分析，请核对后修改",
            {"field": "y", "invalid_indexes": [i for i, _ in bad_scores]},
        )

    # 考试序号应从 1 开始（0、负数或小数序号无法对应真实的第几次考试）
    bad_index = [(i, v) for i, v in enumerate(x, start=1) if v < 1 or v != int(v)]
    if bad_index:
        detail = "、".join(f"第 {i} 个 {v:g}" for i, v in bad_index[:4])
        raise TrendError(
            "INVALID_X",
            f"考试序号 {detail} 无效（应为从 1 开始的整数）",
            "例如四次考试就依次填 1、2、3、4",
            {"field": "x", "invalid_indexes": [i for i, _ in bad_index]},
        )

    # 所有序号完全相同：斜率 sxx=0，数学上无定义（必须在"重复"之前判断，否则会被当成普通重复）
    if len(x) >= 2 and float(np.ptp(np.asarray(x))) == 0.0:
        raise TrendError(
            "DEGENERATE_X",
            f"所有考试序号都是 {x[0]:g}，无法区分第几次考试，斜率没有定义",
            "请按 1、2、3… 依次编号，或直接在下面核对成绩后重新分析",
            {"field": "x", "x": x},
        )
    if len(set(x)) != len(x):
        seen: set[float] = set()
        duplicates: list[int] = []
        for index, value in enumerate(x, start=1):
            if value in seen:
                duplicates.append(index)
            seen.add(value)
        raise TrendError(
            "DUPLICATE_X",
            "考试序号存在重复值（第 " + "、".join(str(i) for i in duplicates[:4]) + " 个），无法唯一确定位置",
            "序号应递增且不重复，可直接在下面核对成绩后重新分析",
            {"field": "x", "invalid_indexes": duplicates, "x": x},
        )

    predict_value: float | None = None
    if predict_x is not None:
        if isinstance(predict_x, bool):
            raise TrendError("BAD_PREDICT_X", "预测位置必须是数字", "例如第 5 次考试就填 5",
                             {"field": "predict_x"})
        try:
            predict_value = float(predict_x)
        except (TypeError, ValueError):
            raise TrendError("BAD_PREDICT_X", f"预测位置「{predict_x}」不是数字", "例如第 5 次考试就填 5",
                             {"field": "predict_x"}) from None
        if not math.isfinite(predict_value):
            raise TrendError("BAD_PREDICT_X", "预测位置不是有限的数字", "请填写正常的考试序号",
                             {"field": "predict_x"})
        if predict_value < 1:
            raise TrendError(
                "BAD_PREDICT_X",
                f"预测位置 {predict_value:g} 无效（考试序号不能小于 1）",
                "例如第五次考试就填 5",
                {"field": "predict_x"},
            )
    return predict_value


def _render_chart(xs: list[float], ys: list[float], slope: float, intercept: float,
                  predict_x: float | None, prediction: float | None,
                  interval: dict[str, Any] | None = None) -> str:
    chinese = _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=140)

    ax.scatter(xs, ys, s=52, color="#2f6fed", zorder=3,
               label="实际成绩" if chinese else "Actual score")

    line_x = np.linspace(min(xs), max(xs), 100)
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color="#f2994a", linewidth=2,
            label="线性拟合" if chinese else "Linear fit")

    if predict_x is not None and prediction is not None:
        ax.scatter([predict_x], [prediction], s=80, marker="*", color="#eb5757", zorder=4,
                   label="预测值" if chinese else "Prediction")
        ax.plot([predict_x, predict_x], [min(min(ys), prediction), prediction],
                linestyle=":", color="#eb5757", linewidth=1)
        ax.annotate(f"{prediction:.1f}", (predict_x, prediction), textcoords="offset points",
                    xytext=(6, 8), color="#eb5757", fontsize=10)
        # 把 95% 预测区间画成误差棒：让人一眼看出这是"一个范围"，不是一个确定分数
        if interval:
            ax.errorbar(
                [predict_x], [prediction],
                yerr=[[prediction - interval["low"]], [interval["high"] - prediction]],
                fmt="none", ecolor="#eb5757", elinewidth=1.2, capsize=6, alpha=0.7, zorder=4,
            )
            ax.annotate(
                f"95%: {interval['low']:g}—{interval['high']:g}",
                (predict_x, min(prediction, interval["low"])), textcoords="offset points",
                xytext=(6, -14), color="#eb5757", fontsize=8,
            )

    for xi, yi in zip(xs, ys):
        ax.annotate(f"{yi:g}", (xi, yi), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color="#5b6472")

    if predict_x is not None and prediction is not None:
        ax.set_xlim(min(xs) - 0.5, max(max(xs), predict_x) + 0.5)
        low = min(min(ys), prediction - (interval["margin"] if interval else 0))
        high = max(max(ys), prediction + (interval["margin"] if interval else 0))
        ax.set_ylim(low - 5, high + 5)

    ax.set_xlabel("考试序号" if chinese else "Exam index")
    ax.set_ylabel("成绩" if chinese else "Score")
    ax.set_title(("成绩趋势：斜率 %.3f / 次" % slope) if chinese else ("Trend: slope %.3f" % slope), fontsize=11)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()

    filename = f"chart_{uuid.uuid4().hex[:12]}.png"
    fig.savefig(settings.chart_dir / filename)
    plt.close(fig)
    return f"/charts/{filename}"


def analyze_trend(x: Any, y: Any, predict_x: float | None = None) -> dict:
    """对成绩序列做一元线性最小二乘拟合，可选出指定位置的预测值。"""
    xs = _as_float_list(x, "x（考试序号）", "x")
    ys = _as_float_list(y, "y（成绩）", "y")
    predict_value = _validate_pair(xs, ys, predict_x)

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    n = x_arr.size

    # ---- 最小二乘闭式解（逐项用 NumPy 数组运算实现）----
    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    dx = x_arr - x_mean
    dy = y_arr - y_mean
    sxx = float(np.sum(dx * dx))
    sxy = float(np.sum(dx * dy))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean

    fitted = slope * x_arr + intercept
    ss_res = float(np.sum((y_arr - fitted) ** 2))
    ss_tot = float(np.sum((y_arr - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # ---- 内部交叉校验：与 np.polyfit 比对，证明闭式解实现正确 ----
    poly_slope, poly_intercept = np.polyfit(x_arr, y_arr, 1)
    cross_check = {
        "polyfit_slope": round(float(poly_slope), 10),
        "polyfit_intercept": round(float(poly_intercept), 10),
        "slope_diff": round(abs(float(poly_slope) - slope), 12),
        "intercept_diff": round(abs(float(poly_intercept) - intercept), 12),
    }

    prediction: float | None = None
    if predict_value is not None:
        prediction = round(slope * predict_value + intercept, 2)

    # ---- 预测区间：残差标准误 + t 临界值（自由度 n-2）----
    # 只给一个点估计会让人误以为"第五次就是 85.37"，把不确定性一并算出来更诚实。
    # 公式：ŷ ± t(0.975, n-2) · s · sqrt(1 + 1/n + (x0-x̄)²/Sxx)
    interval: dict[str, Any] | None = None
    if predict_value is not None and n >= 3 and ss_tot >= 0:
        dof = n - 2
        residual_se = math.sqrt(ss_res / dof)
        t_crit = _T975.get(dof, 1.96)
        margin = t_crit * residual_se * math.sqrt(1 + 1 / n + (predict_value - x_mean) ** 2 / sxx)
        interval = {
            "level": 0.95,
            "dof": dof,
            "residual_se": round(residual_se, 3),
            "low": round(prediction - margin, 2),
            "high": round(prediction + margin, 2),
            "margin": round(margin, 2),
        }

    warnings: list[str] = []
    if n < 3:
        warnings.append("样本只有 2 个点，任意两点都能连成直线，拟合优度不具参考意义")
    if r2 < 0.5:
        warnings.append(f"拟合优度 R² = {r2:.2f} 偏低，线性模型对该组数据解释力有限")
    if predict_value is not None:
        if predict_value > max(xs):
            gap = predict_value - max(xs)
            warnings.append(f"预测位置 {predict_value:g} 超出已有数据范围 {gap:g} 次，属于外推，误差会放大")
        elif predict_value < min(xs):
            warnings.append(f"预测位置 {predict_value:g} 早于已有数据，属于反向外推")
    else:
        warnings.append("用户未指定预测位置，结论中应说明并询问要预测第几次")
    if all(value == ys[0] for value in ys):
        warnings.append("所有成绩完全相同，趋势线为水平线，预测值等于原成绩")
    if interval and interval["margin"] >= 3:
        warnings.append(
            f"95% 预测区间约 {interval['low']:g}—{interval['high']:g} 分（±{interval['margin']:g}），"
            "单次考试的实际成绩可能落在该范围内"
        )

    chart_url = _render_chart(xs, ys, slope, intercept, predict_value, prediction, interval)

    return {
        "ok": True,
        "method": "closed-form OLS with NumPy arrays; cross-checked by np.polyfit",
        "n": n,
        "x": xs,
        "y": ys,
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r2": round(r2, 4),
        "predict_x": predict_value,
        "prediction": prediction,
        "prediction_interval": interval,
        "fitted_y": [round(float(v), 4) for v in fitted],
        "chart_url": chart_url,
        "warnings": warnings,
        "cross_check": cross_check,
    }
