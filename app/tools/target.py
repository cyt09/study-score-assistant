"""目标分倒推：用户说"想考到 X 分"时，算出差距与所需斜率。

完全由程序计算，不让模型心算——和趋势拟合一样，数字必须可核对。
"""

from __future__ import annotations

from typing import Any

_MAX_SCORE = 100


def plan_for_target(
    slope: float,
    intercept: float,
    target: float,
    x: float | None = None,
    prediction: float | None = None,
) -> dict[str, Any]:
    """给出"按当前趋势差多少、需要多大斜率"的结构化结论。

    - x：目标位置（第几次考试）。没有就退化成"下一次"（拟合位置未知时不编造）。
    - prediction：该位置的预测值（工具已经算好时直接传入，避免重复计算口径不一致）。
    """
    try:
        slope = float(slope)
        intercept = float(intercept)
        target = float(target)
    except (TypeError, ValueError):
        return {"ok": False, "error_code": "INVALID_TARGET", "message": "目标分或拟合参数不是有效数字"}

    if not (-1e9 < target <= _MAX_SCORE) or target < 0:
        return {
            "ok": False,
            "error_code": "INVALID_TARGET",
            "message": f"目标分需要在 0—{_MAX_SCORE} 分之间，收到 {target:g}",
        }

    if x is None:
        return {
            "ok": False,
            "error_code": "NO_TARGET_POSITION",
            "message": "没有明确目标是第几次考试（例如“第五次考到 95”），无法计算差距",
        }

    try:
        x = float(x)
    except (TypeError, ValueError):
        return {"ok": False, "error_code": "NO_TARGET_POSITION", "message": "目标位置不是有效数字"}

    if x <= 0:
        return {"ok": False, "error_code": "INVALID_TARGET_X", "message": "目标位置必须从第 1 次开始"}

    # 目标位置的预测值：优先用工具算好的，保证与页面上显示的数字一致
    at_target = prediction if prediction is not None else intercept + slope * x
    at_target = float(at_target)

    gap = target - at_target
    needed_slope = (target - intercept) / x
    extra = needed_slope - slope

    if gap <= 0:
        verdict = "reached"
        message = (
            f"按当前趋势，第 {x:g} 次预计 {at_target:.1f} 分，已经达到 {target:g} 分的目标"
            f"（高出 {abs(gap):.1f} 分）"
        )
    else:
        verdict = "short"
        message = (
            f"按当前趋势，第 {x:g} 次预计 {at_target:.1f} 分，距目标 {target:g} 分还差 {gap:.1f} 分；"
            f"要在第 {x:g} 次达到目标，斜率需要从 {slope:.2f} 提高到 {needed_slope:.2f} 分/次"
            f"（每次多提 {extra:.2f} 分）"
        )

    return {
        "ok": True,
        "target": round(target, 2),
        "x": x,
        "prediction": round(at_target, 2),
        "gap": round(gap, 2),
        "current_slope": round(slope, 2),
        "needed_slope": round(needed_slope, 2),
        "extra_per_exam": round(extra, 2),
        "verdict": verdict,
        "message": message,
    }
