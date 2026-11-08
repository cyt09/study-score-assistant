"""检索质量评估：把"阈值是试出来的"变成可复现的数字。

用法
    python scripts/eval_retrieval.py            # 打印结果 + 阈值扫描
    python scripts/eval_retrieval.py --write    # 同时生成 docs/检索评估.md

指标
    recall@k   —— 应该命中的问题里，期望文件出现在前 k 条命中的比例（越高越好）
    fp          —— 不该命中的问题里，却给出命中的数量（越低越好，理想为 0）
    precision@k —— 命中结果中期望文件所占比例
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.tools import retrieval  # noqa: E402
from app.tools.retrieval import knowledge_base  # noqa: E402

CASES_PATH = ROOT / "eval" / "retrieval_cases.json"
DOC_PATH = ROOT / "docs" / "检索评估.md"


def load_cases() -> tuple[list[dict], int]:
    payload = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    return payload["cases"], int(payload.get("top_k", 3))


def evaluate(cases: list[dict], top_k: int, min_score: float | None = None) -> dict:
    positives = [c for c in cases if c["should_hit"]]
    negatives = [c for c in cases if not c["should_hit"]]

    hits = 0
    correct = 0
    total_returned = 0
    misses: list[dict] = []
    for case in positives:
        result = knowledge_base.search(case["query"], top_k=top_k, min_score=min_score)
        titles = [item["title"] for item in result.get("hits") or []]
        total_returned += len(titles)
        if not titles:
            misses.append({"query": case["query"], "reason": "未命中任何片段", "expect": case["expect"]})
            continue
        if case["expect"] in titles:
            hits += 1
            correct += titles.index(case["expect"]) + 1
        else:
            misses.append({"query": case["query"], "reason": f"命中 {titles}", "expect": case["expect"]})

    false_positives: list[dict] = []
    for case in negatives:
        result = knowledge_base.search(case["query"], top_k=top_k, min_score=min_score)
        if result.get("hits"):
            false_positives.append({"query": case["query"], "hits": [h["chunk_id"] for h in result["hits"]]})

    return {
        "positives": len(positives),
        "negatives": len(negatives),
        "recall": round(hits / len(positives), 3) if positives else 0.0,
        "precision": round(hits / total_returned, 3) if total_returned else 0.0,
        "mean_rank": round(correct / hits, 2) if hits else None,
        "false_positives": len(false_positives),
        "misses": misses,
        "fp_detail": false_positives,
        "min_score": min_score,
    }


def sweep(cases: list[dict], top_k: int) -> list[dict]:
    """扫描"覆盖率 × 余弦"两道门槛：它们决定"判不判资料不足"，比 min_score 更关键。"""
    rows = []
    gates = [(0.2, 0.05), (0.2, 0.10), (0.34, 0.05), (0.34, 0.10), (0.34, 0.20), (0.5, 0.10), (0.7, 0.10)]
    old = (retrieval._MIN_COVERAGE, retrieval._MIN_COSINE)
    try:
        for coverage, cosine in gates:
            retrieval._MIN_COVERAGE = coverage
            retrieval._MIN_COSINE = cosine
            row = evaluate(cases, top_k)
            rows.append(
                {
                    "coverage": coverage,
                    "cosine": cosine,
                    "recall": row["recall"],
                    "false_positives": row["false_positives"],
                }
            )
    finally:
        retrieval._MIN_COVERAGE, retrieval._MIN_COSINE = old
    return rows


def render_markdown(cases: list[dict], top_k: int, base: dict, rows: list[dict]) -> str:
    lines = [
        "# 检索评估",
        "",
        "这份文件是脚本生成的，不用手改；要更新就跑 `python scripts/eval_retrieval.py --write`。",
        "",
        f"- 评估集：`eval/retrieval_cases.json`，共 {len(cases)} 条"
        f"（应命中 {base['positives']} 条 / 应空结果 {base['negatives']} 条），top_k={top_k}",
        "- 判定：期望文件出现在前 k 条命中即算命中；负例只要返回任何片段即算误报",
        "",
        "## 当前默认阈值",
        "",
        f"覆盖率阈值 `{retrieval._MIN_COVERAGE}`，余弦阈值 `{retrieval._MIN_COSINE}`，"
        f"片段分数下限 `{base['min_score'] if base['min_score'] is not None else '取配置默认'}`",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| recall@{top_k} | {base['recall']:.0%} |",
        f"| 命中时平均排名 | {base['mean_rank']} |",
        f"| 负例误报数 | {base['false_positives']} / {base['negatives']} |",
        "",
    ]
    if base["misses"]:
        lines += ["## 漏检明细", "", "| 问题 | 期望文件 | 实际情况 |", "| --- | --- | --- |"]
        lines += [f"| {m['query']} | {m['expect']} | {m['reason']} |" for m in base["misses"]]
        lines.append("")
    if base["fp_detail"]:
        lines += ["## 误报明细", "", "| 问题 | 返回片段 |", "| --- | --- |"]
        lines += [f"| {f['query']} | {', '.join(f['hits'])} |" for f in base["fp_detail"]]
        lines.append("")

    lines += [
        "## 门槛扫描",
        "",
        "覆盖率与余弦门槛决定「判不判资料不足」，改动前后用手上这套集子对比：",
        "",
        "| 覆盖率门槛 | 余弦门槛 | recall@k | 负例误报 |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        mark = " ← 当前" if (
            abs(row["coverage"] - retrieval._MIN_COVERAGE) < 1e-9
            and abs(row["cosine"] - retrieval._MIN_COSINE) < 1e-9
        ) else ""
        lines.append(
            f"| {row['coverage']} | {row['cosine']} | {row['recall']:.0%} | {row['false_positives']}{mark} |"
        )
    lines += [
        "",
        "选型原则：先保住负例误报为 0（不编造引用），再在这一档里取 recall 最高的组合。"
        "当前值不是拍脑袋定的，而是在这套集子上扫出来的；知识库扩容或改写后重跑即可复核。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="生成 docs/检索评估.md")
    args = parser.parse_args()

    cases, top_k = load_cases()
    base = evaluate(cases, top_k)
    rows = sweep(cases, top_k)

    print(f"评估集 {len(cases)} 条（正例 {base['positives']} / 负例 {base['negatives']}），top_k={top_k}")
    print(f"recall@{top_k} = {base['recall']:.0%} | 命中平均排名 = {base['mean_rank']}")
    print(f"负例误报 = {base['false_positives']} / {base['negatives']}")
    for miss in base["misses"]:
        print(f"  漏检：{miss['query']} → 期望 {miss['expect']}，实际 {miss['reason']}")
    for fp in base["fp_detail"]:
        print(f"  误报：{fp['query']} → {fp['hits']}")

    print("\n门槛扫描（覆盖率门槛 × 余弦门槛 → recall / 误报）")
    for row in rows:
        print(f"  {row['coverage']:<5} × {row['cosine']:<5} → {row['recall']:.0%} / {row['false_positives']}")

    if args.write:
        DOC_PATH.write_text(render_markdown(cases, top_k, base, rows), encoding="utf-8", newline="\n")
        print(f"\n已写入 {DOC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
