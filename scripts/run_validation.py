"""一键验证：跑完 6 类必验场景，生成 docs/验证记录.md 和脱敏运行日志。

用法：
  1. 启动服务：  python -m uvicorn app.main:app --port 8000
  2. 执行验证：  python scripts/run_validation.py            # 不配 Key 时只做离线部分
                python scripts/run_validation.py --base http://127.0.0.1:8000

场景 7（模型/工具调用失败）通过"故意使用无效 API Key"真实触发一次 401，
不需要人为破坏配置，也不会影响 .env。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.runlog import sanitize  # noqa: E402
from config import BASE_DIR, settings  # noqa: E402

DOCS = BASE_DIR / "docs"
LOGS = BASE_DIR / "logs"

# 告诉后端这些请求来自验证脚本：记录会标记为 validation，不混进用户在页面上看到的分析记录
VALIDATION_HEADERS = {"X-Client-Source": "validation"}

SCENARIOS = [
    {
        "key": "normal",
        "title": "场景 1：正常成绩数据",
        "expect": "工具被调用，斜率/预测与手算一致（slope=3.3, pred=91.5），图表与数据一致",
        "messages": ["这是我最近四次考试的成绩：78、82、85、88，帮我看看变化趋势"],
    },
    {
        "key": "followup",
        "title": "场景 2：连续追问（复用会话数据）",
        "expect": "不再次询问成绩，直接基于上一轮数据调用工具并给出第 5 次预测",
        "messages": ["继续按这个趋势，第五次可能是多少？"],
    },
    {
        "key": "unspecified_predict",
        "title": "场景 3：未指定预测位置时中断询问",
        "expect": "不给出任何预测分数，回答说明需要指定第几次，并附第 N 次选项卡片（fix.kind=choose_predict）",
        "messages": ["这是我最近四次考试的成绩：78、82、85、88，帮我预测一下成绩"],
    },
    {
        "key": "invalid_data",
        "title": "场景 4：无效数据 / 无法拟合",
        "expect": "工具拒绝执行，回答说明具体原因，用户可以修改后继续",
        "messages": ["我这次考了八十五分，帮我分析一下变化趋势"],
    },
    {
        "key": "kb_hit",
        "title": "场景 5：知识库内的问题",
        "expect": "实际调用检索工具，并展示命中的资料片段与来源",
        "messages": ["如何提高复习效率？有什么具体做法？"],
    },
    {
        "key": "kb_miss",
        "title": "场景 6：知识库未覆盖的问题",
        "expect": "明确说明资料不足，不编造引用",
        "messages": ["量子纠缠的物理机制是什么？"],
    },
]


def http_json(base: str, path: str, payload: dict | None = None, timeout: float = 120) -> tuple[int, dict]:
    if payload is None:
        request = urllib.request.Request(base + path, headers=VALIDATION_HEADERS)
    else:
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **VALIDATION_HEADERS},
        )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode())


def summarize(response: dict) -> dict:
    """压缩响应为可读摘要，完整 JSON 单独放日志文件。"""
    reply = response.get("reply") or {}
    return {
        "error": response.get("error"),
        "answer": (reply.get("answer") or "")[:220] or None,
        "trend": reply.get("trend"),
        "slope": reply.get("slope"),
        "prediction": reply.get("prediction"),
        "limitations": reply.get("limitations"),
        "fix": response.get("fix"),
        "sources": [
            {"title": s.get("title"), "chunk_id": s.get("chunk_id"), "snippet": (s.get("snippet") or "")[:120]}
            for s in (reply.get("sources") or [])
        ],
        "tool_calls": [
            {"name": t.get("name"), "ok": t.get("ok"), "summary": t.get("summary")}
            for t in (response.get("tool_calls") or [])
        ],
        "chart_url": response.get("chart_url"),
        "run_id": response.get("run_id"),
    }


def scenario_failure() -> dict:
    """场景 7：用无效 Key 真实触发一次模型调用失败，验证失败状态与排查信息。"""
    from app.llm import LLMError, LLMClient
    from app.session import store

    original_key = settings.api_key
    settings.api_key = "sk-invalid-key-for-failure-test"
    try:
        client = LLMClient()
        session = store.get_or_create("failure-demo")
        try:
            client.chat([{"role": "user", "content": "78、82、85、88 什么趋势？"}], tools=None)
            return {"ok": False, "note": "预期失败但没有失败（Key 居然有效？请检查环境）"}
        except LLMError as exc:
            return {
                "ok": True,
                "error_message": exc.message,
                "error_detail": exc.detail[:200],
                "note": "前端会把 error/error_detail/run_id 展示给用户，日志保留排查信息",
            }
    finally:
        settings.api_key = original_key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--skip-failure", action="store_true", help="跳过场景 7（模型调用失败）")
    args = parser.parse_args()

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    log_path = LOGS / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    LOGS.mkdir(exist_ok=True)
    log_lines: list[str] = []

    health_code, health = http_json(args.base, "/api/health")
    mock_mode = bool(health.get("mock_mode"))

    sections: list[str] = []
    sections.append(f"# 验证记录\n")
    sections.append(f"- 生成时间：{timestamp}")
    sections.append(f"- 被测服务：{args.base}")
    sections.append(f"- 服务状态：HTTP {health_code}，mock_mode={mock_mode}，model={health.get('model')}")
    sections.append(f"- 知识库：{health.get('knowledge_files')} 个文件 / {health.get('knowledge_chunks')} 个片段\n")

    # ---------- 离线数值验证 ----------
    print(">> 运行数值验证（不需要模型）…")
    import subprocess

    calc_run = subprocess.run(
        [sys.executable, str(BASE_DIR / "scripts" / "verify_calc.py")],
        capture_output=True, text=True, encoding="utf-8",
    )
    sections.append("## 一、数值计算验证（独立实现核对）\n")
    sections.append("用与工具实现无关的纯 Python 闭式解 + 手算答案双重核对，脚本：`scripts/verify_calc.py`。\n")
    sections.append("```text")
    sections.append(calc_run.stdout.strip())
    sections.append("```\n")
    sections.append(f"结果：{'全部通过' if calc_run.returncode == 0 else '存在失败'}\n")

    # ---------- 在线场景 ----------
    session_id = None
    if health_code != 200:
        sections.append("## 二、功能场景验证\n\n服务未启动，在线场景未执行。请先启动服务再运行本脚本。\n")
    else:
        sections.append("## 二、功能场景验证（真实运行）\n")
        for index, scenario in enumerate(SCENARIOS, start=1):
            print(f">> 场景 {index}: {scenario['title']}")
            sections.append(f"### {scenario['title']}\n")
            sections.append(f"- 预期：{scenario['expect']}")
            if scenario["key"] == "followup" and session_id:
                sections.append("- 说明：沿用场景 1 的 session_id，验证连续追问。")

            payload = {"message": scenario["messages"][0]}
            if scenario["key"] == "followup" and session_id:
                payload["session_id"] = session_id

            try:
                code, response = http_json(args.base, "/api/chat", payload)
            except Exception as exc:  # 网络层失败
                code, response = -1, {"error": f"请求失败：{exc}"}

            if session_id is None and response.get("session_id"):
                session_id = response["session_id"]

            summary = summarize(response)
            log_lines.append(json.dumps(
                {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                 "scenario": scenario["key"], "request": payload,
                 "http_status": code, "response": response},
                ensure_ascii=False, default=str))

            sections.append(f"- 实际（HTTP {code}）：")
            sections.append("```json")
            sections.append(json.dumps(summary, ensure_ascii=False, indent=2))
            sections.append("```\n")

        # ---------- 场景 7：失败状态 ----------
        if not args.skip_failure:
            print(">> 场景 7: 模型调用失败（无效 Key 触发 401）")
            sections.append("### 场景 7：模型或工具调用失败\n")
            sections.append("- 预期：展示失败状态，保留必要的排查信息，会话可继续使用")
            result = scenario_failure()
            log_lines.append(json.dumps(
                {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                 "scenario": "failure", "result": result}, ensure_ascii=False))
            sections.append("```json")
            sections.append(json.dumps(result, ensure_ascii=False, indent=2))
            sections.append("```\n")

    sections.append("## 三、如何复核\n")
    sections.append("1. 数值部分：直接运行 `python scripts/verify_calc.py`，退出码 0 即全部通过；")
    sections.append("2. 场景部分：按 docs/录制指南.md 启动服务后，在页面上重现上述对话；")
    sections.append(f"3. 完整请求/响应原文（含 token 用量）见 `{log_path.name}`，已脱敏（无密钥信息）。")

    DOCS.mkdir(exist_ok=True)
    report = "\n".join(sections)
    (DOCS / "验证记录.md").write_text(report, encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        for line in log_lines:
            handle.write(sanitize(line) + "\n")

    print(f"\n验证记录已写入 docs/验证记录.md")
    print(f"运行日志已写入 {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
