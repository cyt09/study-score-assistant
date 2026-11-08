"""生成多条"不同失败原因"的分析记录（kind=error），用于演示失败状态展示。

全部走 handle_message 的真实失败路径落库，每条都带失败原因、排查信息、
已执行工具与 run_id（logs/run.jsonl 可检索完整调用过程）：

1. 模型鉴权失败（401）       —— 临时换成无效 API Key，走真实 DeepSeek 接口
2. 无法连接模型服务          —— base_url 指向不可达地址，真实连接异常
3. 模型请求超时              —— 把超时调成 1ms，触发真实 APITimeoutError
4. 工具调用轮次超限          —— 桩模型每轮都请求工具，触发死循环保护
5. 模型输出未通过结构化校验  —— 工具成功但两次输出都不是合法 JSON

运行：.venv/Scripts/python.exe scripts/make_failure_demos.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import agent, db  # noqa: E402
from app.session import store  # noqa: E402


# ---------- 桩模型工具（与 tests/test_pipeline.py 同款） ----------

def tool_call_response(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
        "usage": {"total_tokens": 100},
        "model": "fake",
    }


def content_response(text: str) -> dict:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [],
        "usage": {"total_tokens": 120},
        "model": "fake",
    }


class ScriptedClient:
    """按脚本返回预设响应，用完即抛（说明模型被多调用了一次）。"""

    def __init__(self, scripted: list[dict]):
        self.scripted = list(scripted)

    def chat(self, messages, tools=None, temperature=0.2, json_mode=False):
        if not self.scripted:
            raise AssertionError("脚本已用尽")
        return self.scripted.pop(0)


class AlwaysToolClient:
    """每一轮都请求调用趋势工具、永远不给最终回答，用于触发轮次超限。"""

    def __init__(self):
        self.rounds = 0

    def chat(self, messages, tools=None, temperature=0.2, json_mode=False):
        self.rounds += 1
        return tool_call_response(
            "analyze_score_trend",
            {"x": [1, 2, 3], "y": [78, 84, 90], "predict_x": 4},
        )


def run(name: str, question: str, patch, restore) -> None:
    """跑一个失败场景并落库；patch() 制造故障，restore() 恢复现场。"""
    session = store.get_or_create(f"fail-demo-{name}")
    patch()
    try:
        response = agent.handle_message(question, session)
    finally:
        restore()
    print(f"[{name}] error={response.error!r}")
    print(f"[{name}] detail={response.error_detail!r}\n")


def main() -> None:
    db.init_db()

    # 1. 鉴权失败（401）：换一个无效 Key，请求真实 DeepSeek 接口
    real_key = agent.settings.api_key
    run(
        "401",
        "这是我最近四次考试的成绩：78、82、85、88，帮我看看趋势，顺便预测第五次",
        lambda: setattr(agent.settings, "api_key", "sk-invalid-demo-key-for-failure-demo"),
        lambda: setattr(agent.settings, "api_key", real_key),
    )

    # 2. 无法连接模型服务：base_url 指向本机不可达端口
    real_url = agent.settings.base_url
    run(
        "conn",
        "第一次 75、第二次 80、第三次 86，帮我看下进步情况",
        lambda: setattr(agent.settings, "base_url", "https://127.0.0.1:9/v1"),
        lambda: setattr(agent.settings, "base_url", real_url),
    )

    # 3. 请求超时：把超时压到 1 秒，真实触发 APITimeoutError（模型正常响应通常超过 1 秒）
    real_timeout = agent.settings.timeout
    run(
        "timeout",
        "70、76、83、91 这几次成绩帮我分析一下，第五次能到多少",
        lambda: setattr(agent.settings, "timeout", 1.0),
        lambda: setattr(agent.settings, "timeout", real_timeout),
    )

    # 4. 工具调用轮次超限：桩模型每轮都请求工具，触发死循环保护
    original_build = agent.build_client
    run(
        "rounds",
        "我的成绩是 78、84、90，帮我预测第四次多少分？",
        lambda: setattr(agent, "build_client", lambda: AlwaysToolClient()),
        lambda: setattr(agent, "build_client", original_build),
    )

    # 5. 结构化校验失败：工具成功，但最终回答两次都不是合法 JSON
    run(
        "schema",
        "85 和 92 这两次成绩帮我算算斜率",
        lambda: setattr(agent, "build_client", lambda: ScriptedClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2], "y": [85, 92]}),
            content_response("抱歉，我暂时无法生成结构化的分析结果。"),
            content_response("确实无法生成，请稍后再试。"),
        ])),
        lambda: setattr(agent, "build_client", original_build),
    )

    # 汇总：列出刚生成的失败记录
    print("=" * 60)
    print("最近生成的失败记录（kind=error）：")
    with db._LOCK:
        pass
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(agent.settings.db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, session_id, question, substr(answer_text, 1, 60) AS head "
            "FROM analyses WHERE kind = 'error' ORDER BY id DESC LIMIT 5"
        ).fetchall()
        for row in rows:
            print(f"  #{row['id']} [{row['session_id']}] {row['question']}")
            print(f"      {row['head']}...")


if __name__ == "__main__":
    main()
