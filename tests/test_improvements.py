"""本轮改进项的回归测试。

覆盖：会话 LRU/TTL 与数据库回填、LLM 退避重试与错误码、工具 warnings 强制合并、
预测区间、检索评估指标、进度事件、搜索转义与多关键词。
不联网：用假模型与桩异常驱动真实流程。
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402
from openai import APITimeoutError, AuthenticationError, RateLimitError  # noqa: E402

from config import settings  # noqa: E402

from app import agent, db  # noqa: E402
from app.llm import LLMClient, LLMError  # noqa: E402
from app.schemas import FinalAnswer  # noqa: E402
from app.session import SessionStore, store  # noqa: E402
from app.tools.retrieval import search_knowledge  # noqa: E402
from app.tools.trend import analyze_trend  # noqa: E402

TEMP_DB = Path(tempfile.mkdtemp()) / "improvements.db"
REPO = Path(__file__).resolve().parents[1]


def payload(**overrides) -> dict:
    data = {
        "answer": "四次成绩整体呈上升趋势，斜率约每次 3.3 分，第五次约 91.5 分。",
        "trend": "上升",
        "slope": 3.3,
        "intercept": 75.0,
        "prediction": 91.5,
        "prediction_note": "假设趋势保持不变",
        "limitations": ["样本只有 4 个点"],
        "sources": [],
        "need_followup": None,
    }
    data.update(overrides)
    return data


def content_response(data: dict) -> dict:
    return {
        "role": "assistant",
        "content": json.dumps(data, ensure_ascii=False),
        "tool_calls": [],
        "usage": {"total_tokens": 100},
        "model": "fake",
    }


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


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools=None, temperature=0.2, json_mode=False):
        self.calls += 1
        if not self.responses:
            raise AssertionError("假模型没有更多预设响应")
        return self.responses.pop(0)


class DBTestCase(unittest.TestCase):
    """每个测试用独立的库文件，结束后还原，避免污染其他测试与真实数据。"""

    def setUp(self) -> None:
        self._old_db = settings.db_path
        settings.db_path = TEMP_DB
        db.init_db()
        with db._LOCK, db.closing(db._connect()) as conn:
            conn.execute("DELETE FROM analyses")
            conn.commit()

    def tearDown(self) -> None:
        settings.db_path = self._old_db


class SessionPersistenceTest(DBTestCase):
    def test_lru_evicts_oldest(self):
        small = SessionStore(max_sessions=2, ttl_seconds=0)
        small.get_or_create("s-lru-1")
        small.get_or_create("s-lru-2")
        small.get_or_create("s-lru-3")
        self.assertEqual(small.stats()["in_memory"], 2)
        self.assertNotIn("s-lru-1", small._sessions)

    def test_ttl_sweeps_idle_sessions(self):
        small = SessionStore(max_sessions=10, ttl_seconds=0.05)
        small.get_or_create("s-ttl-1")
        time.sleep(0.08)
        small.get_or_create("s-ttl-2")  # 触发一次清理
        self.assertNotIn("s-ttl-1", small._sessions)
        self.assertIn("s-ttl-2", small._sessions)

    def test_missing_session_is_restored_from_db(self):
        db.save_analysis(
            "restore-me", "四次成绩 78、82、85、88，第五次呢？",
            {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "slope": 3.3, "intercept": 75.0, "r2": 0.99},
            answer_text="成绩在稳定上升",
        )
        # 模拟服务重启：内存为空，但数据库里有记录
        fresh = SessionStore()
        session = fresh.get_or_create("restore-me")
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.last_dataset["y"], [78, 82, 85, 88])
        self.assertEqual(fresh.stats()["restored_from_db"], 1)

    def test_unknown_session_stays_empty(self):
        fresh = SessionStore()
        session = fresh.get_or_create("no-such-session")
        self.assertEqual(session.messages, [])
        self.assertIsNone(session.last_dataset)


class LLMRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._old = (settings.llm_max_retries, settings.llm_retry_backoff)
        settings.llm_max_retries = 2
        settings.llm_retry_backoff = 0  # 测试不真的等

    def tearDown(self) -> None:
        settings.llm_max_retries, settings.llm_retry_backoff = self._old

    @staticmethod
    def _ok_response() -> SimpleNamespace:
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            model="fake",
        )

    def _client(self, errors: list[Exception]) -> tuple[LLMClient, list[int]]:
        attempts: list[int] = []

        def create(**kwargs):
            attempts.append(1)
            if errors:
                raise errors.pop(0)
            return self._ok_response()

        client = LLMClient.__new__(LLMClient)  # 跳过构造里的 Key 检查
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return client, attempts

    def test_retry_rate_limit_then_success(self):
        request = httpx.Request("POST", "https://example.com")
        errors = [
            RateLimitError("429", response=httpx.Response(429, request=request), body=None),
            APITimeoutError(request=request),
        ]
        client, attempts = self._client(errors)
        result = client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(result["content"], "ok")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(result["retries"], 2)

    def test_retry_stops_after_max_attempts(self):
        request = httpx.Request("POST", "https://example.com")
        errors = [APITimeoutError(request=request) for _ in range(3)]
        client, attempts = self._client(errors)
        with self.assertRaises(LLMError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.code, "TIMEOUT")
        self.assertEqual(len(attempts), 3)  # 1 次原始调用 + 2 次重试

    def test_auth_error_is_not_retried(self):
        request = httpx.Request("POST", "https://example.com")
        error = AuthenticationError("401", response=httpx.Response(401, request=request), body=None)
        client, attempts = self._client([error])
        with self.assertRaises(LLMError) as ctx:
            client.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(ctx.exception.code, "AUTH")
        self.assertEqual(len(attempts), 1)


class LimitationMergeTest(unittest.TestCase):
    def test_tool_warning_is_appended(self):
        answer = FinalAnswer(**payload(limitations=["样本只有 4 个点"]))
        trend = {"warnings": ["预测位置 5 超出已有数据范围 1 次，属于外推，误差会放大"]}
        merged = agent._merge_tool_warnings(answer, trend)
        self.assertEqual(len(merged.limitations), 2)
        self.assertIn("外推", merged.limitations[1])

    def test_same_topic_is_not_repeated(self):
        answer = FinalAnswer(**payload(limitations=["第 5 次属于外推，实际值可能偏离"]))
        trend = {"warnings": ["预测位置 5 超出已有数据范围 1 次，属于外推，误差会放大"]}
        merged = agent._merge_tool_warnings(answer, trend)
        self.assertEqual(len(merged.limitations), 1)

    def test_instruction_style_warning_is_skipped(self):
        answer = FinalAnswer(**payload())
        trend = {"warnings": ["用户未指定预测位置，结论中应说明并询问要预测第几次"]}
        merged = agent._merge_tool_warnings(answer, trend)
        self.assertEqual(merged.limitations, answer.limitations)

    def test_warnings_are_merged_end_to_end(self):
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 5}),
            # 模型只写了一条笼统的限制条件
            content_response(payload(limitations=["样本只有 4 个点"])),
        ])
        old_path = settings.db_path
        settings.db_path = TEMP_DB
        db.init_db()
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("四次 78、82、85、88，第五次呢？", store.get_or_create(None))
        finally:
            agent.build_client = original
            settings.db_path = old_path
        self.assertIsNone(response.error)
        self.assertTrue(any("外推" in item for item in response.reply.limitations))


class PredictionIntervalTest(DBTestCase):
    def test_interval_wraps_prediction(self):
        result = analyze_trend([1, 2, 3, 4], [78, 82, 85, 88], 5)
        band = result["prediction_interval"]
        self.assertIsNotNone(band)
        self.assertEqual(band["level"], 0.95)
        self.assertEqual(band["dof"], 2)
        self.assertLess(band["low"], result["prediction"])
        self.assertGreater(band["high"], result["prediction"])
        # 手算核对：残差标准误 × t(0.975,2) × sqrt(1 + 1/n + (x0-x̄)²/Sxx)
        import math

        residuals = [y - (result["slope"] * x + result["intercept"]) for x, y in zip(result["x"], result["y"])]
        se = math.sqrt(sum(r * r for r in residuals) / 2)
        sxx = sum((x - 2.5) ** 2 for x in result["x"])
        expected = 4.303 * se * math.sqrt(1 + 1 / 4 + (5 - 2.5) ** 2 / sxx)
        self.assertAlmostEqual(band["margin"], expected, places=1)

    def test_two_points_have_no_interval(self):
        result = analyze_trend([1, 2], [80, 90], 3)
        self.assertIsNone(result["prediction_interval"])
        self.assertTrue(any("2 个点" in w for w in result["warnings"]))

    def test_no_prediction_no_interval(self):
        result = analyze_trend([1, 2, 3, 4], [78, 82, 85, 88])
        self.assertIsNone(result["prediction_interval"])

    def test_interval_is_persisted_with_record(self):
        result = analyze_trend([1, 2, 3, 4], [78, 82, 85, 88], 5)
        record_id = db.save_analysis("interval-session", "第五次多少", result, answer_text="约 91.5 分")
        record = next(r for r in db.list_session_records("interval-session") if r["id"] == record_id)
        self.assertAlmostEqual(record["pred_low"], result["prediction_interval"]["low"], places=2)
        self.assertAlmostEqual(record["pred_high"], result["prediction_interval"]["high"], places=2)


class RetrievalEvalTest(DBTestCase):
    """把评估集变成回归门槛：改了检索参数导致召回掉下来，测试会直接失败。"""

    def test_recall_and_false_positive_rate(self):
        cases_path = REPO / "eval" / "retrieval_cases.json"
        data = json.loads(cases_path.read_text(encoding="utf-8"))
        top_k = int(data.get("top_k", 3))
        positives = [c for c in data["cases"] if c["should_hit"]]
        negatives = [c for c in data["cases"] if not c["should_hit"]]

        hits = 0
        for case in positives:
            titles = [hit["title"] for hit in search_knowledge(case["query"], top_k=top_k).get("hits") or []]
            if case["expect"] in titles:
                hits += 1
        false_positives = sum(1 for case in negatives if search_knowledge(case["query"], top_k=top_k)["hits"])

        recall = hits / len(positives)
        self.assertGreaterEqual(recall, 0.9, f"检索召回下降到 {recall:.0%}")
        self.assertEqual(false_positives, 0, "知识库不覆盖的问题不应该返回片段")

    def test_r2_query_is_searchable(self):
        # 单个字母 + 上标的术语以前会被分词器整个丢掉
        result = search_knowledge("R² 是什么意思")
        self.assertTrue(result["hits"])
        self.assertIn("线性拟合", result["hits"][0]["title"])

    def test_paraphrase_falls_back_to_expansion(self):
        result = search_knowledge("这次突然掉了20分是什么原因")
        self.assertTrue(result["hits"])
        self.assertIn("成绩波动", result["hits"][0]["title"])


class ProgressEventTest(DBTestCase):
    def test_phases_are_emitted_in_order(self):
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 5}),
            content_response(payload()),
        ])
        seen: list[tuple[str, dict]] = []
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            agent.handle_message(
                "四次 78、82、85、88，第五次呢？",
                store.get_or_create(None),
                progress=lambda phase, data: seen.append((phase, data)),
            )
        finally:
            agent.build_client = original

        phases = [phase for phase, _ in seen]
        self.assertEqual(phases[0], "thinking")
        self.assertIn("tool", phases)
        self.assertIn("tool_done", phases)
        self.assertLess(phases.index("tool"), phases.index("tool_done"))
        tool_event = next(data for phase, data in seen if phase == "tool")
        self.assertEqual(tool_event["name"], "analyze_score_trend")

    def test_progress_callback_failure_does_not_break_chat(self):
        fake = FakeClient([content_response(payload())])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            def boom(phase, data):
                raise RuntimeError("前端已断开")

            response = agent.handle_message("你好", store.get_or_create(None), progress=boom)
        finally:
            agent.build_client = original
        self.assertIsNone(response.error)


class SearchImprovementTest(DBTestCase):
    def _seed(self) -> None:
        result = analyze_trend([1, 2, 3], [70, 75, 80], 4)
        db.save_analysis("s1", "成绩波动是因为难度吗", result, answer_text="相关不等于因果")
        db.save_analysis("s2", "成绩稳定上升", result, answer_text="斜率稳定")

    def test_like_wildcard_is_escaped(self):
        self._seed()
        self.assertEqual(db.search_records(query="%"), [])
        self.assertEqual(db.search_records(query="_"), [])

    def test_multi_keyword_requires_all_terms(self):
        self._seed()
        both = db.search_records(query="成绩 波动")
        self.assertEqual(len(both), 1)
        self.assertEqual(both[0]["session_id"], "s1")

    def test_session_index_exists(self):
        with db.closing(db._connect()) as conn:
            names = {row["name"] for row in conn.execute("PRAGMA index_list(analyses)")}
        self.assertIn("idx_analyses_session", names)


class StreamErrorEventTest(unittest.TestCase):
    """流式接口失败时也要把错误码推给前端，否则卡片上只剩一句"请求失败"。"""

    def test_stream_error_carries_error_code(self):
        from fastapi.testclient import TestClient

        import app.main as main

        original = main.handle_message

        def boom(*args, **kwargs):
            raise LLMError("无法连接模型服务", "Connection error: getaddrinfo failed", code="CONNECTION")

        main.handle_message = boom
        try:
            client = TestClient(main.app)
            response = client.post("/api/chat/stream", json={"message": "第五次呢？"})
        finally:
            main.handle_message = original

        events = [
            json.loads(line[len("data:"):].strip())
            for line in response.text.splitlines()
            if line.startswith("data:")
        ]
        error = next(event for event in events if event.get("type") == "error")
        self.assertEqual(error["error_code"], "CONNECTION")
        self.assertEqual(error["error"], "无法连接模型服务")
        self.assertIn("getaddrinfo", error["error_detail"])


if __name__ == "__main__":
    unittest.main()
