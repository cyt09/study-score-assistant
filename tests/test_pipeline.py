"""流程测试：不联网，用假模型驱动真实的 Agent / 工具 / 校验 / 入库代码。

运行：python -m unittest tests.test_pipeline -v
这类测试覆盖的是"程序行为"，不是模型回答质量，因此不需要 API Key 也能跑。
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402

TEMP_DB = Path(tempfile.mkdtemp()) / "test_history.db"
settings.db_path = TEMP_DB

from app import agent, db  # noqa: E402
from app.session import store  # noqa: E402
from app.tools.retrieval import search_knowledge  # noqa: E402
from app.tools.trend import TrendError, analyze_trend  # noqa: E402


def final_payload(slope=3.3, intercept=75.0, prediction=91.5, trend="上升", limitations=None, answer=None):
    return {
        "answer": answer or f"四次成绩整体呈{trend}趋势，平均每次提高 {slope} 分。",
        "trend": trend,
        "slope": slope,
        "intercept": intercept,
        "prediction": prediction,
        "prediction_note": "基于最小二乘直线的外推，仅代表趋势。",
        "limitations": limitations if limitations is not None else ["只有 4 个数据点，样本量小"],
        "sources": [],
        "need_followup": None,
    }


class FakeClient:
    """按脚本返回预设响应，用来制造各种"模型输出异常"的场景。"""

    def __init__(self, scripted: list[dict]):
        self.scripted = list(scripted)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None, temperature=0.2, json_mode=False):
        self.calls.append(messages)
        if not self.scripted:
            raise AssertionError("脚本已用尽，说明代码多调用了一次模型")
        return self.scripted.pop(0)


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


def tool_calls_response(*calls: tuple[str, dict]) -> dict:
    """一轮里返回多个工具调用（真实场景里模型常这么干）。"""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
            for index, (name, args) in enumerate(calls, start=1)
        ],
        "usage": {"total_tokens": 100},
        "model": "fake",
    }


def content_response(payload) -> dict:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [],
        "usage": {"total_tokens": 120},
        "model": "fake",
    }


class TrendToolTest(unittest.TestCase):
    def test_known_dataset(self):
        result = analyze_trend([1, 2, 3, 4], [78, 82, 85, 88], 5)
        self.assertAlmostEqual(result["slope"], 3.3, places=4)
        self.assertAlmostEqual(result["intercept"], 75.0, places=4)
        self.assertAlmostEqual(result["prediction"], 91.5, places=2)
        self.assertLess(result["cross_check"]["slope_diff"], 1e-9)

    def test_no_predict_position(self):
        result = analyze_trend([1, 2, 3], [80, 84, 86])
        self.assertIsNone(result["prediction"])
        self.assertTrue(any("未指定预测位置" in w for w in result["warnings"]))

    def test_reject_non_numeric(self):
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([1, 2, 3], ["78", "八十二", 85])
        self.assertEqual(ctx.exception.code, "NON_NUMERIC")

    def test_reject_insufficient(self):
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([1], [80])
        self.assertEqual(ctx.exception.code, "INSUFFICIENT_DATA")

    def test_reject_same_x(self):
        """所有序号相同：斜率无定义，错误码要明确指出这一点。"""
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([2, 2], [80, 90])
        self.assertEqual(ctx.exception.code, "DEGENERATE_X")
        self.assertIn("所有考试序号", ctx.exception.message)
        self.assertEqual(ctx.exception.data["field"], "x")

    def test_reject_partial_duplicate_x(self):
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([1, 1, 3], [70, 80, 90])
        self.assertEqual(ctx.exception.code, "DUPLICATE_X")
        self.assertEqual(ctx.exception.data["invalid_indexes"], [2])

    def test_non_numeric_lists_all_bad_positions(self):
        """一次列出所有坏值的位置，用户不用改一个试一次。"""
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([1, 2, 3, 4], [78, "八十二", 85, "缺考"])
        self.assertEqual(ctx.exception.code, "NON_NUMERIC")
        self.assertEqual(ctx.exception.data["invalid_indexes"], [2, 4])
        self.assertEqual(ctx.exception.data["field"], "y")

    def test_reject_negative_score(self):
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([1, 2, 3, 4], [78, 82, -20, 88])
        self.assertEqual(ctx.exception.code, "INVALID_SCORE")
        self.assertIn("-20", ctx.exception.message)

    def test_reject_over_max_score(self):
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([1, 2, 3], [120, 180, 90])
        self.assertEqual(ctx.exception.code, "INVALID_SCORE")

    def test_reject_invalid_x(self):
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([1, -2, 3], [70, 80, 90])
        self.assertEqual(ctx.exception.code, "INVALID_X")

    def test_reject_negative_predict_x(self):
        with self.assertRaises(TrendError) as ctx:
            analyze_trend([1, 2, 3, 4], [70, 80, 85, 90], -3)
        self.assertEqual(ctx.exception.code, "BAD_PREDICT_X")


class RetrievalTest(unittest.TestCase):
    def test_hit_with_source(self):
        result = search_knowledge("如何提高复习效率")
        self.assertTrue(result["hits"])
        top = result["hits"][0]
        self.assertIn("学习效率", top["title"])
        self.assertIn("#", top["chunk_id"])

    def test_out_of_scope_returns_no_hit(self):
        result = search_knowledge("量子纠缠的物理机制是什么")
        self.assertEqual(result["hits"], [])
        self.assertIn("资料不足", result["message"])


class AgentFlowTest(unittest.TestCase):
    def setUp(self):
        db.init_db()
        self.session = store.get_or_create(None)

    def test_tool_then_answer_and_persist(self):
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 5}),
            content_response(final_payload()),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("这四次成绩 78、82、85、88，第五次大概多少？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(response.reply.slope, 3.3)
        self.assertEqual(response.reply.prediction, 91.5)
        self.assertEqual(len(response.tool_calls), 1)
        self.assertTrue(response.chart_url.startswith("/charts/"))
        self.assertIsNotNone(self.session.last_dataset)
        self.assertEqual(db.list_analyses(1)[0]["slope"], 3.3)
        # 完整回答也要落库，点开历史记录才能看到原文
        self.assertIn("四次成绩整体呈", db.list_analyses(1)[0]["answer"])

    def test_missing_predict_position_asks_instead_of_guessing(self):
        """用户没说要预测第几次：模型自己填的 predict_x 被丢弃，改为询问用户。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 5}),
            content_response(final_payload(
                prediction=None,
                answer="四次成绩从 78 升到 88，斜率 3.3 分/次，整体稳步上升。",
            )),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("这四次成绩 78、82、85、88，帮我分析一下趋势", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        # 工具入参里已经没有 predict_x，本次不产生任何预测值
        self.assertNotIn("predict_x", response.tool_calls[0].arguments)
        self.assertIsNone(response.reply.prediction)
        # 不猜位置，改成让用户点选第几次
        self.assertIsNotNone(response.fix)
        self.assertEqual(response.fix["kind"], "choose_predict")
        self.assertEqual(response.fix["options"], [5, 6, 7])

    def test_guessed_prediction_is_rejected_until_position_specified(self):
        """模型绕过工具硬写"第 5 次 91.5 分"会被校验拦下并要求重写。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 5}),
            content_response(final_payload(answer="按这条趋势外推，第 5 次考试大约 91.5 分。")),
            content_response(final_payload(
                prediction=None,
                answer="四次成绩整体上升，斜率 3.3 分/次；你还没有说要预测第几次。",
            )),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("我的成绩是 78、82、85、88，帮我预测一下", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(len(fake.calls), 3)  # 首答被拒 + 一次修复
        self.assertIsNone(response.reply.prediction)
        self.assertNotIn("91.5", response.reply.answer)
        self.assertEqual(response.fix["kind"], "choose_predict")

    def test_explicit_position_overrides_model_guess(self):
        """用户明确说第五次时，即使模型传了别的位置也会被校正成 5。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 3}),
            content_response(final_payload()),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("我的成绩是 78、82、85、88，帮我预测第五次", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(response.tool_calls[0].arguments["predict_x"], 5)
        self.assertEqual(response.reply.prediction, 91.5)
        self.assertIsNone(response.fix)

    def test_next_exam_resolves_to_following_index(self):
        """用户说"下次"时，按已有数据换算成下一个序号，而不是让模型自己算。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88]}),
            content_response(final_payload()),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("那下次呢？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(response.tool_calls[0].arguments["predict_x"], 5)
        self.assertEqual(response.reply.prediction, 91.5)

    def test_resume_restores_context_from_db(self):
        """点开历史会话（或服务重启后）能恢复对话与上次数据，追问直接复用。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 5}),
            content_response(final_payload()),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("四次成绩 78、82、85、88，第五次呢？", self.session)
        finally:
            agent.build_client = original
        self.assertIsNone(response.error)

        session_id = self.session.session_id
        records = db.list_session_records(session_id)
        self.assertEqual(len(records), 1)

        # 模拟服务重启：内存里的会话被清空，只剩数据库
        store.reset(session_id)
        restored = store.get_or_create(session_id)
        self.assertIsNone(restored.last_dataset)
        restored.restore_from_records(records)

        self.assertIsNotNone(restored.last_dataset)
        self.assertEqual(restored.last_dataset["y"], [78, 82, 85, 88])
        self.assertEqual(len(restored.messages), 2)
        self.assertIn("四次成绩整体呈", restored.messages[1]["content"])

    def test_invalid_json_triggers_repair(self):
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 5}),
            content_response("好的，成绩在上升。"),  # 不是 JSON
            content_response(final_payload()),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("四次 78、82、85、88，第五次呢？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(response.reply.prediction, 91.5)
        self.assertEqual(len(fake.calls), 3)  # 第一次回答 + 一次修复

    def test_empty_limitations_rejected_then_repaired(self):
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3], "y": [80, 84, 86], "predict_x": None}),
            content_response(final_payload(prediction=None, limitations=[])),
            content_response(final_payload(prediction=None, limitations=["样本只有 3 个点"])),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("三次成绩 80、84、86 看看趋势", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(response.reply.limitations, ["样本只有 3 个点"])

    def test_slope_mismatch_triggers_repair(self):
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3, 4], "y": [78, 82, 85, 88], "predict_x": 5}),
            content_response(final_payload(slope=9.9)),  # 与工具返回值不符
            content_response(final_payload()),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("四次 78、82、85、88，第五次大概多少？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(response.reply.slope, 3.3)

    def test_tool_error_is_explained_not_faked(self):
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2], "y": ["78", "八十二"], "predict_x": 3}),
            content_response({
                "answer": "这组数据里“八十二”不是数字，工具拒绝计算，请改成 82 后再试。",
                "trend": "无法判断",
                "slope": None,
                "intercept": None,
                "prediction": None,
                "prediction_note": None,
                "limitations": ["输入包含非数字字符，无法拟合"],
                "sources": [],
                "need_followup": "请确认第二次成绩是否为 82 分",
            }),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("第一次 78，第二次八十二，第三次会多少？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(response.reply.trend, "无法判断")
        self.assertFalse(response.tool_calls[0].ok)
        self.assertTrue(db.list_analyses(50) == [] or db.list_analyses(1)[0]["y"] != ["78", "八十二"])

    # ---- 输入异常：给明确说明 + 可操作卡片，且绝不输出无效预测 ----

    @staticmethod
    def _cannot_judge(answer: str, followup: str | None = None) -> dict:
        return {
            "answer": answer,
            "trend": "无法判断",
            "slope": None,
            "intercept": None,
            "prediction": None,
            "prediction_note": None,
            "limitations": ["本次工具未给出可用结果"],
            "sources": [],
            "need_followup": followup,
        }

    def test_insufficient_data_offers_blank_slots(self):
        """只有一次成绩：说明数据不足，并给空位让用户补，而不是让他重新打字。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1], "y": [82]}),
            content_response(self._cannot_judge(
                "目前只有 1 个成绩，无法拟合趋势；再补一次成绩我就能算斜率。",
                "第二次考试考了多少分？",
            )),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("我只有一次成绩 82 分，能看趋势吗？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(response.reply.trend, "无法判断")
        self.assertIsNone(response.reply.slope)
        self.assertIsNone(response.reply.prediction)
        self.assertEqual(response.fix["kind"], "add_scores")
        self.assertEqual(response.fix["scores"], ["82"])
        self.assertIn("至少需要 2 次", response.fix["hint"])

    def test_non_numeric_marks_only_the_bad_cell(self):
        """非数字成绩：预填原数据、只标红出错那一格。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2, 3], "y": [78, "八十二", 85]}),
            content_response(self._cannot_judge(
                "第 2 个成绩「八十二」不是数字，我无法计算，请把它改成阿拉伯数字。",
                "第二次成绩是 82 分吗？",
            )),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("三次成绩 78、八十二、85，看看趋势", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertIsNone(response.reply.prediction)
        self.assertEqual(response.fix["kind"], "edit_scores")
        self.assertEqual(response.fix["scores"], ["78", "八十二", "85"])
        self.assertEqual(response.fix["invalid"], [2])
        self.assertIn("不是数字", response.fix["hint"])

    def test_all_same_index_gives_renumber_card(self):
        """所有序号相同：斜率无定义，序号交给程序按 1、2、3… 重排。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [2, 2, 2], "y": [78, 82, 85]}),
            content_response(self._cannot_judge(
                "三次考试的序号都是 2，无法区分先后顺序，所以斜率没有定义。",
                "请确认这是第几次到第几次考试？",
            )),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("三次成绩 78、82、85", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertIsNone(response.reply.slope)
        self.assertEqual(response.fix["kind"], "edit_scores")
        self.assertTrue(response.fix["regenerate_index"])
        self.assertIn("所有考试序号", response.fix["hint"])

    def test_failed_tool_never_outputs_invented_prediction(self):
        """工具失败后模型仍编斜率/预测值：两次都被拦下，最终改用程序生成的说明。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 2], "y": [78, "八十二"]}),
            content_response(final_payload()),  # 编造 slope 3.3 / prediction 91.5
            content_response(final_payload()),  # 修复后又编了一次
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("第一次 78，第二次八十二，趋势怎么样？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(response.reply.trend, "无法判断")
        self.assertIsNone(response.reply.slope)
        self.assertIsNone(response.reply.intercept)
        self.assertIsNone(response.reply.prediction)
        self.assertNotIn("3.3", response.reply.answer)
        self.assertNotIn("91.5", response.reply.answer)
        self.assertIn("不是数字", response.reply.answer)
        # 仍然给出可操作的修改卡片
        self.assertEqual(response.fix["kind"], "edit_scores")
        self.assertIsNone(response.chart_url)

    def test_no_retry_after_invalid_input(self):
        """输入无效后不允许在同一轮里"删掉坏值再算一次"，避免替用户改数据。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1], "y": [82]}),
            tool_call_response("analyze_score_trend", {"x": [1, 2], "y": [82, 90]}),
            content_response(self._cannot_judge(
                "只给了 1 个成绩，需要再补一次才能算趋势。",
                "第二次考了多少分？",
            )),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("我只有一次成绩 82 分，能看趋势吗？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertEqual(len(response.tool_calls), 2)
        self.assertFalse(response.tool_calls[1].ok)
        self.assertIn("暂停", response.tool_calls[1].summary)
        self.assertIsNone(response.chart_url)
        self.assertIsNone(response.reply.prediction)

    def test_score_list_gap_detection(self):
        """成绩序列里的空缺要认出来；普通句子不能误伤。"""
        self.assertEqual(
            agent._score_list_gap("第一次 78，第二次缺考，第三次 85，帮我看看趋势"),
            ["78", "缺考", "85"],
        )
        self.assertEqual(agent._score_list_gap("我的成绩是 78、abc、85"), ["78", "abc", "85"])
        self.assertEqual(agent._score_list_gap("成绩：78、—、85"), ["78", "—", "85"])
        # 没有空缺、或不是"成绩列表"的说法，一律不判定
        self.assertIsNone(agent._score_list_gap("四次成绩 78、82、85、88，帮我看看趋势"))
        self.assertIsNone(agent._score_list_gap("第四次考试我没考，前三次是 78、82、85"))
        self.assertIsNone(agent._score_list_gap("我的成绩是 78、82、85、88 这四次，缺考那次不算"))

    def test_missing_score_blocks_fitting_and_asks(self):
        """用户原话里有一次缺考：模型偷偷丢掉那次也不行，必须让用户补全。"""
        fake = FakeClient([
            tool_call_response("analyze_score_trend", {"x": [1, 3], "y": [78, 85]}),
            content_response(self._cannot_judge(
                "第二次考试记的是“缺考”，我需要先确认这一次的分数才能算趋势。",
                "第二次考试考了多少分？",
            )),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("第一次 78，第二次缺考，第三次 85，帮我看看趋势", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        self.assertFalse(response.tool_calls[0].ok)
        self.assertIn("MISSING_VALUE", response.tool_calls[0].summary)
        self.assertIsNone(response.chart_url)
        self.assertEqual(response.reply.trend, "无法判断")
        self.assertEqual(response.fix["kind"], "edit_scores")
        self.assertEqual(response.fix["scores"], ["78", "缺考", "85"])
        self.assertEqual(response.fix["invalid"], [2])

    def test_program_notes_keep_tool_message_order(self):
        """一轮里多个工具调用：程序提示必须排在所有 tool 结果之后，否则服务方会判非法请求。"""
        fake = FakeClient([
            tool_calls_response(
                ("analyze_score_trend", {"x": [1, 3], "y": [78, 85]}),
                ("search_study_knowledge", {"query": "缺考对成绩趋势的影响"}),
            ),
            content_response(self._cannot_judge(
                "第二次考试记的是“缺考”，先确认这一次的分数我再算趋势。",
                "第二次考试考了多少分？",
            )),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("第一次 78，第二次缺考，第三次 85，帮我看看趋势", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        for sent in fake.calls:
            for index, message in enumerate(sent):
                if not message.get("tool_calls"):
                    continue
                expected = len(message["tool_calls"])
                block = []
                cursor = index + 1
                while cursor < len(sent) and sent[cursor].get("role") == "tool":
                    block.append(sent[cursor])
                    cursor += 1
                self.assertEqual(len(block), expected, "tool 结果必须紧跟在 tool_calls 之后，中间不能插入 system 消息")

    def test_retrieval_answer_persists_record(self):
        """知识检索问答也应写入分析记录，并带检索方式与来源片段。"""
        search_result = search_knowledge("如何提高复习效率", mode="keyword")
        sources = [
            {"title": h["title"], "chunk_id": h["chunk_id"], "snippet": h["snippet"]}
            for h in search_result["hits"]
        ]
        fake = FakeClient([
            tool_call_response("search_study_knowledge", {"query": "如何提高复习效率", "mode": "keyword"}),
            content_response({
                "answer": "根据资料，提高复习效率的关键是主动回忆与间隔重复。",
                "trend": "无法判断",
                "slope": None,
                "intercept": None,
                "prediction": None,
                "prediction_note": None,
                "limitations": ["检索结果仅覆盖资料库内容"],
                "sources": sources,
                "need_followup": None,
            }),
        ])
        original = agent.build_client
        agent.build_client = lambda: fake
        try:
            response = agent.handle_message("如何提高复习效率？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNone(response.error)
        # 只认本条测试写进去的记录（同一临时库里还有别的用例写的检索记录）
        records = [
            r
            for r in db.list_analyses(50)
            if r["kind"] == "retrieval" and "复习效率" in r["question"]
        ]
        self.assertTrue(records)
        record = records[-1]
        self.assertEqual(record["mode"], "keyword")
        self.assertTrue(record["sources"])
        self.assertIn("主动回忆", record["answer"])

        # 会话聚合接口对检索记录同样有效
        sessions = db.list_sessions(20)
        target = [s for s in sessions if s["session_id"] == self.session.session_id]
        self.assertTrue(target)
        self.assertEqual(target[0]["kind"], "retrieval")

    def test_llm_failure_returns_error_but_keeps_session(self):
        class Boom:
            def chat(self, *args, **kwargs):
                from app.llm import LLMError

                raise LLMError("模型鉴权失败（401）", "API Key 无效")

        original = agent.build_client
        agent.build_client = lambda: Boom()
        try:
            response = agent.handle_message("78、82、85、88 什么趋势？", self.session)
        finally:
            agent.build_client = original

        self.assertIsNotNone(response.error)
        self.assertIn("401", response.error)
        self.assertTrue(response.run_id)


class RecordDeletionTest(unittest.TestCase):
    """单条删除：只删指定那一条，同会话其他轮次保留；删空后会话卡片消失。"""

    def setUp(self):
        db.init_db()

    @staticmethod
    def _payload():
        return {
            "x": [1, 2, 3],
            "y": [70, 75, 80],
            "predict_x": 4,
            "slope": 5.0,
            "intercept": 65.0,
            "prediction": 85.0,
            "r2": 1.0,
            "chart_url": "/charts/fake.png",
        }

    def test_delete_one_record_keeps_other_turns(self):
        sid = "del-test-keep"
        first = db.save_analysis(sid, "删除测试第一轮", self._payload(), answer_text="第一轮回答")
        second = db.save_analysis(sid, "删除测试第二轮", self._payload(), answer_text="第二轮回答")

        result = db.delete_analysis(first)
        self.assertEqual(result["deleted"], 1)
        self.assertEqual(result["session_id"], sid)
        self.assertEqual(result["remaining"], 1)

        left = [r["id"] for r in db.list_session_records(sid)]
        self.assertEqual(left, [second])
        # 再删同一条：不报错，只是没删到东西
        again = db.delete_analysis(first)
        self.assertEqual(again["deleted"], 0)
        self.assertEqual(again["remaining"], 0)

    def test_delete_last_record_removes_session_from_list(self):
        sid = "del-test-last"
        rid = db.save_retrieval(
            sid,
            "错题本怎么用？",
            {"hits": [{"title": "学习效率与成绩提升", "chunk_id": "c1", "snippet": "片段"}], "mode": "keyword"},
            "错题本用于复盘。",
        )
        self.assertIn(sid, [s["session_id"] for s in db.list_sessions(100)])

        db.delete_analysis(rid)
        self.assertEqual(db.list_session_records(sid), [])
        self.assertNotIn(sid, [s["session_id"] for s in db.list_sessions(100)])

    def test_delete_whole_session_keeps_other_sessions(self):
        """删除整段会话：该会话所有轮次一起消失，别的会话不受影响。"""
        sid = "del-test-session"
        db.save_analysis(sid, "整段删除第一轮", self._payload(), answer_text="第一轮")
        db.save_analysis(sid, "整段删除第二轮", self._payload(), answer_text="第二轮")
        other = "del-test-session-other"
        db.save_analysis(other, "另一个会话", self._payload(), answer_text="别人的")

        result = db.delete_session(sid)
        self.assertEqual(result["deleted"], 2)
        self.assertEqual(db.list_session_records(sid), [])
        self.assertNotIn(sid, [s["session_id"] for s in db.list_sessions(100)])
        # 其他会话原样保留
        self.assertEqual(len(db.list_session_records(other)), 1)

        # 会话已经不存在：再删一次返回 0，接口层会转成 404
        self.assertEqual(db.delete_session(sid)["deleted"], 0)


class RecordSearchTest(unittest.TestCase):
    """搜索分析记录：关键词 / 会话 ID / 时间三种入口，可组合，命中到具体某一条。"""

    def setUp(self):
        db.init_db()
        self._exec("DELETE FROM analyses")

    @staticmethod
    def _payload(slope=5.0):
        return {
            "x": [1, 2, 3],
            "y": [70, 75, 80],
            "predict_x": 4,
            "slope": slope,
            "intercept": 65.0,
            "prediction": 85.0,
            "r2": 1.0,
            "chart_url": "/charts/fake.png",
        }

    @staticmethod
    def _exec(sql: str, params: tuple = ()) -> None:
        """直接改库（造测试数据用）：提交后立刻关闭连接，避免未关闭连接的告警。"""
        conn = sqlite3.connect(settings.db_path)
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _set_time(self, record_id: int, iso: str) -> None:
        self._exec("UPDATE analyses SET created_at = ? WHERE id = ?", (iso, record_id))

    def test_keyword_matches_question_and_answer(self):
        sid = "search-kw-session"
        first = db.save_analysis(sid, "四次成绩看趋势", self._payload(), answer_text="整体呈上升趋势")
        second = db.save_analysis(sid, "如何提高复习效率", self._payload(), answer_text="用错题本复盘")
        db.save_analysis("search-kw-other", "无关问题", self._payload(), answer_text="无关回答")

        by_question = db.search_records(query="看趋势")
        self.assertEqual([r["id"] for r in by_question], [first])

        # 关键词也搜回答原文
        by_answer = db.search_records(query="错题本")
        self.assertEqual([r["id"] for r in by_answer], [second])
        self.assertEqual(by_answer[0]["turn_index"], 2)  # 同一会话的第二轮

        # 搜不到就是空列表，不抛错
        self.assertEqual(db.search_records(query="量子纠缠"), [])

    def test_session_id_prefix_returns_that_session_only(self):
        sid = "a1b2c3d4e5f60718293a4b5c6d7e8f90"
        db.save_analysis(sid, "第一轮问题", self._payload(), answer_text="回答一")
        db.save_analysis(sid, "第二轮问题", self._payload(), answer_text="回答二")
        db.save_analysis("ffffffff0000", "别的会话", self._payload(), answer_text="别人的")

        hits = db.search_records(session_id="a1b2c3d4")
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(r["session_id"] == sid for r in hits))
        self.assertEqual(sorted(r["turn_index"] for r in hits), [1, 2])

    def test_date_filter_uses_record_day(self):
        old = db.save_analysis("search-date-a", "前天的记录", self._payload(), answer_text="旧")
        new = db.save_analysis("search-date-b", "今天的记录", self._payload(), answer_text="新")
        self._set_time(old, "2026-10-28T09:30:00+08:00")
        self._set_time(new, "2026-10-30T11:45:00+08:00")

        self.assertEqual([r["id"] for r in db.search_records(date_from="2026-10-30", date_to="2026-10-30")], [new])
        self.assertEqual([r["id"] for r in db.search_records(date_from="2026-10-01", date_to="2026-10-31")], [new, old])
        self.assertEqual(db.search_records(date_from="2026-11-01", date_to="2026-11-30"), [])

    def test_filters_combine(self):
        sid = "combo-a1b2"
        a = db.save_analysis(sid, "趋势分析一次", self._payload(), answer_text="A")
        db.save_analysis(sid, "学习方法问题", self._payload(), answer_text="B")
        db.save_analysis("combo-ffff", "趋势分析二次", self._payload(), answer_text="C")
        self._set_time(a, "2026-10-30T08:00:00+08:00")
        # 同会话另一轮挪到上个月：只有"关键词 + 会话 + 时间"同时命中才会剩下 a
        for record in db.list_session_records(sid):
            if record["id"] != a:
                self._set_time(record["id"], "2026-09-15T10:00:00+08:00")

        hits = db.search_records(query="趋势分析", session_id=sid, date_from="2026-10-30", date_to="2026-10-30")
        self.assertEqual([r["id"] for r in hits], [a])
        # 去掉时间条件：同会话的两轮里，只有"趋势分析"这条命中
        self.assertEqual([r["id"] for r in db.search_records(query="趋势分析", session_id=sid)], [a])
        # 去掉会话条件：两个会话各有一条含"趋势分析"的记录
        self.assertEqual(len(db.search_records(query="趋势分析")), 2)

    def test_parse_date_range_variants(self):
        self.assertEqual(db.parse_date_range("2026-10-30"), ("2026-10-30", "2026-10-30"))
        self.assertEqual(db.parse_date_range("2026/10/30"), ("2026-10-30", "2026-10-30"))
        self.assertEqual(db.parse_date_range("2026-10"), ("2026-10-01", "2026-10-31"))
        self.assertEqual(db.parse_date_range("2026-12"), ("2026-12-01", "2026-12-31"))
        self.assertEqual(db.parse_date_range("2026"), ("2026-01-01", "2026-12-31"))
        self.assertIsNone(db.parse_date_range("上周"))
        self.assertIsNone(db.parse_date_range("2026-13-40"))
        self.assertEqual(db.parse_date_range("10-30")[0][5:], "10-30")


class FailureRecordTest(unittest.TestCase):
    """模型/工具调用失败要落库留痕（kind=error），保留排查信息，且不覆盖成功记录。"""

    def setUp(self):
        db.init_db()
        FailureRecordTest._exec("DELETE FROM analyses")

    @staticmethod
    def _exec(sql: str, params: tuple = ()) -> None:
        conn = sqlite3.connect(settings.db_path)
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _payload():
        return {
            "x": [1, 2],
            "y": [70, 78],
            "predict_x": None,
            "slope": 8.0,
            "intercept": 62.0,
            "prediction": None,
            "r2": 1.0,
            "chart_url": None,
        }

    def test_failure_record_persists_troubleshooting_info(self):
        sid = "fail-record-a"
        rid = db.save_failure(sid, "帮我预测成绩", "失败原因：连接超时\n排查信息：run_id=abc123")
        record = db.list_session_records(sid)[0]
        self.assertEqual(record["kind"], "error")
        self.assertIn("run_id=abc123", record["answer"])
        self.assertIsNone(record["slope"])  # 失败记录没有任何数值结果

        # 同问题再次失败：更新原记录而不是新增
        rid2 = db.save_failure(sid, "帮我预测成绩", "失败原因：401\n排查信息：run_id=def456")
        self.assertEqual(rid, rid2)
        self.assertEqual(len(db.list_session_records(sid)), 1)
        self.assertIn("run_id=def456", db.list_session_records(sid)[0]["answer"])

    def test_failure_does_not_overwrite_success_record(self):
        sid = "fail-record-b"
        db.save_analysis(sid, "四次成绩看趋势", self._payload(), answer_text="成功回答")
        db.save_failure(sid, "四次成绩看趋势", "失败原因：模型超时")

        # 成功记录保留，失败另起一条（真实历史：同一个问题后来一次失败了）
        records = db.list_session_records(sid)
        self.assertEqual(
            sorted(r["kind"] for r in records),
            ["error", "trend"],
        )
        success = next(r for r in records if r["kind"] == "trend")
        self.assertEqual(success["answer"], "成功回答")
        failure = next(r for r in records if r["kind"] == "error")
        self.assertIn("模型超时", failure["answer"])

    def test_error_session_shows_in_list_and_search(self):
        sid = "fail-record-c"
        db.save_failure(sid, "连接失败的问题", "失败原因：网络错误\n排查信息：run_id=xyz789")
        sessions = db.list_sessions(50)
        target = next(s for s in sessions if s["session_id"] == sid)
        self.assertEqual(target["kind"], "error")
        # 搜索也能按排查信息里的关键词命中失败记录
        hits = db.search_records(query="run_id=xyz789")
        self.assertTrue(any(h["session_id"] == sid for h in hits))


if __name__ == "__main__":
    unittest.main()
