"""会话记忆：保存每个会话的对话历史与最近一次数据集，支持连续追问。

连续追问的关键不是"记住所有文字"，而是把上一轮的结构化数据（x/y）显式带到下一轮上下文，
模型在用户说"第五次呢"时才能直接复用上次数据，而不是重新询问。
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_dataset: dict[str, Any] | None = None

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def context_note(self) -> str:
        """把最近一次数据集渲染成给模型看的上下文说明。"""
        if not self.last_dataset:
            return "当前会话还没有任何成绩数据，用户首次提问若未提供数据，应先请用户提供。"
        data = self.last_dataset
        return (
            "当前会话最近一次分析的数据（可直接复用，不需要用户再提供）：\n"
            f"- x（考试序号）= {data['x']}\n"
            f"- y（成绩）= {data['y']}\n"
            f"- 上次拟合斜率 = {data.get('slope')}，截距 = {data.get('intercept')}，R² = {data.get('r2')}\n"
            "用户再次追问（例如“第五次呢”）时，直接基于这组数据调用工具，不要重复询问成绩。"
        )

    def trim(self, max_messages: int = 24) -> None:
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def restore_from_records(self, records: list[dict[str, Any]]) -> None:
        """用数据库里的历史记录恢复会话上下文（点开旧会话继续追问、或服务重启后续聊）。

        只恢复两部分：对话文本（供模型理解指代）+ 最近一次数据集（供“第五次呢”复用）。
        """
        if not records:
            return
        self.messages = []
        for record in records:
            text = record.get("answer") or _fallback_answer(record)
            self.messages.append({"role": "user", "content": record["question"]})
            self.messages.append({"role": "assistant", "content": text})

        # 从最近一条有数据的趋势记录恢复 last_dataset，保证追问能直接复用
        for record in reversed(records):
            if record.get("kind", "trend") == "trend" and record.get("y"):
                self.last_dataset = {
                    "x": record.get("x"),
                    "y": record.get("y"),
                    "slope": record.get("slope"),
                    "intercept": record.get("intercept"),
                    "r2": record.get("r2"),
                }
                break
        self.trim()


def _fallback_answer(record: dict[str, Any]) -> str:
    """早期记录没有保存回答原文时，用已存的数值拼一句，避免恢复出空对话。"""
    if record.get("kind") == "retrieval":
        count = len(record.get("sources") or [])
        return f"（历史记录）检索了本地资料，命中 {count} 个片段。" if count else "（历史记录）知识库未覆盖该问题，属于资料不足。"
    parts = []
    if record.get("slope") is not None:
        parts.append(f"斜率 {record['slope']} 分/次")
    if record.get("intercept") is not None:
        parts.append(f"截距 {record['intercept']}")
    if record.get("r2") is not None:
        parts.append(f"R² {record['r2']}")
    if record.get("prediction") is not None:
        parts.append(f"预测第 {record.get('predict_x')} 次为 {record['prediction']} 分")
    detail = "，".join(parts) or "无拟合结果"
    return f"（早期记录未保存回答原文，仅保留计算结果：{detail}）"


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            key = session_id or uuid.uuid4().hex
            if key not in self._sessions:
                self._sessions[key] = Session(session_id=key)
            return self._sessions[key]

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


store = SessionStore()
