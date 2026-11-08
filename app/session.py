"""会话记忆：保存每个会话的对话历史与最近一次数据集，支持连续追问。

连续追问的关键不是"记住所有文字"，而是把上一轮的结构化数据（x/y）显式带到下一轮上下文，
模型在用户说"第五次呢"时才能直接复用上次数据，而不是重新询问。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app import db


@dataclass
class Session:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    last_dataset: dict[str, Any] | None = None
    # 每个会话一把锁：同一会话的并发请求排队执行，避免两条消息交错写进 messages
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # 最后活跃时间（monotonic 秒），供 SessionStore 做过期清理
    last_active: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_active = time.monotonic()

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
    """内存会话表：有上限、会过期、未命中时从数据库回填。

    之前只用一个只增不减的 dict：长跑会缓慢涨内存，服务重启后同一 session_id
    会拿到一个空白会话（前端 localStorage 里还存着这个 id），追问就"失忆"了。
    现在未命中先查库，把历史对话与最近一次数据集恢复回来。
    """

    def __init__(self, max_sessions: int = 200, ttl_seconds: float = 2 * 3600) -> None:
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds
        self.restored_count = 0

    def get_or_create(self, session_id: str | None) -> Session:
        key = session_id or uuid.uuid4().hex
        now = time.monotonic()

        with self._lock:
            self._sweep(now)
            existing = self._sessions.get(key)
            if existing is not None:
                existing.touch()
                self._sessions.move_to_end(key)
                return existing

        # 未命中：可能是服务重启过，也可能是很久没说话被清掉了 —— 从数据库回填。
        # IO 放在锁外，避免一次查库把整个会话表卡住。
        session = Session(session_id=key)
        if session_id:
            try:
                records = db.list_session_records(key)
            except Exception:  # 数据库异常不应阻断对话，退化成空会话
                records = []
            if records:
                session.restore_from_records(records)
                with self._lock:
                    self.restored_count += 1

        with self._lock:
            # 并发下可能已有别的请求先建好了，以先到者为准
            existing = self._sessions.get(key)
            if existing is not None:
                return existing
            session.touch()
            self._sessions[key] = session
            self._sweep(time.monotonic())
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)
        return session

    def _sweep(self, now: float) -> None:
        """清理空闲超时的会话（调用方需已持有 self._lock）。"""
        if self._ttl <= 0:
            return
        expired = [key for key, item in self._sessions.items() if now - item.last_active > self._ttl]
        for key in expired:
            self._sessions.pop(key, None)

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "in_memory": len(self._sessions),
                "max_sessions": self._max_sessions,
                "ttl_seconds": int(self._ttl),
                "restored_from_db": self.restored_count,
            }


store = SessionStore()
