"""SQLite 持久化：保存每次分析记录，支持按时间倒序查看。"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from contextlib import closing
from datetime import date as _date, datetime, timedelta
from typing import Any

from config import settings

_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    question     TEXT    NOT NULL,
    x_json       TEXT    NOT NULL,
    y_json       TEXT    NOT NULL,
    predict_x    REAL,
    slope        REAL,
    intercept    REAL,
    prediction   REAL,
    r2           REAL,
    chart_url    TEXT,
    created_at   TEXT    NOT NULL,
    source       TEXT    NOT NULL DEFAULT 'web',
    kind         TEXT    NOT NULL DEFAULT 'trend',
    answer_text  TEXT,
    sources_json TEXT,
    mode         TEXT
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
"""

# 来源标记：web = 真人在页面上提问；validation = scripts/run_validation.py 生成的验证证据。
# 验证记录只作为证据留档，默认不进用户的分析记录列表，否则跑几次验证就会把列表刷满重复项。
SOURCE_WEB = "web"
SOURCE_VALIDATION = "validation"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _LOCK, closing(_connect()) as conn:
        conn.executescript(_SCHEMA)
        # 老库就地迁移：缺哪列补哪列，避免用户已有记录丢失
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(analyses)")}
        migrations = {
            "source": "ALTER TABLE analyses ADD COLUMN source TEXT NOT NULL DEFAULT 'web'",
            "kind": "ALTER TABLE analyses ADD COLUMN kind TEXT NOT NULL DEFAULT 'trend'",
            "answer_text": "ALTER TABLE analyses ADD COLUMN answer_text TEXT",
            "sources_json": "ALTER TABLE analyses ADD COLUMN sources_json TEXT",
            "mode": "ALTER TABLE analyses ADD COLUMN mode TEXT",
        }
        for column, statement in migrations.items():
            if column not in columns:
                conn.execute(statement)
        conn.commit()


def save_analysis(session_id: str, question: str, result: dict[str, Any],
                  source: str = SOURCE_WEB, answer_text: str | None = None) -> int:
    """保存一次分析。

    同一个会话里重复问完全相同的问题时做覆盖更新而不是新增，
    避免用户手抖重复发送、或反复点按钮把列表刷成一堆重复项。
    """
    payload = {
        "session_id": session_id,
        "question": question,
        "x_json": json.dumps(result.get("x"), ensure_ascii=False),
        "y_json": json.dumps(result.get("y"), ensure_ascii=False),
        "predict_x": result.get("predict_x"),
        "slope": result.get("slope"),
        "intercept": result.get("intercept"),
        "prediction": result.get("prediction"),
        "r2": result.get("r2"),
        "chart_url": result.get("chart_url"),
        "answer_text": answer_text,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
    }
    with _LOCK, closing(_connect()) as conn:
        existing = conn.execute(
            "SELECT id FROM analyses WHERE session_id = ? AND question = ? AND source = ?",
            (session_id, question, source),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE analyses SET x_json = :x_json, y_json = :y_json, predict_x = :predict_x,
                       slope = :slope, intercept = :intercept, prediction = :prediction, r2 = :r2,
                       chart_url = :chart_url, answer_text = :answer_text, created_at = :created_at
                   WHERE id = :id""",
                {**payload, "id": existing["id"]},
            )
            conn.commit()
            return int(existing["id"])
        cursor = conn.execute(
            """INSERT INTO analyses
               (session_id, question, x_json, y_json, predict_x, slope, intercept, prediction, r2,
                chart_url, answer_text, created_at, source)
               VALUES (:session_id, :question, :x_json, :y_json, :predict_x, :slope, :intercept,
                       :prediction, :r2, :chart_url, :answer_text, :created_at, :source)""",
            payload,
        )
        conn.commit()
        return int(cursor.lastrowid)


def save_retrieval(session_id: str, question: str, result: dict[str, Any],
                   answer_text: str, source: str = SOURCE_WEB) -> int:
    """保存一次知识检索问答（kind=retrieval），记录检索方式与命中片段。

    覆盖规则与 save_analysis 一致：同会话同问题做更新而不是新增。
    """
    payload = {
        "session_id": session_id,
        "question": question,
        "kind": "retrieval",
        "answer_text": answer_text,
        "sources_json": json.dumps(result.get("hits") or [], ensure_ascii=False),
        "mode": result.get("mode") or "hybrid",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
    }
    with _LOCK, closing(_connect()) as conn:
        existing = conn.execute(
            "SELECT id FROM analyses WHERE session_id = ? AND question = ? AND source = ?",
            (session_id, question, source),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE analyses SET kind = :kind, answer_text = :answer_text,
                       sources_json = :sources_json, mode = :mode, created_at = :created_at
                   WHERE id = :id""",
                {**payload, "id": existing["id"]},
            )
            conn.commit()
            return int(existing["id"])
        cursor = conn.execute(
            """INSERT INTO analyses
               (session_id, question, x_json, y_json, kind, answer_text, sources_json, mode,
                created_at, source)
               VALUES (:session_id, :question, '[]', '[]', :kind, :answer_text, :sources_json,
                       :mode, :created_at, :source)""",
            payload,
        )
        conn.commit()
        return int(cursor.lastrowid)


def save_failure(
    session_id: str,
    question: str,
    detail: str,
    source: str = SOURCE_WEB,
) -> int:
    """保存一次"模型或工具调用失败"的记录（kind=error），留作失败状态与排查信息的证据。

    排查信息写入 answer_text：失败原因 + run_id + 已执行的工具等，前端详情弹窗直接展示。
    覆盖规则：只在"同会话同问题的历史失败记录"上更新；绝不覆盖同问题的成功分析记录。
    """
    payload = {
        "session_id": session_id,
        "question": question,
        "kind": "error",
        "answer_text": detail,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
    }
    with _LOCK, closing(_connect()) as conn:
        existing = conn.execute(
            """SELECT id FROM analyses
               WHERE session_id = ? AND question = ? AND source = ? AND kind = 'error'""",
            (session_id, question, source),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE analyses SET answer_text = :answer_text, created_at = :created_at
                   WHERE id = :id""",
                {**payload, "id": existing["id"]},
            )
            conn.commit()
            return int(existing["id"])
        cursor = conn.execute(
            """INSERT INTO analyses
               (session_id, question, x_json, y_json, kind, answer_text, created_at, source)
               VALUES (:session_id, :question, '[]', '[]', :kind, :answer_text, :created_at, :source)""",
            payload,
        )
        conn.commit()
        return int(cursor.lastrowid)


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    kind = row["kind"] if "kind" in row.keys() else "trend"
    record: dict[str, Any] = {
        "id": row["id"],
        "session_id": row["session_id"],
        "question": row["question"],
        "kind": kind,
        "created_at": row["created_at"],
        "source": row["source"] if "source" in row.keys() else SOURCE_WEB,
    }
    if kind == "error":
        # 失败记录：answer 存排查信息（原因 + run_id + 已执行工具），无任何数值结果
        record["answer"] = row["answer_text"] or ""
        record["x"] = []
        record["y"] = []
        record["slope"] = None
        record["intercept"] = None
        record["prediction"] = None
        record["r2"] = None
        record["chart_url"] = None
        return record
    if kind == "retrieval":
        record["answer"] = row["answer_text"] or ""
        record["sources"] = json.loads(row["sources_json"] or "[]")
        record["mode"] = row["mode"] or "hybrid"
        record["x"] = []
        record["y"] = []
    else:
        record["x"] = json.loads(row["x_json"])
        record["y"] = json.loads(row["y_json"])
        record["answer"] = row["answer_text"] or ""
        record["predict_x"] = row["predict_x"]
        record["slope"] = row["slope"]
        record["intercept"] = row["intercept"]
        record["prediction"] = row["prediction"]
        record["r2"] = row["r2"]
        record["chart_url"] = row["chart_url"]
    return record


def list_analyses(limit: int = 20, include_validation: bool = False) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 100))
    where = "" if include_validation else f"WHERE source = '{SOURCE_WEB}'"
    with _LOCK, closing(_connect()) as conn:
        rows = conn.execute(
            f"SELECT * FROM analyses {where} ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def parse_date_range(text: str) -> tuple[str, str] | None:
    """把用户输入的时间说法解析成 [起始日, 结束日]（YYYY-MM-DD，含首尾）。

    支持这几种常见写法：
        2026-10-30 / 2026/10/30 / 2026.10.30  → 当天
        2026-10                               → 整月
        2026                                  → 整年
        10-30（不带年份）                      → 当年的这一天
    解析不出来返回 None，由调用方给出"格式看不懂"的提示。
    """
    raw = (text or "").strip()
    if not raw:
        return None
    parts = [p for p in re.split(r"\D+", raw) if p]
    if not parts or len(parts) > 3:
        return None
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        return None

    today = datetime.now()
    try:
        if len(numbers) == 1:
            # 只有一段：4 位数字按年份（整年），否则无法判断是月还是日
            if len(parts[0]) == 4:
                year = numbers[0]
                return f"{year:04d}-01-01", f"{year:04d}-12-31"
            return None
        if len(numbers) == 2:
            if len(parts[0]) == 4:  # 2026-10 → 整月
                year, month = numbers
                start = _date(year, month, 1)
                nxt = _date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1)
                return start.isoformat(), (nxt - timedelta(days=1)).isoformat()
            month, day = numbers  # 10-30 → 当年这一天
            start = _date(today.year, month, day)
            return start.isoformat(), start.isoformat()
        # 三段：年月日；年份不足 4 位时按"月日+忽略"处理没有意义，仍按年份补全
        year, month, day = numbers
        start = _date(year, month, day)
        return start.isoformat(), start.isoformat()
    except ValueError:
        return None


def search_records(
    query: str | None = None,
    session_id: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 50,
    include_validation: bool = False,
) -> list[dict[str, Any]]:
    """搜索分析记录：关键词 / 会话 ID / 时间三种条件，可任意组合（组合即"同时满足"）。

    - 关键词：同时匹配问题、回答原文和会话 ID（所以只记得会话号的一部分也能搜到）；
    - 会话 ID：按前缀匹配，输入前 8 位即可；
    - 时间：传入 YYYY-MM-DD 区间，按 created_at 的日期部分比较。

    返回命中的每一条记录（不是会话聚合），并带上 turn_index 标明它是该会话的第几轮。
    """
    limit = max(1, min(int(limit), 100))
    clauses: list[str] = [] if include_validation else [f"source = '{SOURCE_WEB}'"]
    params: list[Any] = []

    keyword = (query or "").strip()
    sid = (session_id or "").strip()

    if sid:
        clauses.append("session_id LIKE ?")
        params.append(f"{sid}%")
    if keyword:
        clauses.append("(question LIKE ? OR IFNULL(answer_text, '') LIKE ? OR session_id LIKE ?)")
        like = f"%{keyword}%"
        params += [like, like, f"{keyword}%"]
    if date_from:
        clauses.append("substr(created_at, 1, 10) >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("substr(created_at, 1, 10) <= ?")
        params.append(date_to)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _LOCK, closing(_connect()) as conn:
        rows = conn.execute(
            f"SELECT * FROM analyses {where} ORDER BY id DESC LIMIT ?", (*params, limit)
        ).fetchall()
        records: list[dict[str, Any]] = []
        for row in rows:
            record = _row_to_record(row)
            # 第几轮：同一会话里 id 不大于它的记录数，和列表里"第N轮"的口径一致
            record["turn_index"] = int(
                conn.execute(
                    "SELECT COUNT(*) AS c FROM analyses WHERE session_id = ? AND id <= ?",
                    (row["session_id"], row["id"]),
                ).fetchone()["c"]
            )
            records.append(record)
    return records


def list_sessions(limit: int = 20, include_validation: bool = False) -> list[dict[str, Any]]:
    """按会话聚合分析记录：一次会话（含多轮追问）只占一条记录。

    排序按会话最近一次活动倒序；组内按轮次正序，便于从第一轮看到最新一轮。
    """
    limit = max(1, min(int(limit), 100))
    where = "" if include_validation else f"WHERE source = '{SOURCE_WEB}'"
    with _LOCK, closing(_connect()) as conn:
        rows = conn.execute(f"SELECT * FROM analyses {where} ORDER BY id DESC").fetchall()

    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:  # 已按 id 倒序，第一次遇到的即该会话最新一轮
        sid = row["session_id"]
        if sid not in groups:
            if len(order) >= limit:
                continue
            groups[sid] = {
                "session_id": sid,
                "updated_at": row["created_at"],
                "started_at": row["created_at"],
                "title": row["question"],
                "records": [],
            }
            order.append(sid)
        group = groups[sid]
        group["records"].append(_row_to_record(row))
        # 倒序遍历，最后写入的就是最早一轮
        group["started_at"] = row["created_at"]
        group["title"] = row["question"]

    sessions = []
    for sid in order:
        group = groups[sid]
        group["records"].reverse()
        group["turn_count"] = len(group["records"])
        latest = group["records"][-1]
        group["slope"] = latest.get("slope")
        group["prediction"] = latest.get("prediction")
        group["kind"] = latest.get("kind", "trend")
        sessions.append(group)
    return sessions


def list_session_records(session_id: str) -> list[dict[str, Any]]:
    """取出某个会话的全部记录（按时间正序），用于点开会话时恢复对话上下文。"""
    with _LOCK, closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT * FROM analyses WHERE session_id = ? ORDER BY id ASC", (session_id,)
        ).fetchall()
    return [_row_to_record(row) for row in rows]


def delete_analysis(record_id: int) -> dict[str, Any]:
    """删除单条分析记录（按 id）。

    返回删除结果，并带回该记录所属会话还剩几轮，
    前端据此决定是刷新列表还是提示"这个会话已空，卡片已移除"。
    只动数据库，不影响内存里正在进行的对话上下文。
    """
    with _LOCK, closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, session_id, question FROM analyses WHERE id = ?", (int(record_id),)
        ).fetchone()
        if row is None:
            return {"deleted": 0, "session_id": None, "question": None, "remaining": 0}
        conn.execute("DELETE FROM analyses WHERE id = ?", (int(record_id),))
        conn.commit()
        remaining = conn.execute(
            "SELECT COUNT(*) AS c FROM analyses WHERE session_id = ?", (row["session_id"],)
        ).fetchone()["c"]
    return {
        "deleted": 1,
        "session_id": row["session_id"],
        "question": row["question"],
        "remaining": int(remaining),
    }


def delete_session(session_id: str) -> dict[str, Any]:
    """删除某个会话的全部记录（整段对话）。

    返回删掉的轮数，前端据此提示"已删除 N 轮"。
    """
    with _LOCK, closing(_connect()) as conn:
        found = conn.execute(
            "SELECT COUNT(*) AS c FROM analyses WHERE session_id = ?", (session_id,)
        ).fetchone()["c"]
        deleted = 0
        if found:
            cursor = conn.execute("DELETE FROM analyses WHERE session_id = ?", (session_id,))
            deleted = int(cursor.rowcount)
            conn.commit()
    return {"deleted": deleted, "session_id": session_id, "remaining": 0}


def clear_analyses(source: str | None = None) -> int:
    """清空分析记录；指定 source 时只清该来源。返回删除条数。"""
    with _LOCK, closing(_connect()) as conn:
        if source:
            cursor = conn.execute("DELETE FROM analyses WHERE source = ?", (source,))
        else:
            cursor = conn.execute("DELETE FROM analyses")
        conn.commit()
        return int(cursor.rowcount)
