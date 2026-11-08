"""FastAPI 入口：网页 → 后端接口 → Agent → 工具 → 数据库。

启动：python -m uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import db
from app.agent import handle_message
from app.session import store
from app.tools.retrieval import knowledge_base
from config import BASE_DIR, settings

app = FastAPI(title="学习与成绩分析助手", version="1.0.0")

STATIC_DIR = Path(BASE_DIR) / "app" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/charts", StaticFiles(directory=settings.chart_dir), name="charts")


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "mock_mode": settings.mock,
        "model": settings.model,
        "base_url": settings.base_url,
        "api_key_configured": bool(settings.api_key),
        "knowledge_files": knowledge_base.file_count,
        "knowledge_chunks": len(knowledge_base.chunks),
        # 会话表是内存结构：有多少条、有没有过期清理、有多少次从库里回填
        "sessions": store.stats(),
    }


@app.post("/api/chat")
def chat(payload: ChatRequest, x_client_source: str | None = Header(default=None)) -> JSONResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="问题不能为空")

    # 验证脚本会带 X-Client-Source: validation，这类记录只作证据留档，不进用户的分析记录列表
    source = db.SOURCE_VALIDATION if x_client_source == db.SOURCE_VALIDATION else db.SOURCE_WEB
    session = store.get_or_create(payload.session_id)
    response = handle_message(message, session, source=source)
    return JSONResponse(content=response.model_dump())


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest, x_client_source: str | None = Header(default=None)):
    """带进度的对话接口（SSE）。

    一次分析可能要经历"理解 → 调工具 → 校验 → 可能修复"好几轮，十几秒没有反馈
    用户会以为卡死。这里把每个阶段推给前端；最终结果仍是一条完整 JSON，
    前端解析逻辑与 /api/chat 完全一致，接口挂了也能回退。
    """
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="问题不能为空")

    source = db.SOURCE_VALIDATION if x_client_source == db.SOURCE_VALIDATION else db.SOURCE_WEB
    session = store.get_or_create(payload.session_id)

    def event_stream():
        events: queue.Queue = queue.Queue()

        def progress(phase: str, data: dict) -> None:
            events.put({"type": "progress", "phase": phase, **(data or {})})

        def worker() -> None:
            try:
                response = handle_message(message, session, source=source, progress=progress)
                events.put({"type": "result", "payload": response.model_dump()})
            except Exception as exc:  # 后台线程里的异常必须转成事件，否则前端会一直等
                events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            finally:
                events.put(None)

        yield _sse({"type": "start", "session_id": session.session_id})
        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = events.get()
            if item is None:
                break
            yield _sse(item)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/history")
def history(limit: int = 20, include_validation: bool = False, group: bool = True) -> JSONResponse:
    if group:
        # 默认按会话聚合：一次会话（含多轮追问）显示为一条记录
        return JSONResponse(
            content={"sessions": db.list_sessions(limit=limit, include_validation=include_validation)}
        )
    return JSONResponse(
        content={"items": db.list_analyses(limit=limit, include_validation=include_validation)}
    )


@app.get("/api/search")
def search_history(
    q: str = "",
    session_id: str = "",
    date: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 50,
) -> JSONResponse:
    """搜索分析记录：关键词 / 会话 ID / 时间，可组合使用。

    date 支持「2026-10-30」「2026-10」「2026」「10-30」等写法，由 db.parse_date_range 解析；
    解析不出来直接 400，避免用户以为"没搜到"其实是格式没被识别。
    """
    q = q.strip()[:100]
    session_id = session_id.strip()[:64]
    raw_date = date.strip()

    if not (q or session_id or raw_date or date_from or date_to):
        raise HTTPException(status_code=400, detail="请给出关键词、会话 ID 或时间中的至少一项")

    if raw_date:
        parsed = db.parse_date_range(raw_date)
        if parsed is None:
            raise HTTPException(
                status_code=400,
                detail=f"时间「{raw_date}」无法识别，可以试试 2026-10-30、2026-10 或 2026",
            )
        date_from, date_to = parsed

    items = db.search_records(
        query=q or None,
        session_id=session_id or None,
        date_from=date_from.strip() or None,
        date_to=date_to.strip() or None,
        limit=limit,
    )
    return JSONResponse(
        content={
            "items": items,
            "count": len(items),
            "filters": {
                "q": q,
                "session_id": session_id,
                "date_from": date_from or None,
                "date_to": date_to or None,
            },
        }
    )


@app.delete("/api/history")
def clear_history(source: str | None = None) -> dict:
    deleted = db.clear_analyses(source=source)
    return {"ok": True, "deleted": deleted}


@app.delete("/api/history/{record_id}")
def delete_history_record(record_id: int) -> JSONResponse:
    """删除单条分析记录。

    只删这条历史记录本身；如果这条记录是当前正在对话的会话产生的，
    内存里的上下文不会被清掉，用户可以继续追问。
    """
    result = db.delete_analysis(record_id)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="这条记录不存在，可能已被删除")
    return JSONResponse(content={"ok": True, **result})


@app.get("/api/session/{session_id}")
def get_session(session_id: str) -> JSONResponse:
    """读取某个历史会话：返回全部轮次，并把对话上下文恢复到会话记忆里。

    这样点开旧会话后可以直接继续追问，服务重启过也能恢复（上下文从数据库回填）。
    """
    records = db.list_session_records(session_id)
    if not records:
        raise HTTPException(status_code=404, detail="该会话没有可恢复的记录")

    session = store.get_or_create(session_id)
    if not session.messages:
        session.restore_from_records(records)

    return JSONResponse(
        content={
            "session_id": session_id,
            "turns": records,
            "has_dataset": session.last_dataset is not None,
            "turn_count": len(records),
        }
    )


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str) -> JSONResponse:
    """删除整个会话（该会话的全部轮次），并清掉内存里的对话上下文。

    会话删掉后再追问也不会复活：记录没了，上下文也重置了。
    """
    result = db.delete_session(session_id)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="这个会话不存在，可能已被删除")
    store.reset(session_id)
    return JSONResponse(content={"ok": True, **result})


@app.post("/api/session/{session_id}/reset")
def reset(session_id: str) -> dict:
    store.reset(session_id)
    return {"ok": True, "session_id": session_id}
