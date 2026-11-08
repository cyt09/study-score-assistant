"""结构化输出定义与字段校验。

模型返回的文本必须能被解析成 FinalAnswer 并通过这里的校验，否则由 agent 触发一次修复重试；
仍然失败就把原始输出连同错误信息返回给前端（不静默丢弃）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Trend = Literal["上升", "下降", "基本平稳", "无法判断"]


class Source(BaseModel):
    """检索命中的资料片段，必须能追溯到具体文件与片段。"""

    title: str = Field(min_length=1, description="资料文件名或标题")
    chunk_id: str = Field(min_length=1, description="片段编号，如 学习效率与成绩提升.md#2")
    snippet: str = Field(min_length=1, description="命中的原文片段")


class FinalAnswer(BaseModel):
    """最终回答的结构。字段缺失或类型错误都会被拒绝。"""

    answer: str = Field(min_length=10, description="给用户看的自然语言结论")
    trend: Trend
    slope: float | None = None
    intercept: float | None = None
    prediction: float | None = None
    prediction_note: str | None = None
    limitations: list[str] = Field(min_length=1, description="本次结论的限制条件，至少一条")
    sources: list[Source] = Field(default_factory=list)
    need_followup: str | None = None

    @field_validator("limitations")
    @classmethod
    def _clean_limitations(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        if not cleaned:
            raise ValueError("limitations 不能为空，必须说明结论的适用边界")
        return cleaned

    @field_validator("sources")
    @classmethod
    def _dedup_sources(cls, value: list[Source]) -> list[Source]:
        seen: set[str] = set()
        result: list[Source] = []
        for item in value:
            key = f"{item.title}|{item.chunk_id}"
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    @model_validator(mode="after")
    def _check_prediction(self) -> "FinalAnswer":
        if self.prediction is not None and self.slope is None:
            raise ValueError("给出 prediction 时必须同时给出 slope，否则无法核对预测依据")
        if self.trend in ("上升", "下降") and self.slope is not None:
            if self.trend == "上升" and self.slope <= 0:
                raise ValueError("trend 为上升但 slope <= 0，二者矛盾")
            if self.trend == "下降" and self.slope >= 0:
                raise ValueError("trend 为下降但 slope >= 0，二者矛盾")
        return self


class ToolCallRecord(BaseModel):
    """一次工具调用的可审计记录。"""

    name: str
    arguments: dict
    ok: bool
    summary: str


class ChatResponse(BaseModel):
    session_id: str
    reply: FinalAnswer | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    chart_url: str | None = None
    # 预测区间由工具计算、程序直接下发，不经过模型转述，避免被写错或省略
    prediction_interval: dict | None = None
    # 需要用户补充/修正时下发的可操作卡片：前端据此渲染输入框和快捷选项，
    # 而不是只让用户自己看文字说明后重新打字。
    fix: dict | None = None
    error: str | None = None
    error_detail: str | None = None
    # 失败分类（AUTH/TIMEOUT/RATE_LIMIT/...），前端据此选择展示样式，也便于统计
    error_code: str | None = None
    run_id: str | None = None
