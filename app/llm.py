"""模型客户端：DeepSeek（OpenAI 兼容接口）+ 无 Key 时的本地桩模型。

桩模型只用于验证前后端链路是否通畅，不参与考核演示的"真实运行证据"。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from config import settings

try:  # openai>=1.x
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AuthenticationError,
        OpenAI,
        RateLimitError,
    )
except Exception:  # pragma: no cover - 依赖缺失时给出可读错误
    OpenAI = None  # type: ignore[assignment]


class LLMError(Exception):
    def __init__(self, message: str, detail: str = "", code: str = "UNKNOWN") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.code = code


def _normalize_tool_calls(raw_calls: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for index, call in enumerate(raw_calls or []):
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", "") or "{}"
        calls.append(
            {
                "id": getattr(call, "id", None) or f"call_{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return calls


class LLMClient:
    """真实模型客户端。每次请求都记录 token 用量，便于核对是否真的调用了模型。"""

    def __init__(self) -> None:
        if OpenAI is None:
            raise LLMError("缺少 openai 依赖", "请先执行 pip install -r requirements.txt")
        if not settings.api_key:
            raise LLMError("未配置 DEEPSEEK_API_KEY", "请在 .env 中填入密钥后重启服务；或设置 LLM_MOCK=1 使用桩模型")
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=0,
        )

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None,
             temperature: float = 0.2, json_mode: bool = False) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": settings.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: LLMError | None = None
        attempts = max(0, settings.llm_max_retries) + 1
        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except AuthenticationError as exc:
                # 鉴权失败重试没有意义，直接报错
                raise LLMError(
                    "模型鉴权失败（401）",
                    "API Key 无效或已过期，请检查 .env 中的 DEEPSEEK_API_KEY",
                    code="AUTH",
                ) from exc
            except RateLimitError as exc:
                last_error = LLMError("模型限流或余额不足（429）", str(exc)[:300], code="RATE_LIMIT")
            except APITimeoutError:
                last_error = LLMError(
                    f"模型请求超时（>{settings.timeout:.0f}s）",
                    "可调大 LLM_TIMEOUT 或稍后重试",
                    code="TIMEOUT",
                )
            except APIConnectionError as exc:
                last_error = LLMError("无法连接模型服务", str(exc)[:300], code="CONNECTION")
            except APIStatusError as exc:
                if exc.status_code >= 500:
                    last_error = LLMError(
                        f"模型服务返回异常状态码 {exc.status_code}",
                        str(exc)[:300],
                        code="SERVER",
                    )
                else:
                    raise LLMError(
                        f"模型服务返回异常状态码 {exc.status_code}", str(exc)[:300], code="API_ERROR"
                    ) from exc
            except Exception as exc:  # 兜底：不让未预期异常打断会话状态
                raise LLMError(
                    "模型调用发生未预期错误", f"{type(exc).__name__}: {exc}"[:300], code="UNKNOWN"
                ) from exc

            # 能走到这里的都是瞬时故障：429 / 超时 / 连接失败 / 5xx
            if attempt < attempts - 1:
                time.sleep(settings.llm_retry_backoff * (2**attempt))
                continue
            raise last_error

        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)
        return {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": _normalize_tool_calls(getattr(message, "tool_calls", None)),
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            },
            "model": getattr(response, "model", settings.model),
            # 实际发生的退避重试次数，写进运行日志，便于判断是不是模型侧在抖
            "retries": attempt,
        }


class MockLLMClient:
    """本地桩模型：不联网，用固定规则模拟"决定调工具 → 生成结构化回答"。

    仅用于 LLM_MOCK=1 时验证页面交互与结构化校验流程，输出会明确标注为 mock。
    """

    model = "mock-llm"

    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None,
             temperature: float = 0.2, json_mode: bool = False) -> dict[str, Any]:
        last_user = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
        text = last_user.get("content", "")
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        last_tool = json.loads(tool_messages[-1]["content"]) if tool_messages else None

        # 第二轮：已经拿到工具结果，直接产出结构化回答
        if last_tool is not None:
            return self._final_from_tool(last_tool, text)

        # 会话上下文里带了上一轮数据，追问时可以直接复用（模拟真实模型的上下文记忆）
        context_x, context_y = self._parse_context(messages)
        numbers = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]
        # 中文分数（如“八十五分”）也提取出来，便于演示数据不足的场景
        for cn in re.findall(r"([一二三四五六七八九十]{1,3})\s*分", text):
            value = _cn_to_int(cn)
            if value is not None:
                numbers.append(float(value))
        predict = None
        match = re.search(r"第\s*([0-9一二三四五六七八九十]+)\s*次", text)
        if match:
            predict = _cn_to_number(match.group(1))

        if len(numbers) >= 4:
            ys = numbers[-4:]
            xs = list(range(1, len(ys) + 1))
            return self._tool_call("analyze_score_trend", {"x": xs, "y": ys, "predict_x": predict}, 0)

        if 1 <= len(numbers) < 4 and any(k in text for k in ("考", "成绩", "分", "趋势")):
            ys = numbers
            return self._tool_call(
                "analyze_score_trend",
                {"x": list(range(1, len(ys) + 1)), "y": ys, "predict_x": predict},
                0,
            )

        if context_y and len(context_y) >= 2 and any(k in text for k in ("趋势", "第", "次", "预测", "多少")):
            return self._tool_call(
                "analyze_score_trend", {"x": context_x, "y": context_y, "predict_x": predict}, 0
            )

        return self._tool_call("search_study_knowledge", {"query": text or "学习方法"}, 0)

    @staticmethod
    def _parse_context(messages: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
        for message in messages:
            if message.get("role") != "system":
                continue
            content = str(message.get("content", ""))
            x_match = re.search(r"x（考试序号）\s*=\s*\[([^\]]*)\]", content)
            y_match = re.search(r"y（成绩）\s*=\s*\[([^\]]*)\]", content)
            if x_match and y_match:
                try:
                    xs = [float(v) for v in x_match.group(1).split(",") if v.strip()]
                    ys = [float(v) for v in y_match.group(1).split(",") if v.strip()]
                    if len(xs) == len(ys) and xs:
                        return xs, ys
                except ValueError:
                    continue
        return [], []


    @staticmethod
    def _tool_call(name: str, arguments: dict, index: int) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"mock_call_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": "mock-llm",
        }

    @staticmethod
    def _final_from_tool(tool: dict, question: str) -> dict[str, Any]:
        if not tool.get("ok"):
            # 工具失败：说明原因与修改建议，不输出任何预测
            return {
                "role": "assistant",
                "content": json.dumps({
                    "answer": f"这次没能完成分析：{tool.get('message', '工具执行失败')}。{tool.get('hint', '')}请按建议修改后重新发送。",
                    "trend": "无法判断",
                    "slope": None,
                    "intercept": None,
                    "prediction": None,
                    "prediction_note": None,
                    "limitations": [f"工具返回错误（{tool.get('error_code', 'UNKNOWN')}），本次没有产出任何数值结论"],
                    "sources": [],
                    "need_followup": tool.get("hint") or "请检查输入数据后重试",
                }, ensure_ascii=False),
                "tool_calls": [],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": "mock-llm",
            }

        if tool.get("ok") and "slope" in tool:
            slope = tool["slope"]
            trend = "上升" if slope > 0.5 else "下降" if slope < -0.5 else "基本平稳"
            prediction = tool.get("prediction")
            answer = {
                "answer": (
                    f"这组成绩整体呈{trend}趋势，平均每次变化 {slope:+.2f} 分。"
                    + (f"按趋势外推，第 {tool.get('predict_x'):g} 次约为 {prediction} 分。" if prediction is not None else "")
                    + "（当前为 mock 模式输出，未调用真实模型）"
                ),
                "trend": trend,
                "slope": slope,
                "intercept": tool.get("intercept"),
                "prediction": prediction,
                "prediction_note": "预测基于最小二乘直线外推，仅代表趋势，不是确定的考试结果。",
                "limitations": tool.get("warnings") or ["样本量有限，趋势仅供参考"],
                "sources": [],
                "need_followup": None if prediction is not None else "需要你告诉我预测第几次考试",
            }
        elif tool.get("hits"):
            top = tool["hits"][0]
            answer = {
                "answer": f"根据资料《{top['title']}》的相关片段：{top['snippet']}（当前为 mock 模式输出）",
                "trend": "无法判断",
                "slope": None,
                "intercept": None,
                "prediction": None,
                "prediction_note": None,
                "limitations": ["该结论仅来自检索到的资料片段，未结合你的具体成绩数据"],
                "sources": [
                    {"title": top["title"], "chunk_id": top["chunk_id"], "snippet": top["snippet"]}
                ],
                "need_followup": None,
            }
        else:
            message = tool.get("message") or "知识库没有覆盖这个问题"
            answer = {
                "answer": f"现有资料不足，无法回答这个问题：{message}。我没有找到可引用的资料，因此不做推测。",
                "trend": "无法判断",
                "slope": None,
                "intercept": None,
                "prediction": None,
                "prediction_note": None,
                "limitations": ["知识库未覆盖该问题，缺少可引用的证据"],
                "sources": [],
                "need_followup": None,
            }
        return {
            "role": "assistant",
            "content": json.dumps(answer, ensure_ascii=False),
            "tool_calls": [],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": "mock-llm",
        }


_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_to_int(token: str) -> int | None:
    """把“八十五”“二十”“十”这类 1—99 的中文数转成整数，失败返回 None。"""
    if not token:
        return None
    if token == "十":
        return 10
    if "十" in token:
        tens, _, ones = token.partition("十")
        tens_value = _CN_DIGITS.get(tens, 1) if tens else 1
        ones_value = _CN_DIGITS.get(ones, 0) if ones else 0
        if (tens and tens not in _CN_DIGITS) or (ones and ones not in _CN_DIGITS):
            return None
        return tens_value * 10 + ones_value
    if token in _CN_DIGITS:
        return _CN_DIGITS[token]
    return None


def _cn_to_number(token: str) -> float | None:
    if token.isdigit():
        return float(token)
    value = _cn_to_int(token)
    return float(value) if value is not None else None


def build_client():
    if settings.mock:
        return MockLLMClient()
    return LLMClient()
