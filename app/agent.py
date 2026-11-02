"""Agent 主流程：模型决定是否调工具 → 程序执行工具 → 结果回填模型 → 最终回答结构化校验。

这一层是"程序"和"模型"职责的分界线：
- 模型只负责理解问题、决定调用哪个工具、组织语言；
- 数值计算、检索、入库、字段校验全部由程序完成，模型无法绕过。
"""

from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any

from pydantic import ValidationError

from app import db
from config import settings
from app.llm import LLMError, build_client
from app.llm import _cn_to_int as cn_to_int
from app.runlog import RunLogger
from app.schemas import ChatResponse, FinalAnswer, ToolCallRecord
from app.tools.retrieval import search_knowledge
from app.tools.trend import _MAX_SCORE, TrendError, analyze_trend

SYSTEM_PROMPT = """你是"学习与成绩分析助手"，帮助学生分析考试成绩的变化趋势，并结合学习资料给出可执行的建议。

工作要求（必须遵守）：
1. 只要问题涉及成绩（包括只给了一个分数、只问"能分析吗"），都必须先调用 analyze_score_trend 工具。数据够不够、能不能拟合由工具判断，不要自己下结论；禁止自己心算或口算预测值。
2. 只要问题涉及学习方法、效率、复习策略、成绩波动原因，或者你不确定知识库有没有覆盖，都必须先调用 search_study_knowledge 检索。命中资料时在 sources 中给出标题、片段编号和原文片段；检索返回空才说明资料不足，禁止编造引用，也禁止因为"看起来不相关"就跳过检索。检索时必须选择 mode：用户问题中直接出现资料术语（如 错题本、间隔重复、R²、外推、拟合）时用 "keyword"；用户用口语化、同义改写的说法（如 总记不住、发挥不稳定、忽高忽低）时用 "vector"；不确定时不要传 mode（默认混合检索）。
3. 用户问"是不是因为增加了学习时间才进步"这类因果问题时，要明确说明：现有数据只能看出相关，不能证明因果，并给出还需要什么证据。
4. 工具返回错误（数据不足、非数字、序号重复、无法拟合）时：trend 必须填"无法判断"，slope/intercept/prediction 必须为 null，answer 只说明原因和用户该怎么改，不得给出任何趋势判断或预测分数，也不要在同一轮里重复调用工具试图绕过错误。
5. 预测位置只能来自用户原话。用户没说第几次（例如只说"帮我预测成绩""按这个趋势预测一下"）时：调用 analyze_score_trend 不要传 predict_x，回答只讲趋势、拟合质量与限制条件，并用 need_followup 询问想预测第几次；此时回答里不得出现任何具体预测分数，也不允许把"下一次"当默认位置。用户明确说了第几次（含"下次/下一次"）时，才传 predict_x，并原样使用工具返回的 prediction。
6. 会话已经保存了用户上一次提供的数据，用户追问"第五次呢"时直接复用，不要重复索要成绩。

输出格式（只输出一个 JSON 对象，不要加 Markdown 代码块、不要加解释文字）：
{
  "answer": "给用户看的自然语言结论，2-4 句，包含关键数值",
  "trend": "上升|下降|基本平稳|无法判断",
  "slope": 每次考试的分数变化（没有计算则为 null）,
  "intercept": 拟合截距（没有计算则为 null）,
  "prediction": 预测值（没有预测则为 null）,
  "prediction_note": 预测的前提说明,
  "limitations": ["至少一条限制条件，例如样本量、外推风险、相关不等于因果"],
  "sources": [{"title": "资料文件名", "chunk_id": "文件名#片段号", "snippet": "原文片段"}],
  "need_followup": "需要继续追问用户的问题，没有则为 null"
}
字段要求：limitations 至少一条且不能为空；slope/prediction 必须与工具返回值完全一致；trend 与 slope 的正负必须一致。"""

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analyze_score_trend",
            "description": (
                "对一组考试成绩做一元线性最小二乘拟合，返回斜率、截距、R²、指定位置的预测值和趋势图。"
                "任何涉及成绩趋势或预测的问题都必须调用本工具，不要自行计算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "考试序号，例如 [1,2,3,4]",
                    },
                    "y": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "与 x 一一对应的成绩，例如 [78,82,85,88]",
                    },
                    "predict_x": {
                        "type": "number",
                        "description": (
                            "要预测的考试序号，例如用户说第五次就填 5。"
                            "只有用户明确说了第几次（或明确说下次/下一次）才能填；"
                            "用户没说清时不要传，更不要默认填下一次或按数据长度顺延——系统会先追问用户。"
                        ),
                    },
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_study_knowledge",
            "description": (
                "在本地学习资料库中检索与学习方法、复习策略、成绩波动、线性拟合原理相关的资料，"
                "返回命中的原文片段和来源。知识性问题必须先检索再回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题，例如“如何提高复习效率”"},
                    "mode": {
                        "type": "string",
                        "enum": ["keyword", "vector"],
                        "description": (
                            "检索方式：keyword=关键词检索（BM25，适合问题里直接出现资料术语）；"
                            "vector=向量检索（tf-idf 余弦，适合口语化、同义改写的问题）。"
                            "不确定时不传，默认混合检索。"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]

GUARD_PROMPT = """你上一条回答没有调用任何工具，但规则要求先取证再作答：
- 涉及成绩（哪怕只给了一个分数）→ 必须先调用 analyze_score_trend，数据是否够由工具判断；
- 涉及学习方法、复习策略、成绩波动，或你不确定知识库是否覆盖 → 必须先调用 search_study_knowledge，
  检索返回空才能说"资料不足"。
请现在调用合适的工具，拿到工具结果后再组织最终 JSON 回答。"""

REPAIR_PROMPT = """你上一条输出无法通过程序的字段校验，错误如下：
{errors}

请严格按要求的 JSON 结构重新输出，只输出 JSON，不要任何额外文字。数值字段必须与工具返回结果一致。"""


def _needs_tool(text: str) -> bool:
    """判断这条消息是否要求"先取证再回答"（问候类闲聊不触发）。"""
    text = text.strip()
    if len(text) < 4:
        return False
    keywords = ("成绩", "分", "考", "复习", "学习", "趋势", "预测", "怎么", "如何",
                "为什么", "是什么", "吗", "？", "?", "能帮")
    return any(keyword in text for keyword in keywords)


# 只在出现"预测类"措辞时，才把"第 N 次"当成预测目标；
# 否则"第一次 78、第二次 82"这种数据描述会被误判成"要预测第 1 次"。
_PREDICT_INTENT = ("预测", "预估", "预计", "推算", "推断", "估计", "会考", "能考",
                   "多少", "几分", "大概", "猜")
_POSITION_PATTERN = re.compile(r"第\s*([0-9]+|[一二三四五六七八九十两]+)\s*(?:次|回|场)")
# "第五次呢？/ 第五次会多少"这类追问，即使整句较长也算明确指定了位置
_POSITION_FOLLOW_RE = re.compile(
    r"第\s*(?:[0-9]+|[一二三四五六七八九十两]+)\s*(?:次|回|场)\s*(?:呢|会|能|大概|多少|是多少|怎么样|如何|[?？]|$)"
)
_NEXT_PATTERN = re.compile(r"(?:下一|下下|下)\s*(?:次|回|场)")


def _predict_position_hint(text: str) -> int | str | None:
    """解析用户原话有没有明确指定"要预测第几次"。

    返回值：
      int     —— 明确说了第 N 次（阿拉伯数字或中文数字）
      "next"  —— 说的是"下次 / 下一次"
      None    —— 没指定（此时必须中断预测、回来询问用户，不允许模型自己猜一个位置）
    """
    text = (text or "").strip()
    if not text:
        return None

    positions: list[int] = []
    for token in _POSITION_PATTERN.findall(text):
        value = int(token) if token.isdigit() else cn_to_int(token)
        if value:
            positions.append(value)

    wants_predict = any(word in text for word in _PREDICT_INTENT)
    follow_up = bool(_POSITION_FOLLOW_RE.search(text))
    short_ask = len(text) <= 16 and any(mark in text for mark in ("呢", "多少", "？", "?"))

    # 一句话里出现两个以上"第 N 次"：在列举历史成绩，不能当成预测目标
    if len(positions) >= 2:
        return None
    if positions:
        return positions[0] if (wants_predict or follow_up) else None
    if _NEXT_PATTERN.search(text) and (wants_predict or short_ask):
        return "next"
    return None


_SCORE_DELIM = re.compile(r"[、,，;；|]+")
_SLOT_PREFIX = re.compile(r"^第\s*[0-9一二三四五六七八九十两]*\s*(?:次|回|场)?\s*[:：]?\s*")
_SLOT_LEAD = re.compile(
    r"^(?:我的|我|这|最近|之前|本次|目前)?(?:[0-9一二三四五六七八九十几]+次|几次)?的?"
    r"(?:考试)?(?:成绩|分数)?(?:分别|依次)?(?:是|为|有|:|：)?\s*"
)
_SCORE_UNIT = re.compile(r"\s*分\s*$")
# 用户写成这些词，说明这一次成绩"存在但没有数值"，必须让用户补上再分析
_SCORE_BLANKS = {
    "缺考", "没考", "未考", "没参加", "待定", "请假", "免考", "弃考", "不知道", "不确定",
    "忘了", "空", "无", "暂无", "—", "-", "?", "？",
}


def _slot_display(raw: str) -> str:
    """去掉"第 2 次""78 分"这类前后缀，留下用户真正写的内容。"""
    text = _SLOT_PREFIX.sub("", (raw or "").strip())
    return _SCORE_UNIT.sub("", text).strip()


def _slot_kind(raw: str) -> tuple[str | None, float | None]:
    """判断一个片段是"数字成绩""空缺"还是"看不懂"。"""
    text = (raw or "").strip().strip("。.！!？? ")
    if not text:
        return None, None
    text = _SLOT_PREFIX.sub("", text).strip()
    text = _SCORE_UNIT.sub("", text).strip()
    if text in _SCORE_BLANKS:
        return "gap", None
    if re.fullmatch(r"\d{1,3}(?:\.\d+)?", text):
        return "num", float(text)
    if re.fullmatch(r"[一二三四五六七八九十百两]{1,3}", text):
        value = cn_to_int(text)
        if value:
            return "num", float(value)
    return None, None


def _score_list_gap(text: str) -> list[str] | None:
    """从用户原话里找出"成绩序列中的空格"。

    只在能明确认出"一串成绩"时才判定，例如：
        第一次 78，第二次缺考，第三次 85   → 返回 ['78', '缺考', '85']
        我的成绩是 78、abc、85            → 返回 ['78', 'abc', '85']
    普通句子（含数字但不成列表）一律返回 None，避免误伤。
    """
    tokens = [t for t in _SCORE_DELIM.split(text or "") if t.strip()]
    if len(tokens) < 3:
        return None

    kinds: list[tuple[str | None, float | None]] = []
    for index, token in enumerate(tokens):
        # 第一个片段常带"我的成绩是"之类的前缀
        candidate = _SLOT_LEAD.sub("", token) if index == 0 else token
        kinds.append(_slot_kind(candidate))

    # 结尾常是"帮我看看趋势"这类句子尾巴，整段去掉；若它本身带数字，说明不是在列成绩
    while kinds and kinds[-1][0] is None:
        tail = tokens[-1]
        if re.search(r"\d", tail) or len(tail.strip()) < 4:
            return None
        kinds.pop()
        tokens.pop()
    if not kinds:
        return None

    # 中间出现看不懂的短片段（abc、xx、待补）同样说明"这一次没填"；
    # 长句说明这句话不是在列成绩，直接放弃判断，避免误伤
    for index, (kind, _) in enumerate(kinds):
        if kind is None:
            display = _slot_display(tokens[index])
            if len(display) > 4 or re.search(r"[。！？!?]", display):
                return None
            kinds[index] = ("gap", None)

    if sum(1 for kind, _ in kinds if kind == "num") < 2:
        return None
    if not any(kind == "gap" for kind, _ in kinds):
        return None

    cleaned: list[str] = []
    for token, (kind, value) in zip(tokens, kinds):
        if kind == "num" and value is not None:
            cleaned.append(str(int(value)) if float(value).is_integer() else f"{value:g}")
        elif kind == "gap":
            cleaned.append(_slot_display(token) or token.strip())
        else:
            cleaned.append(token.strip())
    return cleaned


def _missing_slots_error(scores: list[str]) -> dict[str, Any]:
    """成绩序列里有空缺/非数字：直接拒绝分析，让用户先补上。"""
    gaps = [i for i, item in enumerate(scores, start=1) if _slot_kind(item)[0] != "num"]
    labels = "、".join(f"第 {i} 个" for i in gaps)
    values = "、".join(f"「{scores[i - 1]}」" for i in gaps)
    return {
        "ok": False,
        "error_code": "MISSING_VALUE",
        "message": f"成绩里有 {len(gaps)} 次没有有效数字：{labels} {values}",
        "hint": "请把这一次补上；如果确实缺考，也请说明这次按多少分算，或者去掉这一次",
        "detail": {"field": "y", "invalid_indexes": gaps, "scores": scores},
    }


def _align_predict_arguments(
    arguments: dict[str, Any], hint: int | str | None
) -> tuple[dict[str, Any], bool, str | None]:
    """按用户原话校正 predict_x，保证"没指定位置就不预测"。

    - hint 是整数：强制使用用户说的位置（模型写错也会被改写）
    - hint 是 "next"：换算成已有数据的下一次
    - hint 是 None：删掉模型自作主张传入的 predict_x，本次不做预测

    返回 (校正后参数, 是否改动, 追加给模型的系统提示)。
    """
    args = dict(arguments or {})
    model_value = args.get("predict_x")

    if hint is None:
        if model_value is None:
            return args, False, None
        args.pop("predict_x", None)
        return args, True, (
            "【程序校验】用户没有说明要预测第几次，已忽略你传入的 predict_x，本次不做预测。"
            "answer 里只能讲趋势、拟合质量与限制条件，不得出现任何具体预测分数"
            "（也不要写“按趋势外推大概多少分”）；请用 need_followup 询问用户想预测第几次。"
        )

    if hint == "next":
        xs = [v for v in (args.get("x") or [])]
        try:
            target = int(max(float(v) for v in xs)) + 1
        except (TypeError, ValueError, OverflowError):
            try:
                target = len(args.get("y") or []) + 1
            except TypeError:
                return args, False, None
    else:
        target = int(hint)

    if model_value is not None:
        try:
            if float(model_value) == float(target):
                return args, False, None
        except (TypeError, ValueError):
            pass

    args["predict_x"] = target
    return args, True, (
        f"【程序校验】已按用户原话把预测位置校正为第 {target} 次；"
        "回答里的预测值必须原样使用工具返回的 prediction，不要自己计算。"
    )



def _execute_tool(name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """执行工具，返回 (结果, 一句话摘要)。工具异常统一转成结构化错误，交给模型解释。"""
    if name == "analyze_score_trend":
        try:
            result = analyze_trend(
                arguments.get("x"),
                arguments.get("y"),
                arguments.get("predict_x"),
            )
        except TrendError as exc:
            return exc.to_dict(), f"工具拒绝执行：{exc.code} {exc.message}"
        summary = (
            f"拟合完成：斜率 {result['slope']}，截距 {result['intercept']}，R² {result['r2']}，"
            f"预测 {result['predict_x']} → {result['prediction']}"
        )
        return result, summary

    if name == "search_study_knowledge":
        result = search_knowledge(str(arguments.get("query", "")), mode=arguments.get("mode"))
        if result.get("hits"):
            titles = "、".join(hit["title"] for hit in result["hits"])
            return result, f"[{result.get('mode')}] 检索命中 {len(result['hits'])} 个片段：{titles}"
        return result, f"[{result.get('mode')}] 检索无命中：{result.get('message', '')}"

    return {"ok": False, "error_code": "UNKNOWN_TOOL", "message": f"未注册的工具 {name}"}, "工具不存在"


def _predict_options(xs: Any, count: int = 3) -> list[int]:
    """给出几个合理的预测位置供用户一键选择（已有数据的后续 1—3 次）。"""
    try:
        last = int(max(float(v) for v in (xs or [])))
    except (TypeError, ValueError, OverflowError):
        last = 4
    return [last + i for i in range(1, count + 1)]


def _invalid_score_indexes(values: Any) -> list[int]:
    """挑出明显不合法的成绩位置（1-based）：非数字、非有限数、负数、超过 150 分。"""
    bad: list[int] = []
    for index, raw in enumerate(values or [], start=1):
        try:
            number = float(raw)
        except (TypeError, ValueError):
            bad.append(index)
            continue
        if not math.isfinite(number) or number < 0 or number > _MAX_SCORE:
            bad.append(index)
    return bad


def _fmt_score(value: Any) -> str:
    """70.0 → "70"，保留原始文本形态，避免预填框里出现多余小数点。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


# 每类工具错误对应一张"改哪里"的卡片，标题按原因区分，避免所有错误都长一样
_FIX_TITLE = {
    "EMPTY_INPUT": "还没有可用的成绩",
    "BAD_INPUT_TYPE": "成绩格式不对",
    "NON_NUMERIC": "有成绩不是数字",
    "NON_FINITE": "有成绩不是有效数字",
    "INVALID_SCORE": "有成绩超出有效范围",
    "LENGTH_MISMATCH": "序号和成绩数量对不上",
    "INSUFFICIENT_DATA": "再加一次成绩就能分析",
    "INVALID_X": "考试序号无效",
    "DUPLICATE_X": "考试序号有重复",
    "DEGENERATE_X": "考试序号全都一样",
    "MISSING_VALUE": "有一次成绩没填或不是数字",
    "BAD_PREDICT_X": "预测位置无效",
}
# 这些错误都出在"序号"上：序号由程序按 1、2、3… 自动重排，用户只需要核对成绩
_INDEX_ERROR_CODES = ("INVALID_X", "DUPLICATE_X", "DEGENERATE_X", "LENGTH_MISMATCH", "BAD_INPUT_TYPE")


def _error_text(result: dict[str, Any]) -> str:
    """把工具的错误说明 + 修改建议拼成一句给用户看的话。"""
    message = str(result.get("message") or "").strip().rstrip("。")
    hint = str(result.get("hint") or "").strip().rstrip("。")
    if message and hint:
        return f"{message}。{hint}。"
    return (message or hint) + ("。" if (message or hint) else "")


def _build_fix(last_tool_error: dict[str, Any] | None, last_trend: dict[str, Any] | None) -> dict | None:
    """把"需要用户修正/补充"转成可操作卡片：预填原数据、标出错的位置、给快捷按钮。

    返回结构由前端渲染：
      kind=edit_scores     逐次编辑成绩（invalid 标红），可重新提交
      kind=add_scores      在已有成绩后追加几次新成绩
      kind=choose_predict  选择要预测第几次考试
    """
    if last_tool_error:
        result = last_tool_error.get("result") or {}
        arguments = last_tool_error.get("arguments") or {}
        code = str(result.get("error_code") or "")
        detail = result.get("detail") or {}
        # 模型可能悄悄丢掉坏值再调用工具，这时以用户原话解析出的序列为准
        scores = detail.get("scores") or [_fmt_score(v) for v in (arguments.get("y") or [])]
        xs = arguments.get("x") or []
        predict_x = arguments.get("predict_x")
        title = _FIX_TITLE.get(code, "修正后重新分析")

        if code == "BAD_PREDICT_X":
            return {
                "kind": "choose_predict",
                "title": "选择要预测第几次考试",
                "scores": scores,
                "options": _predict_options(xs),
                "hint": _error_text(result),
            }

        # 数据不足 / 没有数据：直接给空位让用户补，而不是让他重新打一遍
        if code in ("INSUFFICIENT_DATA", "EMPTY_INPUT") or len(scores) < 2:
            return {
                "kind": "add_scores",
                "title": title,
                "scores": scores,
                "invalid": [],
                "hint": _error_text(result),
            }

        # 序号类问题：序号由程序按 1、2、3… 重排，用户只要核对成绩
        if code in _INDEX_ERROR_CODES:
            return {
                "kind": "edit_scores",
                "title": title,
                "scores": scores,
                "invalid": [],
                "regenerate_index": True,
                "predict_x": predict_x,
                "hint": _error_text(result),
            }

        # 非数字 / 超范围：逐个标红，用户直接改那几格
        invalid = [int(i) for i in (detail.get("invalid_indexes") or []) if isinstance(i, int)]
        if detail.get("field") and detail.get("field") != "y":
            invalid = []
        return {
            "kind": "edit_scores",
            "title": title,
            "scores": scores,
            "invalid": invalid or _invalid_score_indexes(arguments.get("y")),
            "regenerate_index": True,
            "predict_x": predict_x,
            "hint": _error_text(result),
        }

    # 拟合成功但用户没说预测第几次 → 不做预测，先把位置问清楚
    if last_trend and last_trend.get("predict_x") is None:
        return {
            "kind": "choose_predict",
            "title": "还没指定要预测第几次",
            "scores": [_fmt_score(v) for v in (last_trend.get("y") or [])],
            "options": _predict_options(last_trend.get("x")),
            "hint": "选一个位置后，我再按同一条拟合直线给出预测值；不选就不预测。",
        }
    return None


def _failed_tool_note(result: dict[str, Any]) -> str:
    """工具失败后追加给模型的硬性约束：本轮不允许再出现任何数值结论。"""
    return (
        f"【程序校验】analyze_score_trend 返回错误（{result.get('error_code')}）：{result.get('message')}。"
        "本轮没有任何可用的拟合结果：trend 必须为「无法判断」，slope、intercept、prediction 必须为 null；"
        "answer 只能说明原因和怎么修改，不得出现预测分数、每次提高/下降多少分之类的数值结论；"
        "不要在回答里出现错误码、工具名等内部术语，用用户能看懂的话说明。"
        "不要再次调用该工具，直接引导用户按提示修改。"
    )


def _error_answer(last_tool_error: dict[str, Any]) -> FinalAnswer:
    """模型坚持编数字时的兜底：用工具的错误信息直接生成回答，保证不出现无效预测。"""
    result = last_tool_error.get("result") or {}
    code = str(result.get("error_code") or "TOOL_ERROR")
    text = _error_text(result) or "输入数据无法完成拟合。"
    hint = str(result.get("hint") or "请按提示修改后重新分析").strip()
    return FinalAnswer(
        answer=f"这次没有给你结果：{text}改好之后我再重新计算，不会用不可靠的数据给出趋势或分数。",
        trend="无法判断",
        slope=None,
        intercept=None,
        prediction=None,
        prediction_note=None,
        limitations=[
            f"工具返回错误 {code}，本次不存在可用的拟合结果",
            "为避免误导，未给出斜率、截距或任何预测分数",
        ],
        sources=[],
        need_followup=hint,
    )


def _extract_json(text: str) -> str:
    """模型偶尔会加代码围栏或前后缀，这里做最小限度的容错提取。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _parse_answer(text: str) -> tuple[FinalAnswer | None, list[str]]:
    try:
        payload = json.loads(_extract_json(text))
    except json.JSONDecodeError as exc:
        return None, [f"输出不是合法 JSON：{exc.msg}（位置 {exc.pos}）"]
    try:
        return FinalAnswer.model_validate(payload), []
    except ValidationError as exc:
        errors = [
            f"字段 {'.'.join(str(part) for part in err['loc']) or '(根)'}：{err['msg']}"
            for err in exc.errors()
        ]
        return None, errors


# 数字前面紧挨着负号/小数点/数字时不算"结论性数值"，避免把"-20 分超出范围"这类错误说明误判成预测
_NO_NUM_PREFIX = r"(?<![\d.,\-−])"
# 工具失败时，回答里出现这些形态就等于"仍在给无效结论"
_FAILED_CLAIM_PATTERNS = (
    r"第\s*(?:\d+|[一二三四五六七八九十两]+)\s*(?:次|回)[^。！？；]{0,20}?" + _NO_NUM_PREFIX + r"\d+(?:\.\d+)?\s*分",
    r"(?:每次|平均)[^。！？；\d]{0,10}(?:提高|提升|增加|上升|下降|减少|降低|涨|掉)"
    r"[^。！？；\d]{0,8}" + _NO_NUM_PREFIX + r"\d+(?:\.\d+)?\s*分",
    r"(?:预测为|预测值是|预测值为|预测结果|预计为|外推为|大概是|大约是)"
    r"[^\d。！？；]{0,6}" + _NO_NUM_PREFIX + r"\d+(?:\.\d+)?\s*分",
)


def _cross_check(
    answer: FinalAnswer,
    tool_result: dict[str, Any] | None,
    failure: dict[str, Any] | None = None,
) -> list[str]:
    """程序侧一致性校验：模型写的数值必须与工具真实返回值一致。

    failure 非空表示本轮拟合工具报错（且之后没有成功过）——此时任何趋势/斜率/预测
    都是模型凭空编的，必须全部清空，否则就是"继续输出无效预测"。
    """
    if failure:
        code = failure.get("error_code") or "TOOL_ERROR"
        problems: list[str] = []
        if answer.trend != "无法判断":
            problems.append(f"工具返回错误（{code}），但回答把 trend 写成了「{answer.trend}」；必须为「无法判断」")
        for field, label in (("slope", "斜率"), ("intercept", "截距"), ("prediction", "预测值")):
            value = getattr(answer, field)
            if value is not None:
                problems.append(f"工具返回错误（{code}），但回答给出了{label} {value}；必须为 null")
        for pattern in _FAILED_CLAIM_PATTERNS:
            claim = re.search(pattern, answer.answer)
            if claim:
                problems.append(
                    f"工具返回错误（{code}），回答里仍出现了结论性数值「{claim.group(0)}」；"
                    "只应说明原因和修改办法，不能给趋势或分数"
                )
                break
        return problems

    if not tool_result or not tool_result.get("ok") or "slope" not in tool_result:
        return []
    problems = []
    if answer.slope is not None and abs(answer.slope - tool_result["slope"]) > 0.011:
        problems.append(
            f"slope 与工具返回值不一致：模型写 {answer.slope}，工具返回 {tool_result['slope']}"
        )
    if tool_result.get("prediction") is None:
        # 工具没有预测位置时，模型不得自行给具体分数（避免"模型自己算"绕过工具）
        claims = [
            r"第\s*[0-9一二三四五六七八九十]+\s*次[^。！？；]{0,24}?\d+(?:\.\d+)?\s*分(?!\s*/\s*次)",
            r"(?:预测|预计|预估|推算|外推|大概|估计)[^。！？；]{0,14}?\d+(?:\.\d+)?\s*分(?!\s*/\s*次)",
        ]
        for pattern in claims:
            claim = re.search(pattern, answer.answer)
            if claim:
                problems.append(
                    f"工具未返回预测值，但回答中出现了具体预测「{claim.group(0)}」；"
                    "应改为询问预测位置，或带 predict_x 重新调用工具"
                )
                break
    if answer.prediction is not None:
        expected = tool_result.get("prediction")
        if expected is None:
            problems.append("工具没有给出预测值，但回答中出现了 prediction")
        elif abs(answer.prediction - expected) > 0.11:
            problems.append(
                f"prediction 与工具返回值不一致：模型写 {answer.prediction}，工具返回 {expected}"
            )
    return problems


def _save_failure_record(
    session, user_text: str, message: str, detail: str, tool_records: list, source: str, run_id: str
) -> None:
    """把"模型或工具调用失败"落库（kind=error），保留失败状态与排查信息。

    只做留痕，不抛错：落库万一失败也不能影响错误响应本身。
    """
    tools = "、".join(f"{t.name}({'成功' if t.ok else '失败'})" for t in tool_records) or "未执行任何工具"
    detail_text = (
        f"失败原因：{message}\n"
        f"排查信息：{detail}\n"
        f"已执行工具：{tools}\n"
        f"run_id：{run_id}（可在 logs/run.jsonl 中检索该编号查看完整调用过程）"
    )
    try:
        db.save_failure(session.session_id, user_text, detail_text, source=source)
    except Exception as exc:  # noqa: BLE001 - 留痕失败只记日志
        logger = RunLogger(run_id)
        logger.log("db", "save_failure_failed", {"message": str(exc)})


def handle_message(user_text: str, session, source: str = db.SOURCE_WEB) -> ChatResponse:
    run_id = uuid.uuid4().hex[:12]
    logger = RunLogger(run_id)
    logger.log("start", "user_message",
               {"session_id": session.session_id, "message": user_text, "source": source})

    try:
        client = build_client()
    except LLMError as exc:
        logger.log("init", "llm_error", {"message": exc.message, "detail": exc.detail})
        _save_failure_record(session, user_text, exc.message, exc.detail, [], source, run_id)
        return ChatResponse(
            session_id=session.session_id,
            error=exc.message,
            error_detail=exc.detail,
            run_id=run_id,
        )

    session.add_message("user", user_text)
    session.trim()

    # 用户到底有没有说"预测第几次"：没说的话，后面一律不预测，先停下来问
    position_hint = _predict_position_hint(user_text)
    # 用户原话里的成绩序列如果本身有空缺（缺考、写成文字），先补全再算
    score_gap = _score_list_gap(user_text)

    base_messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    base_messages.extend(session.messages)

    tool_records: list[ToolCallRecord] = []
    last_trend: dict[str, Any] | None = None
    last_retrieval: dict[str, Any] | None = None
    last_tool_error: dict[str, Any] | None = None
    chart_url: str | None = None
    final_text = ""
    messages = list(base_messages)

    try:
        for round_index in range(settings.max_tool_rounds):
            context = {"role": "system", "content": session.context_note()}
            response = client.chat([messages[0], context, *messages[1:]], tools=TOOLS)
            logger.log(
                f"round-{round_index + 1}",
                "llm_response",
                {
                    "model": response.get("model"),
                    "usage": response.get("usage"),
                    "tool_calls": response.get("tool_calls"),
                    "content": response.get("content"),
                },
            )

            tool_calls = response.get("tool_calls") or []
            if not tool_calls:
                # 护栏：首轮直接作答且从未调用工具时，提醒一次（只提醒一次，避免死循环）
                if not tool_records and round_index == 0 and _needs_tool(user_text):
                    logger.log("guard", "no_tool_call_nudge", {"content": response.get("content", "")[:200]})
                    messages.append({"role": "assistant", "content": response.get("content") or ""})
                    messages.append({"role": "system", "content": GUARD_PROMPT})
                    continue
                final_text = response.get("content", "")
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.get("content") or "",
                    "tool_calls": tool_calls,
                }
            )

            # 程序侧的提示统一放到本轮所有工具结果之后再插入：
            # 工具结果必须紧跟在 assistant 的 tool_calls 后面，中间夹 system 会被服务方判为非法请求
            round_notes: list[str] = []

            for call in tool_calls:
                name = call["function"]["name"]
                raw_args = call["function"]["arguments"] or "{}"
                try:
                    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                except json.JSONDecodeError:
                    arguments = {}
                    logger.log("tool", "bad_arguments", {"name": name, "raw": raw_args})

                # 本轮已经判定输入无效：不再允许重新拟合。
                # 否则模型可能"少传一个坏值"再算一次，等于替用户删掉数据后给出趋势。
                if name == "analyze_score_trend" and last_tool_error is not None:
                    blocked = {
                        "ok": False,
                        "error_code": "BLOCKED_AFTER_ERROR",
                        "message": "本轮已判定输入数据无效，拟合工具暂停执行",
                        "hint": "请直接说明上一次的错误并引导用户按提示修改，不要重试工具",
                    }
                    logger.log("guard", "blocked_retry", {"arguments": arguments})
                    tool_records.append(
                        ToolCallRecord(
                            name=name,
                            arguments=arguments,
                            ok=False,
                            summary="工具已暂停：等待用户修正输入后重新分析",
                        )
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(blocked, ensure_ascii=False),
                        }
                    )
                    round_notes.append("【程序校验】输入数据尚未修正，本轮不要再调用 analyze_score_trend。")
                    continue

                # 用户原话里的成绩有空缺（缺考/写成文字）：不拿残缺数据拟合，
                # 直接把"哪一次没填"还给模型，让用户补全后再算。
                if name == "analyze_score_trend" and score_gap:
                    result = _missing_slots_error(score_gap)
                    logger.log("guard", "missing_score_value", {"scores": score_gap})
                    tool_records.append(
                        ToolCallRecord(
                            name=name,
                            arguments=arguments,
                            ok=False,
                            summary=f"工具拒绝执行：{result['error_code']} {result['message']}",
                        )
                    )
                    last_tool_error = {"result": result, "arguments": arguments}
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                    round_notes.append(_failed_tool_note(result))
                    continue

                # 预测位置：用户说了就按用户说的，没说就不允许模型自己猜"下一次"
                guard_note = None
                if name == "analyze_score_trend":
                    arguments, adjusted, guard_note = _align_predict_arguments(arguments, position_hint)
                    if adjusted:
                        logger.log(
                            "guard",
                            "predict_position",
                            {"hint": position_hint, "arguments": arguments, "note": guard_note},
                        )

                result, summary = _execute_tool(name, arguments)
                logger.log(
                    "tool",
                    "executed",
                    {"name": name, "arguments": arguments, "ok": bool(result.get("ok")), "summary": summary},
                )
                tool_records.append(
                    ToolCallRecord(name=name, arguments=arguments, ok=bool(result.get("ok")), summary=summary)
                )
                if name == "analyze_score_trend" and result.get("ok"):
                    last_trend = result
                    chart_url = result.get("chart_url")
                    # 后面成功了就不需要再让用户改数据
                    last_tool_error = None
                elif name == "analyze_score_trend":
                    # 记录失败上下文（原始输入 + 错误码 + 出错位置），用于生成"改哪里"的可操作卡片
                    last_tool_error = {"result": result, "arguments": arguments}
                    failure_note = _failed_tool_note(result)
                    guard_note = f"{guard_note}\n{failure_note}" if guard_note else failure_note
                elif name == "search_study_knowledge":
                    # 检索问答也落库（含资料不足的情况），留作检索功能的验证证据
                    last_retrieval = {"result": result, "arguments": arguments}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                # 工具结果之后再补一句程序侧要求，让模型知道预测位置被系统改过/忽略了
                if guard_note:
                    round_notes.append(guard_note)

            # 本轮工具结果全部写完之后，再插入程序侧提示
            for note in round_notes:
                messages.append({"role": "system", "content": note})
        else:
            raise LLMError("工具调用轮次超限", f"连续 {settings.max_tool_rounds} 轮仍在调用工具，已终止以避免死循环")
    except LLMError as exc:
        logger.log("error", "llm_error", {"message": exc.message, "detail": exc.detail})
        _save_failure_record(
            session, user_text, exc.message,
            f"{exc.detail}（run_id={run_id}，可在 logs/run.jsonl 中检索该编号排查）",
            tool_records, source, run_id,
        )
        return ChatResponse(
            session_id=session.session_id,
            tool_calls=tool_records,
            chart_url=chart_url,
            error=exc.message,
            error_detail=f"{exc.detail}（run_id={run_id}，可在 logs/run.jsonl 中检索该编号排查）",
            run_id=run_id,
        )

    if not final_text.strip():
        logger.log("error", "empty_final", {})
        _save_failure_record(
            session, user_text, "模型没有返回最终回答", f"run_id={run_id}", tool_records, source, run_id
        )
        return ChatResponse(
            session_id=session.session_id,
            tool_calls=tool_records,
            chart_url=chart_url,
            error="模型没有返回最终回答",
            error_detail=f"run_id={run_id}",
            run_id=run_id,
        )

    failure = (last_tool_error or {}).get("result") if last_tool_error else None
    answer, errors = _parse_answer(final_text)
    if answer is not None:
        errors = _cross_check(answer, last_trend, failure)

    if errors:
        logger.log("validate", "rejected", {"errors": errors, "raw": final_text})
        repair_messages = messages + [
            {"role": "assistant", "content": final_text},
            {"role": "user", "content": REPAIR_PROMPT.format(errors="\n".join(f"- {e}" for e in errors))},
        ]
        try:
            repaired = client.chat(repair_messages, tools=None, temperature=0.0, json_mode=True)
            raw_repaired = repaired.get("content", "")
            logger.log("validate", "repair_attempt", {"raw": raw_repaired})
            answer, errors = _parse_answer(raw_repaired)
            if answer is not None:
                errors = _cross_check(answer, last_trend, failure)
        except LLMError as exc:
            logger.log("validate", "repair_failed", {"message": exc.message})
            errors = errors + [f"二次修复调用失败：{exc.message}"]

    # 拟合工具已经失败：模型再坚持给数值就直接换成程序生成的说明，
    # 用户看到的一定是"为什么算不了 + 怎么改"，而不是一个编出来的趋势或分数。
    if errors and failure:
        logger.log("validate", "tool_error_fallback", {"errors": errors, "code": failure.get("error_code")})
        answer = _error_answer(last_tool_error)
        errors = []

    if answer is None:
        logger.log("validate", "failed", {"errors": errors, "raw": final_text})
        _save_failure_record(
            session, user_text, "模型输出未通过结构化校验",
            "；".join(errors[:4]) + f"（原始输出已写入日志，run_id={run_id}）",
            tool_records, source, run_id,
        )
        return ChatResponse(
            session_id=session.session_id,
            tool_calls=tool_records,
            chart_url=chart_url,
            error="模型输出未通过结构化校验",
            error_detail="；".join(errors[:4]) + f"（原始输出已写入日志，run_id={run_id}）",
            run_id=run_id,
        )

    # 通过校验：落库 + 更新会话记忆
    if last_trend:
        record_id = db.save_analysis(
            session.session_id, user_text, last_trend, source=source, answer_text=answer.answer
        )
        logger.log("db", "saved", {"id": record_id, "slope": last_trend.get("slope"), "source": source})
        session.last_dataset = {
            "x": last_trend["x"],
            "y": last_trend["y"],
            "slope": last_trend["slope"],
            "intercept": last_trend["intercept"],
            "r2": last_trend["r2"],
        }
    elif last_retrieval is not None:
        record_id = db.save_retrieval(
            session.session_id, user_text, last_retrieval["result"], answer.answer, source=source
        )
        logger.log(
            "db",
            "saved_retrieval",
            {
                "id": record_id,
                "mode": last_retrieval["result"].get("mode"),
                "hits": len(last_retrieval["result"].get("hits") or []),
                "source": source,
            },
        )

    session.add_message("assistant", answer.answer)
    session.trim()
    logger.log(
        "finish",
        "answered",
        {"trend": answer.trend, "slope": answer.slope, "prediction": answer.prediction},
    )

    return ChatResponse(
        session_id=session.session_id,
        reply=answer,
        tool_calls=tool_records,
        chart_url=chart_url,
        fix=_build_fix(last_tool_error, last_trend),
        run_id=run_id,
    )
