"""学习资料检索：本地 Markdown 知识库，支持关键词检索与向量检索两种模式。

两种检索方式都是本地真实实现、不依赖外部服务：
- keyword（关键词检索）：BM25 得分排序，对"术语型"问题（问题里直接出现资料中的词）召回稳定；
- vector（向量检索）：查询与片段各表示为 tf-idf 权重向量，按余弦相似度排序，
  对口语化、同义改写的问题（用词与资料不一致）比纯关键词更稳；
- hybrid（默认）：BM25 × (1 + 余弦)，兼顾两种信号。

两个关键设计（对应"知识库未覆盖的问题要说明资料不足"这一要求）：
1. 中文只保留字符二元组，并过滤含虚词的二元组，避免"的/是/什么"这类噪声把无关文档顶上来；
2. 除排序得分外，再计算"查询词覆盖率"，覆盖率过低直接判定资料不足，不硬凑引用。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from config import settings

_CJK = re.compile(r"[\u4e00-\u9fff]")
# 允许单字母 + 上标（R²、t 值）这类术语：原来的 {2,} 会把 "R²" 整个丢掉，
# 导致"R² 是什么意思"只剩"意思/解读"两个词，直接判成资料不足。
_WORD = re.compile(r"[a-zA-Z0-9_²]+")

# 查询扩展：口语说法与资料用词对不上时（"突然掉了20分" vs "成绩波动"），
# 用一张小型同义词表把领域术语补进查询。只补领域词，不补通用词，避免把无关问题拉进来。
_SYNONYMS = (
    (("掉分", "掉了", "跌了", "降了", "下滑", "退步", "起伏", "忽高忽低", "考差", "没考好", "不稳定"),
     "成绩波动 原因"),
    (("紧张", "失常", "发挥", "心态", "焦虑", "状态不好"), "临场状态 发挥"),
    (("两个点", "两点", "数据点", "次数太少", "样本"), "样本量 趋势判断"),
    (("因为", "导致", "因果", "原因", "是不是"), "相关 因果"),
    (("r²", "r2", "拟合优度", "决定系数"), "R² 拟合优度"),
    (("外推", "往后推", "未来"), "外推 风险"),
    (("复习", "记忆", "遗忘", "错题", "拖延", "专注", "效率"), "学习效率 复习"),
)


def _expansion_terms(query: str) -> list[str]:
    """把口语化说法映射到资料里的术语，返回**独立切好的**扩展词。

    不能把扩展文本直接拼进原句：中文二元组按字符相邻切分，拼接处会产出
    "率复""率高"这种跨词垃圾词，反而拉低覆盖率。
    """
    lowered = query.lower()
    terms: list[str] = []
    for triggers, expansion in _SYNONYMS:
        if any(trigger in lowered for trigger in triggers):
            terms.extend(_tokenize(expansion))
    # 去重但保持顺序，避免同一批词被重复计入词频
    return list(dict.fromkeys(terms))

# 高频虚词：出现在二元组里说明该二元组多半是语法片段而非术语
_STOP_CHARS = set(
    "的了是在和与有也就都而及或我你他她它这那之其为不很会要上下个吗呢吧啊把被对从到还没最更再又才只"
    "什么怎怎样如何谁哪能可以想说做时候让给但所因果着过里中后前多少好大小"
)

# 覆盖率门槛：查询中有意义词被命中的比例低于该值，视为知识库未覆盖
_MIN_COVERAGE = 0.34
# 余弦相似度门槛：BM25 只看词命中，容易被单个偶然词抬高；cos 低于该值且覆盖率不足时判定资料不足
_MIN_COSINE = 0.10


@dataclass
class Chunk:
    doc: str
    chunk_id: str
    text: str
    tokens: list[str]


def _tokenize(text: str) -> list[str]:
    """英文/数字按词切分；中文取相邻二元组，并丢掉含虚词的二元组。"""
    text = text.lower().replace("²", "2")
    tokens = [token for token in _WORD.findall(text) if len(token) > 1]
    chars = _CJK.findall(text)
    for i in range(len(chars) - 1):
        bigram = chars[i] + chars[i + 1]
        if chars[i] in _STOP_CHARS or chars[i + 1] in _STOP_CHARS:
            continue
        tokens.append(bigram)
    return tokens


def _split_chunks(doc_name: str, content: str, max_len: int = 420) -> list[Chunk]:
    """按段落聚合，单段过长则按句号再切，保证片段可独立阅读。"""
    # 标题行保留在片段里：小标题往往承载了核心术语（如"间隔重复""学习效率"），丢掉会直接损失召回
    paragraphs = [block.strip().lstrip("#").strip() for block in re.split(r"\n\s*\n", content) if block.strip()]

    merged: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) + 2 <= max_len:
            buffer = f"{buffer}\n{para}".strip()
        else:
            if buffer:
                merged.append(buffer)
            while len(para) > max_len:
                cut = para.rfind("。", 0, max_len)
                cut = cut + 1 if cut > 100 else max_len
                merged.append(para[:cut].strip())
                para = para[cut:].strip()
            buffer = para
    if buffer:
        merged.append(buffer)

    # 文件名即文档主题，参与词表可让"效率""间隔重复"这类只出现在标题里的术语可被检索到
    doc_title = doc_name.rsplit(".", 1)[0]
    return [
        Chunk(doc=doc_name, chunk_id=f"{doc_name}#{index}", text=text,
              tokens=_tokenize(f"{doc_title} {text}"))
        for index, text in enumerate(merged, start=1)
    ]


def _cosine(query_tf: Counter[str], chunk_tf: Counter[str], idf) -> float:
    """tf-idf 余弦相似度：衡量查询与片段在"术语分布"上的整体接近程度。"""
    shared = set(query_tf) & set(chunk_tf)
    numerator = sum(query_tf[t] * chunk_tf[t] * idf(t) ** 2 for t in shared)
    if numerator == 0:
        return 0.0
    q_norm = math.sqrt(sum((v * idf(t)) ** 2 for t, v in query_tf.items()))
    c_norm = math.sqrt(sum((v * idf(t)) ** 2 for t, v in chunk_tf.items()))
    if q_norm == 0 or c_norm == 0:
        return 0.0
    return numerator / (q_norm * c_norm)


class KnowledgeBase:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.chunks: list[Chunk] = []
        self._df: Counter[str] = Counter()
        self._avg_len: float = 0.0
        self.file_count: int = 0
        self.reload()

    def reload(self) -> None:
        chunks: list[Chunk] = []
        files = [p for p in sorted(self.directory.glob("*.md")) if p.name.lower() != "readme.md"]
        self.file_count = len(files)
        for path in files:
            content = path.read_text(encoding="utf-8", errors="ignore")
            chunks.extend(_split_chunks(path.name, content))
        self.chunks = chunks
        df: Counter[str] = Counter()
        for chunk in chunks:
            df.update(set(chunk.tokens))
        self._df = df
        self._avg_len = sum(len(c.tokens) for c in chunks) / len(chunks) if chunks else 0.0

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        total = len(self.chunks)
        return math.log(1 + (total - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int | None = None,
               min_score: float | None = None, mode: str = "hybrid") -> dict:
        """先按原话检索；原话一个片段都没命中时，再用同义词扩展重试一次。

        之所以做"兜底式"扩展而不是一开始就扩展：扩展会引入额外词，
        可能把本来排第一的正确片段挤下去（"如何提高复习效率"就踩过）。
        只在原话完全无命中时启用，既救回口语化问题，又不影响已经正确的排序。
        """
        result = self._search_once(query, top_k=top_k, min_score=min_score, mode=mode)
        if result.get("hits"):
            return result

        expanded = _expansion_terms(query)
        if not expanded:
            return result

        retry = self._search_once(query, top_k=top_k, min_score=min_score, mode=mode, extra_terms=expanded)
        if retry.get("hits"):
            retry["expanded_from"] = query
            retry["query"] = query
            retry["note"] = "原话未命中，已按同义术语扩展后重试"
            return retry
        return result

    def _search_once(self, query: str, top_k: int | None = None,
                     min_score: float | None = None, mode: str = "hybrid",
                     extra_terms: list[str] | None = None) -> dict:
        top_k = top_k or settings.retrieval_top_k
        min_score = settings.retrieval_min_score if min_score is None else min_score
        mode = str(mode or "hybrid").lower()
        if mode not in ("keyword", "vector", "hybrid"):
            mode = "hybrid"
        query = (query or "").strip()
        if not query:
            return {"ok": False, "error_code": "EMPTY_QUERY", "message": "检索问题为空", "hits": []}
        if not self.chunks:
            return {"ok": False, "error_code": "EMPTY_KB", "message": "知识库为空，没有任何资料", "hits": []}

        terms = _tokenize(query) + list(extra_terms or [])
        query_terms = sorted(set(terms))
        if not query_terms:
            return {"ok": False, "error_code": "EMPTY_QUERY", "message": "检索问题中没有可用关键词", "hits": []}

        k1, b = 1.5, 0.75
        query_tf = Counter(terms)
        scored: list[tuple[float, float, Chunk, list[str]]] = []
        for chunk in self.chunks:
            tf = Counter(chunk.tokens)
            score = 0.0
            matched: list[str] = []
            for term in query_terms:
                if tf.get(term, 0) == 0:
                    continue
                matched.append(term)
                denom = tf[term] + k1 * (1 - b + b * len(chunk.tokens) / (self._avg_len or 1))
                score += self._idf(term) * tf[term] * (k1 + 1) / denom
            if score > 0:
                scored.append((score, _cosine(query_tf, tf, self._idf), chunk, matched))

        # 排序方式由 mode 决定：
        # keyword → 纯 BM25（术语命中强度）
        # vector  → 纯 tf-idf 余弦（向量相似度，容忍用词改写）
        # hybrid  → BM25 × (1 + 余弦)，兼顾两种信号
        if mode == "keyword":
            scored.sort(key=lambda item: item[0], reverse=True)
        elif mode == "vector":
            scored.sort(key=lambda item: item[1], reverse=True)
        else:
            scored.sort(key=lambda item: item[0] * (1 + item[1]), reverse=True)
        matched_terms_all = {term for _, _, _, matched in scored[:top_k] for term in matched}
        coverage = len(matched_terms_all) / len(query_terms)
        best_cos = scored[0][1] if scored else 0.0

        # 覆盖率与余弦都过低 → 明确判定资料不足，而不是返回弱相关片段
        if not scored or (coverage < _MIN_COVERAGE and best_cos < _MIN_COSINE):
            return {
                "ok": True,
                "mode": mode,
                "hits": [],
                "coverage": round(coverage, 2),
                "query_terms": query_terms,
                "message": "知识库中没有与该问题相关的资料，应明确告知资料不足，不要编造引用",
                "query": query,
            }

        best = scored[0][0]
        hits = [
            {
                "title": chunk.doc,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 2),
                "similarity": round(cos, 3),
                "matched_terms": matched[:8],
                "snippet": chunk.text[:280] + ("…" if len(chunk.text) > 280 else ""),
            }
            for score, cos, chunk, matched in scored[:top_k]
            if score >= min_score and score >= best * 0.45
        ]

        if not hits:
            return {
                "ok": True,
                "mode": mode,
                "hits": [],
                "coverage": round(coverage, 2),
                "query_terms": query_terms,
                "message": "命中的片段相关度过低，应当说明资料不足",
                "query": query,
            }

        return {
            "ok": True,
            "mode": mode,
            "hits": hits,
            "coverage": round(coverage, 2),
            "query_terms": query_terms,
            "query": query,
        }


knowledge_base = KnowledgeBase(settings.knowledge_dir)


def search_knowledge(query: str, top_k: int | None = None, mode: str = "hybrid") -> dict:
    """工具入口：检索本地学习资料，返回命中片段与来源。

    mode: keyword=关键词检索（BM25）/ vector=向量检索（tf-idf 余弦）/ hybrid=混合（默认）。
    """
    return knowledge_base.search(query, top_k=top_k, mode=mode)
