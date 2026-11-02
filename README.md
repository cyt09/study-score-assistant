# 学习与成绩分析助手

一个本地运行的 Agent 小应用：用户在网页输入成绩或提问，后端调用真实大模型，模型按需调用
**成绩趋势分析工具**（NumPy 手写一元线性最小二乘）或**学习资料检索工具**（BM25 + 余弦相似度），
程序执行后把结果回填模型生成最终回答，回答经过**结构化校验**后连同图表展示给用户，并把每次
分析写入 SQLite。

## 项目用途

- 输入几次考试的成绩 → 得到趋势（斜率/截距/R²）、指定次数的预测值、拟合图，以及明确的限制条件；
- 预测位置必须由用户指定：只说「帮我预测成绩」时程序会中断预测、只讲趋势，并给出「第 N 次」一键选项继续询问；模型自己填的 predict_x 会被程序改写或丢弃；
- 追问「第五次呢」→ 会话记忆直接复用上一轮数据，不重复索要成绩；
- 问学习方法类问题 → 真实检索本地资料（关键词 BM25 / 向量 tf-idf 余弦两种模式）并展示命中片段与来源；资料没覆盖时明确说"资料不足"；
- 输入非法数据（数据不足、成绩里混着文字、缺考、序号重复/全部相同）→ 工具拒绝计算并说明原因，前端给出对应的填空/选项卡片，标红出错位置；修复前绝不下发任何趋势或预测数值；
- 模型或工具调用失败（鉴权失败、网络错误、轮次超限、结构化校验不通过）→ 聊天区展示失败状态，同时落库一条「失败记录」（kind=error），保留失败原因、run_id、已执行工具等排查信息，记录列表里用红色"调用失败"标签区分，详情弹窗可查看完整排查线索；
- 分析记录按会话聚合展示，点开任意一轮可查看完整问答（含来源片段与图表），任意会话可一键恢复上下文继续多轮追问（服务重启后同样可恢复）；
- 记录搜索（`GET /api/search`）：关键词 / 会话 ID / 时间三种入口，可组合（组合即"同时满足"）。关键词同时匹配问题、回答原文和会话 ID；会话 ID 支持前 8 位前缀匹配；时间支持 2026-10-30、2026-10、2026、10-30 等写法。搜索框输入时自动识别类型，结果直接定位到具体某一条（带轮次序号），点击即查看完整问答；
- 支持记录管理：单轮记录可逐条删除（悬停显示删除按钮，删除前二次确认，后端对不存在/已删除的记录返回 404），悬停会话卡片可删除整段会话（全部轮次连同内存上下文一起清除），整份记录可一键清空；
- 应用级界面（浅色科技风）：玻璃拟态三栏工作台（记录侧栏 / 对话区 / 数据看板），毛玻璃、柔阴影、渐变网格与颗粒纹理背景，Three.js 粒子场与漂浮几何体 + 鼠标视差，微交互、弹性动画与滚动渐入；看板含 KPI、记录构成与最近斜率走势，数据全部来自已有接口。

## 架构

```
浏览器(原生 HTML/CSS/JS，Three.js 背景粒子本地打包于 app/static/vendor)
   │  POST /api/chat
   ▼
FastAPI (app/main.py)
   ▼
Agent 循环 (app/agent.py) ── 系统提示词 + 会话记忆(app/session.py)
   │  工具调用(OpenAI function calling 协议)
   ├── analyze_score_trend ── NumPy 闭式解最小二乘 + matplotlib 图 (app/tools/trend.py)
   ├── search_study_knowledge ─ BM25 + tf-idf 余弦 + 覆盖率门槛 (app/tools/retrieval.py)
   ▼
结构化输出校验 (app/schemas.py, Pydantic) ── 不通过则触发一次修复重试
   ▼
SQLite 落库 (app/db.py) + 运行日志 (app/runlog.py, logs/*.jsonl)
```

设计原则：**模型只负责理解和表达，所有计算、检索、校验、落库都由程序完成**。
模型给出的 slope/prediction 必须与工具真实返回值一致，否则被程序拒绝并要求重写。

无效数据校验（`INVALID_SCORE` / `INVALID_X`）：负数或超过 150 分的成绩、小于 1 的考试序号
都会被工具拒绝并说明具体是哪个值无效，用户修改后可直接重发继续。

## 依赖版本

- Python 3.11+（开发环境 3.13.14 / Windows 11）
- 依赖清单见 `requirements.txt`（fastapi 0.162.0 / uvicorn 0.54.0 / numpy 2.5.4 /
  matplotlib 3.11.2 / openai 3.41.0 / pydantic 2.14.1 / python-dotenv 1.2.4）

## 安装与启动

**最省事的方式（Windows）**：双击项目根目录的 `start.bat`，脚本会自动检查虚拟环境、
安装依赖、启动服务并打开浏览器；检查测试可双击 `verify.bat`。

命令行方式：

```bash
# 1. 创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置密钥（见下节"配置方法"）
copy .env.example .env      # Windows
# cp .env.example .env      # macOS / Linux
#   然后把 .env 中的 DEEPSEEK_API_KEY 改成自己的 Key

# 4. 启动
python -m uvicorn app.main:app --port 8000

# 5. 打开 http://127.0.0.1:8000
```

## 配置方法

所有配置都在 `.env`（参考 `.env.example`，仓库中不含任何密钥）：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | DeepSeek 开放平台 API Key（[申请地址](https://platform.deepseek.com/api_keys)） | 无，必填 |
| `LLM_BASE_URL` | 任何 OpenAI 兼容接口 | `https://api.deepseek.com` |
| `LLM_MODEL` | 模型名 | `deepseek-chat` |
| `LLM_MOCK` | `1` 时使用本地桩模型（不联网，仅验证链路） | `0` |
| `MAX_TOOL_ROUNDS` | 工具调用最大轮次（防死循环） | `4` |
| `RETRIEVAL_TOP_K` / `RETRIEVAL_MIN_SCORE` | 检索条数与最低得分 | `3` / `1.2` |

## 测试与验证

```bash
# 单元/流程测试（44 项，不需要 API Key）
python -m unittest tests.test_pipeline -v

# 数值验证：独立实现 + 手算答案双重核对
python scripts/verify_calc.py

# 一键场景验证：生成 docs/验证记录.md 与脱敏日志 logs/validation_*.jsonl
# （需先启动服务；配置真实 Key 后重新运行即可得到正式记录）
python scripts/run_validation.py --base http://127.0.0.1:8000
```

七类必验场景（正常数据 / 连续追问 / 未指定预测位置时中断询问 / 无效数据 / 知识库命中 /
知识库未覆盖 / 调用失败）的真实运行记录见 `docs/验证记录.md`，对应的脱敏运行日志
（自动过滤密钥，见 `app/runlog.py`）随仓库提交于 `logs/validation_*.jsonl`。

## 目录结构

```
app/
  main.py        FastAPI 入口与接口
  agent.py       Agent 主循环、工具执行、输出校验与修复
  llm.py         DeepSeek 客户端（OpenAI 兼容）+ 本地桩模型
  schemas.py     结构化输出定义与字段校验
  session.py     会话记忆（对话历史 + 最近数据集）
  db.py          SQLite 分析记录
  runlog.py      JSONL 运行日志（自动脱敏）
  tools/
    trend.py     最小二乘趋势工具 + 图表
    retrieval.py 本地知识库 BM25 检索
  static/        前端单页
knowledge/       检索用学习资料（4 篇，获取说明见 knowledge/README.md）
scripts/         verify_calc.py 数值验证 / run_validation.py 场景验证
tests/           44 项单元与流程测试
docs/            验证记录、AI 使用说明、录制指南、提交清单
```

## 已知限制

1. **预测是外推**：模型假设"趋势保持线性"，真实学习存在平台期与难度跃升；预测距离已有数据越远，误差越大。回答中会附限制条件提示。
2. **相关不等于因果**：对"是不是因为增加学习时间才提高"这类问题，系统只报告相关性并解释因果需要的证据，不做因果断言。
3. **样本量敏感**：数据点少于 3 个时拟合优度不具参考意义，工具会输出警告，回答会降调。
4. **检索是词元统计级的**：BM25 关键词与 tf-idf 余弦向量都是本地统计实现，不是深度语义向量模型；同义改写或资料中完全没有对应措辞的问题可能召回不足，此时会明确说资料不足而不是硬答。
5. **会话上下文可从记录恢复**：实时会话记忆在内存中，服务重启后清空；点开"分析记录"里的任意会话（或继续对其提问）时会从 SQLite 回填对话与最近数据集。未做多用户并发隔离，仅适合本机单人使用。
6. **结构化输出靠解析 + 校验**：未使用厂商的原生 JSON mode 强约束（工具调用与 JSON mode 不能同时启用），异常输出会触发一次修复重试，仍失败则返回明确错误。
7. 图表中文字体依赖系统字体（Windows 下使用微软雅黑）；缺失字体的系统会退回英文标签，数值不受影响。

## AI 使用披露

本项目在开发过程中使用了 AI 辅助（说明与自查过程见 `docs/AI使用说明.md`）。

## 发布到 GitHub

仓库已初始化 git 并完成首次提交，推送步骤：

```bash
# 1. 在 GitHub 网页上新建一个空仓库（不要勾选自动生成 README/.gitignore）
# 2. 关联远程并推送
git remote add origin https://github.com/<你的用户名>/<仓库名>.git
git push -u origin main
```

推送前自检（敏感信息确认）：

```bash
git ls-files | grep -E "\.env$|\.venv|data/.*\.db|data/charts" || echo "无敏感文件被跟踪"
grep -rn "sk-" --include="*.py" --include="*.js" --include="*.md" app scripts tests docs \
  | grep -v "sk-your-key-here\|sk-invalid" || echo "无硬编码密钥"
```

`.env`、`data/history.db`、`data/charts/`、`logs/run.jsonl` 均在 `.gitignore` 中，
不会进入仓库；随仓库提交的 `logs/validation_*.jsonl` 是脱敏后的验证日志。
