# 🎬 AI Comic Drama — 全自动漫剧生成流水线

> 通过聊天描述需求，系统自动完成：剧本生成 → 资产创建 → 图生视频 → 剪辑包装，最终输出成品视频。

**硬件基准**：MacBook Pro M5 Max · 128GB 统一内存 · 2TB SSD

---

## 架构概览

```
用户描述
   │
   ▼
┌──────────────────────────────────────────────────────┐
│                 Pipeline Orchestrator                  │
│   INIT → SCRIPTING → ASSET_GEN → VIDEO_GEN → EDITING  │
└──────────────────────────────────────────────────────┘
        │            │           │           │
        ▼            ▼           ▼           ▼
  ScriptWriter  AssetGenerator VideoGenerator  Editor
   (Ollama)      (ComfyUI)     (Wan2.1+TTS)  (FFmpeg)
        │            │           │           │
        ▼            ▼           ▼           ▼
   剧本 JSON      资产图片     视频片段+音频  成品 .mp4
```

---

## 目录结构

```
ai-comic-drama/
├── main.py                          # CLI 入口
├── requirements.txt
├── .gitignore
├── README.md
├── docs/
│   └── technical-roadmap.md        # 技术路线文档
├── configs/
│   └── services.yaml               # 服务端口配置
├── pipeline/
│   ├── orchestrator.py             # 流水线编排（状态机）
│   └── state.py                    # 状态模型 + 持久化
├── skills/
│   ├── script_writer/
│   │   ├── skill.py                # ScriptWriter（Ollama API）
│   │   └── prompts/
│   │       ├── system_outline.txt  # 大纲生成提示词
│   │       └── system_scene.txt    # 分镜细化提示词
│   ├── asset_generator/
│   │   ├── skill.py                # AssetGenerator（ComfyUI API）
│   │   └── workflows/
│   │       ├── character_gen.json  # 角色生成工作流
│   │       ├── scene_gen.json      # 场景生成工作流
│   │       └── shot_gen.json       # 分镜图生成工作流
│   ├── video_generator/
│   │   ├── skill.py                # VideoGenerator（Wan2.1+ChatTTS+SadTalker）
│   │   └── configs/
│   │       ├── video_config.yaml   # 视频生成参数
│   │       └── voice_config.yaml   # 角色-音色映射
│   └── editor/
│       ├── skill.py                # Editor（FFmpeg自动剪辑）
│       └── templates/
│           ├── transitions.yaml    # 转场效果配置
│           ├── title_card.py       # 片头生成脚本
│           └── subtitle_style.ass  # 字幕样式模板
└── utils/
    ├── logger.py                   # 彩色日志 + 文件记录
    ├── http_client.py              # 异步 HTTP 客户端（含重试）
    └── validators.py               # 剧本 JSON Schema 验证
```

---

## 快速开始

### 1. 安装依赖

```bash
# Python 3.12+
pip install -r requirements.txt

# FFmpeg（视频剪辑）
brew install ffmpeg  # macOS
```

### 2. 启动依赖服务

| 服务 | 端口 | 启动命令 |
|---|---|---|
| Ollama | 11434 | `ollama serve` |
| ComfyUI | 8188 | `python main.py --listen 0.0.0.0 --port 8188` |
| ChatTTS | 9966 | `python app.py --port 9966` |
| SadTalker | 7860 | `python inference.py --port 7860` |

> **注意**：ChatTTS 和 SadTalker 是可选服务——若不可用，流水线会自动跳过配音和口型同步步骤。

### 3. 配置服务地址

编辑 `configs/services.yaml` 填入实际服务地址（默认 localhost）。

### 4. 运行

```bash
# 一键启动全流程
python main.py --prompt "写一个赛博朋克风格的3分钟漫剧，主角是黑客少女"

# 断点续跑（使用上次运行的 project_id）
python main.py --resume <project_id>

# 查看进度
python main.py --status <project_id>
```

---

## 输出说明

```
output/
├── state/<project_id>.json     # 流水线状态（可断点续跑）
├── videos/<shot_id>.mp4        # 各镜头原始视频
├── audio/<shot_id>.wav         # 各镜头配音
├── lipsync/<shot_id>_lipsync.mp4  # 口型同步视频
└── final/<project_id>_ep01.mp4   # 最终成品
```

---

## 依赖模型

| 模型 | 用途 | 推荐规格 |
|---|---|---|
| Llama 3.1 70B-Q4 / Qwen2.5-72B-Q4 | 剧本生成 | ~40GB 内存 |
| SDXL 1.0 + IP-Adapter | 角色/场景图生成 | ~12-16GB |
| Wan2.1-14B | 图生视频 | ~60-80GB |
| ChatTTS | AI 配音 | ~4-6GB |
| SadTalker | 口型同步 | ~8GB |
| Whisper | 字幕生成 | ~2-4GB |

---

## 技术栈

- **Python 3.12+** + `asyncio` + `aiohttp`（异步 HTTP）
- **Pydantic v2**（数据模型与验证）
- **PyYAML**（配置读取）
- **MoviePy** + **FFmpeg**（视频剪辑）
- **Ollama**（本地 LLM 推理）
- **ComfyUI**（图像/视频生成）

---

详细技术路线请参阅 [`docs/technical-roadmap.md`](docs/technical-roadmap.md)。
