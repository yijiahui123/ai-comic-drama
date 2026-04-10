# 🎬 AI漫剧全自动化流水线 — 技术路线

> **目标形态**：与 OpenClaw 对话 → 全自动生成漫剧成品视频  
> **硬件基准**：MacBook Pro M5 Max · 128GB 统一内存 · 2TB SSD

---

## 一、整体架构

```
用户 ←→ OpenClaw Chat
              │
              ▼
     ┌──────────────────┐
     │ Pipeline          │
     │ Orchestrator      │  ← 编排层（状态机）
     └────────┬─────────┘
              │
   ┌──────────┼──────────────────────┐
   ▼          ▼          ▼           ▼
Skill 1    Skill 2    Skill 3     Skill 4
剧本生成    资产创建    图生视频    剪辑包装
(LLM)     (ComfyUI)  (Wan2.1)   (FFmpeg)
              │
              ▼
         成品视频 .mp4
```

---

## 二、四步流程 × 技术栈

### 步骤 1：文生剧本（ScriptWriter）

| 项目 | 说明 |
|---|---|
| **子任务** | 故事大纲 → 分镜脚本 → 角色对白 |
| **推荐模型** | Llama 3.1 70B-Q4 / Qwen2.5-72B-Q4 |
| **部署方式** | `ollama` / `llama.cpp` (Metal 加速) |
| **内存需求** | ~40GB 统一内存 |
| **输出格式** | 结构化 JSON |
| **版权策略** | 自定义 Prompt + RAG 知识库（原创世界观语料），降低侵权风险 |

**输出示例：**

```json
{
  "title": "星际漫游记",
  "episodes": [
    {
      "episode": 1,
      "scenes": [
        {
          "scene_id": "S01",
          "location": "太空站控制室",
          "time": "夜晚",
          "shots": [
            {
              "shot_id": "S01-001",
              "type": "全景",
              "characters": ["凯", "艾拉"],
              "dialogue": "凯：信号源来自M87星系的边缘。",
              "visual_prompt": "Wide shot of a futuristic space station control room, holographic star map glowing blue, two characters in uniform looking at screen, anime style",
              "camera_move": "缓慢推进",
              "duration": 4
            }
          ]
        }
      ]
    }
  ]
}
```

---

### 步骤 2：资产创建（AssetGenerator）

| 项目 | 说明 |
|---|---|
| **角色一致性立绘** | SDXL + IP-Adapter + InstantID |
| **场景/道具/技能图** | SDXL + ControlNet（深度图/线稿） |
| **风格统一** | 训练漫画风格 LoRA（kohya_ss / ComfyUI Train） |
| **部署方式** | ComfyUI（MPS 后端） |
| **内存需求** | ~12-16GB |
| **LoRA 训练数据** | 约 50-200 张风格参考图 |

**资产管理结构：**

```
assets/
├── characters/
│   ├── kai/
│   │   ├── reference.png          # 参考立绘
│   │   ├── ip_adapter_embed.safetensors  # IP-Adapter 嵌入
│   │   └── expressions/           # 表情变体
│   └── aila/
├── scenes/
│   ├── space_station_control/
│   └── planet_surface/
├── props/
├── effects/
└── style_lora/
    └── comic_style_v1.safetensors
```

---

### 步骤 3：图生视频（VideoGenerator）

| 项目 | 说明 |
|---|---|
| **图生视频模型** | Wan2.1-14B（首选）/ CogVideoX-5B（备选） |
| **口型同步** | SadTalker / MuseTalk |
| **AI 配音** | ChatTTS / GPT-SoVITS（可克隆角色声线） |
| **部署方式** | ComfyUI 节点 / 官方推理脚本 |
| **内存需求** | Wan2.1-14B: 60-80GB / CogVideoX-5B: ~40GB |
| **生成速度** | 约 5-15 min/镜头（4s clip），M5 Max |

**性能预估：**

| 模型 | 分辨率 | 时长/clip | M5 Max 耗时 |
|---|---|---|---|
| Wan2.1-14B | 720p | 4s | ~10-15 min |
| CogVideoX-5B | 480p | 6s | ~5-8 min |
| SadTalker | 512x512 | 按音频长度 | ~1-2 min |
| ChatTTS | - | 按文本长度 | ~10-30s |

---

### 步骤 4：剪辑包装（Editor）

| 项目 | 说明 |
|---|---|
| **自动剪辑** | FFmpeg + MoviePy 脚本 |
| **字幕生成** | Whisper（语音→时间轴字幕） |
| **特效/转场** | Pillow / FFmpeg 滤镜 |
| **背景音乐** | MusicGen / Stable Audio（按场景情绪标签配乐） |
| **内存需求** | CPU 为主；Whisper ~2-4GB；MusicGen ~8-12GB |

**剪辑脚本逻辑：**

```python
# 伪代码
for scene in script["scenes"]:
    for shot in scene["shots"]:
        video = load(f"output/videos/{shot['shot_id']}.mp4")
        audio = load(f"output/audio/{shot['shot_id']}.wav")
        subtitle = generate_subtitle(shot["dialogue"])
        
        clip = compose(video, audio, subtitle)
        clip = apply_transition(clip, shot.get("transition", "crossfade"))
        timeline.append(clip)

bgm = generate_music(scene["mood"])
final = merge(timeline, bgm)
final.export("output/final_episode.mp4")
```

---

## 三、OpenClaw Skills 开发清单

| Skill 名称 | 职责 | 调用接口 | 输入 | 输出 |
|---|---|---|---|---|
| **ScriptWriter** | LLM 生成结构化剧本 | Ollama API (`localhost:11434`) | 用户自然语言描述 | 剧本 JSON |
| **AssetGenerator** | ComfyUI 批量生成图片资产 | ComfyUI API (`localhost:8188/prompt`) | 剧本 JSON + 角色参考图 | 角色图/场景图文件 |
| **VideoGenerator** | 图生视频 + 配音 + 口型 | ComfyUI Wan2.1 节点 + ChatTTS | 图片资产 + 对白文本 | 视频片段 .mp4 |
| **Editor** | 自动剪辑合成 | FFmpeg CLI / MoviePy | 视频片段 + 剧本 JSON | 成品视频 .mp4 |

---

## 四、依赖服务 & 端口规划

| 服务 | 默认端口 | 启动命令 |
|---|---|---|
| Ollama | `11434` | `ollama serve` |
| ComfyUI | `8188` | `python main.py --listen 0.0.0.0 --port 8188` |
| ChatTTS | `9966` | `python app.py --port 9966` |
| SadTalker | `7860` | `python inference.py --port 7860` |
| OpenClaw | `3000` | 待定（取决于 OpenClaw 框架） |

---

## 五、成本估算

| 项目 | 费用 | 说明 |
|---|---|---|
| 硬件 | ¥0（已有） | M5 Max 128GB 是苹果端跑大模型的天花板 |
| 模型 | ¥0 | 全部开源模型，无 API 费用 |
| ComfyUI + 插件 | ¥0 | 开源 |
| LoRA 训练数据 | ¥0-500 | 如需购买风格参考图/画师授权 |
| 存储 | 关注 2TB 用量 | 单个视频模型 ~20-30GB；素材库增长快，建议外挂 NAS |
| 开发时间 | ~4-8 周 | 搭建全流程 + 调试 Skill + 质量优化 |
| 电费 | 较高 | M5 Max 满载 ~80-120W，长时间生成注意散热 |

---

## 六、风险 & 注意事项

| 风险 | 说明 | 缓解方案 |
|---|---|---|
| **MPS 兼容性** | 并非所有模型完美支持 Apple MPS | 优先用 MLX 社区转换权重；关注 `mlx-community` HuggingFace |
| **生成速度** | 图生视频是最慢环节 | 缩小分辨率 / 用 CogVideoX-5B 替代 / 分批夜间生成 |
| **角色一致性** | IP-Adapter 并非 100% 一致 | 多次生成+筛选 / 训练角色 LoRA / 人工校验环节 |
| **版权风险** | AI 生成内容版权归属各国法律有争议 | 用完全原创世界观 + 保留全部创作过程记录 |
| **内存压力** | 多模型同时加载可能 OOM | Pipeline 串行执行，用完一个模型释放后再加载下一个 |

---

## 七、开发路线图

| 阶段 | 周次 | 任务 | 产出 |
|---|---|---|---|
| **环境搭建** | Week 1 | 安装 Ollama + ComfyUI + SDXL + IP-Adapter | 手动跑通单张图生成 |
| **Skill 1** | Week 2 | 编写 ScriptWriter Skill | LLM → 结构化剧本 JSON |
| **Skill 2** | Week 3 | 编写 AssetGenerator Skill | ComfyUI API 批量出图 |
| **Skill 3** | Week 4-5 | 部署 Wan2.1/CogVideoX + 编写 VideoGenerator Skill | 图生视频 + 配音 |
| **Skill 4** | Week 6 | 编写 Editor Skill | FFmpeg 自动剪辑 |
| **集成调试** | Week 7-8 | 集成到 OpenClaw + 端到端调试 + 质量优化 | 完整 Pipeline 可用 |

---

## 八、本地开发环境配置

```bash
# 1. 安装 Ollama
brew install ollama
ollama pull llama3.1:70b-instruct-q4_K_M

# 2. 安装 ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
# 安装自定义节点
cd custom_nodes
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git

# 3. 下载模型
# SDXL
wget -P models/checkpoints/ https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors
# IP-Adapter
wget -P models/ipadapter/ https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors

# 4. 安装 ChatTTS
git clone https://github.com/2noise/ChatTTS.git
cd ChatTTS
pip install -r requirements.txt

# 5. 安装 FFmpeg
brew install ffmpeg

# 6. Python 依赖
pip install moviepy whisper-openai requests aiohttp pydantic
```

---

## 九、项目目录结构（建议）

```
ai-comic-drama/
├── README.md
├── docs/
│   └── technical-roadmap.md      # 本文档
├── skills/
│   ├── script_writer/            # Skill 1: 剧本生成
│   │   ├── __init__.py
│   │   ├── skill.py
│   │   └── prompts/
│   ├── asset_generator/          # Skill 2: 资产创建
│   │   ├── __init__.py
│   │   ├── skill.py
│   │   └── workflows/            # ComfyUI 工作流 JSON
│   ├── video_generator/          # Skill 3: 图生视频
│   │   ├── __init__.py
│   │   ├── skill.py
│   │   └── configs/
│   └── editor/                   # Skill 4: 剪辑包装
│       ├── __init__.py
│       ├── skill.py
│       └── templates/            # 转场/字幕模板
├── pipeline/
│   ├── orchestrator.py           # 流水线编排
│   └── state.py                  # 状态管理
├── assets/                       # 生成的资产（gitignore）
├── output/                       # 输出视频（gitignore）
├── configs/
│   └── services.yaml             # 服务端口配置
├── requirements.txt
└── .gitignore
```

---

*最后更新：2026-04-10*