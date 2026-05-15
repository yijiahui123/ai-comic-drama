# 🎬 AI漫剧全自动化流水线 — 技术路线

> **目标形态**：与 AI 对话 → 全自动生成漫剧成品视频  
> **硬件基准**：MacBook Pro M5 Max · 128GB 统一内存 · 2TB SSD

---

## 一、整体架构

```
用户 ←→ Chat
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
(oMLX)    (ComfyUI)  (Wan 2.2)  (FFmpeg)
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
| **推荐模型** | Qwen3.6-35B-A3B-MLX-8bit（本地）或 Llama 3.1 70B-Q4 |
| **部署方式** | `oMLX`（OpenAI 兼容 API） |
| **API 端点** | `POST http://127.0.0.1:8000/v1/chat/completions` |
| **内存需求** | ~20-25GB 统一内存 |
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
| **API 端点** | `POST http://localhost:8188/prompt` |
| **内存需求** | ~8-16GB |
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
| **图生视频模型** | Wan 2.2-14B FP16 两阶段采样 + LoRA 加速（首选）/ Wan 2.2-5B TI2V（轻量） |
| **口型同步** | SadTalker / MuseTalk |
| **AI 配音** | ChatTTS / GPT-SoVITS（可克隆角色声线） |
| **部署方式** | ComfyUI 节点（两阶段 KSamplerAdvanced + LoraLoaderModelOnly） |
| **内存需求** | Wan 2.2-14B FP16: ~30-40GB / Wan 2.2-5B: ~15-20GB |
| **生成速度** | 约 2-5 min/镜头（4s clip，LoRA 4步加速），M5 Max |

**性能预估：**

| 模型 | 分辨率 | 时长/clip | M5 Max 耗时 |
|---|---|---|---|
| Wan 2.2-14B FP16 + LoRA（两阶段 4步×2） | 720p | 4s | ~2-5 min |
| Wan 2.2-5B TI2V | 480p | 4s | ~2-4 min |
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

## 三、Skills 开发清单

| Skill 名称 | 职责 | 调用接口 | 输入 | 输出 |
|---|---|---|---|---|
| **ScriptWriter** | LLM 生成结构化剧本 | oMLX API (`127.0.0.1:8000/v1`) | 用户自然语言描述 | 剧本 JSON |
| **AssetGenerator** | ComfyUI 批量生成图片资产 | ComfyUI API (`localhost:8188/prompt`) | 剧本 JSON + 角色参考图 | 角色图/场景图文件 |
| **VideoGenerator** | 图生视频 + 配音 + 口型 | ComfyUI Wan 2.2 节点 + ChatTTS | 图片资产 + 对白文本 | 视频片段 .mp4 |
| **Editor** | 自动剪辑合成 | FFmpeg CLI / MoviePy | 视频片段 + 剧本 JSON | 成品视频 .mp4 |

---

## 四、依赖服务 & 端口规划

| 服务 | 默认端口 | conda 环境 | 启动命令 |
|---|---|---|---|
| oMLX | `8000` | `ai-comic` | `omlx serve --host 127.0.0.1 --port 8000` |
| ComfyUI (Wan 2.2) | `8188` | `comfyui` | `python main.py --listen 0.0.0.0 --port 8188 --enable-manager` |
| ChatTTS | `9966` | `ai-comic` | `python -m chattts --port 9966` |
| SadTalker | `7860` | `sadtalker` | `python inference.py --port 7860` |

---

## 五、成本估算

| 项目 | 费用 | 说明 |
|---|---|---|
| 硬件 | ¥0（已有） | M5 Max 128GB 是苹果端跑大模型的天花板 |
| 模型 | ¥0 | 全部开源模型，无 API 费用 |
| ComfyUI + 插件 | ¥0 | 开源 |
| LoRA 训练数据 | ¥0-500 | 如需购买风格参考图/画师授权 |
| 存储 | 关注 2TB 用量 | Wan 2.2 模型 ~65GB（含 LoRA）；素材库增长快，建议外挂 NAS |
| 开发时间 | ~4-8 周 | 搭建全流程 + 调试 Skill + 质量优化 |
| 电费 | 较高 | M5 Max 满载 ~80-120W，长时间生成注意散热 |

---

## 六、风险 & 注意事项

| 风险 | 说明 | 缓解方案 |
|---|---|---|
| **MPS 兼容性** | FP8 模型不支持 MPS，必须用 FP16/BF16 | 已使用 FP16 版本；关注 Comfy-Org MPS 兼容更新 |
| **生成速度** | 图生视频是最慢环节 | LoRA 4步加速（~2-5 min/镜头）/ 5B 轻量模型 / 分批夜间生成 |
| **角色一致性** | IP-Adapter 并非 100% 一致 | 多次生成+筛选 / 训练角色 LoRA / 人工校验环节 |
| **版权风险** | AI 生成内容版权归属各国法律有争议 | 用完全原创世界观 + 保留全部创作过程记录 |
| **内存压力** | 多模型同时加载可能 OOM | Pipeline 串行执行，oMLX 用完直接杀进程释放 ~25GB；ComfyUI 用 /free 接口释放模型 |

---

## 七、开发路线图

| 阶段 | 周次 | 任务 | 产出 |
|---|---|---|---|
| **环境搭建** | Week 1 | 安装 oMLX + ComfyUI + SDXL + IP-Adapter + Wan 2.2 | 手动跑通单张图生成 |
| **Skill 1** | Week 2 | 编写 ScriptWriter Skill | LLM → 结构化剧本 JSON |
| **Skill 2** | Week 3 | 编写 AssetGenerator Skill | ComfyUI API 批量出图 |
| **Skill 3** | Week 4-5 | 部署 Wan 2.2 两阶段采样 + LoRA + 编写 VideoGenerator Skill | 图生视频 + 配音 |
| **Skill 4** | Week 6 | 编写 Editor Skill | FFmpeg 自动剪辑 |
| **集成调试** | Week 7-8 | 集成 + 端到端调试 + 质量优化 | 完整 Pipeline 可用 |

---

## 八、本地开发环境配置

```bash
# 1. 安装 miniforge3（如果还没装）
# https://github.com/conda-forge/miniforge3

# 2. 创建 conda 环境
conda create -n ai-comic python=3.12 pip -y
conda create -n comfyui python=3.13 pip -y
conda create -n sadtalker python=3.10 pip -y

# 3. 安装 oMLX
conda activate ai-comic
pip install omlx -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
omlx pull Qwen3.6-35B-A3B-MLX-8bit

# 4. 安装 ComfyUI
conda activate comfyui
cd ~/backend
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git

# 下载 Wan 2.2 模型（FP16 版，MPS 兼容，约 65GB）
# 从 https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged 下载：
# Diffusion Models（models/diffusion_models/）：
#   wan2.2_i2v_high_noise_14B_fp16.safetensors (27GB)
#   wan2.2_i2v_low_noise_14B_fp16.safetensors (27GB)
#   wan2.2_ti2v_5B_fp16.safetensors (9.3GB, 备选轻量模型)
# LoRA（models/loras/）：
#   wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors (1.2GB)
#   wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors (1.2GB)
# Text Encoder + VAE：
#   umt5_xxl_fp8_e4m3fn_scaled.safetensors → models/text_encoders/
#   wan_2.1_vae.safetensors → models/vae/（注意：2.2 I2V 模型用的是 2.1 VAE）

# 5. 安装 SadTalker 依赖
conda activate sadtalker
cd ~/backend/SadTalker
pip install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 6. 安装 ChatTTS
conda activate ai-comic
pip install ChatTTS -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 7. 安装 FFmpeg
brew install ffmpeg

# 8. 安装项目 Python 依赖
conda activate ai-comic
cd ~/backend/ai-comic-drama
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

---

## 九、项目目录结构

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

*最后更新：2026-05-12 — 已更新至 Wan 2.2 两阶段采样 + LoRA 加速，MPS FP16 兼容*
