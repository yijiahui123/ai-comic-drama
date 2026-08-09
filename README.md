# 🎬 AI Comic Drama — 全自动漫剧生成流水线

> 通过聊天描述需求，系统自动完成：剧本生成 → 资产创建 → 图生视频 → 剪辑包装，最终输出成品视频。

**硬件基准**：MacBook Pro M5 Max · 128GB 统一内存 · 2TB SSD

---

## 当前推荐运行模式

项目现在有两条入口：

- **Web 生产台**：`web_server.py`，默认不启动真实模型服务，适合做项目管理、资源库管理、镜头编辑、质检、审核、重试标记、导出包。
- **CLI 真实流水线**：`main.py`，按原始流程调用 LLM、ComfyUI、Wan2.2、TTS、剪辑服务，适合真实生成。

Web 生产台默认是安全模式：

```bash
conda run -n ai-comic uvicorn web_server:app --host 127.0.0.1 --port 8080
```

默认情况下，Web 端创建项目、恢复、重跑、失败重试都会进入本地队列，但不会真实调用 ComfyUI/Wan/LLM/TTS。这样可以在不占显存的情况下完成控制台、状态、审核和导出闭环。

如果确认本地模型服务都已经启动，并且希望 Web 端真实执行生成任务，再显式打开：

```bash
AI_COMIC_ENABLE_MODEL_TASKS=1 conda run -n ai-comic uvicorn web_server:app --host 127.0.0.1 --port 8080
```

生产建议：

- 没调通模型服务前，先用 Web 安全模式检查资源、脚本、镜头状态、质检和导出。
- 真正出片时，再启动对应模型服务，并开启 `AI_COMIC_ENABLE_MODEL_TASKS=1`。
- CLI `python main.py` 仍保留为直接跑完整流水线的入口。

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
   (oMLX)        (ComfyUI)     (Wan 2.2+TTS) (FFmpeg)
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
│   │   ├── skill.py                # ScriptWriter（OpenAI-compatible API）
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
│   │   ├── skill.py                # VideoGenerator（Wan2.2+ChatTTS+SadTalker）
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

## 依赖总览

本项目需要 4 个独立服务 + 1 个 CLI 工具，各自运行在独立的 conda 环境中：

| 服务 | 用途 | conda 环境 | Python 版本 | 端口 |
|---|---|---|---|---|
| **oMLX** | LLM 剧本生成 | `ai-comic` | 3.12 | 8000 |
| **ComfyUI** | 图片/视频生成（含 Wan 2.2） | `comfyui` | 3.13 | 8188 |
| **ChatTTS** | AI 配音 | `ai-comic` | 3.12 | 9966 |
| **SadTalker** | 口型同步 | `sadtalker` | 3.10 | 7860 |
| **FFmpeg** | 视频剪辑 | 系统级 | — | — |

> **注意**：ChatTTS 和 SadTalker 是可选服务——若不可用，流水线会自动跳过配音和口型同步步骤。

---

## 完整部署指南

### 前置条件

- macOS（Apple Silicon M 系列芯片）
- [miniforge3](https://github.com/conda-forge/miniforge3)（conda 包管理器）
- FFmpeg（`brew install ffmpeg`）
- Git

---

### 第一步：克隆项目

```bash
cd ~/backend
git clone https://github.com/YJH-Lab/ai-comic-drama.git
cd ai-comic-drama
```

---

### 第二步：创建 conda 环境

本项目需要 3 个独立的 conda 环境（避免依赖冲突）：

```bash
# 1. ai-comic 环境（oMLX 剧本生成 + ChatTTS 配音）
conda create -n ai-comic python=3.12 pip -y
conda activate ai-comic
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 2. comfyui 环境（ComfyUI 图片/视频生成）
conda create -n comfyui python=3.13 pip -y

# 3. sadtalker 环境（SadTalker 口型同步）
conda create -n sadtalker python=3.10 pip -y
```

---

### 第三步：部署 oMLX（LLM 剧本生成）

oMLX 是本地 LLM 推理引擎，提供 OpenAI 兼容 API。

```bash
conda activate ai-comic

# 安装 oMLX（参照官方文档）
pip install omlx -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 下载模型（Qwen3.6-35B-A3B-MLX-8bit，约 20GB）
omlx pull Qwen3.6-35B-A3B-MLX-8bit

# 启动服务
omlx serve --host 127.0.0.1 --port 8000
```

**验证**：
```bash
curl http://127.0.0.1:8000/v1/models
# 应返回包含 Qwen3.6-35B-A3B-MLX-8bit 的模型列表
```

**内存占用**：约 20-25GB 统一内存

---

### 第四步：部署 ComfyUI（图片/视频生成）

```bash
conda activate comfyui

# 安装 PyTorch（MPS 后端）
pip install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 克隆 ComfyUI
cd ~/backend
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 安装依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 安装 ComfyUI Manager（节点管理器）
cd custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# 安装必要自定义节点
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git

# 启动 ComfyUI
cd ~/backend/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --enable-manager
```

**验证**：浏览器打开 `http://127.0.0.1:8188`

**内存占用**：约 8-16GB（取决于加载的模型）

#### 下载 ComfyUI 模型

```bash
cd ~/backend/ComfyUI/models

# Flux.2 Klein 4B 图片生成（约 12GB）
export HF_ENDPOINT=https://hf-mirror.com  # 国内镜像
cd diffusion_models && curl -L -O "$HF_ENDPOINT/black-forest-labs/FLUX.2-klein-base-4b-fp8/resolve/main/flux-2-klein-base-4b-fp8.safetensors"
cd ../text_encoders && curl -L -O "$HF_ENDPOINT/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"
cd ../vae && curl -L -O "$HF_ENDPOINT/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors"

# Wan 2.2 视频生成（FP16 版，约 65GB）
HF_WAN="$HF_ENDPOINT/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
cd ../diffusion_models
curl -L -O "$HF_WAN/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
curl -L -O "$HF_WAN/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
curl -L -O "$HF_WAN/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"
cd ../loras && mkdir -p ../loras
curl -L -O "$HF_WAN/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
curl -L -O "$HF_WAN/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
cd ../text_encoders && curl -L -O "$HF_ENDPOINT/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
cd ../vae && curl -L -O "$HF_WAN/vae/wan_2.1_vae.safetensors"
```

> **注意**：ComfyUI 模型需在 ComfyUI Web UI 中手动下载或通过 ComfyUI Manager 安装。

---

### 第五步：部署 ChatTTS（AI 配音）

```bash
conda activate ai-comic
cd ~/backend

# ChatTTS 已在 ai-comic 环境中安装（步骤二）
# 如果还没装：
pip install ChatTTS -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 启动 ChatTTS 服务（端口 9966）
# 具体启动方式取决于 ChatTTS 版本，通常：
python -m chattts --port 9966
# 或使用 WebUI：
# python app.py --port 9966
```

**验证**：
```bash
curl http://localhost:9966
# 应返回 ChatTTS 服务响应
```

**内存占用**：约 4-6GB

---

### 第六步：部署 SadTalker（口型同步）

```bash
conda activate sadtalker

# 安装 PyTorch（MPS 后端）
pip install torch torchvision torchaudio -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 安装 SadTalker 依赖
cd ~/backend/SadTalker
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 下载模型（见下方模型下载章节）
# 模型放在 ~/backend/SadTalker/checkpoints/ 目录

# 启动 SadTalker（端口 7860）
python inference.py --port 7860
```

**验证**：浏览器打开 `http://127.0.0.1:7860`

**内存占用**：约 6-8GB

---

### 第七步：安装 FFmpeg

```bash
# macOS
brew install ffmpeg

# 验证
ffmpeg -version
```

---

### 第八步：配置服务地址

编辑 `configs/services.yaml`，填入实际服务地址：

```yaml
# Service endpoints for the AI Comic Drama pipeline
llm:
  url: "http://127.0.0.1:8000/v1"
  model: "Qwen3.6-35B-A3B-MLX-8bit"
  timeout: 300

comfyui:
  url: "http://localhost:8188"
  timeout: 600

chattts:
  url: "http://localhost:9966"
  timeout: 60
  enabled: true

sadtalker:
  url: "http://localhost:7860"
  timeout: 300
  enabled: true

memory:
  unload_between_stages: true
  gc_delay: 2

output:
  root: "output"
  state_dir: "output/state"
  videos_dir: "output/videos"
  audio_dir: "output/audio"
  lipsync_dir: "output/lipsync"
  final_dir: "output/final"
```

---

### 第九步：运行流水线

```bash
conda activate ai-comic
cd ~/backend/ai-comic-drama

# 一键启动全流程
python main.py --prompt "写一个赛博朋克风格的3分钟漫剧，主角是黑客少女"

# 断点续跑（使用上次运行的 project_id）
python main.py --resume <project_id>

# 查看进度
python main.py --status <project_id>
```

---

## 模型下载清单

### SadTalker 模型（必需）

SadTalker 需要以下模型文件，放在 `~/backend/SadTalker/checkpoints/` 目录：

| 文件名 | 大小 | 用途 | 下载地址 |
|---|---|---|---|
| `mapping_00109-model.pth.tar` | 156MB | MappingNet（全身模式） | [HuggingFace](https://hf-mirror.com/vinthony/SadTalker/resolve/main/mapping_00109-model.pth.tar) |
| `mapping_00229-model.pth.tar` | 156MB | MappingNet（裁剪模式） | [HuggingFace](https://hf-mirror.com/vinthony/SadTalker/resolve/main/mapping_00229-model.pth.tar) |
| `auido2exp_00300-model.pth` | 34MB | 音频→表情 | [HuggingFace](https://hf-mirror.com/vinthony/SadTalker/resolve/main/auido2exp_00300-model.pth) |
| `auido2pose_00140-model.pth` | 96MB | 音频→头部姿态 | [HuggingFace](https://hf-mirror.com/vinthony/SadTalker/resolve/main/auido2pose_00140-model.pth) |
| `shape_predictor_68_face_landmarks.dat` | 100MB | 人脸特征点检测 | [HuggingFace](https://hf-mirror.com/vinthony/SadTalker/resolve/main/shape_predictor_68_face_landmarks.dat) |
| `epoch_20.pth` | 289MB | 3D 人脸重建 | [HuggingFace](https://hf-mirror.com/vinthony/SadTalker/resolve/main/epoch_20.pth) |
| `wav2lip.pth` | 436MB | 唇形同步 | [HuggingFace](https://hf-mirror.com/vinthony/SadTalker/resolve/main/wav2lip.pth) |
| `facevid2vid_00189-model.pth.tar` | 2.1GB | 面部渲染 | [HuggingFace](https://hf-mirror.com/vinthony/SadTalker/resolve/main/facevid2vid_00189-model.pth.tar) |

**BFM_Fitting 目录**（3D 人脸模型库，放在 `checkpoints/BFM_Fitting/`）：

| 文件名 | 用途 |
|---|---|
| `01_MorphableModel.mat` | 3D 可变形模型 |
| `BFM09_model_info.mat` | 模型信息 |
| `BFM_exp_idx.mat` | 表情索引 |
| `BFM_front_idx.mat` | 正面索引 |
| `Exp_Pca.bin` | 表情 PCA |
| `facemodel_info.mat` | 人脸模型信息 |
| `select_vertex_id.mat` | 顶点选择 |
| `similarity_Lm3D_all.mat` | 3D 特征点相似度 |
| `std_exp.txt` | 标准表情 |

**hub/checkpoints 目录**（人脸检测模型，放在 `checkpoints/hub/checkpoints/`）：

| 文件名 | 用途 |
|---|---|
| `s3fd-619a316812.pth` | 人脸检测 |
| `2DFAN4-cd938726ad.zip` | 人脸对齐 |

#### 一键下载脚本

```bash
cd ~/backend/SadTalker
bash scripts/download_models.sh
```

或手动下载（使用 HuggingFace 镜像）：

```bash
cd ~/backend/SadTalker
mkdir -p checkpoints/BFM_Fitting checkpoints/hub/checkpoints

# 下载核心模型
curl -L -o checkpoints/mapping_00109-model.pth.tar "https://hf-mirror.com/vinthony/SadTalker/resolve/main/mapping_00109-model.pth.tar"
curl -L -o checkpoints/mapping_00229-model.pth.tar "https://hf-mirror.com/vinthony/SadTalker/resolve/main/mapping_00229-model.pth.tar"
curl -L -o checkpoints/auido2exp_00300-model.pth "https://hf-mirror.com/vinthony/SadTalker/resolve/main/auido2exp_00300-model.pth"
curl -L -o checkpoints/auido2pose_00140-model.pth "https://hf-mirror.com/vinthony/SadTalker/resolve/main/auido2pose_00140-model.pth"
curl -L -o checkpoints/shape_predictor_68_face_landmarks.dat "https://hf-mirror.com/vinthony/SadTalker/resolve/main/shape_predictor_68_face_landmarks.dat"
curl -L -o checkpoints/epoch_20.pth "https://hf-mirror.com/vinthony/SadTalker/resolve/main/epoch_20.pth"
curl -L -o checkpoints/wav2lip.pth "https://hf-mirror.com/vinthony/SadTalker/resolve/main/wav2lip.pth"
curl -L -o checkpoints/facevid2vid_00189-model.pth.tar "https://hf-mirror.com/vinthony/SadTalker/resolve/main/facevid2vid_00189-model.pth.tar"

# 下载 BFM_Fitting
for f in 01_MorphableModel.mat BFM09_model_info.mat BFM_exp_idx.mat BFM_front_idx.mat Exp_Pca.bin facemodel_info.mat select_vertex_id.mat similarity_Lm3D_all.mat std_exp.txt; do
  curl -L -o "checkpoints/BFM_Fitting/$f" "https://hf-mirror.com/vinthony/SadTalker/resolve/main/BFM_Fitting/$f"
done

# 下载 hub 模型
curl -L -o checkpoints/hub/checkpoints/s3fd-619a316812.pth "https://hf-mirror.com/vinthony/SadTalker/resolve/main/hub/checkpoints/s3fd-619a316812.pth"
curl -L -o checkpoints/hub/checkpoints/2DFAN4-cd938726ad.zip "https://hf-mirror.com/vinthony/SadTalker/resolve/main/hub/checkpoints/2DFAN4-cd938726ad.zip"
```

#### SadTalker 可选模型（面部增强）

放在 `~/backend/SadTalker/gfpgan/weights/` 目录：

| 文件名 | 大小 | 用途 | 下载地址 |
|---|---|---|---|
| `alignment_WFLW_4HG.pth` | 184MB | 人脸对齐 | [GitHub](https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth) |
| `detection_Resnet50_Final.pth` | ~100MB | 人脸检测 | [GitHub](https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth) |
| `GFPGANv1.4.pth` | ~330MB | 面部增强 | [GitHub](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth) |
| `parsing_parsenet.pth` | ~85MB | 面部分割 | [GitHub](https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth) |

> **注意**：GFPGAN 权重是可选的，用于 `--enhancer gfpgan` 参数启用面部增强。没有这些文件 SadTalker 也能正常运行。

---

### ComfyUI 模型

ComfyUI 需要以下模型（通过 ComfyUI Web UI 或手动下载）：

#### Flux.2 Klein 4B 图片生成模型

从 [black-forest-labs/FLUX.2-klein-base-4b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4b-fp8) 和 [Comfy-Org](https://huggingface.co/Comfy-Org) 下载：

| 文件 | 大小 | 用途 | 目录 |
|---|---|---|---|
| `flux-2-klein-base-4b-fp8.safetensors` | ~8GB | 图像扩散模型 | `diffusion_models/` |
| `qwen_3_4b.safetensors` | ~4GB | 文本编码器 | `text_encoders/` |
| `flux2-vae.safetensors` | ~200MB | VAE 解码器 | `vae/` |

#### Wan 2.2 视频生成模型（FP16 版，MPS 兼容）

从 [Comfy-Org/Wan_2.2_ComfyUI_Repackaged](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged) 下载。

> **注意**：Apple Silicon MPS 不支持 FP8（Float8_e4m3fn），必须使用 FP16/BF16 版本。

**Diffusion Models**（放到 `models/diffusion_models/`）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `wan2.2_i2v_high_noise_14B_fp16.safetensors` | 27GB | 图生视频 — 高噪声阶段 |
| `wan2.2_i2v_low_noise_14B_fp16.safetensors` | 27GB | 图生视频 — 低噪声阶段 |
| `wan2.2_ti2v_5B_fp16.safetensors` | 9.3GB | 轻量模型（文本+图片→视频，备选） |

**LoRA**（放到 `models/loras/`）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | 1.2GB | 高噪声 LoRA（4步加速） |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | 1.2GB | 低噪声 LoRA（4步加速） |

**Text Encoder + VAE**（放到 `models/text_encoders/` 和 `models/vae/`）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | 6.3GB | 文本编码器 |
| `wan_2.1_vae.safetensors` | 242MB | VAE 解码器（注意：虽然叫 2.1，但 2.2 I2V 模型需要用这个） |

**下载命令**：

```bash
HF="https://hf-mirror.com/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"
cd ~/backend/ComfyUI/models

# Diffusion models
curl -L -o diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors "$HF/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
curl -L -o diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors "$HF/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
curl -L -o diffusion_models/wan2.2_ti2v_5B_fp16.safetensors "$HF/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors"

# LoRA
curl -L -o loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors "$HF/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
curl -L -o loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors "$HF/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

# VAE + Text Encoder
curl -L -o vae/wan_2.1_vae.safetensors "$HF/vae/wan_2.1_vae.safetensors"
curl -L -o text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors "https://hf-mirror.com/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
```

> **轻量方案**：如果内存紧张，可只用 `wan2.2_ti2v_5B_fp16.safetensors`（9.3GB），5B 模型同时支持文生视频和图生视频。

> **注意**：ComfyUI 模型需在 ComfyUI Web UI 中通过 ComfyUI Manager 安装，或手动下载后放到 `~/backend/ComfyUI/models/` 对应目录。

---

## 输出说明

```
output/
├── state/<project_id>.json        # 流水线状态（可断点续跑）
├── videos/<shot_id>.mp4           # 各镜头原始视频
├── audio/<shot_id>.wav            # 各镜头配音
├── lipsync/<shot_id>_lipsync.mp4  # 口型同步视频
└── final/<project_id>_ep01.mp4    # 最终成品
```

---

## 内存占用估算

| 组件 | 内存占用 | 备注 |
|---|---|---|
| oMLX (Qwen3.6-35B) | ~20-25GB | 剧本生成阶段 |
| ComfyUI (Flux.2 4B) | ~12-16GB | 图片生成阶段 |
| ComfyUI (Wan 2.2 14B FP16 两阶段+LoRA) | ~30-40GB | 视频生成阶段（两阶段采样，4步×2） |
| ChatTTS | ~4-6GB | 配音阶段 |
| SadTalker | ~6-8GB | 口型同步阶段 |
| **总计（峰值）** | ~80-100GB | 流水线串行执行，不会同时加载所有模型 |

> **内存管理**：流水线在每个阶段完成后会自动卸载上一阶段的模型，避免 OOM。可在 `configs/services.yaml` 中设置 `memory.unload_between_stages: false` 关闭此行为（调试时有用）。

---

## 故障排查

### oMLX 无法启动
- 检查端口 8000 是否被占用：`lsof -i :8000`
- 检查模型是否下载完成：`omlx list`

### ComfyUI 启动失败
- 检查 PyTorch MPS 支持：`python -c "import torch; print(torch.backends.mps.is_available())"`
- 检查端口 8188 是否被占用：`lsof -i :8188`

### ChatTTS 无响应
- 检查端口 9966 是否被占用：`lsof -i :9966`
- ChatTTS 是可选服务，可在 `configs/services.yaml` 中设置 `chattts.enabled: false` 跳过

### SadTalker 模型加载失败
- 确认所有模型文件已下载完整（检查文件大小）
- 确认 `checkpoints/BFM_Fitting/` 目录包含所有 .mat 文件
- SadTalker 是可选服务，可在 `configs/services.yaml` 中设置 `sadtalker.enabled: false` 跳过

### 流水线中途失败
- 使用 `python main.py --resume <project_id>` 从上次断点继续
- 检查 `output/state/<project_id>.json` 查看失败阶段

---

## 技术栈

- **Python 3.12+** + `asyncio` + `aiohttp`（异步 HTTP）
- **Pydantic v2**（数据模型与验证）
- **PyYAML**（配置读取）
- **MoviePy** + **FFmpeg**（视频剪辑）
- **oMLX**（本地 LLM 推理，OpenAI 兼容 API）
- **ComfyUI**（图像/视频生成，Wan 2.2）
- **ChatTTS**（AI 配音）
- **SadTalker**（口型同步）

---

详细技术路线请参阅 [`docs/technical-roadmap.md`](docs/technical-roadmap.md)。
