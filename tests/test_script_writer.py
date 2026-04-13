"""单独测试 ScriptWriter Skill

Usage::

    # 确保 Ollama 正在运行
    ollama serve

    # 运行测试
    python tests/test_script_writer.py
"""
import asyncio
import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills.script_writer.skill import ScriptWriter


async def test():
    # 如果用小模型先测试，改成你实际 pull 的模型名
    writer = ScriptWriter(
        ollama_url="http://localhost:11434",
        model="qwen2.5:14b",  # 正式用: llama3.1:70b-instruct-q4_K_M
    )

    prompt = (
        "故事发生在被永夜笼罩的近未来都市，讲述了靠修复'记忆残片'为生的边缘少年，"
        "意外得到了一枚刻有自己名字、却属于三十年前英雄战死的残存芯片。"
        "随着记忆数据的逐层解码，少年发现这个世界的繁华盛世不过是巨型AI构建的虚假幻象，"
        "而他日夜躲避的'执法者'，竟是曾经拼死守护人类的战友。用1集讲完。"
    )

    print("🎬 开始生成剧本...\n")
    script = await writer.generate(prompt)

    # 保存到文件方便查看
    output_path = Path(__file__).resolve().parent.parent / "test_script_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(script, f, ensure_ascii=False, indent=2)

    # 打印摘要
    print(f"✅ 标题: {script.get('title')}")
    print(f"🎨 风格: {script.get('style')}")

    total_shots = 0
    total_duration = 0.0

    for ep in script.get("episodes", []):
        print(f"\n📺 Episode {ep.get('episode')}: {ep.get('title', '')}")
        for scene in ep.get("scenes", []):
            shots = scene.get("shots", [])
            print(f"  🎬 {scene.get('scene_id')} — {scene.get('location')} ({len(shots)} 个镜头)")
            for shot in shots:
                total_shots += 1
                total_duration += float(shot.get("duration", 4))
                print(f"    📷 {shot.get('shot_id')} [{shot.get('type')}] {shot.get('duration')}s")
                if shot.get("dialogue"):
                    print(f"       💬 {shot.get('dialogue')[:50]}...")
                print(f"       🖼️  {shot.get('visual_prompt', '')[:60]}...")

    print(f"\n{'='*50}")
    print(f"📊 总计: {total_shots} 个镜头, 预估时长 {total_duration:.0f}s ({total_duration/60:.1f}min)")
    print(f"📄 完整 JSON 已保存到: {output_path}")


if __name__ == "__main__":
    asyncio.run(test())