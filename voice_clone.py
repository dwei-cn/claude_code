import sys
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

OUTPUT_DIR = r"D:\vscode\outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 三个参数 =====
ref_file = input("参考音频路径（用 Windows 录音机录的 wav/m4a 都行）: ").strip().strip('"')
ref_text = input("参考音频的文本（你录的内容）: ").strip()
gen_text = input("想生成什么文本: ").strip()

if not all([ref_file, ref_text, gen_text]):
    print("三个参数缺一不可！")
    sys.exit(1)

if not os.path.exists(ref_file):
    print(f"文件不存在: {ref_file}")
    sys.exit(1)

# ===== 加载模型 =====
print("\n加载 F5-TTS 模型...")
from f5_tts.api import F5TTS

f5tts = F5TTS(model="F5TTS_v1_Base", device="cuda")

# ===== 克隆 + 生成 =====
print("正在生成...")
wav, sr, spec = f5tts.infer(
    ref_file=ref_file,
    ref_text=ref_text,
    gen_text=gen_text,
)

import soundfile as sf

output_file = os.path.join(OUTPUT_DIR, "output.wav")
sf.write(output_file, wav, sr)
print(f"\n完成: {output_file}")
