import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

def demo0():
    model_dir = "/sharedir/nlp/workspace/yuqiangz_living_data/models/Qwen3-TTS-12Hz-0.6B-Base"
    # Load the model
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cuda:7",
        dtype=torch.bfloat16,
    )
    print("Model loaded successfully.")

    # Reference audio for cloning
    ref_audio = "./zero_shot_prompt.wav"
    ref_text  = "希望你以后能够做的比我还好呦。"

    # Generate speech
    wavs, sr = model.generate_voice_clone(
        text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
        language="English",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    # Save the resulting audio
    sf.write("output_voice_clone.wav", wavs[0], sr)

def demo1_custom_voice_gen():
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:7",
    dtype=torch.bfloat16,
    )

    # single inference
    wavs, sr = model.generate_custom_voice(
        text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
        speaker="Vivian",
        instruct="用特别愤怒的语气说", # Omit if not needed.
    )
    sf.write("output_custom_voice.wav", wavs[0], sr)

    # batch inference
    wavs, sr = model.generate_custom_voice(
        text=[
            "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
            "She said she would be here by noon."
        ],
        language=["Chinese", "English"],
        speaker=["Vivian", "Ryan"],
        instruct=["", "Very happy."]
    )
    sf.write("output_custom_voice_1.wav", wavs[0], sr)
    sf.write("output_custom_voice_2.wav", wavs[1], sr)

demo1_custom_voice_gen()