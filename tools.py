import os
import json
from glob import glob

def demo1_prepare_data():
    speech_dir = "/sharedir/nlp/workspace/yuqiangz_living_data/models/LJSpeech-1.1-48kHz/LJSpeech-1.1-48kHz/wavs/MossFormer2_SR_48K"
    metadata_path = "/sharedir/nlp/workspace/yuqiangz_living_data/models/LJSpeech-1.1-48kHz/LJSpeech-1.1-48kHz/metadata.csv"
    filename2text = {}
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                filename2text[parts[0]] = parts[1].strip()

    wav_paths = glob(os.path.join(speech_dir, "*.wav"))
    wav_paths.sort()
    line_list = []
    for wav_path in wav_paths:
        filename = os.path.basename(wav_path).replace(".wav", "")
        assert filename in filename2text, f"{filename} not in metadata"
        text = filename2text[filename]
        line = {
            "audio": wav_path,
            "text": text,
            "ref_audio": wav_paths[0],
        }
        line_list.append(line)
    with open("/sharedir/nlp/workspace/yuqiangz_living_data/master_models/Qwen3-TTS/finetuning/train_raw.jsonl", "w", encoding="utf-8") as f:
        lines = [json.dumps(item, ensure_ascii=False) + "\n" for item in line_list]
        f.writelines(lines)
    
demo1_prepare_data()