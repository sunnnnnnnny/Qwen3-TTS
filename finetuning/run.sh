# python prepare_data.py \
#   --device cuda:7 \
#   --tokenizer_model_path /sharedir/nlp/workspace/yuqiangz_living_data/models/Qwen3-TTS-Tokenizer-12Hz \
#   --input_jsonl train_raw.jsonl \
#   --output_jsonl train_with_codes.jsonl

CUDA_VISIBLE_DEVICES='7' python sft_12hz.py \
  --init_model_path /sharedir/nlp/workspace/yuqiangz_living_data/models/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test