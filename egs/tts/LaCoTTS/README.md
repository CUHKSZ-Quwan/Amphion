# Download codec ckpts and pretrained tts model
https://huggingface.co/Hecheng0625/latent_codec_gpt_tts/tree/main/ckpts_latent_codec_gpt_tts

# modify config for your task
egs/tts/LaCoTTS/exp_config_base.json

# train
参考 egs/tts/LaCoTTS/run_train.sh
models/tts/gpt_tts/gpt_tts_trainer.py

# inference
参考 latent_gpt_tts.ipynb