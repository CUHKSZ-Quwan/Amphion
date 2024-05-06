export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

SAVE_DIR=/blob/v-shenkai/checkpoints/tts/vocoder/bigvgan/v2
# SAVE_DIR=exp/bigvgan

mkdir -p ${SAVE_DIR}

python train.py \
    --config configs/bigvgan_librilight_16k.json \
    --input_training_file LibriLight/train_meta_file.txt \
    --input_validation_file LibriLight/valid_meta_file.txt \
    --input_test_file LibriLight/test_meta_file.txt \
    --checkpoint_path ${SAVE_DIR} 2>&1 | tee ${SAVE_DIR}/train.log


