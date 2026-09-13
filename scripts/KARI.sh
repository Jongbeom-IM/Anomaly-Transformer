export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 1.2 --num_epochs 3   --batch_size 128  --mode train --dataset KARI  --data_path dataset/KARI --input_c 29    --output_c 29
python main.py --anormly_ratio 1.2  --num_epochs 10      --batch_size 128     --mode test    --dataset KARI   --data_path dataset/KARI  --input_c 29    --output_c 29  --pretrained_model 20
