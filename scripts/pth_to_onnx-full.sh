export CUDA_VISIBLE_DEVICES=0
python pth_to_onnx.py --dataset SMD --input_c 38 --output_c 38 --win_size 100 --device cuda --full_output
python pth_to_onnx.py --dataset MSL --input_c 55 --output_c 55 --win_size 100 --device cuda --full_output
python pth_to_onnx.py --dataset SMAP --input_c 25 --output_c 25 --win_size 100 --device cuda --full_output
python pth_to_onnx.py --dataset PSM --input_c 25 --output_c 25 --win_size 100 --device cuda --full_output
