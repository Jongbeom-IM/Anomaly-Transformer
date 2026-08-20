# PyTorch(.pth) → ONNX 변환

`pth_to_onnx.py`로 학습된 체크포인트(`checkpoints/{dataset}_checkpoint.pth`)를 ONNX로 변환합니다.

## 요구 패키지

```bash
pip install onnx onnxruntime        # CPU
# 또는
pip install onnx onnxruntime-gpu    # GPU 검증까지 하려면
```

## 사용법

```bash
python pth_to_onnx.py --dataset <DATASET> --input_c <IN> --output_c <OUT> --win_size <WIN>
```

`--input_c`, `--output_c`, `--win_size`는 해당 데이터셋을 학습시킬 때 `scripts/*.sh`에 준 값과 반드시 일치해야 합니다 (모델 구조가 그 값으로 결정됨).

### 인자

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--dataset` | (필수) | 체크포인트 파일명 접두사 (`checkpoints/{dataset}_checkpoint.pth`) |
| `--input_c` | 38 | 입력 채널 수 |
| `--output_c` | 38 | 출력 채널 수 |
| `--win_size` | 100 | 윈도우 크기 |
| `--model_save_path` | `checkpoints` | 체크포인트가 있는 디렉터리 |
| `--onnx_path` | `{model_save_path}/{dataset}.onnx` (`--full_output` 시 `{dataset}_full.onnx`) | 저장 경로 |
| `--device` | `cpu` | `cpu` 또는 `cuda` |
| `--batch_size` | 1 | export용 더미 입력 배치 크기 (실제 추론 시 배치 축은 dynamic이라 자유롭게 바뀜) |
| `--opset` | 14 | ONNX opset 버전 |
| `--full_output` | off | reconstruction 외에 series/prior/sigma(레이어별)까지 export |

### 예시

```bash
# SMD, reconstruction만, CPU
python pth_to_onnx.py --dataset SMD --input_c 38 --output_c 38 --win_size 100

# MSL, GPU로 export
python pth_to_onnx.py --dataset MSL --input_c 55 --output_c 55 --win_size 100 --device cuda

# association discrepancy 계산에 필요한 series/prior/sigma까지 전부
python pth_to_onnx.py --dataset SMD --input_c 38 --output_c 38 --win_size 100 --full_output
```

실행하면 더미 입력으로 PyTorch 결과와 ONNX Runtime 결과를 비교해 각 출력의 최대 절대오차를 출력합니다 (`onnxruntime`이 설치되어 있을 때만).

## 출력 구성

- 기본: `reconstruction` 1개 (`[B, win_size, output_c]`)
- `--full_output`: `reconstruction`, `series_0..2`, `prior_0..2`, `sigma_0..2` (encoder layer 수 = 3 기준)

이 저장소의 anomaly score는 reconstruction error와 `series`/`prior` 사이의 association discrepancy(KL divergence)를 함께 쓰므로, 논문 방식 그대로 스코어(inference time)를 재현하려면 `--full_output`이 필요합니다. reconstruction만 있으면 되는 경량 배포라면 기본 옵션으로 충분합니다.

## 참고: 함께 고친 기존 버그

- `model/attn.py`의 `AnomalyAttention.distances`가 `.cuda()`로 하드코딩되어 있어 CPU에서 모델 자체를 생성할 수 없었습니다. `register_buffer(..., persistent=False)`로 바꿔서 CPU/GPU 어디서든 동작하도록 고쳤고, 기존 체크포인트와도 호환됩니다.
- `output_attention=False`로 모델을 만들면 `AttentionLayer.forward`가 4개 값을 언패킹하려다 크래시하는 기존 버그가 있어서, 모델은 항상 `output_attention=True`로 생성한 뒤 `ReconstructionOnly` wrapper로 감싸 reconstruction만 export하는 방식으로 우회했습니다.
