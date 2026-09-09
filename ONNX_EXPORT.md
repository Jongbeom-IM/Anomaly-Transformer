# PyTorch(.pth) → ONNX 변환

`pth_to_onnx.py`로 학습된 체크포인트(`checkpoints/{dataset}_checkpoint.pth`)를 ONNX로 변환합니다.

## 요구 패키지

```bash
pip install onnx onnxruntime onnxsim        # CPU
# 또는
pip install onnx onnxruntime-gpu onnxsim    # GPU 검증까지 하려면
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

## 제한된 ONNX op set 대응 (NPU 등 임베디드 백엔드용)

일부 NPU/임베디드 추론 엔진은 ONNX op을 제한된 whitelist로만 지원합니다 (예: `LayerNormalization`/`ReduceMean`/`Sqrt`/`Pow`/`Erf`/`Einsum`/`Reciprocal`/`Neg`/`Unsqueeze`/`Tile`/`Expand` 미지원, `BatchNormalization`/`InstanceNormalization`/`Softmax`/`Sigmoid`/`Tanh`/`MatMul`/`Gemm`/`Conv` 등만 지원). 기존 Anomaly-Transformer 구현은 `nn.LayerNorm`, erf 기반 `F.gelu`, `torch.einsum`, `torch.pow`/`torch.exp`, `.repeat()`/`.unsqueeze()`를 광범위하게 써서 이 whitelist를 만족하지 못했습니다. `model/onnx_ops.py`에 수학적으로 동치인 대체 구현을 추가하고 `model/attn.py`, `model/AnomalyTransformer.py`에 적용했습니다:

- **LayerNorm → `ExportableLayerNorm`**: `InstanceNormalization`(단일 채널)으로 평균/분산 정규화를 수행한 뒤 `weight`/`bias`를 수동으로 곱하고 더합니다. 파라미터 이름(`weight`, `bias`)이 `nn.LayerNorm`과 동일해 기존 체크포인트를 그대로 로드할 수 있습니다.
- **GELU(erf) → `gelu_tanh`**: 표준 tanh 근사(`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`)로 대체했습니다. `F.gelu`와 최대 절대오차 ~3e-4 수준의 근사 오차가 있습니다 (재학습 없이 임베디드 배포 시 통상 허용되는 수준).
- **`torch.einsum` → `permute` + `matmul`**: 어텐션의 두 einsum(`QK^T`, `attn·V`)을 전치+행렬곱으로 재작성해 `Einsum` 노드가 생기지 않게 했습니다.
- **`torch.pow(3, sigma)`, `torch.exp(...)` → `safe_exp`**: `exp(x) = sigmoid(x)/(1-sigmoid(x))` 항등식으로 `Exp`/`Pow` 없이 계산합니다. 이 모델에서 인자값이 항상 유계(bounded)임을 확인하고 적용했습니다.
- **`1.0/x` → `safe_reciprocal`**: PyTorch가 `1.0/x`를 `Reciprocal` 노드로 최적화하는 것을 피하기 위해, 텐서 분자로 나누도록 바꿨습니다(`Div` 노드로 export됨).
- **`.repeat()`/`.unsqueeze()` 제거**: `Tile`/`Expand`/`Unsqueeze` 대신, `.reshape()`로 브로드캐스트 축만 만들고 이후 `Mul`/`Div`의 자연스러운 브로드캐스팅에 맡겼습니다.

### 검증

- 실제 체크포인트(`SMD_checkpoint.pth`)로 수정 전/후 출력을 비교: `reconstruction`/`series`/`sigma`는 최대 절대오차 ~2e-4 이하(gelu 근사에서 기인), `prior`는 유효값(>1e-6) 기준 최대 상대오차 0.24%, 중앙값 4.6e-7 — sigma가 하한(~1e-5)에 가까운 대각선 근처에서 1/sigma 특이점이 부동소수점 오차를 증폭시키는 것으로, 정상적인 부동소수점 거동입니다.
- export 후 `pth_to_onnx.py`가 자동으로 `onnxsim`을 돌려 tracing이 남긴 `Identity` 노드를 제거합니다. `--op_whitelist`에 콤마로 구분된 op 목록을 주면 최종 그래프의 op이 그 목록 안에 있는지 자동으로 체크합니다.

```bash
python pth_to_onnx.py --dataset SMD --input_c 38 --output_c 38 --win_size 100 --full_output --static_batch \
  --op_whitelist "Abs,Add,AveragePool,BatchNormalization,Cast,Clip,Concat,Constant,Conv,ConvTranspose,DequantizeLinear,Div,Dropout,Elu,Flatten,Gemm,GlobalAveragePool,InstanceNormalization,LSTM,LeakyRelu,MatMul,MaxPool,Mul,PRelu,Pad,QGemm,QLinearAdd,QLinearAveragePool,QLinearConcat,QLinearConv,QLinearGlobalAveragePool,QLinearLeakyRelu,QLinearMatMul,QLinearMul,QLinearSigmoid,QuantizeLinear,Relu,Reshape,Resize,Shape,Sigmoid,Slice,Softmax,Squeeze,Sub,Sum,Tanh,Transpose"
```

**`--static_batch`는 필수입니다.** 이 플래그 없이 export하면(batch 축을 dynamic으로 두면) PyTorch가 배치 크기를 런타임에 읽어오려고 `Shape`+`Gather`+`Unsqueeze`+`ConstantOfShape`/`Expand` 노드를 추가로 생성하는데, 이 중 `Gather`/`Unsqueeze`는 whitelist에 없습니다. `--static_batch`를 주면 `--batch_size`(기본 1)가 그래프에 고정되어 이 노드들이 아예 생기지 않습니다. NPU 배포는 대개 배치=1로 고정 운용하므로 실사용에 문제 없습니다.

### 2026-09-07: 과거 export본이 whitelist를 어기고 있었음 (재-export 완료)

`checkpoints/onnx/*.onnx`에 있던 기존 파일들은 위 op-whitelist 대응 코드(`model/onnx_ops.py` 등, 2026-09-05 작성)가 반영되기 **이전**(2026-08-21)에 만들어진 것이었습니다. 실제로 열어보면 `ReduceMean`/`Sqrt`/`Pow`/`Erf`/`Einsum`/`Gather`/`Unsqueeze`/`Exp`/`Reciprocal`/`Neg`/`Tile`/`Expand`/`ConstantOfShape`가 그대로 남아 있어 whitelist를 위반했고, 이 때문에 Klepsydra 등 NPU 변환기가 그래프를 읽지 못했습니다. `--static_batch`를 추가해 8개 파일(4개 데이터셋 × reconstruction-only/full) 전부 재-export했고, 최종 그래프는 아래 op만 사용합니다(둘 다 whitelist 부분집합):

- reconstruction-only: `Add, Concat, Constant, Conv, InstanceNormalization, MatMul, Mul, Reshape, Slice, Softmax, Tanh, Transpose`
- full_output: 위 + `Div, Sigmoid, Sub`

`--op_whitelist`로 자동 체크한 결과 모두 `All ops are within --op_whitelist.`. PyTorch 대비 최대 절대오차는 `reconstruction`/`series`/`sigma` 전부 1e-2 이하(gelu tanh 근사 기인), `prior`는 대부분 데이터셋·레이어에서 1e-3~1e-4대이나 SMD의 `prior_0`처럼 sigma가 하한(~1e-5)에 가까운 지점에서 `1/sigma` 특이점이 부동소수점 오차를 키워 절대오차가 커지는 레이어가 있습니다 — 값 자체가 그 지점에서 매우 크기 때문이며(상대오차는 여전히 작음), 기존에 파악된 정상적인 부동소수점 거동입니다(§검증 참고).

### 남은 주의사항

- `AnomalyAttention`은 이 저장소에서 항상 `mask_flag=False`로 생성되므로(인코더 셀프어텐션, causal mask 미사용) `masked_fill_`(→ `Where`) 경로는 실제로 export되지 않습니다. causal mask를 쓰는 다른 용도로 재사용한다면 그 경로도 별도로 손봐야 합니다.
- Conv1d의 `circular` padding(`model/embed.py`)은 opset 14에서 이미 `Slice`+`Concat`으로 분해되어 export되므로 추가 수정이 필요 없었습니다.
