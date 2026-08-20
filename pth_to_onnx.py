import argparse
import os

import torch

from model.AnomalyTransformer import AnomalyTransformer


class ReconstructionOnly(torch.nn.Module):
    # AttentionLayer.forward always unpacks 4 values from the inner
    # attention, so AnomalyTransformer only actually works with
    # output_attention=True; this wrapper exposes just the reconstruction.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def export(config):
    device = torch.device(config.device)

    model = AnomalyTransformer(
        win_size=config.win_size,
        enc_in=config.input_c,
        c_out=config.output_c,
        e_layers=3,
        output_attention=True,
    )
    ckpt_path = os.path.join(config.model_save_path, config.dataset + '_checkpoint.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    export_model = model if config.full_output else ReconstructionOnly(model)

    dummy_input = torch.randn(config.batch_size, config.win_size, config.input_c, device=device)

    dynamic_axes = {'input': {0: 'batch'}}
    if config.full_output:
        e_layers = len(model.encoder.attn_layers)
        output_names = ['reconstruction']
        output_names += [f'series_{i}' for i in range(e_layers)]
        output_names += [f'prior_{i}' for i in range(e_layers)]
        output_names += [f'sigma_{i}' for i in range(e_layers)]
    else:
        output_names = ['reconstruction']
    for name in output_names:
        dynamic_axes[name] = {0: 'batch'}

    torch.onnx.export(
        export_model,
        dummy_input,
        config.onnx_path,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=config.opset,
        dynamo=False,
    )
    print(f'Exported to {config.onnx_path} (outputs: {output_names})')

    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed; skipping numerical verification '
              '(pip install onnxruntime-gpu or onnxruntime).')
        return

    with torch.no_grad():
        out = export_model(dummy_input)
    flat_torch = [out[0]] + [t for group in out[1:] for t in group] if config.full_output else [out]

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
    sess = ort.InferenceSession(config.onnx_path, providers=providers)
    ort_out = sess.run(None, {'input': dummy_input.cpu().numpy()})

    for name, expected, actual in zip(output_names, flat_torch, ort_out):
        diff = expected.cpu().numpy() - actual
        print(f'{name}: max abs diff = {abs(diff).max():.3e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--onnx_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=1,
                         help='batch size used for the dummy export input; batch axis is dynamic in the ONNX graph')
    parser.add_argument('--opset', type=int, default=14)
    parser.add_argument('--full_output', action='store_true',
                         help='export series/prior/sigma attention outputs in addition to the reconstruction '
                              '(needed to reproduce the association-discrepancy anomaly score outside PyTorch)')
    config = parser.parse_args()

    if config.onnx_path is None:
        suffix = '_full.onnx' if config.full_output else '.onnx'
        config.onnx_path = os.path.join(config.model_save_path, config.dataset + suffix)

    export(config)
