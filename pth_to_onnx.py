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

    if config.full_output:
        e_layers = len(model.encoder.attn_layers)
        output_names = ['reconstruction']
        output_names += [f'series_{i}' for i in range(e_layers)]
        output_names += [f'prior_{i}' for i in range(e_layers)]
        output_names += [f'sigma_{i}' for i in range(e_layers)]
    else:
        output_names = ['reconstruction']

    if config.static_batch:
        # No dynamic_axes: the batch size is baked into the graph as a constant. Most NPU/
        # embedded backends require fixed input shapes anyway, and it also avoids the
        # Shape/Gather/Unsqueeze/ConstantOfShape nodes PyTorch emits to rebuild reshape target
        # shapes at runtime when a dimension is dynamic (those ops aren't in most such
        # backends' supported-op whitelists either).
        dynamic_axes = None
    else:
        dynamic_axes = {'input': {0: 'batch'}}
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

    import onnx

    onnx_model = onnx.load(config.onnx_path)
    try:
        from onnxsim import simplify
        onnx_model, ok = simplify(onnx_model)
        if ok:
            onnx.save(onnx_model, config.onnx_path)
            print('Simplified with onnxsim (removes Identity/Dropout no-ops left over from tracing).')
        else:
            print('onnxsim simplification could not be verified; keeping the unsimplified graph.')
            onnx_model = onnx.load(config.onnx_path)
    except ImportError:
        print('onnxsim not installed; exported graph may still contain harmless Identity nodes '
              '(pip install onnxsim to clean those up).')

    op_types = sorted(set(n.op_type for n in onnx_model.graph.node))
    print(f'Ops used in exported graph: {op_types}')
    if config.op_whitelist:
        allowed = set(config.op_whitelist.split(','))
        disallowed = [op for op in op_types if op not in allowed]
        if disallowed:
            print(f'WARNING: ops not in --op_whitelist: {disallowed}')
        else:
            print('All ops are within --op_whitelist.')

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
    parser.add_argument('--static_batch', action='store_true',
                         help='bake --batch_size into the graph instead of leaving the batch axis dynamic; '
                              'avoids Shape/Gather/Unsqueeze/ConstantOfShape nodes that most NPU/embedded '
                              'backends with a restricted op whitelist do not support either')
    parser.add_argument('--op_whitelist', type=str, default=None,
                         help='comma-separated list of allowed ONNX op types; if given, the exported graph is '
                              'checked against it and any other op is reported as a WARNING')
    config = parser.parse_args()

    if config.onnx_path is None:
        suffix = '_full.onnx' if config.full_output else '.onnx'
        config.onnx_path = os.path.join(config.model_save_path, config.dataset + suffix)

    export(config)
