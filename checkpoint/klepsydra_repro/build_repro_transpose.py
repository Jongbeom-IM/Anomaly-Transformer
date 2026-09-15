"""Minimal repro for the second Klepsydra ONNX importer issue: a 3D Transpose used to
convert a channel-last [B,L,C] tensor to channel-first for a Conv1d (and back) is rejected
as an "unsupported permutation", even though it is the standard NCHW<->NHWC boundary pattern
Klepsydra's own docs say it recognizes and skips for 4D image tensors.
Run: python3 build_repro_transpose.py
Then: kpsr_ai_onnx_converter -m repro_transpose_conv1d.onnx -o /tmp/out2.kpsr
"""
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper, shape_inference

B, L, C_in, D = 1, 100, 38, 512
inp = helper.make_tensor_value_info('input', TensorProto.FLOAT, [B, L, C_in])
out = helper.make_tensor_value_info('output', TensorProto.FLOAT, [B, L, D])

w = np.random.randn(D, C_in, 3).astype(np.float32) * 0.02
w_init = numpy_helper.from_array(w, name='conv_w')

nodes = [
    helper.make_node('Transpose', ['input'], ['x_nchw'], perm=[0, 2, 1]),   # NHWC -> NCHW for Conv1d
    helper.make_node('Conv', ['x_nchw', 'conv_w'], ['y_nchw'],
                      kernel_shape=[3], pads=[1, 1], strides=[1], dilations=[1], group=1),
    helper.make_node('Transpose', ['y_nchw'], ['output'], perm=[0, 2, 1]),  # NCHW -> NHWC
]
graph = helper.make_graph(nodes, 'repro_transpose_conv1d', [inp], [out], initializer=[w_init])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
model.ir_version = 7
onnx.checker.check_model(model)
model = shape_inference.infer_shapes(model)
onnx.save(model, 'repro_transpose_conv1d.onnx')
print('saved repro_transpose_conv1d.onnx')
