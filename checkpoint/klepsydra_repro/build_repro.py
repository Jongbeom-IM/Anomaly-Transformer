"""Minimal repro for the Klepsydra ONNX importer issue: a plain MatMul between two
runtime (non-initializer) tensors is rejected, even with zero Transpose ops involved.
Run: python3 build_repro.py
Then: kpsr_ai_onnx_converter -m repro_matmul_dynamic.onnx -o /tmp/out.kpsr
"""
import onnx
from onnx import helper, TensorProto, shape_inference

L, S, Dv = 100, 100, 64
a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [L, S])   # e.g. attention weights
v = helper.make_tensor_value_info('v', TensorProto.FLOAT, [S, Dv])  # e.g. values
out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [L, Dv])

nodes = [
    helper.make_node('Softmax', ['a'], ['series'], axis=1),
    helper.make_node('MatMul', ['series', 'v'], ['out']),  # both inputs are runtime tensors
]
graph = helper.make_graph(nodes, 'repro_matmul_dynamic', [a, v], [out])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
model.ir_version = 7
onnx.checker.check_model(model)
model = shape_inference.infer_shapes(model)
onnx.save(model, 'repro_matmul_dynamic.onnx')
print('saved repro_matmul_dynamic.onnx')
