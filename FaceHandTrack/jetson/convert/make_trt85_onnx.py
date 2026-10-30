"""Rewrite the EdgeTAM ONNX graphs so TensorRT 8.5 (JetPack 5) can build them.

TensorRT 8.5 has no LayerNormalization layer (added in 8.6), and it cannot use a
OneHot result to compute a tensor shape, which the mask decoder's variable
number of prompt points needs. This script:
  1. fixes the mask decoder's point count: n=1 (tracking, no prompt) and n=2 (box),
  2. folds every shape computation into constants (onnxsim),
  3. splits each LayerNormalization into ReduceMean / Sub / Mul / Add / Sqrt / Div,
  4. checks each new graph against the original with ONNX Runtime on test inputs.

    python3 make_trt85_onnx.py --src ../../onnx/models --out ../models
"""
import argparse
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper, shape_inference
from onnxsim import simplify


def fix_inputs(model, shapes):
    for inp in model.graph.input:
        if inp.name in shapes:
            dims = inp.type.tensor_type.shape.dim
            for d, v in zip(dims, shapes[inp.name]):
                d.ClearField("dim_param")
                d.dim_value = v
    return model


def split_layernorm(model):
    """Replace every LayerNormalization node with primitive ops (opset <= 17 semantics)."""
    model = shape_inference.infer_shapes(model)
    ranks = {}
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.type.tensor_type.HasField("shape"):
            ranks[vi.name] = len(vi.type.tensor_type.shape.dim)
    g = model.graph
    new_nodes, n_split = [], 0
    for node in g.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(node)
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        axis, eps = attrs.get("axis", -1), attrs.get("epsilon", 1e-5)
        x, scale = node.input[0], node.input[1]
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else None
        assert len([o for o in node.output if o]) == 1, f"{node.name}: Mean/InvStdDev outputs are used"
        if axis < 0:
            assert axis == -1 or x in ranks, f"{node.name}: rank of {x} unknown"
            axes = [-1] if axis == -1 else list(range(ranks[x] + axis, ranks[x]))
        else:
            axes = list(range(axis, ranks[x]))
        p = (node.name or f"ln{n_split}").replace("/", "_") + "_split"
        eps_name = p + "_eps"
        g.initializer.append(numpy_helper.from_array(np.array(eps, np.float32), eps_name))
        new_nodes += [
            helper.make_node("ReduceMean", [x], [p + "_mean"], axes=axes, keepdims=1),
            helper.make_node("Sub", [x, p + "_mean"], [p + "_d"]),
            helper.make_node("Mul", [p + "_d", p + "_d"], [p + "_d2"]),
            helper.make_node("ReduceMean", [p + "_d2"], [p + "_var"], axes=axes, keepdims=1),
            helper.make_node("Add", [p + "_var", eps_name], [p + "_vare"]),
            helper.make_node("Sqrt", [p + "_vare"], [p + "_std"]),
            helper.make_node("Div", [p + "_d", p + "_std"], [p + "_n"]),
            helper.make_node("Mul", [p + "_n", scale], [p + "_s"] if bias else [node.output[0]]),
        ]
        if bias:
            new_nodes.append(helper.make_node("Add", [p + "_s", bias], [node.output[0]]))
        n_split += 1
    del g.node[:]
    g.node.extend(new_nodes)
    return model, n_split


def test_feed(model, rng):
    feed = {}
    for inp in model.graph.input:
        t = inp.type.tensor_type
        shape = [d.dim_value for d in t.shape.dim]
        name = inp.name
        if name == "point_labels":
            a = np.array([[2.0, 3.0]], np.float32)[:, : shape[1]] if shape[1] == 2 else -np.ones(shape, np.float32)
        elif name == "point_coords":
            a = rng.uniform(100, 400, shape).astype(np.float32)
        elif name == "slot_valid":
            a = (rng.uniform(size=shape) > 0.4).astype(np.float32)
            a[0, 0] = 1.0
        elif name == "binarize":
            a = np.ones(shape, np.float32)
        elif name == "image":
            a = rng.uniform(-2, 2, shape).astype(np.float32)
        else:
            a = rng.normal(0, 1, shape).astype(np.float32)
        assert t.elem_type == TensorProto.FLOAT, f"{name}: not float"
        feed[name] = a
    return feed


def run(path_or_model, feed):
    data = path_or_model if isinstance(path_or_model, str) else path_or_model.SerializeToString()
    s = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    return [o for o in s.run(None, feed)], [o.name for o in s.get_outputs()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../onnx/models")
    ap.add_argument("--out", default="../models")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    jobs = [
        ("edgetam_image_encoder", None, "edgetam_image_encoder"),
        ("edgetam_memory_attention", None, "edgetam_memory_attention"),
        ("edgetam_memory_encoder", None, "edgetam_memory_encoder"),
        ("edgetam_mask_decoder", 1, "edgetam_mask_decoder_n1"),
        ("edgetam_mask_decoder", 2, "edgetam_mask_decoder_n2"),
    ]
    rng = np.random.default_rng(0)
    for src, n, dst in jobs:
        src_path = os.path.join(args.src, src + ".onnx")
        m = onnx.load(src_path)
        if n is not None:
            m = fix_inputs(m, {"point_coords": [1, n, 2], "point_labels": [1, n]})
        m, ok = simplify(m)
        assert ok, f"{dst}: onnxsim check failed"
        m, n_split = split_layernorm(m)
        onnx.checker.check_model(m)
        out_path = os.path.join(args.out, dst + ".onnx")
        onnx.save(m, out_path)

        ops = sorted({nd.op_type for nd in m.graph.node})
        feed = test_feed(m, rng)
        ref, names = run(src_path, feed)
        new, _ = run(out_path, feed)
        errs = ", ".join(f"{nm} {np.abs(a - b).max():.2e}" for nm, a, b in zip(names, ref, new))
        left = [o for o in ("LayerNormalization", "OneHot", "Shape") if o in ops]
        print(f"{dst:28s} {len(m.graph.node):4d} nodes, {n_split:2d} LayerNorm split, "
              f"left: {left or 'none'} | max |diff| {errs}")


if __name__ == "__main__":
    main()
