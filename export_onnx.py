"""Export the fine-tuned ResNet18 to ONNX (for the browser version) and check it.

Usage (inside the project folder, with .venv active):
    python export_onnx.py                          # export + check on random input
    python export_onnx.py photo1.jpg photo2.webp   # also compare on real photos
"""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image

from classifier import CLASSES, load_model, predict

OUT = Path("web/model.onnx")
OUT.parent.mkdir(exist_ok=True)

# 1) Load the PyTorch model (fp16 file -> float32 weights)
model = load_model()
dummy = torch.randn(1, 3, 224, 224)

# 2) Export to ONNX
kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=17)
try:
    torch.onnx.export(model, dummy, str(OUT), dynamo=False, **kwargs)   # classic exporter
except TypeError:
    torch.onnx.export(model, dummy, str(OUT), **kwargs)                 # older torch: no `dynamo` arg

# make sure everything is in ONE file (the browser needs a single model file)
m = onnx.load(str(OUT))
onnx.save_model(m, str(OUT), save_as_external_data=False)
for extra in OUT.parent.glob("model.onnx.data"):
    extra.unlink()
onnx.checker.check_model(str(OUT))
print(f"Saved {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")

# 3) Compare PyTorch vs ONNX Runtime on the same input
sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
with torch.no_grad():
    pt = model(dummy).numpy()
ox = sess.run(None, {"input": dummy.numpy()})[0]
print(f"Random input: max difference in logits = {np.abs(pt - ox).max():.6f}")


def onnx_predict(img):
    """Same preprocessing the website will use: resize 224, /255, ImageNet mean/std."""
    x = np.asarray(img.convert("RGB").resize((224, 224)), dtype=np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)
    logits = sess.run(None, {"input": x})[0][0]
    e = np.exp(logits - logits.max())
    return e / e.sum()


for path in sys.argv[1:]:
    img = Image.open(path)
    p_pt, p_ox = predict(model, img), onnx_predict(img)
    print(f"\n{path}")
    for c, a, b in zip(CLASSES, p_pt, p_ox):
        print(f"   {c:7s} PyTorch {a:.4f}   ONNX {b:.4f}")
