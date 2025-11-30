# NNEngine interface assumptions (to refine)

What we know from `use_NN.md`:
- Model: DeepLabV3+ with EfficientNet-b4 encoder.
- 25 segmentation classes (listed in config `nn.class_names`).
- Checkpoints: `NN/model_traced.pt` and `NN/best_model.pth` exist in the repository.

Open points to clarify before wiring a real inference path:
- Exact runtime API: TorchScript vs. eager PyTorch; which checkpoint should be used (`model_traced.pt` preferred if fully self-contained).
- Expected input format:
  - Image size (width, height) for inference; current config uses `[512, 512]` as a placeholder.
  - Color space (RGB/BGR) and channel order.
  - Normalization values (mean/std) and scaling (0..1 or 0..255).
- Output format:
  - Whether the model returns per-pixel class logits, masks, or already postprocessed polygons/lines.
  - If output is a mask, how to derive object instances (connected components, thresholding, or provided indices).
  - Confidence definition for each detection (per-pixel average, max logit, softmax score, etc.).
- Performance constraints:
  - Preferred device (`cpu` vs `cuda`) and memory footprint.
  - Batch size (likely 1) and allowed latency.

Until these points are clarified, `core/nn_engine.py` contains a stub that loads the path and raises `NotImplementedError` for `infer`.
