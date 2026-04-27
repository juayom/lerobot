# Mesh renderer replacement plan

Current state:
- `render_proxy_control.py` generates a geometry-preserving proxy control image using the original object mask's support region.
- This is a temporary stand-in for the paper's rendered mesh step.

To become more paper-faithful, replace the proxy stage with:
1. object mesh asset loading
2. known camera pose / intrinsics loading
3. mesh scaling and placement inside the original mask support region
4. depth or shaded render export
5. pass that render as `--rendered-control` to `run_fi_genaug.py`

Expected swap point:
- keep `run_fi_genaug.py` and `depth_guided_editor.py` unchanged
- replace only the producer of `rendered-control`
