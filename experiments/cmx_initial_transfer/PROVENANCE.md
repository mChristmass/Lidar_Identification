# CMX source provenance

- Upstream: `https://github.com/huaaaliu/RGBX_Semantic_Segmentation`
- Upstream commit: `e251d860aebc2f583a6c4919877e6bebe7f1aff3`
- Upstream license: MIT, copied verbatim to `CMX_LICENSE`
- Retrieved: 2026-09-21

Adapted upstream files:

- `models/encoders/dual_segformer.py` -> `official/dual_segformer.py`
- `models/net_utils.py` -> `official/net_utils.py`
- `models/decoders/MLPDecoder.py` -> `official/mlp_decoder.py`

Minimal compatibility/adaptation changes:

1. Replaced `timm.models.layers` helpers with local equivalents backed by PyTorch.
2. Removed the upstream global logger dependency.
3. Exposed `img_size`, `in_chans`, and `norm_fuse` in the MiT-B2 constructor.
4. Set the two input stems to one channel in the experiment wrapper.
5. Added a two-class MLP decoder wrapper and full-resolution interpolation.
6. Added explicit ImageNet MiT-B2 loading that duplicates encoder weights into both
   branches and averages the two first-stem RGB kernels across their input channels.

FRM, FFM, MiT-B2 stage depths, head counts, embedding dimensions, spatial-reduction
ratios, stochastic-depth rate, and MLP decoder logic remain the upstream CMX design.
