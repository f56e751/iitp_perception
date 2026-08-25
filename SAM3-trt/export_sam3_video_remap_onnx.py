import argparse
import os
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None


TRT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRT_ROOT.parent
SAM3_ROOT = REPO_ROOT / "sam3"
if str(SAM3_ROOT) not in sys.path:
    sys.path.insert(0, str(SAM3_ROOT))

DEFAULT_SAM3_WEIGHTS = SAM3_ROOT / "sam3" / "assets" / "weights"
TorchModule = torch.nn.Module if torch is not None else object


def require_torch():
    if torch is None:
        raise ImportError(
            "Missing torch. Run this inside the SAM3/FastGS environment before exporting ONNX."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Experimental ONNX export for the fixed 2-frame SAM3 tracker remap path. "
            "This exports two staged graphs: mask-prompt init and one-step propagation."
        )
    )
    parser.add_argument("--sam3-weights-dir", default=str(DEFAULT_SAM3_WEIGHTS))
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--bpe-path", default=None)
    parser.add_argument("--output-dir", default="onnx_weights")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamo", action="store_true")
    parser.add_argument(
        "--force-fp32",
        action="store_true",
        help="Disable SAM3 tracker autocast and export a float32 graph for ONNX Runtime CPU checks.",
    )
    parser.add_argument("--export-backbone", action="store_true")
    parser.add_argument("--export-init", action="store_true")
    parser.add_argument("--export-propagate", action="store_true")
    parser.add_argument(
        "--export-split",
        action="store_true",
        help=(
            "Export backbone plus feature-input init/propagate graphs. "
            "This avoids recomputing image features for every object."
        ),
    )
    parser.add_argument(
        "--compact-backbone-outputs",
        action="store_true",
        help=(
            "When exporting the split backbone, only output tensors used by "
            "init/propagate-from-features."
        ),
    )
    parser.add_argument(
        "--minimal-tracking-outputs",
        action="store_true",
        help=(
            "When exporting split init/propagate, only expose tensors needed for "
            "two-frame tracking."
        ),
    )
    parser.add_argument(
        "--disable-onnx-patches",
        action="store_true",
        help="Disable SAM3 compatibility patches used only for ONNX export.",
    )
    parser.add_argument(
        "--disable-real-rope-patch",
        dest="disable_onnx_patches",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def _reshape_rope_freqs_for_broadcast(freqs, x):
    ndim = x.ndim
    assert freqs.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs.view(*shape)


def _complex_mul_as_real(x, freqs_real, freqs_imag):
    x = x.float().reshape(*x.shape[:-1], -1, 2)
    x_real = x[..., 0]
    x_imag = x[..., 1]
    real = x_real * freqs_real - x_imag * freqs_imag
    imag = x_real * freqs_imag + x_imag * freqs_real
    return torch.stack((real, imag), dim=-1).flatten(3)


def _apply_rotary_enc_real_for_onnx(
    xq,
    xk,
    freqs_cis_real,
    freqs_cis_imag,
    repeat_freqs_k=False,
):
    xq_pairs = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 0]
    freqs_cis_real = _reshape_rope_freqs_for_broadcast(freqs_cis_real, xq_pairs)
    freqs_cis_imag = _reshape_rope_freqs_for_broadcast(freqs_cis_imag, xq_pairs)
    xq_out = _complex_mul_as_real(xq, freqs_cis_real, freqs_cis_imag)

    if xk.shape[-2] == 0:
        return xq_out.type_as(xq).to(xq.device), xk

    if repeat_freqs_k:
        xk_pairs = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 0]
        repeat = xk_pairs.shape[-2] // xq_pairs.shape[-2]
        freqs_cis_real = freqs_cis_real.repeat(
            *([1] * (freqs_cis_real.ndim - 2)), repeat, 1
        )
        freqs_cis_imag = freqs_cis_imag.repeat(
            *([1] * (freqs_cis_imag.ndim - 2)), repeat, 1
        )

    xk_out = _complex_mul_as_real(xk, freqs_cis_real, freqs_cis_imag)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


def patch_sam3_for_onnx_export():
    import sam3.model.vitdet as vitdet
    import sam3.sam.mask_decoder as mask_decoder
    import sam3.sam.transformer as transformer
    import torch.nn.functional as F

    original_interpolate = F.interpolate

    def interpolate_without_antialias_for_onnx(*args, **kwargs):
        if kwargs.get("antialias", False):
            kwargs = dict(kwargs)
            kwargs["antialias"] = False
        return original_interpolate(*args, **kwargs)

    F.interpolate = interpolate_without_antialias_for_onnx

    def vitdet_apply_rope_real(self, q, k):
        if not self.use_rope:
            return q, k
        if not hasattr(self, "freqs_cis_real") or not hasattr(self, "freqs_cis_imag"):
            raise RuntimeError(
                "Missing real RoPE buffers. Call prepare_real_rope_buffers() before ONNX export."
            )
        freqs_real = self.freqs_cis_real.to(device=q.device, dtype=torch.float32)
        freqs_imag = self.freqs_cis_imag.to(device=q.device, dtype=torch.float32)
        if freqs_real.shape[0] != q.shape[-2]:
            raise RuntimeError(
                "The real RoPE ONNX patch expects the fixed SAM3 export token length. "
                f"Got freqs={freqs_real.shape[0]} and tokens={q.shape[-2]}."
            )
        return _apply_rotary_enc_real_for_onnx(q, k, freqs_real, freqs_imag)

    def transformer_apply_rotary_enc_real(
        xq, xk, freqs_cis, repeat_freqs_k=False
    ):
        freqs_real = freqs_cis.real.to(dtype=torch.float32)
        freqs_imag = freqs_cis.imag.to(dtype=torch.float32)
        return _apply_rotary_enc_real_for_onnx(
            xq, xk, freqs_real, freqs_imag, repeat_freqs_k=repeat_freqs_k
        )

    vitdet.Attention._apply_rope = vitdet_apply_rope_real
    transformer.apply_rotary_enc = transformer_apply_rotary_enc_real

    def repeat_interleave_scalar_for_onnx(x, repeats, dim):
        dim = dim if dim >= 0 else x.dim() + dim
        view_shape = list(x.shape)
        view_shape.insert(dim + 1, 1)
        expand_shape = list(x.shape)
        expand_shape.insert(dim + 1, repeats)
        out_shape = list(x.shape)
        out_shape[dim] = out_shape[dim] * repeats
        return x.reshape(*view_shape).expand(*expand_shape).reshape(*out_shape)

    def mask_decoder_predict_masks_for_onnx(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        repeat_image,
        high_res_features=None,
    ):
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat(
                [self.iou_token.weight, self.mask_tokens.weight], dim=0
            )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if repeat_image:
            src = repeat_interleave_scalar_for_onnx(
                image_embeddings, tokens.shape[0], dim=0
            )
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.size(0) == 1, (
            "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        )
        pos_src = repeat_interleave_scalar_for_onnx(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    mask_decoder.MaskDecoder.predict_masks = mask_decoder_predict_masks_for_onnx


def _set_non_persistent_buffer(module, name, tensor):
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor, persistent=False)


def prepare_real_rope_buffers(module):
    for child in module.modules():
        freqs_cis = getattr(child, "freqs_cis", None)
        if freqs_cis is None or not torch.is_complex(freqs_cis):
            continue
        _set_non_persistent_buffer(child, "freqs_cis_real", freqs_cis.real.detach())
        _set_non_persistent_buffer(child, "freqs_cis_imag", freqs_cis.imag.detach())
        if hasattr(child, "use_rope_real"):
            child.use_rope_real = True


def build_tracker(args):
    from sam3.model_builder import build_sam3_video_model

    weights_dir = Path(args.sam3_weights_dir)
    checkpoint_path = (
        Path(args.checkpoint_path) if args.checkpoint_path else weights_dir / "sam3.pt"
    )
    bpe_path = (
        Path(args.bpe_path)
        if args.bpe_path
        else SAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
    if not bpe_path.exists():
        raise FileNotFoundError(f"SAM3 BPE vocab not found: {bpe_path}")

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam3_video_model(
        checkpoint_path=str(checkpoint_path),
        bpe_path=str(bpe_path),
        device=device,
    )
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    if args.force_fp32:
        bf16_context = getattr(tracker, "bf16_context", None)
        if bf16_context is not None:
            try:
                bf16_context.__exit__(None, None, None)
            except RuntimeError:
                pass
        model.float()
        tracker.float()
    tracker.eval()
    return tracker, torch.device(device)


def flatten_backbone_out(backbone_out):
    outputs = []
    names = []
    for i, feat in enumerate(backbone_out["backbone_fpn"]):
        outputs.append(feat)
        names.append(f"backbone_fpn_{i}")
    for i, pos in enumerate(backbone_out["vision_pos_enc"]):
        outputs.append(pos)
        names.append(f"vision_pos_enc_{i}")
    return tuple(outputs), names


def unflatten_backbone_out(values, num_fpn, num_pos):
    values = list(values)
    return {
        "backbone_fpn": values[:num_fpn],
        "vision_pos_enc": values[num_fpn : num_fpn + num_pos],
    }


def expand_backbone_out_batch(backbone_out, batch_size):
    expanded = {"backbone_fpn": [], "vision_pos_enc": []}
    for key in ("backbone_fpn", "vision_pos_enc"):
        for tensor in backbone_out[key]:
            if tensor.shape[0] == batch_size:
                expanded[key].append(tensor)
                continue
            if tensor.shape[0] != 1:
                raise RuntimeError(
                    f"Cannot expand {key} batch {tensor.shape[0]} to {batch_size}"
                )
            expanded[key].append(tensor.expand(batch_size, *tensor.shape[1:]))
    return expanded


def expand_image_batch(image, batch_size):
    if image.shape[0] == batch_size:
        return image
    if image.shape[0] != 1:
        raise RuntimeError(f"Cannot expand image batch {image.shape[0]} to {batch_size}")
    return image.expand(batch_size, *image.shape[1:])


class BackboneWrapper(TorchModule):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, image):
        backbone_out = self.tracker.forward_image(image)
        outputs, _ = flatten_backbone_out(backbone_out)
        return outputs


class CompactBackboneWrapper(TorchModule):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, image):
        backbone_out = self.tracker.forward_image(image)
        return (
            backbone_out["backbone_fpn"][0],
            backbone_out["backbone_fpn"][1],
            backbone_out["backbone_fpn"][2],
            backbone_out["vision_pos_enc"][2],
        )


class SelectOutputWrapper(TorchModule):
    def __init__(self, module, indices):
        super().__init__()
        self.module = module
        self.indices = tuple(int(i) for i in indices)

    def forward(self, *args):
        outputs = self.module(*args)
        return tuple(outputs[i] for i in self.indices)


class InitMaskPromptFromFeaturesWrapper(TorchModule):
    def __init__(self, tracker, num_fpn, num_pos):
        super().__init__()
        self.tracker = tracker
        self.num_fpn = int(num_fpn)
        self.num_pos = int(num_pos)

    def forward(self, image0, mask0, *backbone_values):
        batch_size = int(mask0.shape[0])
        backbone_out = unflatten_backbone_out(
            backbone_values, self.num_fpn, self.num_pos
        )
        backbone_out = expand_backbone_out_batch(backbone_out, batch_size)
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self.tracker._prepare_backbone_features(backbone_out)
        )
        out = self.tracker.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=expand_image_batch(image0, batch_size),
            point_inputs=None,
            mask_inputs=mask0,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
            use_prev_mem_frame=True,
        )
        outputs = [
            out["pred_masks"],
            out["pred_masks_high_res"],
            out["maskmem_features"],
            out["obj_ptr"],
            out["object_score_logits"],
        ]
        if "iou_score" in out:
            outputs.append(out["iou_score"])
        else:
            outputs.append(mask0.new_ones((batch_size, 1)))
        if "eff_iou_score" in out:
            outputs.append(out["eff_iou_score"].reshape(1))
        else:
            outputs.append(mask0.new_ones((1,)))
        for pos in out["maskmem_pos_enc"]:
            outputs.append(pos)
        return tuple(outputs)


class InitMaskPromptWrapper(TorchModule):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(self, image0, mask0):
        batch_size = int(image0.shape[0])
        backbone_out = self.tracker.forward_image(image0)
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self.tracker._prepare_backbone_features(backbone_out)
        )
        out = self.tracker.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image0,
            point_inputs=None,
            mask_inputs=mask0,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=True,
            prev_sam_mask_logits=None,
            use_prev_mem_frame=True,
        )
        outputs = [
            out["pred_masks"],
            out["pred_masks_high_res"],
            out["maskmem_features"],
            out["obj_ptr"],
            out["object_score_logits"],
        ]
        if "iou_score" in out:
            outputs.append(out["iou_score"])
        else:
            outputs.append(image0.new_ones((batch_size, 1)))
        if "eff_iou_score" in out:
            outputs.append(out["eff_iou_score"].reshape(1))
        else:
            outputs.append(image0.new_ones((1,)))
        for pos in out["maskmem_pos_enc"]:
            outputs.append(pos)
        return tuple(outputs)


class PropagateOneStepFromFeaturesWrapper(TorchModule):
    def __init__(self, tracker, num_fpn, num_pos):
        super().__init__()
        self.tracker = tracker
        self.num_fpn = int(num_fpn)
        self.num_pos = int(num_pos)

    def forward(
        self,
        cond_pred_masks,
        cond_maskmem_features,
        cond_obj_ptr,
        cond_object_score_logits,
        cond_iou_score,
        cond_eff_iou_score,
        *values,
    ):
        num_backbone_values = self.num_fpn + self.num_pos
        cond_maskmem_pos_enc = values[:-num_backbone_values]
        backbone_values = values[-num_backbone_values:]
        batch_size = int(cond_maskmem_features.shape[0])
        backbone_out = unflatten_backbone_out(
            backbone_values, self.num_fpn, self.num_pos
        )
        backbone_out = expand_backbone_out_batch(backbone_out, batch_size)
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self.tracker._prepare_backbone_features(backbone_out)
        )
        cond_out = {
            "pred_masks": cond_pred_masks,
            "maskmem_features": cond_maskmem_features,
            "maskmem_pos_enc": list(cond_maskmem_pos_enc),
            "obj_ptr": cond_obj_ptr,
            "object_score_logits": cond_object_score_logits,
            "iou_score": cond_iou_score,
            "eff_iou_score": cond_eff_iou_score,
        }
        out = self.tracker.track_step(
            frame_idx=1,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=None,
            point_inputs=None,
            mask_inputs=None,
            output_dict={
                "cond_frame_outputs": {0: cond_out},
                "non_cond_frame_outputs": {},
            },
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
            use_prev_mem_frame=True,
        )
        return (
            out["pred_masks"],
            out["pred_masks_high_res"],
            out["obj_ptr"],
            out["object_score_logits"],
        )


class PropagateOneStepWrapper(TorchModule):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def forward(
        self,
        image1,
        cond_pred_masks,
        cond_maskmem_features,
        cond_obj_ptr,
        cond_object_score_logits,
        cond_iou_score,
        cond_eff_iou_score,
        *cond_maskmem_pos_enc,
    ):
        backbone_out = self.tracker.forward_image(image1)
        _, vision_feats, vision_pos_embeds, feat_sizes = (
            self.tracker._prepare_backbone_features(backbone_out)
        )
        cond_out = {
            "pred_masks": cond_pred_masks,
            "maskmem_features": cond_maskmem_features,
            "maskmem_pos_enc": list(cond_maskmem_pos_enc),
            "obj_ptr": cond_obj_ptr,
            "object_score_logits": cond_object_score_logits,
            "iou_score": cond_iou_score,
            "eff_iou_score": cond_eff_iou_score,
        }
        out = self.tracker.track_step(
            frame_idx=1,
            is_init_cond_frame=False,
            current_vision_feats=vision_feats,
            current_vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            image=image1,
            point_inputs=None,
            mask_inputs=None,
            output_dict={
                "cond_frame_outputs": {0: cond_out},
                "non_cond_frame_outputs": {},
            },
            num_frames=2,
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
            use_prev_mem_frame=True,
        )
        return (
            out["pred_masks"],
            out["pred_masks_high_res"],
            out["obj_ptr"],
            out["object_score_logits"],
        )


def export_onnx(
    module,
    inputs,
    path,
    input_names,
    output_names,
    opset,
    dynamo=False,
    do_constant_folding=True,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            dynamo=dynamo,
            do_constant_folding=do_constant_folding,
        )
    print(f"wrote {path}")


def main():
    args = parse_args()
    require_torch()
    if not args.disable_onnx_patches:
        patch_sam3_for_onnx_export()
    if not (
        args.all
        or args.export_backbone
        or args.export_init
        or args.export_propagate
        or args.export_split
    ):
        args.export_init = True
        args.export_propagate = True

    output_dir = Path(args.output_dir)
    tracker, device = build_tracker(args)
    if not args.disable_onnx_patches:
        prepare_real_rope_buffers(tracker)
    batch = int(args.batch_size)
    size = int(args.image_size)
    image0 = torch.randn(batch, 3, size, size, device=device, dtype=torch.float32)
    image1 = torch.randn(batch, 3, size, size, device=device, dtype=torch.float32)
    mask0 = torch.zeros(batch, 1, size, size, device=device, dtype=torch.float32)
    mask0[:, :, size // 4 : size * 3 // 4, size // 4 : size * 3 // 4] = 1.0
    frame_image0 = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)
    frame_image1 = torch.randn(1, 3, size, size, device=device, dtype=torch.float32)

    if args.all or args.export_backbone or args.export_split:
        backbone = (
            CompactBackboneWrapper(tracker)
            if args.compact_backbone_outputs
            else BackboneWrapper(tracker)
        )
        with torch.inference_mode():
            backbone_sample_image = frame_image0 if args.export_split else image0
            backbone_out = tracker.forward_image(backbone_sample_image)
            backbone_outputs, backbone_names = flatten_backbone_out(backbone_out)
            feature_backbone_names = list(backbone_names)
            num_backbone_fpn = len(backbone_out["backbone_fpn"])
            num_backbone_pos = len(backbone_out["vision_pos_enc"])
            if args.compact_backbone_outputs:
                backbone_outputs = (
                    backbone_out["backbone_fpn"][0],
                    backbone_out["backbone_fpn"][1],
                    backbone_out["backbone_fpn"][2],
                    backbone_out["vision_pos_enc"][2],
                )
                backbone_names = [
                    "backbone_fpn_0",
                    "backbone_fpn_1",
                    "backbone_fpn_2",
                    "vision_pos_enc_2",
                ]
        print("backbone output shapes:")
        for name, tensor in zip(backbone_names, backbone_outputs):
            print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")
        export_onnx(
            backbone,
            (backbone_sample_image,),
            output_dir / "sam3_tracker_backbone.onnx",
            ["image"],
            backbone_names,
            args.opset,
            args.dynamo,
        )
    else:
        with torch.inference_mode():
            backbone_out = tracker.forward_image(frame_image0)
            backbone_outputs, backbone_names = flatten_backbone_out(backbone_out)
            feature_backbone_names = list(backbone_names)
            num_backbone_fpn = len(backbone_out["backbone_fpn"])
            num_backbone_pos = len(backbone_out["vision_pos_enc"])

    if args.export_split:
        with torch.inference_mode():
            frame0_backbone_out = tracker.forward_image(frame_image0)
            frame0_backbone_outputs, _ = flatten_backbone_out(frame0_backbone_out)
            frame1_backbone_out = tracker.forward_image(frame_image1)
            frame1_backbone_outputs, _ = flatten_backbone_out(frame1_backbone_out)
            frame0_backbone_inputs = tuple(
                torch.randn_like(t).detach() for t in frame0_backbone_outputs
            )
            frame1_backbone_inputs = tuple(
                torch.randn_like(t).detach() for t in frame1_backbone_outputs
            )

        split_init_wrapper = InitMaskPromptFromFeaturesWrapper(
            tracker, num_backbone_fpn, num_backbone_pos
        )
        with torch.inference_mode():
            split_init_outputs = split_init_wrapper(
                frame_image0, mask0, *frame0_backbone_inputs
            )
        split_init_names = [
            "pred_masks",
            "pred_masks_high_res",
            "maskmem_features",
            "obj_ptr",
            "object_score_logits",
            "iou_score",
            "eff_iou_score",
        ] + [f"maskmem_pos_enc_{i}" for i in range(len(split_init_outputs) - 7)]
        print("feature-init output shapes:")
        for name, tensor in zip(split_init_names, split_init_outputs):
            print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")
        init_export_wrapper = split_init_wrapper
        init_export_outputs = split_init_outputs
        init_export_names = split_init_names
        if args.minimal_tracking_outputs:
            init_export_wrapper = SelectOutputWrapper(split_init_wrapper, [2, 3, 7])
            init_export_outputs = (
                split_init_outputs[2],
                split_init_outputs[3],
                split_init_outputs[7],
            )
            init_export_names = [
                "maskmem_features",
                "obj_ptr",
                "maskmem_pos_enc_0",
            ]
            print("minimal feature-init output shapes:")
            for name, tensor in zip(init_export_names, init_export_outputs):
                print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")
        export_onnx(
            init_export_wrapper,
            (frame_image0, mask0, *frame0_backbone_inputs),
            output_dir / "sam3_tracker_init_mask_prompt_from_features.onnx",
            ["image0", "mask0"] + [f"frame0_{name}" for name in feature_backbone_names],
            init_export_names,
            args.opset,
            args.dynamo,
            do_constant_folding=False,
        )

        split_propagate = PropagateOneStepFromFeaturesWrapper(
            tracker, num_backbone_fpn, num_backbone_pos
        )
        split_propagate_inputs = (
            split_init_outputs[0],
            split_init_outputs[2],
            split_init_outputs[3],
            split_init_outputs[4],
            split_init_outputs[5],
            split_init_outputs[6],
            *split_init_outputs[7:],
            *frame1_backbone_inputs,
        )
        split_propagate_input_names = [
            "cond_pred_masks",
            "cond_maskmem_features",
            "cond_obj_ptr",
            "cond_object_score_logits",
            "cond_iou_score",
            "cond_eff_iou_score",
        ] + [f"cond_maskmem_pos_enc_{i}" for i in range(len(split_init_outputs) - 7)] + [
            f"frame1_{name}" for name in feature_backbone_names
        ]
        with torch.inference_mode():
            split_prop_outputs = split_propagate(*split_propagate_inputs)
        split_prop_names = [
            "pred_masks",
            "pred_masks_high_res",
            "obj_ptr",
            "object_score_logits",
        ]
        print("feature-propagate output shapes:")
        for name, tensor in zip(split_prop_names, split_prop_outputs):
            print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")
        prop_export_wrapper = split_propagate
        prop_export_names = split_prop_names
        if args.minimal_tracking_outputs:
            prop_export_wrapper = SelectOutputWrapper(split_propagate, [1])
            prop_export_names = ["pred_masks_high_res"]
            print("minimal feature-propagate output shapes:")
            print(f"  pred_masks_high_res: {tuple(split_prop_outputs[1].shape)} {split_prop_outputs[1].dtype}")
        export_onnx(
            prop_export_wrapper,
            split_propagate_inputs,
            output_dir / "sam3_tracker_propagate_one_step_from_features.onnx",
            split_propagate_input_names,
            prop_export_names,
            args.opset,
            args.dynamo,
            do_constant_folding=False,
        )

    if not (args.all or args.export_init or args.export_propagate):
        return

    init_wrapper = InitMaskPromptWrapper(tracker)
    with torch.inference_mode():
        init_outputs = init_wrapper(image0, mask0)
    init_names = [
        "pred_masks",
        "pred_masks_high_res",
        "maskmem_features",
        "obj_ptr",
        "object_score_logits",
        "iou_score",
        "eff_iou_score",
    ] + [f"maskmem_pos_enc_{i}" for i in range(len(init_outputs) - 7)]

    print("init output shapes:")
    for name, tensor in zip(init_names, init_outputs):
        print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")

    if args.all or args.export_init:
        export_onnx(
            init_wrapper,
            (image0, mask0),
            output_dir / "sam3_tracker_init_mask_prompt.onnx",
            ["image0", "mask0"],
            init_names,
            args.opset,
            args.dynamo,
        )

    if args.all or args.export_propagate:
        propagate = PropagateOneStepWrapper(tracker)
        propagate_inputs = (
            image1,
            init_outputs[0],
            init_outputs[2],
            init_outputs[3],
            init_outputs[4],
            init_outputs[5],
            init_outputs[6],
            *init_outputs[7:],
        )
        propagate_input_names = [
            "image1",
            "cond_pred_masks",
            "cond_maskmem_features",
            "cond_obj_ptr",
            "cond_object_score_logits",
            "cond_iou_score",
            "cond_eff_iou_score",
        ] + [f"cond_maskmem_pos_enc_{i}" for i in range(len(init_outputs) - 7)]
        with torch.inference_mode():
            prop_outputs = propagate(*propagate_inputs)
        prop_names = [
            "pred_masks",
            "pred_masks_high_res",
            "obj_ptr",
            "object_score_logits",
        ]
        print("propagate output shapes:")
        for name, tensor in zip(prop_names, prop_outputs):
            print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")
        export_onnx(
            propagate,
            propagate_inputs,
            output_dir / "sam3_tracker_propagate_one_step.onnx",
            propagate_input_names,
            prop_names,
            args.opset,
            args.dynamo,
        )


if __name__ == "__main__":
    main()
