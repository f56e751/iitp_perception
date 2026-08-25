import torch
from pathlib import Path
from transformers.models.sam3 import Sam3Processor, Sam3Model
from PIL import Image
import argparse


SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent
DEFAULT_WEIGHTS_DIR = BUNDLE_ROOT / "weights" / "sam3_hf"
DEFAULT_SAMPLE_IMAGE = BUNDLE_ROOT / "data" / "perception_dataset_260526" / "val" / "images" / "000000.jpg"
DEFAULT_OUTPUT_DIR = BUNDLE_ROOT / "weights" / "SAM3-trt" / "onnx"
TEXT_SEQ_LEN = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Export the SAM3 mixed-prompt image model to static-batch ONNX.")
    parser.add_argument("--weights-dir", default=str(DEFAULT_WEIGHTS_DIR))
    parser.add_argument("--sample-image", default=str(DEFAULT_SAMPLE_IMAGE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-seq-len", type=int, default=TEXT_SEQ_LEN)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bake-prompts", action="store_true")
    return parser.parse_args()


class Sam3ONNXWrapper(torch.nn.Module):
    def __init__(self, sam3):
        super().__init__()
        self.sam3 = sam3

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.sam3(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.pred_masks, outputs.semantic_seg, outputs.pred_logits


class Sam3BakedPromptONNXWrapper(torch.nn.Module):
    def __init__(self, sam3, input_ids, attention_mask):
        super().__init__()
        self.sam3 = sam3
        self.register_buffer("input_ids", input_ids)
        self.register_buffer("attention_mask", attention_mask)

    def forward(self, pixel_values):
        outputs = self.sam3(
            pixel_values=pixel_values,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )
        return outputs.pred_masks, outputs.semantic_seg, outputs.pred_logits


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = Sam3Model.from_pretrained(args.weights_dir).to(device)
    processor = Sam3Processor.from_pretrained(args.weights_dir)
    model.eval()

    image = Image.open(args.sample_image).convert("RGB")
    images = [image] * args.batch_size
    prompts = [
        "plastic beverage bottle",
        "recyclable metal packaging",
        "crushed can",
        "cardboard box",
        "cardboard package",
    ]
    while len(prompts) < args.batch_size:
        prompts.append(prompts[-1])
    prompts = prompts[:args.batch_size]

    image_inputs = processor.image_processor(images=images, return_tensors="pt")
    text_inputs = processor.tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=args.text_seq_len,
        return_tensors="pt",
    )
    inputs = {**image_inputs, **text_inputs}
    inputs = {k: v.to(device) for k, v in inputs.items()}

    pixel_values = inputs["pixel_values"]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    if args.bake_prompts:
        wrapper = Sam3BakedPromptONNXWrapper(model, input_ids, attention_mask).to(device).eval()
        export_args = (pixel_values,)
        input_names = ["pixel_values"]
        output_name_default = f"sam3_mixed_prompt_b{args.batch_size}_baked.onnx"
    else:
        wrapper = Sam3ONNXWrapper(model).to(device).eval()
        export_args = (pixel_values, input_ids, attention_mask)
        input_names = ["pixel_values", "input_ids", "attention_mask"]
        output_name_default = f"sam3_mixed_prompt_b{args.batch_size}.onnx"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or output_name_default
    onnx_path = str(output_dir / output_name)

    print(f"Exporting batch={args.batch_size} ONNX to {onnx_path}")
    print(f"bake_prompts={args.bake_prompts} prompts={prompts}")
    print(f"pixel_values={tuple(pixel_values.shape)} input_ids={tuple(input_ids.shape)}")
    torch.onnx.export(
        wrapper,
        export_args,
        onnx_path,
        input_names=input_names,
        output_names=["instance_masks", "semantic_seg", "pred_logits"],
        dynamo=False,
        opset_version=17,
    )
    print(f"Exported to {onnx_path}")


if __name__ == "__main__":
    main()
