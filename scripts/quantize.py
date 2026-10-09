#!/usr/bin/env python3
"""
Dynamic INT8 quantization script for Nemotron Streaming ONNX models.
Reduces model size from ~2.4GB to ~650MB and accelerates CPU inference.
"""

import os
import shutil
import argparse


def quantize_model(input_dir, output_dir):
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please install dependencies: pip install onnx onnxruntime")
        return False

    os.makedirs(output_dir, exist_ok=True)

    models_to_quantize = [
        ("encoder.onnx", "encoder.onnx"),
        ("decoder_joint.onnx", "decoder_joint.onnx"),
    ]

    for input_file, output_file in models_to_quantize:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)

        if not os.path.exists(input_path):
            print(f"⚠️  Skipping {input_file} (not found in {input_dir})")
            continue

        print(f"Quantizing {input_file} to INT8...")

        try:
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                optimize_model=True
            )

            orig_mb = os.path.getsize(input_path) / (1024 * 1024)
            quant_mb = os.path.getsize(output_path) / (1024 * 1024)
            reduction = ((orig_mb - quant_mb) / orig_mb * 100) if orig_mb > 0 else 0

            print(f"✅ {input_file} quantized successfully!")
            print(f"   Original: {orig_mb:.1f} MB -> Quantized: {quant_mb:.1f} MB (-{reduction:.1f}%)\n")
        except Exception as e:
            print(f"❌ Error quantizing {input_file}: {e}")

    # Copy tokenizer.model and config.json
    for extra in ["tokenizer.model", "config.json", "vocab.txt"]:
        src = os.path.join(input_dir, extra)
        dst = os.path.join(output_dir, extra)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {extra} to {output_dir}")

    print(f"\n✅ Quantized model ready in: {output_dir}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Nemotron ONNX models to INT8")
    parser.add_argument("--input-dir", default="./crane-nemo-asr-1.1-onnx", help="Input ONNX directory")
    parser.add_argument("--output-dir", default="./crane-nemo-asr-1.1-onnx-int8", help="Output INT8 directory")
    args = parser.parse_args()

    quantize_model(args.input_dir, args.output_dir)