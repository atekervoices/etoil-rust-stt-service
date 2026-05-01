import os
import shutil

def quantize_model(input_dir, output_dir):
    """Quantize ONNX models to int8 using dynamic quantization"""
    
    # Import here after dependencies are installed
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please install dependencies first: pip3 install onnx onnxruntime")
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Files to quantize
    models_to_quantize = [
        ("encoder-model.onnx", "encoder-model.int8.onnx"),
        ("decoder-model.onnx", "decoder-model.int8.onnx")
    ]
    
    for input_file, output_file in models_to_quantize:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        print(f"Quantizing {input_file}...")
        
        try:
            # Apply dynamic quantization
            quantize_dynamic(
                model_input=input_path,
                model_output=output_path,
                weight_type=QuantType.QInt8,
                optimize_model=True
            )
            
            # Get file sizes for comparison
            original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
            quantized_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            
            print(f" {input_file} quantized successfully!")
            print(f"  Original: {original_size:.1f} MB")
            print(f"  Quantized: {quantized_size:.1f} MB")
            print(f"  Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")
            print()
            
        except Exception as e:
            print(f" Error quantizing {input_file}: {str(e)}")
            return False
    
    # Copy vocab.txt to output directory
    vocab_input = os.path.join(input_dir, "vocab.txt")
    vocab_output = os.path.join(output_dir, "vocab.txt")
    
    if os.path.exists(vocab_input):
        shutil.copy2(vocab_input, vocab_output)
        print(f" vocab.txt copied to output directory")
    
    print("Quantization complete!")
    return True

if __name__ == "__main__":
    # Input and output directories
    input_directory = "canary-180m-flash"
    output_directory = "canary-180m-flash-int8"
    
    print("Starting ONNX model quantization...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print()
    
    # Install dependencies using uv pip
    print("Installing dependencies...")
    import subprocess
    import sys
    
    try:
        # Use uv pip to install to the current Python environment
        result = subprocess.run([sys.executable, "-m", "uv", "pip", "install", "onnx", "onnxruntime"], 
                              check=True, capture_output=True, text=True)
        print("✓ Dependencies installed successfully with uv pip")
    except subprocess.CalledProcessError:
        print("Trying pip as fallback...")
        subprocess.run([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"], check=True)
    print()
    
    # Check if input directory exists
    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' not found!")
        print("Make sure you have downloaded the original model files first.")
        exit(1)
    
    # Run quantization
    success = quantize_model(input_directory, output_directory)
    
    if success:
        print("\n All models quantized successfully!")
        print(f"You can now use the quantized models from: {output_directory}")
        print("Update your code to point to this directory for faster inference.")
    else:
        print("\n Quantization failed. Check the error messages above.")