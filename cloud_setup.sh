#!/bin/bash
# Shimmer Cloud Setup Script
# Run this after SSH-ing into a cloud GPU instance

set -e

echo "=== Shimmer Cloud Setup ==="

# Install dependencies
echo "[1/4] Installing dependencies..."
pip install -q -r requirements.txt

# Optional: Install sentence-transformers for evaluation
echo "[2/4] Installing optional evaluation dependencies..."
pip install -q sentence-transformers 2>/dev/null || echo "  (skipped - not critical)"

# Create directories
echo "[3/4] Creating directories..."
mkdir -p checkpoints tokenizers

# Verify GPU
echo "[4/4] Checking GPU..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Example training commands:"
echo ""
echo "  # 100M model (recommended next step)"
echo "  python train.py --progressive \\"
echo "      --model lira --dataset tinystories \\"
echo "      --num_samples 2000000 --vocab_size 10000 \\"
echo "      --hidden_size 768 --num_layers 12 --num_heads 12 \\"
echo "      --batch_size 64 --stage_epochs 3 \\"
echo "      --device cuda --fp16 \\"
echo "      --checkpoint_name lira_100m_progressive"
echo ""
echo "  # 300M model"
echo "  python train.py --progressive \\"
echo "      --model lira --dataset blend \\"
echo "      --num_samples 5000000 --vocab_size 10000 \\"
echo "      --hidden_size 1024 --num_layers 16 --num_heads 16 \\"
echo "      --batch_size 32 --stage_epochs 3 \\"
echo "      --device cuda --fp16 \\"
echo "      --checkpoint_name lira_300m_progressive"
echo ""
