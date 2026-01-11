#!/bin/bash
# RunPod Setup Script for Comify Virtual Try-On
# Run this script after SSH'ing into your RunPod

set -e

echo "=========================================="
echo "  Comify Virtual Try-On - RunPod Setup"
echo "=========================================="

# Check GPU
echo ""
echo "[1/7] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Navigate to workspace
cd /workspace

# Clone repository
echo "[2/7] Cloning repository..."
if [ -d "comify" ]; then
    echo "Repository already exists, pulling latest..."
    cd comify
    git pull
else
    git clone https://github.com/nandha030/comify.git
    cd comify
fi

# Create virtual environment
echo ""
echo "[3/7] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
echo ""
echo "[4/7] Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo ""
echo "[5/7] Installing dependencies..."
pip install -r requirements.txt

# Install additional AI packages
pip install insightface onnxruntime-gpu segment-anything controlnet-aux accelerate transformers diffusers safetensors

# Install frontend dependencies
echo ""
echo "[6/7] Setting up frontend..."
cd frontend
npm install
npm run build
cd ..

# Create data directories
mkdir -p models data backend/data/results backend/data/uploads

# Create start script
echo ""
echo "[7/7] Creating start script..."
cat > start_comify.sh << 'STARTSCRIPT'
#!/bin/bash
source /workspace/comify/venv/bin/activate
cd /workspace/comify

# Start backend
echo "Starting backend on port 8000..."
cd backend
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Start frontend
echo "Starting frontend on port 3000..."
cd ../frontend
nohup npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

cd ..
mkdir -p logs

echo ""
echo "=========================================="
echo "  Comify is running!"
echo "=========================================="
echo ""
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Access via RunPod proxy:"
echo "  Backend:  https://YOUR_POD_ID-8000.proxy.runpod.net"
echo "  Frontend: https://YOUR_POD_ID-3000.proxy.runpod.net"
echo ""
echo "Logs:"
echo "  tail -f logs/backend.log"
echo "  tail -f logs/frontend.log"
echo ""
echo "To stop: pkill -f uvicorn && pkill -f 'npm start'"
STARTSCRIPT
chmod +x start_comify.sh

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To start the application, run:"
echo "  cd /workspace/comify"
echo "  ./start_comify.sh"
echo ""
echo "Optional: Download AI models (recommended for best results):"
echo "  python -c \"from installer.model_downloader import ModelDownloader; d = ModelDownloader('./models'); d.download_all()\""
echo ""
