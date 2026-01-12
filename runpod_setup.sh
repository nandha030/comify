#!/bin/bash
# RunPod Setup Script for Comify Virtual Try-On
# Run this script after SSH'ing into your RunPod

set -e

echo "=========================================="
echo "  Comify Virtual Try-On - RunPod Setup"
echo "=========================================="

# Check GPU
echo ""
echo "[1/8] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "No GPU detected (CPU mode)"
echo ""

# Navigate to workspace
cd /workspace

# Clone repository
echo "[2/8] Cloning repository..."
if [ -d "comify" ]; then
    echo "Repository already exists, pulling latest..."
    cd comify
    git fetch --all
    git reset --hard origin/master
    echo "Updated to latest version"
else
    git clone https://github.com/nandha030/comify.git
    cd comify
fi

# Install Node.js if not present
echo ""
echo "[3/8] Checking Node.js..."
if ! command -v npm &> /dev/null; then
    echo "Installing Node.js 20.x..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
    echo "Node.js installed: $(node --version)"
    echo "npm installed: $(npm --version)"
else
    echo "Node.js already installed: $(node --version)"
fi

# Create virtual environment
echo ""
echo "[4/8] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA
echo ""
echo "[5/8] Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
echo ""
echo "[6/8] Installing Python dependencies..."
pip install -r requirements.txt

# Install additional AI packages
pip install insightface onnxruntime-gpu segment-anything controlnet-aux accelerate transformers diffusers safetensors

# Install frontend dependencies
echo ""
echo "[7/8] Setting up frontend..."
cd frontend
npm install
npm run build
cd ..

# Create data directories
mkdir -p models data backend/data/results backend/data/uploads logs

# Create start script
echo ""
echo "[8/8] Creating start script..."
cat > start_comify.sh << 'STARTSCRIPT'
#!/bin/bash
source /workspace/comify/venv/bin/activate
cd /workspace/comify
mkdir -p logs

# Kill any existing processes
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true
sleep 2

# Start backend
echo "Starting backend on port 8000..."
cd backend
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8000 > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting frontend on port 3000..."
cd ../frontend
nohup npm start > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

cd ..

echo ""
echo "=========================================="
echo "  Comify is running!"
echo "=========================================="
echo ""
echo "Backend API:  http://localhost:8000"
echo "Frontend UI:  http://localhost:3000"
echo ""
echo "Health check: curl http://localhost:8000/api/health"
echo ""
echo "View logs:"
echo "  tail -f logs/backend.log"
echo "  tail -f logs/frontend.log"
echo ""
echo "To stop: pkill -f uvicorn && pkill -f 'next start'"
STARTSCRIPT
chmod +x start_comify.sh

# Create stop script
cat > stop_comify.sh << 'STOPSCRIPT'
#!/bin/bash
echo "Stopping Comify..."
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "next start" 2>/dev/null || true
echo "Stopped."
STOPSCRIPT
chmod +x stop_comify.sh

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To start the application:"
echo "  cd /workspace/comify"
echo "  ./start_comify.sh"
echo ""
echo "To stop the application:"
echo "  ./stop_comify.sh"
echo ""
echo "Optional: Download AI models for better results:"
echo "  source venv/bin/activate"
echo "  python -c \"from installer.model_downloader import ModelDownloader; d = ModelDownloader('./models'); d.download_all()\""
echo ""
