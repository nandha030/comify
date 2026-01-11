"""
Boutique Virtual Try-On System
Professional local-only solution for fashion boutiques
Supports all garment categories - no content restrictions
All data stored locally with SQLite
"""

import asyncio
import json
import uuid
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import aiohttp
import websockets

from database import (
    init_database, ClientDB, GarmentDB, TryOnSessionDB,
    TryOnResultDB, SettingsDB, get_db
)

# ============== Configuration ==============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
GARMENTS_DIR = DATA_DIR / "garments"
THUMBNAILS_DIR = DATA_DIR / "thumbnails"

# Create directories
for dir_path in [DATA_DIR, UPLOADS_DIR, RESULTS_DIR, GARMENTS_DIR, THUMBNAILS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_WS_URL = os.getenv("COMFYUI_WS_URL", "ws://127.0.0.1:8188/ws")
COMFYUI_INPUT_DIR = BASE_DIR.parent / "ComfyUI" / "input"
COMFYUI_OUTPUT_DIR = BASE_DIR.parent / "ComfyUI" / "output"

# Ensure ComfyUI directories exist
COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
COMFYUI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============== FastAPI App ==============
app = FastAPI(
    title="Boutique Try-On System",
    description="Professional virtual try-on for fashion boutiques. Local-only, unrestricted.",
    version="2.0.0"
)

# CORS - local only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job tracking (also persisted to DB)
active_jobs = {}


# ============== Pydantic Models ==============

class ClientCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    notes: Optional[str] = None


class GarmentCreate(BaseModel):
    name: str
    category_id: Optional[str] = None
    description: Optional[str] = None
    price: Optional[float] = None
    sku: Optional[str] = None
    tags: Optional[List[str]] = None


class TryOnSettings(BaseModel):
    prompt: str = "person wearing the garment, professional fashion photo, high quality, detailed fabric texture"
    negative_prompt: str = "blurry, distorted, low quality, deformed, bad anatomy, ugly"
    steps: int = 15
    cfg_scale: float = 7.0
    sampler: str = "euler_ancestral"
    seed: int = -1
    denoise: float = 0.85
    model: str = "realisticVision"


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    result_url: Optional[str] = None
    result_id: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str


# ============== Utility Functions ==============

def generate_id() -> str:
    return uuid.uuid4().hex[:12]


async def save_upload_file(file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination"""
    content = await file.read()
    with open(destination, "wb") as f:
        f.write(content)
    return destination


def copy_to_comfyui_input(source_path: Path, prefix: str = "") -> str:
    """Copy file to ComfyUI input directory and return filename"""
    filename = f"{prefix}_{generate_id()}{source_path.suffix}"
    dest = COMFYUI_INPUT_DIR / filename
    shutil.copy(source_path, dest)
    return filename


def get_available_models() -> List[dict]:
    """Get list of available checkpoint models from both ComfyUI and V2 models directories"""
    models = []

    # Check ComfyUI models directory
    comfyui_models_dir = BASE_DIR.parent / "ComfyUI" / "models" / "checkpoints"
    if comfyui_models_dir.exists():
        for f in comfyui_models_dir.glob("*.safetensors"):
            models.append({"name": f.stem, "file": f.name, "type": "safetensors", "source": "comfyui"})
        for f in comfyui_models_dir.glob("*.ckpt"):
            models.append({"name": f.stem, "file": f.name, "type": "ckpt", "source": "comfyui"})

    # Check V2 models directory (our downloaded models)
    v2_models_dir = BASE_DIR.parent / "models"
    if v2_models_dir.exists():
        # Check for base models
        base_models_dir = v2_models_dir / "base_models"
        if base_models_dir.exists():
            for f in base_models_dir.glob("*.safetensors"):
                models.append({"name": f.stem, "file": f.name, "type": "safetensors", "source": "v2"})
            for f in base_models_dir.glob("*.ckpt"):
                models.append({"name": f.stem, "file": f.name, "type": "ckpt", "source": "v2"})

        # Check for checkpoints directory
        checkpoints_dir = v2_models_dir / "checkpoints"
        if checkpoints_dir.exists():
            for f in checkpoints_dir.glob("*.safetensors"):
                models.append({"name": f.stem, "file": f.name, "type": "safetensors", "source": "v2"})
            for f in checkpoints_dir.glob("*.ckpt"):
                models.append({"name": f.stem, "file": f.name, "type": "ckpt", "source": "v2"})

    return models


def check_v2_models_available() -> dict:
    """Check which V2 AI models are available"""
    v2_models_dir = BASE_DIR.parent / "models"
    status = {
        "available": False,
        "face_detection": False,
        "body_pose": False,
        "base_model": False,
        "inpainting": False,
        "models_found": [],
        "models_dir": str(v2_models_dir)
    }

    if not v2_models_dir.exists():
        return status

    # Check for InsightFace models (antelopev2 folder with .onnx files)
    insightface_dir = v2_models_dir / "insightface"
    if insightface_dir.exists():
        # Check for .onnx files recursively (they're in antelopev2 subfolder)
        onnx_files = list(insightface_dir.glob("**/*.onnx"))
        if onnx_files:
            status["face_detection"] = True
            status["models_found"].append("insightface")

    # Check for DWPose models
    dwpose_dir = v2_models_dir / "dwpose"
    if dwpose_dir.exists():
        onnx_files = list(dwpose_dir.glob("*.onnx"))
        if onnx_files:
            status["body_pose"] = True
            status["models_found"].append("dwpose")

    # Check for base models in checkpoints directory (where model_downloader puts them)
    checkpoints_dir = v2_models_dir / "checkpoints"
    base_models_dir = v2_models_dir / "base_models"

    for models_dir in [checkpoints_dir, base_models_dir]:
        if models_dir and models_dir.exists():
            safetensor_files = list(models_dir.glob("*.safetensors"))
            ckpt_files = list(models_dir.glob("*.ckpt"))
            if safetensor_files or ckpt_files:
                status["base_model"] = True
                status["models_found"].append(f"base_model ({len(safetensor_files) + len(ckpt_files)} files)")

                # Check specifically for inpainting model
                inpaint_files = list(models_dir.glob("*inpaint*"))
                if inpaint_files:
                    status["inpainting"] = True
                    status["models_found"].append("inpainting")
                break

    # V2 is available if we have at least face detection or base model
    status["available"] = status["face_detection"] or status["base_model"]

    return status


def build_tryon_workflow(
    person_image: str,
    mask_image: str,
    settings: TryOnSettings,
    garment_image: str = None
) -> dict:
    """Build ComfyUI workflow for try-on - optimized for all garment types"""

    # Select model based on settings
    model_map = {
        "realisticVision": "realisticVisionV60B1_v51VAE.safetensors",
        "sd15_inpainting": "sd-v1-5-inpainting.ckpt",
        "deliberate": "deliberate_v3.safetensors",
        "dreamshaper": "dreamshaper_8.safetensors",
    }
    checkpoint = model_map.get(settings.model, "realisticVisionV60B1_v51VAE.safetensors")

    # Use random seed if -1
    seed = settings.seed if settings.seed != -1 else int(time.time() * 1000) % (2**32)

    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint}
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": person_image}
        },
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image}
        },
        "3b": {
            "class_type": "ImageToMask",
            "inputs": {
                "image": ["3", 0],
                "channel": "red"
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": settings.prompt,
                "clip": ["1", 1]
            }
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": settings.negative_prompt,
                "clip": ["1", 1]
            }
        },
        "6": {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["2", 0],
                "vae": ["1", 2]
            }
        },
        "7": {
            "class_type": "SetLatentNoiseMask",
            "inputs": {
                "samples": ["6", 0],
                "mask": ["3b", 0]
            }
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": settings.steps,
                "cfg": settings.cfg_scale,
                "sampler_name": settings.sampler,
                "scheduler": "normal",
                "denoise": settings.denoise,
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["7", 0]
            }
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0],
                "vae": ["1", 2]
            }
        },
        "10": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "boutique_result",
                "images": ["9", 0]
            }
        }
    }

    return workflow, seed


async def queue_comfyui_prompt(workflow: dict) -> str:
    """Queue workflow in ComfyUI"""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": workflow}
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(status_code=500, detail=f"ComfyUI error: {error_text}")
            result = await response.json()
            return result.get("prompt_id")


async def wait_for_completion(prompt_id: str, job_id: str):
    """Wait for ComfyUI completion via WebSocket with robust timeout handling"""
    try:
        async with websockets.connect(
            f"{COMFYUI_WS_URL}?clientId={job_id}",
            ping_interval=30,
            ping_timeout=60,
            close_timeout=10
        ) as ws:
            while True:
                try:
                    # Long timeout for CPU processing - 30 minutes per message
                    message = await asyncio.wait_for(ws.recv(), timeout=1800)
                    data = json.loads(message)

                    if data.get("type") == "executing":
                        node = data.get("data", {}).get("node")
                        if node is None:
                            # Execution completed
                            active_jobs[job_id]["status"] = "completed"
                            active_jobs[job_id]["progress"] = 100
                            break
                        else:
                            # Update which node is processing
                            active_jobs[job_id]["current_node"] = node

                    elif data.get("type") == "progress":
                        value = data.get("data", {}).get("value", 0)
                        max_val = data.get("data", {}).get("max", 100)
                        progress = int((value / max_val) * 100)
                        active_jobs[job_id]["progress"] = progress

                    elif data.get("type") == "execution_error":
                        error_msg = data.get("data", {}).get("exception_message", "Unknown error")
                        active_jobs[job_id]["status"] = "failed"
                        active_jobs[job_id]["error"] = error_msg
                        break

                    active_jobs[job_id]["updated_at"] = datetime.now().isoformat()

                except asyncio.TimeoutError:
                    # Check if job is still in ComfyUI queue
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(f"{COMFYUI_URL}/queue") as resp:
                                queue = await resp.json()
                                running = queue.get("queue_running", [])
                                pending = queue.get("queue_pending", [])
                                if not running and not pending:
                                    # Queue empty, job might be done
                                    active_jobs[job_id]["status"] = "completed"
                                    active_jobs[job_id]["progress"] = 100
                                    break
                                # Still processing, continue waiting
                                continue
                    except:
                        continue

    except websockets.exceptions.ConnectionClosed:
        # Connection closed, check if job completed
        if active_jobs[job_id]["status"] != "completed":
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = "WebSocket connection closed"
    except Exception as e:
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)


# ============== API Endpoints ==============

@app.get("/")
async def root():
    return {
        "service": "Boutique Try-On System",
        "version": "2.0.0",
        "mode": "local-only",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """Check system health - returns healthy if either ComfyUI OR V2 models are available"""
    comfyui_status = "disconnected"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{COMFYUI_URL}/system_stats", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    comfyui_status = "connected"
    except:
        pass

    # Check V2 models availability
    v2_models = check_v2_models_available()

    # System is healthy if ComfyUI is connected OR V2 models are available
    is_healthy = comfyui_status == "connected" or v2_models["available"]

    return {
        "status": "healthy" if is_healthy else "degraded",
        "comfyui": comfyui_status,
        "v2_engine": "ready" if v2_models["available"] else "not_available",
        "v2_models": v2_models,
        "database": "connected",
        "models": get_available_models()
    }


# ============== Client Endpoints ==============

@app.post("/api/clients")
async def create_client(client: ClientCreate):
    """Create new client"""
    client_id = generate_id()
    result = ClientDB.create(
        id=client_id,
        name=client.name,
        email=client.email,
        phone=client.phone,
        notes=client.notes
    )
    return result


@app.get("/api/clients")
async def list_clients():
    """List all clients"""
    return {"clients": ClientDB.list_all()}


@app.get("/api/clients/{client_id}")
async def get_client(client_id: str):
    """Get client by ID"""
    client = ClientDB.get(client_id)
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")
    return client


# ============== Garment Catalog Endpoints ==============

@app.get("/api/categories")
async def list_categories():
    """List all garment categories"""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM garment_categories ORDER BY sort_order"
        ).fetchall()
        return {"categories": [dict(row) for row in rows]}


@app.post("/api/garments")
async def upload_garment(
    image: UploadFile = File(...),
    name: str = Form(...),
    category_id: str = Form(None),
    description: str = Form(None),
    price: float = Form(None),
    sku: str = Form(None),
    tags: str = Form(None)  # Comma-separated
):
    """Upload new garment to catalog"""
    garment_id = generate_id()

    # Save image
    ext = Path(image.filename).suffix or ".jpg"
    image_path = GARMENTS_DIR / f"{garment_id}{ext}"
    await save_upload_file(image, image_path)

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    result = GarmentDB.create(
        id=garment_id,
        name=name,
        image_path=str(image_path),
        category_id=category_id,
        description=description,
        price=price,
        sku=sku,
        tags=tag_list
    )

    return result


@app.get("/api/garments")
async def list_garments(
    category_id: str = Query(None),
    search: str = Query(None)
):
    """List garments, optionally filtered by category or search"""
    if search:
        garments = GarmentDB.search(search)
    else:
        garments = GarmentDB.list_by_category(category_id)
    return {"garments": garments}


@app.get("/api/garments/{garment_id}")
async def get_garment(garment_id: str):
    """Get garment by ID"""
    garment = GarmentDB.get(garment_id)
    if not garment:
        raise HTTPException(status_code=404, detail="Garment not found")
    return garment


@app.get("/api/garments/{garment_id}/image")
async def get_garment_image(garment_id: str):
    """Get garment image"""
    garment = GarmentDB.get(garment_id)
    if not garment or not garment.get("image_path"):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(garment["image_path"])


# ============== Try-On Endpoints ==============

@app.post("/api/tryon")
async def start_tryon(
    background_tasks: BackgroundTasks,
    person_image: UploadFile = File(...),
    mask_image: UploadFile = File(...),
    garment_image: UploadFile = File(None),
    garment_id: str = Form(None),
    client_id: str = Form(None),
    prompt: str = Form("person wearing elegant garment, professional fashion photography, high quality, detailed"),
    negative_prompt: str = Form("blurry, distorted, low quality, deformed, bad anatomy, ugly, disfigured"),
    steps: int = Form(25),
    cfg_scale: float = Form(7.0),
    sampler: str = Form("euler_a"),
    seed: int = Form(-1),
    denoise: float = Form(0.85),
    model: str = Form("realisticVision")
):
    """Start a try-on generation job"""

    job_id = generate_id()
    session_id = generate_id()
    now = datetime.now().isoformat()

    # Save uploaded images locally
    person_ext = Path(person_image.filename).suffix or ".png"
    person_path = UPLOADS_DIR / f"person_{job_id}{person_ext}"
    await save_upload_file(person_image, person_path)

    mask_ext = Path(mask_image.filename).suffix or ".png"
    mask_path = UPLOADS_DIR / f"mask_{job_id}{mask_ext}"
    await save_upload_file(mask_image, mask_path)

    garment_path = None
    if garment_image:
        garment_ext = Path(garment_image.filename).suffix or ".png"
        garment_path = UPLOADS_DIR / f"garment_{job_id}{garment_ext}"
        await save_upload_file(garment_image, garment_path)

    # Copy to ComfyUI input directory
    person_comfy = copy_to_comfyui_input(person_path, "person")
    mask_comfy = copy_to_comfyui_input(mask_path, "mask")

    # Create settings
    settings = TryOnSettings(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        seed=seed,
        denoise=denoise,
        model=model
    )

    # Save session to database
    TryOnSessionDB.create(
        id=session_id,
        client_photo_id=str(person_path),
        garment_id=garment_id,
        client_id=client_id,
        prompt=prompt,
        settings=settings.model_dump()
    )

    # Initialize job tracking
    active_jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "pending",
        "progress": 0,
        "result_url": None,
        "result_id": None,
        "error": None,
        "created_at": now,
        "updated_at": now
    }

    # Background processing
    async def process_job():
        try:
            active_jobs[job_id]["status"] = "processing"
            TryOnSessionDB.update_status(session_id, "processing")

            # Build and queue workflow
            workflow, used_seed = build_tryon_workflow(
                person_image=person_comfy,
                mask_image=mask_comfy,
                settings=settings
            )

            start_time = time.time()
            prompt_id = await queue_comfyui_prompt(workflow)
            await wait_for_completion(prompt_id, job_id)

            generation_time = time.time() - start_time

            if active_jobs[job_id]["status"] == "completed":
                # Find and save result
                result_files = sorted(
                    COMFYUI_OUTPUT_DIR.glob("boutique_result_*.png"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )

                if result_files:
                    # Copy result to our results directory
                    result_id = generate_id()
                    result_path = RESULTS_DIR / f"{result_id}.png"
                    shutil.copy(result_files[0], result_path)

                    # Save to database
                    TryOnResultDB.create(
                        id=result_id,
                        session_id=session_id,
                        image_path=str(result_path),
                        seed=used_seed,
                        generation_time=generation_time,
                        model_used=settings.model
                    )

                    active_jobs[job_id]["result_id"] = result_id
                    active_jobs[job_id]["result_url"] = f"/api/results/{result_id}/image"

                TryOnSessionDB.update_status(session_id, "completed")
            else:
                TryOnSessionDB.update_status(session_id, "failed")

        except Exception as e:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = str(e)
            TryOnSessionDB.update_status(session_id, "failed")

    background_tasks.add_task(process_job)

    return {"job_id": job_id, "session_id": session_id, "status": "queued"}


@app.get("/api/tryon/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**active_jobs[job_id])


# ============== Results Endpoints ==============

@app.get("/api/results/{result_id}/image")
async def get_result_image(result_id: str):
    """Get result image"""
    result_path = RESULTS_DIR / f"{result_id}.png"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(result_path)


@app.get("/api/results")
async def list_results(limit: int = Query(50)):
    """List recent results"""
    sessions = TryOnSessionDB.list_recent(limit)
    results = []
    for session in sessions:
        session_results = TryOnResultDB.get_by_session(session["id"])
        for r in session_results:
            r["session"] = session
            results.append(r)
    return {"results": results}


@app.post("/api/results/{result_id}/favorite")
async def toggle_favorite(result_id: str, is_favorite: bool = Form(True)):
    """Toggle favorite status"""
    TryOnResultDB.set_favorite(result_id, is_favorite)
    return {"status": "updated", "is_favorite": is_favorite}


@app.get("/api/results/favorites")
async def get_favorites():
    """Get favorite results"""
    return {"favorites": TryOnResultDB.get_favorites()}


# ============== Settings Endpoints ==============

@app.get("/api/settings")
async def get_settings():
    """Get all settings"""
    return {"settings": SettingsDB.get_all()}


@app.post("/api/settings")
async def update_settings(settings: dict):
    """Update settings"""
    for key, value in settings.items():
        SettingsDB.set(key, str(value))
    return {"status": "updated"}


@app.get("/api/models")
async def list_models():
    """List available AI models"""
    return {"models": get_available_models()}


# ============== Data Management ==============

@app.delete("/api/data/clear-results")
async def clear_results():
    """Clear all generated results (keeps garments and clients)"""
    # Clear results directory
    for f in RESULTS_DIR.glob("*"):
        if f.is_file():
            f.unlink()

    # Clear database results
    with get_db() as conn:
        conn.execute("DELETE FROM tryon_results")
        conn.execute("DELETE FROM tryon_sessions")

    # Clear active jobs
    global active_jobs
    active_jobs = {}

    return {"status": "cleared", "message": "All results cleared"}


@app.delete("/api/data/clear-uploads")
async def clear_uploads():
    """Clear temporary uploads"""
    for f in UPLOADS_DIR.glob("*"):
        if f.is_file():
            f.unlink()
    return {"status": "cleared"}


@app.get("/api/data/stats")
async def get_stats():
    """Get system statistics"""
    with get_db() as conn:
        clients = conn.execute("SELECT COUNT(*) as count FROM clients").fetchone()["count"]
        garments = conn.execute("SELECT COUNT(*) as count FROM garments").fetchone()["count"]
        sessions = conn.execute("SELECT COUNT(*) as count FROM tryon_sessions").fetchone()["count"]
        results = conn.execute("SELECT COUNT(*) as count FROM tryon_results").fetchone()["count"]
        favorites = conn.execute("SELECT COUNT(*) as count FROM tryon_results WHERE is_favorite = 1").fetchone()["count"]

    return {
        "clients": clients,
        "garments": garments,
        "sessions": sessions,
        "results": results,
        "favorites": favorites
    }


# ============== Static Files ==============

app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/garments", StaticFiles(directory=str(GARMENTS_DIR)), name="garments")


# ============== V2 API Integration ==============

try:
    from api_v2 import include_v2_router
    include_v2_router(app)
    V2_API_AVAILABLE = True
    print("V2 API (Advanced AI Engine) loaded")
except ImportError as e:
    V2_API_AVAILABLE = False
    print(f"V2 API not available: {e}")


# ============== Startup ==============

@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    init_database()
    print("\n" + "="*50)
    print("   COMIFY - Virtual Try-On System")
    print("="*50)
    print(f"\nDatabase: {BASE_DIR / 'data' / 'boutique.db'}")
    print(f"Results: {RESULTS_DIR}")
    print(f"ComfyUI: {COMFYUI_URL}")
    print(f"V2 API: {'Enabled' if V2_API_AVAILABLE else 'Disabled'}")
    print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
