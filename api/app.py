"""
FastAPI application for serving the Siamese Face Similarity model.
Endpoint:
    POST /predict  – upload two face images, get similarity prediction.
"""
import os
from pathlib import Path
import tempfile
import shutil
from contextlib import asynccontextmanager

#fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

#src
from src.logger import get_logger
from src.utils import load_config, get_device
from src.model_build.model import build_model
from src.inference import preprocess_input
from api.schemas import SimilarityResponse

import torch

logger = get_logger("api")

#global model reference
model = None
config = None
device = None


@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Load the model on startup and clean up on shutdown.
    """
    global model, config, device
    logger.info("Loading configuration and model for API...")
    # Load config from a known path; can be set via environment variable
    config_path = "./config.yaml"
    config = load_config(config_path)
    device = get_device(config["training"]["device"])
    model = build_model(config).to(device)
    checkpoint = torch.load(config["inference"]["saved_model_path"], map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    logger.info("Model loaded and ready.")
    yield
    # Cleanup if needed
    logger.info("Shutting down API.")

app = FastAPI(
    title="Siamese Face Similarity API",
    description="Predict whether two faces images belong to the same person",
    version="1.0.0",
    lifespan=lifespan
)
#serve statis files  (CSS, JS, images) from the "api/static" folder
os.makedirs("api/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="api/static"), name="static")

app.get("/")
async def read_index():
    "serve the main UI page"
    return FileResponse("api/static/index.html")


@app.post("/predict", response_model=SimilarityResponse)
async def predict_similarity(
    img1: UploadFile = File(..., description="First face image (JPEG or PNG)"),
    img2: UploadFile = File(..., description="Second face image (JPEG or PNG)")
):
    """
    Compare two face images and return whether they show the same person.
    """

    #validate file types
    allowed_extensions = {".jpg", ".jpeg", ".png"}
    for f in (img1 ,img2):
        ext = Path(f.filename).suffix.lower()
        if ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail=f"file type {ext} not allowed. Use JPG or PNG")
    #save uploaded files temporarily
    tmp_dir = tempfile.mkdtemp()
    try:
        img1_path = Path(tmp_dir)/img1.filename
        img2_path = Path(tmp_dir)/img2.filename
        with open(img1_path, "wb") as buffer:
            shutil.copyfileobj(img1.file, buffer)
        with open(img2_path, "wb") as buffer:
            shutil.copyfileobj(img2.file, buffer)

        #preprocess and run inference
        img_size = config["data"]["image_size"]
        tensor1 = preprocess_input(str(img1_path), img_size).to(device)
        tensor2 = preprocess_input(str(img2_path), img_size).to(device)

        with torch.no_grad():
            _,_,distance = model(tensor1, tensor2)
            distance_value = distance.item()

        threshold = config["inference"]["threshold"]
        is_same = distance_value < threshold

        logger.info("compared %s vs %s: distance=%.4f, same=%s", img1.filename, img2.filename, distance_value, is_same)

        return SimilarityResponse(
            distance = distance_value,
            is_same_person = is_same,
            threshold = threshold
        )
    except Exception as e:
        logger.exception("production failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

#to run the api:
#use

#uvicorn api.app:app --host 0.0.0.0 --port 5000