#!/usr/bin/env python3
"""
Ears Server - Speech-to-Text using MLX Whisper
OpenAI-compatible API endpoint for audio transcription.
"""

import argparse
import tempfile
import logging
from typing import Optional

import mlx_whisper
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model path
MODEL_PATH: str = "mlx-community/whisper-large-v3-turbo"

# Create FastAPI app
app = FastAPI(
    title="OmniMLX Ears Server",
    description="Speech-to-Text API using MLX Whisper",
    version="1.0.0"
)


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""
    text: str


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "ears",
        "model": MODEL_PATH
    }


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-large-v3-turbo"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0)
):
    """
    Transcribe audio file to text.
    
    OpenAI-compatible endpoint for audio transcription.
    
    Args:
        file: Audio file (WAV, MP3, M4A, FLAC, OGG, etc.)
        model: Model name (ignored, uses server's loaded model)
        language: Language hint (e.g., "en", "es")
        prompt: Optional prompt to guide transcription
        response_format: Output format ("json", "text", "srt", "verbose_json", "vtt")
        temperature: Sampling temperature (0.0 = deterministic)
    
    Returns:
        TranscriptionResponse with transcribed text
    """
    try:
        # Read uploaded file
        audio_data = await file.read()
        logger.info(f"Received audio file: {file.filename} ({len(audio_data)} bytes)")
        
        # Determine file extension
        filename = file.filename or "audio.wav"
        suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".wav"
        
        # Save to temporary file (mlx_whisper needs a file path)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        logger.info(f"Transcribing with model: {MODEL_PATH}")
        
        # Run Whisper transcription
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=MODEL_PATH,
            language=language,
            verbose=False,
            temperature=temperature,
            initial_prompt=prompt
        )
        
        # Clean up temp file
        import os
        os.unlink(tmp_path)
        
        transcribed_text = result.get("text", "").strip()
        logger.info(f"Transcription complete: {len(transcribed_text)} characters")
        
        return TranscriptionResponse(text=transcribed_text)
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/hear")
async def hear_legacy(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-large-v3-turbo")
):
    """
    Legacy endpoint for backward compatibility with Conscious Pebble.
    
    Args:
        file: Audio file
        model: Model name (ignored)
    
    Returns:
        JSON with transcribed text
    """
    return await transcribe_audio(file, model)


def main():
    """Main entry point."""
    global MODEL_PATH
    
    parser = argparse.ArgumentParser(description="OmniMLX Ears Server (Speech-to-Text)")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/whisper-large-v3-turbo",
        help="Path to Whisper model (local path or HuggingFace ID)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port to run the server on (default: 8081)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    
    args = parser.parse_args()
    MODEL_PATH = args.model
    
    logger.info(f"Starting Ears Server on {args.host}:{args.port}")
    logger.info(f"Model: {MODEL_PATH}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()