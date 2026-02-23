#!/usr/bin/env python3
"""
Mouth Server - Text-to-Speech using MLX Kokoro
OpenAI-compatible API endpoint for speech synthesis.
"""

import argparse
import io
import logging
import numpy as np
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn
import soundfile as sf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model path
MODEL_PATH: str = "mlx-community/Kokoro-82M-bf16"

# Sample rate for Kokoro
SAMPLE_RATE = 24000

# Available voices
AVAILABLE_VOICES = [
    {"id": "af_heart", "name": "Pebble", "description": "Default warm voice"},
    {"id": "af_bella", "name": "Bella", "description": "Soft, gentle"},
    {"id": "af_nicole", "name": "Nicole", "description": "Professional"},
    {"id": "af_sarah", "name": "Sarah", "description": "Friendly"},
    {"id": "af_sky", "name": "Emily", "description": "Energetic"},
    {"id": "am_michael", "name": "Michael", "description": "Male, deep"},
    {"id": "am_adam", "name": "Adam", "description": "Male, neutral"},
    {"id": "am_eric", "name": "Eric", "description": "Male, casual"},
    {"id": "am_liam", "name": "Liam", "description": "Male, warm"},
    {"id": "am_onyx", "name": "Onyx", "description": "Male, bold"},
]

# Global model instance
KOKORO_MODEL = None


def load_kokoro_model(model_path: str):
    """Load Kokoro model from path or HuggingFace ID."""
    global KOKORO_MODEL
    
    if KOKORO_MODEL is not None:
        return KOKORO_MODEL
    
    try:
        # Try mlx_audio first (preferred)
        from mlx_audio import load_model
        KOKORO_MODEL = load_model(model_path)
        logger.info(f"Loaded Kokoro model via mlx_audio: {model_path}")
        return KOKORO_MODEL
    except ImportError:
        logger.warning("mlx_audio not available, trying kokoro module...")
    
    try:
        # Fallback to kokoro module
        from kokoro import KModel
        KOKORO_MODEL = KModel(repo_or_path=model_path)
        logger.info(f"Loaded Kokoro model via kokoro: {model_path}")
        return KOKORO_MODEL
    except ImportError:
        logger.error("Neither mlx_audio nor kokoro module available")
        raise ImportError(
            "No TTS backend available. Install with: pip install mlx-audio"
        )


# Create FastAPI app
app = FastAPI(
    title="OmniMLX Mouth Server",
    description="Text-to-Speech API using MLX Kokoro",
    version="1.0.0"
)


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech request."""
    model: str = Field(default="kokoro", description="Model to use for synthesis")
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(default="af_heart", description="Voice ID to use")
    response_format: str = Field(default="wav", description="Audio format (wav, mp3)")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Speech speed multiplier")


class VoiceResponse(BaseModel):
    """Voice information response."""
    id: str
    name: str
    description: str


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "mouth",
        "model": MODEL_PATH
    }


@app.get("/v1/audio/voices", response_model=List[VoiceResponse])
async def list_voices():
    """
    List available voices for text-to-speech.
    
    Returns:
        List of available voice configurations
    """
    return [VoiceResponse(**v) for v in AVAILABLE_VOICES]


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """
    Generate speech from text.
    
    OpenAI-compatible endpoint for text-to-speech synthesis.
    
    Args:
        request: SpeechRequest with text, voice, and options
    
    Returns:
        Audio bytes (WAV format)
    """
    try:
        # Validate voice
        voice_ids = [v["id"] for v in AVAILABLE_VOICES]
        if request.voice not in voice_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid voice '{request.voice}'. Available: {voice_ids}"
            )
        
        logger.info(f"Synthesizing: '{request.input[:50]}...' with voice '{request.voice}'")
        
        # Load model
        model = load_kokoro_model(MODEL_PATH)
        
        # Generate audio
        audio_np = await generate_audio(model, request.input, request.voice, request.speed)
        
        # Convert to WAV bytes
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_np, SAMPLE_RATE, format="WAV")
        wav_buffer.seek(0)
        
        logger.info(f"Generated {len(audio_np)} audio samples ({len(audio_np) / SAMPLE_RATE:.2f}s)")
        
        return Response(
            content=wav_buffer.read(),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def generate_audio(model, text: str, voice: str, speed: float) -> np.ndarray:
    """
    Generate audio from text using Kokoro model.
    
    Args:
        model: Loaded Kokoro model
        text: Text to synthesize
        voice: Voice ID
        speed: Speed multiplier
    
    Returns:
        Numpy array of audio samples at 24kHz
    """
    try:
        # Try mlx_audio style generation
        results = model.generate(text=text, voice=voice, speed=speed)
        
        # Concatenate audio chunks
        audio_chunks = []
        for result in results:
            if hasattr(result, 'audio'):
                audio_chunks.append(np.array(result.audio))
            elif isinstance(result, dict) and 'audio' in result:
                audio_chunks.append(np.array(result['audio']))
            elif isinstance(result, np.ndarray):
                audio_chunks.append(result)
        
        if audio_chunks:
            return np.concatenate(audio_chunks)
        else:
            raise ValueError("No audio generated")
            
    except AttributeError:
        # Fallback for different API style (kokoro module)
        logger.info("Trying alternative generation API...")
        audio = model.generate(text=text, voice=voice, speed=speed)
        return np.array(audio)


@app.post("/speak")
async def speak_legacy(payload: dict):
    """
    Legacy endpoint for backward compatibility with Conscious Pebble.
    
    Args:
        payload: {"text": "...", "voice": "...", "speed": 1.0}
    
    Returns:
        Audio bytes (WAV format)
    """
    request = SpeechRequest(
        model=payload.get("model", "kokoro"),
        input=payload.get("text", ""),
        voice=payload.get("voice", "af_heart"),
        speed=payload.get("speed", 1.0)
    )
    return await create_speech(request)


def main():
    """Main entry point."""
    global MODEL_PATH
    
    parser = argparse.ArgumentParser(description="OmniMLX Mouth Server (Text-to-Speech)")
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/Kokoro-82M-bf16",
        help="Path to Kokoro model (local path or HuggingFace ID)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8082,
        help="Port to run the server on (default: 8082)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    
    args = parser.parse_args()
    MODEL_PATH = args.model
    
    logger.info(f"Starting Mouth Server on {args.host}:{args.port}")
    logger.info(f"Model: {MODEL_PATH}")
    
    # Pre-load model (optional, for faster first response)
    try:
        load_kokoro_model(MODEL_PATH)
    except Exception as e:
        logger.warning(f"Could not pre-load model: {e}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()