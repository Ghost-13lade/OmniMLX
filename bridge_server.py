#!/usr/bin/env python3
"""
Bridge Server - API Proxy/Gateway for External AI Services
Forwards OpenAI-compatible requests to external APIs (OpenAI, OpenRouter, etc.)
"""

import argparse
import logging
import json
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global configuration
TARGET_URL: str = ""
API_KEY: str = ""
SERVER_TYPE: str = "chat"  # chat, stt, tts
MODEL_NAME: str = ""

# Create FastAPI app
app = FastAPI(
    title="OmniMLX Bridge Server",
    description="API Proxy for External AI Services",
    version="1.0.0"
)


def get_headers() -> dict:
    """Build headers for external API requests."""
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "bridge",
        "type": SERVER_TYPE,
        "target": TARGET_URL,
        "model": MODEL_NAME
    }


@app.api_route("/v1/chat/completions", methods=["GET", "POST"])
async def proxy_chat_completions(request: Request):
    """
    Proxy chat completions requests to external API.
    Supports both regular and streaming responses.
    """
    if SERVER_TYPE not in ["chat", "vision"]:
        raise HTTPException(status_code=400, detail="This endpoint is not enabled for this bridge type")
    
    try:
        # Get request body
        body = await request.json()
        
        # Override model if specified in bridge config
        if MODEL_NAME:
            body["model"] = MODEL_NAME
        
        logger.info(f"Proxying chat request to {TARGET_URL}/chat/completions")
        logger.info(f"Model: {body.get('model', 'default')}")
        
        # Check if streaming is requested
        stream = body.get("stream", False)
        
        url = f"{TARGET_URL}/chat/completions"
        
        if stream:
            # Streaming response
            return StreamingResponse(
                stream_response(url, body),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    headers=get_headers(),
                    json=body
                )
                
                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=response.text
                    )
                
                return Response(
                    content=response.content,
                    media_type="application/json"
                )
    
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


async def stream_response(url: str, body: dict):
    """Stream SSE response from external API."""
    headers = get_headers()
    headers["Accept"] = "text/event-stream"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            url,
            headers=headers,
            json=body
        ) as response:
            if response.status_code != 200:
                error_body = await response.aread()
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_body.decode()
                )
            
            async for chunk in response.aiter_bytes():
                yield chunk


@app.post("/v1/audio/transcriptions")
async def proxy_audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    prompt: Optional[str] = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0)
):
    """
    Proxy audio transcription requests to external API.
    Handles multipart/form-data for file uploads.
    """
    if SERVER_TYPE != "stt":
        raise HTTPException(status_code=400, detail="This endpoint is not enabled for this bridge type")
    
    try:
        # Read file data
        file_data = await file.read()
        
        logger.info(f"Proxying transcription request to {TARGET_URL}/audio/transcriptions")
        logger.info(f"File: {file.filename} ({len(file_data)} bytes)")
        
        # Build multipart form
        files = {
            "file": (file.filename, file_data, file.content_type or "audio/wav")
        }
        data = {
            "model": MODEL_NAME or model,
        }
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt
        if response_format:
            data["response_format"] = response_format
        if temperature:
            data["temperature"] = str(temperature)
        
        url = f"{TARGET_URL}/audio/transcriptions"
        
        # Remove Content-Type header for multipart (httpx handles it)
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                headers=headers,
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.text
                )
            
            return Response(
                content=response.content,
                media_type="application/json"
            )
    
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/v1/audio/speech")
async def proxy_audio_speech(request: Request):
    """
    Proxy text-to-speech requests to external API.
    Returns audio bytes.
    """
    if SERVER_TYPE != "tts":
        raise HTTPException(status_code=400, detail="This endpoint is not enabled for this bridge type")
    
    try:
        # Get request body
        body = await request.json()
        
        # Override model if specified in bridge config
        if MODEL_NAME:
            body["model"] = MODEL_NAME
        
        logger.info(f"Proxying TTS request to {TARGET_URL}/audio/speech")
        logger.info(f"Model: {body.get('model', 'default')}")
        logger.info(f"Voice: {body.get('voice', 'default')}")
        
        url = f"{TARGET_URL}/audio/speech"
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                headers=get_headers(),
                json=body
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=response.text
                )
            
            # Return audio bytes with appropriate content type
            content_type = response.headers.get("content-type", "audio/mpeg")
            return Response(
                content=response.content,
                media_type=content_type
            )
    
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@app.api_route("/v1/models", methods=["GET"])
async def proxy_models(request: Request):
    """Proxy models listing request."""
    url = f"{TARGET_URL}/models"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, headers=get_headers())
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text
            )
        
        return Response(
            content=response.content,
            media_type="application/json"
        )


def main():
    """Main entry point."""
    global TARGET_URL, API_KEY, SERVER_TYPE, MODEL_NAME
    
    parser = argparse.ArgumentParser(description="OmniMLX Bridge Server (API Proxy)")
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to run the server on (default: 8080)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--target_url",
        type=str,
        required=True,
        help="Target API base URL (e.g., https://api.openai.com/v1)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for authentication"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["chat", "stt", "tts", "vision"],
        default="chat",
        help="Server type: chat (LLM), stt (Whisper), tts (TTS), vision"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name to use (overrides client-specified model)"
    )
    
    args = parser.parse_args()
    
    # Set global configuration
    TARGET_URL = args.target_url.rstrip("/")
    API_KEY = args.api_key
    SERVER_TYPE = args.type
    MODEL_NAME = args.model
    
    logger.info(f"Starting Bridge Server on {args.host}:{args.port}")
    logger.info(f"Target URL: {TARGET_URL}")
    logger.info(f"Server Type: {SERVER_TYPE}")
    logger.info(f"Model: {MODEL_NAME or '(client-specified)'}")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()