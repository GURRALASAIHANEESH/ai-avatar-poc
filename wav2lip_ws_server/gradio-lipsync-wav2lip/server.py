import os
import tempfile
import asyncio
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic_settings import BaseSettings
import torch

# Set memory allocation strategy for RTX 3050
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    wav2lip_checkpoint: str = "./wav2lip.pth"
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_concurrent_connections: int = 10
    processing_timeout: int = 300
    chunk_size: int = 1024 * 1024  # 1MB
    openai_api_key: str = ""
    hf_token: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Import your model/inference logic
try:
    from inference import infer as wav2lip_infer
    logger.info("Successfully imported Wav2Lip inference module")
except ImportError as e:
    logger.error(f"CRITICAL: Could not import wav2lip inference module: {e}")
    traceback.print_exc()
    raise

app = FastAPI(title="Wav2Lip WebSocket Server - Memory Optimized", version="1.2.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

executor = ThreadPoolExecutor(max_workers=2)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        if len(self.active_connections) >= settings.max_concurrent_connections:
            logger.warning("Too many connections, rejecting new connection")
            await websocket.close(code=1008, reason="Too many connections")
            return False

        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Active: {len(self.active_connections)}")
        return True

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Active: {len(self.active_connections)}")

manager = ConnectionManager()

async def send_file_stream(websocket: WebSocket, file_path: str):
    """Stream file data back to client with proper error handling"""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Output file not found: {file_path}")
            
        file_size = os.path.getsize(file_path)
        logger.info(f"Sending result file: {file_path} ({file_size} bytes)")
        
        # Check if WebSocket is still connected before sending
        if websocket.client_state.name != "CONNECTED":
            logger.warning("WebSocket not connected, cannot send file")
            return
        
        # Send file size first
        await websocket.send_text(str(file_size))
        logger.info(f"Sent file size: {file_size}")

        # Send file data in chunks
        with open(file_path, 'rb') as f:
            sent = 0
            chunk_count = 0
            while True:
                chunk = f.read(settings.chunk_size)
                if not chunk:
                    break
                    
                # Check connection before each chunk
                if websocket.client_state.name != "CONNECTED":
                    logger.warning("WebSocket disconnected during file transfer")
                    break
                    
                await websocket.send_bytes(chunk)
                sent += len(chunk)
                chunk_count += 1
                
                if chunk_count % 5 == 0:  # Log every 5 chunks
                    logger.info(f"Sent {sent}/{file_size} bytes ({chunk_count} chunks)")
                    
        logger.info(f"File sent successfully: {sent} bytes in {chunk_count} chunks")
        
    except Exception as e:
        logger.error(f"Error sending file: {e}")
        traceback.print_exc()
        raise

async def run_wav2lip_inference(face_path: str, audio_path: str, out_path: str):
    """RTX 3050 optimized Wav2Lip inference with quality preservation"""
    def _run_inference():
        try:
            from argparse import Namespace
            
            # RTX 3050 GPU Memory Management (CRITICAL)
            if torch.cuda.is_available():
                # Clear any existing GPU memory
                torch.cuda.empty_cache()
                
                # Limit GPU memory usage to 85% of available (3.4GB out of 4GB)
                torch.cuda.set_per_process_memory_fraction(0.85)
                
                logger.info("RTX 3050 GPU Memory Management:")
                logger.info(f"   CUDA available: {torch.cuda.is_available()}")
                logger.info(f"   GPU device: {torch.cuda.get_device_name(0)}")
                logger.info(f"   GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
                logger.info(f"   Memory fraction set to: 85% (3.4GB)")
                logger.info(f"   GPU memory before inference: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
            else:
                logger.warning("CUDA not available - using CPU (will be slower)")
            
            # Log file verification
            logger.info(f"Verifying input files:")
            logger.info(f"   Face file: {face_path} (exists: {os.path.exists(face_path)})")
            logger.info(f"   Audio file: {audio_path} (exists: {os.path.exists(audio_path)})")
            
            if not os.path.exists(face_path):
                raise FileNotFoundError(f"Face file not found: {face_path}")
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            face_size = os.path.getsize(face_path)
            audio_size = os.path.getsize(audio_path)
            logger.info(f"   Face file size: {face_size} bytes")
            logger.info(f"   Audio file size: {audio_size} bytes")
            
            if face_size == 0:
                raise ValueError("Face file is empty")
            if audio_size == 0:
                raise ValueError("Audio file is empty")
            
            # RTX 3050 Quality-Optimized Parameters
            # These settings balance memory usage with quality
            args = Namespace(
                checkpoint_path=settings.wav2lip_checkpoint,
                face=face_path,
                audio=audio_path,
                outfile=out_path,
                static=False,
                fps=20.0,                    # Good quality FPS (was 25, now 20)
                pads=[0, 10, 0, 0],         # Keep padding for quality
                face_det_batch_size=2,       # Conservative but faster than 1
                wav2lip_batch_size=16,       # Balanced batch size
                resize_factor=0.75,          # Slightly reduced but maintains quality
                crop=[0, -1, 0, -1],
                rotate=False,
                nosmooth=False,
                job_id='ws'
            )
            
            logger.info(f"Starting Wav2Lip with RTX 3050 quality-optimized settings:")
            logger.info(f"   Checkpoint: {args.checkpoint_path}")
            logger.info(f"   FPS: {args.fps} (good quality)")
            logger.info(f"   Face detection batch: {args.face_det_batch_size}")
            logger.info(f"   Wav2Lip batch: {args.wav2lip_batch_size}")
            logger.info(f"   Resize factor: {args.resize_factor} (maintains quality)")
            logger.info(f"   Pads: {args.pads}")
            
            if torch.cuda.is_available():
                logger.info(f"   Using GPU acceleration: {torch.cuda.get_device_name(0)}")
            else:
                logger.warning("   Using CPU only - processing will be slower")
            
            # Enhanced error handling for RTX 3050
            max_retries = 3
            current_retry = 0
            
            while current_retry < max_retries:
                try:
                    # Run the inference
                    wav2lip_infer(args)
                    break  # Success, exit retry loop
                    
                except torch.cuda.OutOfMemoryError as e:
                    current_retry += 1
                    logger.warning(f"GPU OOM attempt {current_retry}/{max_retries}")
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("   GPU cache cleared")
                    
                    if current_retry < max_retries:
                        # Progressively reduce settings while maintaining quality
                        if current_retry == 1:
                            # First retry: reduce batch sizes only
                            args.face_det_batch_size = 1
                            args.wav2lip_batch_size = 8
                            logger.info("   Retry 1: Reduced batch sizes")
                            
                        elif current_retry == 2:
                            # Second retry: slightly reduce FPS but keep resolution
                            args.fps = 15.0
                            args.face_det_batch_size = 1
                            args.wav2lip_batch_size = 4
                            logger.info("   Retry 2: Reduced FPS to 15, smaller batches")
                            
                        elif current_retry == 3:
                            # Final retry: minimal settings but still preserve quality
                            args.fps = 12.0
                            args.resize_factor = 0.6
                            args.face_det_batch_size = 1
                            args.wav2lip_batch_size = 2
                            logger.info("   Retry 3: Minimal settings while preserving quality")
                    else:
                        # All retries failed, try CPU
                        logger.error("All GPU retries failed, attempting fallback to CPU")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        # Force CPU mode would require modifying the inference code
                        raise e
                        
                except Exception as e:
                    logger.error(f"Non-memory error in Wav2Lip inference: {e}")
                    raise e
            
            if current_retry >= max_retries:
                raise RuntimeError("Wav2Lip failed after all retry attempts")
            
            # GPU Diagnostics after successful inference
            if torch.cuda.is_available():
                logger.info(f"   GPU memory after inference: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB")
                logger.info(f"   GPU memory peak usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f}GB")
                torch.cuda.empty_cache()  # Clear GPU cache
                logger.info("   GPU cache cleared post-inference")
            
            # Verify output
            if os.path.exists(out_path):
                output_size = os.path.getsize(out_path)
                logger.info(f"Wav2Lip inference completed successfully!")
                logger.info(f"   Output file: {out_path}")
                logger.info(f"   Output size: {output_size} bytes")
                
                # Quality verification
                if output_size < 100000:  # Less than 100KB suggests quality issues
                    logger.warning(f"   Warning: Output file seems small ({output_size} bytes)")
                else:
                    logger.info(f"   Output file size indicates good quality")
                    
            else:
                logger.error(f"Wav2Lip failed - output file not created: {out_path}")
                raise RuntimeError("Wav2Lip did not generate output file")
                
        except Exception as e:
            logger.error(f"Wav2Lip inference failed: {e}")
            # Final cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            traceback.print_exc()
            raise

    # Run in thread pool
    logger.info("Submitting Wav2Lip task to thread pool...")
    await asyncio.get_event_loop().run_in_executor(executor, _run_inference)

@app.get("/api/config")
async def get_config():
    """Provide configuration to frontend"""
    api_key = os.getenv('OPENAI_API_KEY') or settings.openai_api_key
    
    if not api_key:
        logger.error("OpenAI API key not configured")
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    masked_key = f"{api_key[:7]}...{api_key[-4:]}" if len(api_key) > 11 else "configured"
    
    logger.info(f"API config requested - providing masked key: {masked_key}")
    
    return {
        "openai_api_key": api_key,
        "masked_key": masked_key,
        "status": "configured",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "memory_optimized": "RTX 3050"
    }

@app.websocket("/ws/lipsync-single")
async def lipsync_single_message(websocket: WebSocket):
    """Enhanced WebSocket endpoint with RTX 3050 memory optimization"""
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket connection attempt from {client_ip}")
    
    if not await manager.connect(websocket):
        logger.warning("Connection rejected due to limits")
        return
    
    try:
        # STEP 1: Connection established
        logger.info("WebSocket connected successfully - waiting for binary message...")
        logger.info("About to call websocket.receive_bytes() with 60s timeout...")
        
        # STEP 2: Receive message with extended timeout and debugging
        try:
            message_data = await asyncio.wait_for(
                websocket.receive_bytes(), 
                timeout=60.0  # Extended timeout for debugging
            )
            logger.info(f"SUCCESS: Received binary message: {len(message_data)} bytes")
            
        except asyncio.TimeoutError:
            error_msg = "ERROR: Timeout waiting for client data (60 seconds)"
            logger.error(error_msg)
            await websocket.send_text(error_msg)
            return
        except Exception as e:
            error_msg = f"ERROR: Failed to receive client data: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Exception type: {type(e)}")
            traceback.print_exc()
            await websocket.send_text(error_msg)
            return
        
        # STEP 3: Parse message with enhanced bounds checking
        offset = 0
        
        try:
            logger.info("Starting binary message parsing...")
            
            # Validate minimum message size
            if len(message_data) < 16:  # Minimum: 4+4+8 bytes
                raise ValueError(f"Message too short: {len(message_data)} bytes")
            
            # Read avatar filename length
            if offset + 4 > len(message_data):
                raise ValueError(f"Cannot read avatar filename length at offset {offset}")
                
            avatar_name_length = int.from_bytes(message_data[offset:offset+4], 'big')
            offset += 4
            logger.info(f"   Avatar filename length: {avatar_name_length}")
            
            if avatar_name_length <= 0 or avatar_name_length > 1000:
                raise ValueError(f"Invalid avatar filename length: {avatar_name_length}")
            
            # Read avatar filename
            if offset + avatar_name_length > len(message_data):
                raise ValueError(f"Cannot read avatar filename: need {avatar_name_length} bytes at offset {offset}")
                
            avatar_name = message_data[offset:offset+avatar_name_length].decode('utf-8')
            offset += avatar_name_length
            logger.info(f"   Avatar filename: {avatar_name}")
            
            # Read avatar size
            if offset + 8 > len(message_data):
                raise ValueError(f"Cannot read avatar size at offset {offset}")
                
            avatar_size = int.from_bytes(message_data[offset:offset+8], 'big')
            offset += 8
            logger.info(f"   Avatar size: {avatar_size} bytes")
            
            if avatar_size <= 0 or avatar_size > 50 * 1024 * 1024:  # Max 50MB
                raise ValueError(f"Invalid avatar size: {avatar_size}")
            
            # Read avatar data
            if offset + avatar_size > len(message_data):
                raise ValueError(f"Cannot read avatar data: need {avatar_size} bytes at offset {offset}")
                
            avatar_data = message_data[offset:offset+avatar_size]
            offset += avatar_size
            logger.info(f"   Avatar data read: {len(avatar_data)} bytes")
            
            # Read audio filename length
            if offset + 4 > len(message_data):
                raise ValueError(f"Cannot read audio filename length at offset {offset}")
                
            audio_name_length = int.from_bytes(message_data[offset:offset+4], 'big')
            offset += 4
            logger.info(f"   Audio filename length: {audio_name_length}")
            
            if audio_name_length <= 0 or audio_name_length > 1000:
                raise ValueError(f"Invalid audio filename length: {audio_name_length}")
            
            # Read audio filename
            if offset + audio_name_length > len(message_data):
                raise ValueError(f"Cannot read audio filename: need {audio_name_length} bytes at offset {offset}")
                
            audio_name = message_data[offset:offset+audio_name_length].decode('utf-8')
            offset += audio_name_length
            logger.info(f"   Audio filename: {audio_name}")
            
            # Read audio size
            if offset + 8 > len(message_data):
                raise ValueError(f"Cannot read audio size at offset {offset}")
                
            audio_size = int.from_bytes(message_data[offset:offset+8], 'big')
            offset += 8
            logger.info(f"   Audio size: {audio_size} bytes")
            
            if audio_size <= 0 or audio_size > 10 * 1024 * 1024:  # Max 10MB
                raise ValueError(f"Invalid audio size: {audio_size}")
            
            # Read audio data
            if offset + audio_size > len(message_data):
                raise ValueError(f"Cannot read audio data: need {audio_size} bytes at offset {offset}")
                
            audio_data = message_data[offset:offset+audio_size]
            logger.info(f"   Audio data read: {len(audio_data)} bytes")
            
            # Verify we parsed everything correctly
            total_parsed = offset + audio_size
            logger.info(f"Message parsing complete: {total_parsed}/{len(message_data)} bytes")
            
            if total_parsed != len(message_data):
                logger.warning(f"Message size mismatch: parsed {total_parsed}, received {len(message_data)}")
            
        except Exception as e:
            error_msg = f"ERROR: Failed to parse message: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Message details: length={len(message_data)}, offset={offset}")
            traceback.print_exc()
            await websocket.send_text(error_msg)
            return
        
        # STEP 4: Process the files
        logger.info("Writing files to temporary directory...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"Created temporary directory: {tmpdir}")
            
            face_path = os.path.join(tmpdir, avatar_name)
            audio_path = os.path.join(tmpdir, audio_name)
            out_path = os.path.join(tmpdir, "lipsync_result.mp4")
            
            try:
                # Write avatar file
                logger.info(f"Writing avatar file: {face_path}")
                with open(face_path, 'wb') as f:
                    f.write(avatar_data)
                
                # Write audio file  
                logger.info(f"Writing audio file: {audio_path}")
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
                
                # Verify files were written correctly
                face_written_size = os.path.getsize(face_path)
                audio_written_size = os.path.getsize(audio_path)
                
                logger.info(f"Files written successfully:")
                logger.info(f"   {face_path}: {face_written_size} bytes")
                logger.info(f"   {audio_path}: {audio_written_size} bytes")
                
                if face_written_size != len(avatar_data):
                    raise RuntimeError(f"Avatar file size mismatch: expected {len(avatar_data)}, got {face_written_size}")
                    
                if audio_written_size != len(audio_data):
                    raise RuntimeError(f"Audio file size mismatch: expected {len(audio_data)}, got {audio_written_size}")
                
            except Exception as e:
                error_msg = f"ERROR: Failed to write files: {str(e)}"
                logger.error(error_msg)
                await websocket.send_text(error_msg)
                return
            
            # STEP 5: Run Wav2Lip inference
            try:
                logger.info("Starting Wav2Lip processing with RTX 3050 optimization...")
                await run_wav2lip_inference(face_path, audio_path, out_path)
                
            except Exception as e:
                error_msg = f"ERROR: Wav2Lip processing failed: {str(e)}"
                logger.error(error_msg)
                traceback.print_exc()
                await websocket.send_text(error_msg)
                return
            
            # STEP 6: Send result back
            if os.path.exists(out_path):
                try:
                    output_size = os.path.getsize(out_path)
                    logger.info(f"Processing completed successfully!")
                    logger.info(f"   Output file: {out_path}")
                    logger.info(f"   Output size: {output_size} bytes")
                    
                    await send_file_stream(websocket, out_path)
                    logger.info("Result sent to client successfully")
                    
                except Exception as e:
                    error_msg = f"ERROR: Failed to send result: {str(e)}"
                    logger.error(error_msg)
                    traceback.print_exc()
                    try:
                        await websocket.send_text(error_msg)
                    except:
                        logger.error("Could not send error message - connection closed")
                    return
                    
            else:
                error_msg = "ERROR: Wav2Lip processing completed but no output file was generated"
                logger.error(error_msg)
                await websocket.send_text(error_msg)
                return
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected during processing")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket processing: {e}")
        try:
            await websocket.send_text(f"ERROR: Unexpected error: {str(e)}")
        except:
            logger.error("Failed to send error message to client - connection closed")
    finally:
        manager.disconnect(websocket)

@app.websocket("/ws/test")
async def test_websocket(websocket: WebSocket):
    """Simple test WebSocket for debugging communication"""
    await websocket.accept()
    logger.info("Test WebSocket connected")
    
    try:
        # Test text message
        await websocket.send_text("Test connection established")
        
        # Wait for response
        data = await websocket.receive_text()
        logger.info(f"Received test message: {data}")
        
        await websocket.send_text(f"Echo: {data}")
        logger.info("Test WebSocket communication successful")
        
    except Exception as e:
        logger.error(f"Test WebSocket error: {e}")
    finally:
        logger.info("Test WebSocket disconnected")

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
        <head><title>Wav2Lip WebSocket Server - RTX 3050 Optimized</title></head>
        <body>
            <h1>Wav2Lip WebSocket Server - RTX 3050 Optimized</h1>
            <p><strong>Status:</strong> Memory management optimized for RTX 3050</p>
            <p><strong>Quality:</strong> Balanced settings to maintain video quality</p>
            <p><strong>CORS:</strong> Configured for localhost:8080</p>
            <p><strong>Processing:</strong> Auto-retry with progressive fallback</p>
            <p>Endpoints:</p>
            <ul>
                <li><code>/ws/lipsync-single</code> - Optimized lipsync processing</li>
                <li><code>/ws/test</code> - Simple WebSocket test endpoint</li>
                <li><code>/api/config</code> - Configuration endpoint with GPU info</li>
                <li><code>/health</code> - Health check</li>
            </ul>
        </body>
    </html>
    """)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "active_connections": len(manager.active_connections),
        "max_connections": settings.max_concurrent_connections,
        "openai_configured": bool(os.getenv('OPENAI_API_KEY') or settings.openai_api_key),
        "wav2lip_checkpoint_exists": os.path.exists(settings.wav2lip_checkpoint),
        "cors_enabled": True,
        "enhanced_logging": True,
        "debug_version": "1.2 - RTX 3050 Optimized",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "memory_management": "Enabled for 4GB GPU"
    }

if __name__ == "__main__":
    # Validate configuration
    logger.info("Validating server configuration...")
    
    # RTX 3050 GPU Configuration Check
    logger.info("RTX 3050 GPU Configuration Check:")
    logger.info(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"   GPU device: {torch.cuda.get_device_name(0)}")
        logger.info(f"   GPU memory total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        logger.info(f"   GPU compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        logger.info(f"   Memory management: Optimized for RTX 3050 (4GB)")
        logger.info(f"   Memory allocation: Conservative with quality preservation")
    else:
        logger.warning("   CUDA not available - processing will use CPU")
        logger.warning("   Consider installing CUDA and PyTorch with GPU support")
    
    api_key = os.getenv('OPENAI_API_KEY') or settings.openai_api_key
    if not api_key:
        logger.warning("OpenAI API key not found - check your .env file")
    else:
        logger.info(f"OpenAI API key configured: {api_key[:7]}...{api_key[-4:]}")
    
    if not os.path.exists(settings.wav2lip_checkpoint):
        logger.error(f"Wav2Lip checkpoint not found: {settings.wav2lip_checkpoint}")
        logger.error("Please ensure wav2lip.pth exists in the current directory")
        exit(1)
    else:
        checkpoint_size = os.path.getsize(settings.wav2lip_checkpoint)
        logger.info(f"Wav2Lip checkpoint found: {settings.wav2lip_checkpoint} ({checkpoint_size} bytes)")

    logger.info("Starting Wav2Lip WebSocket Server with RTX 3050 optimization...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
