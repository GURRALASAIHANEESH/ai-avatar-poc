import os
import tempfile
import asyncio
import logging
import traceback
import json
import wave
import io
import struct
import subprocess
import sys
import shutil
import warnings
import time
import uuid
import psutil
import gc
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Set, Optional, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic_settings import BaseSettings
from pydantic import BaseModel
import torch
import requests
from starlette.websockets import WebSocketState

# SPEECH-OPTIMIZED ULTRA-LOW MEMORY: Configure PyTorch for minimal memory but better speech
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,expandable_segments:True'

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_per_process_memory_fraction(0.4)  # Keep 40% for stability
    torch.set_num_threads(2)
    logging.info(f"Speech-Optimized Ultra-Low Memory: {torch.cuda.get_device_name(0)}")
    logging.info(f"GPU Memory Limit: {torch.cuda.get_device_properties(0).total_memory * 0.4 / 1024**3:.1f} GB")
else:
    logging.warning("CUDA not available - running on CPU only")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

if sys.platform == "win32":
    try:
        import windows_config
        logger.info("Windows configurations applied.")
    except ImportError:
        logger.warning("windows_config.py not found.")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    if 'weights_only' in kwargs:
        del kwargs['weights_only']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return original_torch_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

torch.load = patched_torch_load

class Settings(BaseSettings):
    wav2lip_checkpoint: str = "./wav2lip.pth"
    max_concurrent_connections: int = 2
    openai_api_key: str = ""
    avatar_video_path: str = "./avatar_sample_green.mp4"
    silence_threshold: int = 120  # SPEECH FIX: Lower threshold for better detection

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

try:
    from inference import infer as wav2lip_infer
    logger.info("Wav2Lip inference module loaded successfully")
except ImportError as e:
    logger.warning(f"Wav2Lip inference module not available: {e}")
    wav2lip_infer = None

app = FastAPI(title="AI Avatar Speech-Optimized Ultra-Low Memory System", version="24.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

executor = ThreadPoolExecutor(max_workers=1)
video_generation_jobs: Dict[str, dict] = {}
video_storage_dir = os.path.join(tempfile.gettempdir(), "ai_avatar_videos")
os.makedirs(video_storage_dir, exist_ok=True)

class WindowsFileManager:
    def __init__(self):
        self.locked_files = set()
        self.temp_dirs = []
        self.protected_files = set()
    
    def protect_file(self, file_path: str):
        self.protected_files.add(os.path.abspath(file_path))
        logger.info(f"Protected file: {file_path}")
    
    def unprotect_file(self, file_path: str):
        abs_path = os.path.abspath(file_path)
        self.protected_files.discard(abs_path)
        logger.info(f"Unprotected file: {file_path}")
    
    def force_unlock_file(self, file_path: str, max_attempts: int = 2) -> bool:
        try:
            if not os.path.exists(file_path):
                return True
            for attempt in range(max_attempts):
                try:
                    with open(file_path, 'r+b') as f:
                        pass
                    return True
                except IOError:
                    if attempt < max_attempts - 1:
                        time.sleep(0.1)
            return False
        except Exception as e:
            logger.error(f"Error unlocking file: {e}")
            return False
    
    def create_safe_temp_dir(self, prefix: str = "ai_avatar_") -> str:
        try:
            base_dir = os.getcwd()
            safe_temp = os.path.join(base_dir, "temp_processing", f"{prefix}{int(time.time())}")
            os.makedirs(safe_temp, exist_ok=True)
            self.temp_dirs.append(safe_temp)
            return safe_temp
        except Exception as e:
            logger.error(f"Failed to create temp directory: {e}")
            return tempfile.mkdtemp()
    
    def safe_file_copy(self, src: str, dst: str, max_attempts: int = 2) -> bool:
        for attempt in range(max_attempts):
            try:
                if os.path.exists(src):
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(src, dst)
                    return True
                return False
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(0.2)
                else:
                    logger.error(f"Copy failed: {e}")
                    return False
        return False
    
    def cleanup_temp_dirs(self, force: bool = False):
        for temp_dir in self.temp_dirs[:]:
            try:
                if os.path.exists(temp_dir) and (force or not any(
                    pf.startswith(os.path.abspath(temp_dir)) for pf in self.protected_files
                )):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    self.temp_dirs.remove(temp_dir)
            except Exception:
                pass

file_manager = WindowsFileManager()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_states: Dict[str, bool] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_states[session_id] = True
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        self.connection_states[session_id] = False

    def is_connected(self, session_id: str) -> bool:
        if session_id not in self.connection_states or not self.connection_states[session_id]:
            return False
        ws = self.active_connections.get(session_id)
        if not ws:
            return False
        try:
            return ws.client_state == WebSocketState.CONNECTED
        except Exception:
            self.connection_states[session_id] = False
            return False

    async def notify_session(self, session_id: str, message: dict):
        if self.is_connected(session_id):
            try:
                websocket = self.active_connections[session_id]
                await asyncio.wait_for(
                    websocket.send_text(json.dumps(message)), 
                    timeout=5.0
                )
                return True
            except Exception as e:
                logger.warning(f"Send failed to {session_id}: {e}")
                self.connection_states[session_id] = False
                return False
        return False

manager = ConnectionManager()

class SpeechOptimizedAudioProcessor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.speech_buffer = bytearray()
        self.last_speech_time = 0
        self.chunk_count = 0
        self.temp_dir = file_manager.create_safe_temp_dir(f"audio_{session_id}_")
        self.speech_started = False
        self.min_speech_duration = 0.8  # SPEECH FIX: Minimum duration for better accuracy
        self.noise_floor = 0  # For adaptive noise detection
        self.speech_energy_history = []

    async def process_audio_chunk(self, chunk_data: bytes):
        has_speech, rms, confidence = self._advanced_speech_detection(chunk_data)
        
        # SPEECH FIX: Log detailed audio analysis for debugging
        if self.chunk_count % 100 == 0:  # Every 100 chunks (about 6 seconds)
            logger.info(f"Audio Analysis - RMS: {rms:.2f}, Confidence: {confidence:.2f}, Speech: {has_speech}")
        
        if has_speech:
            if not self.speech_started:
                self.speech_started = True
                logger.info(f"SPEECH STARTED for {self.session_id} - RMS: {rms:.2f}")
            
            self.last_speech_time = time.time()
            self.speech_buffer.extend(chunk_data)
            
            # SPEECH FIX: Optimal buffer size for accuracy (3.5 seconds)
            if len(self.speech_buffer) >= 112000:  # 3.5 seconds at 16kHz
                await self._process_speech_buffer("buffer_full")
                
        elif self.speech_buffer and self.speech_started:
            silence_duration = time.time() - self.last_speech_time
            buffer_duration = len(self.speech_buffer) / 16000
            
            # SPEECH FIX: Intelligent silence detection with context
            should_process = False
            
            if buffer_duration >= 4.0:  # Long speech - process immediately
                should_process = True
                reason = "long_speech"
            elif buffer_duration >= self.min_speech_duration and silence_duration > 2.0:  # Standard processing
                should_process = True
                reason = "normal_speech"
            elif buffer_duration >= 0.3 and silence_duration > 3.0:  # Short but definite speech
                should_process = True
                reason = "short_speech"
            
            if should_process:
                logger.info(f"Processing speech: {buffer_duration:.2f}s audio, {silence_duration:.2f}s silence, reason: {reason}")
                await self._process_speech_buffer(reason)

        self.chunk_count += 1

    def _advanced_speech_detection(self, audio_data: bytes) -> tuple[bool, float, float]:
        """SPEECH FIX: Advanced multi-factor speech detection with adaptive noise floor"""
        try:
            samples = struct.unpack(f'<{len(audio_data)//2}h', audio_data)
            
            # Basic energy calculations
            sum_squares = sum(s * s for s in samples)
            rms = (sum_squares / len(samples)) ** 0.5
            max_amplitude = max(abs(s) for s in samples)
            
            # Zero crossing rate (voice characteristic)
            zero_crossings = sum(1 for i in range(1, len(samples)) 
                               if (samples[i] >= 0) != (samples[i-1] >= 0))
            zcr = zero_crossings / len(samples) if len(samples) > 1 else 0
            
            # Spectral characteristics (simplified)
            energy_ratio = rms / max_amplitude if max_amplitude > 0 else 0
            
            # SPEECH FIX: Adaptive noise floor
            self.speech_energy_history.append(rms)
            if len(self.speech_energy_history) > 100:  # Keep last 100 samples
                self.speech_energy_history.pop(0)
            
            if len(self.speech_energy_history) >= 10:
                sorted_energy = sorted(self.speech_energy_history)
                self.noise_floor = sorted_energy[len(sorted_energy) // 4]  # 25th percentile
            
            # SPEECH FIX: Dynamic thresholds based on noise floor
            adaptive_threshold = max(settings.silence_threshold, self.noise_floor * 3)
            peak_threshold = adaptive_threshold * 15
            
            # SPEECH FIX: Multi-factor confidence scoring
            confidence_factors = []
            
            # Energy confidence
            if rms > adaptive_threshold:
                confidence_factors.append(min(rms / adaptive_threshold, 3.0))
            else:
                confidence_factors.append(0)
            
            # Peak confidence
            if max_amplitude > peak_threshold:
                confidence_factors.append(min(max_amplitude / peak_threshold, 2.0))
            else:
                confidence_factors.append(0)
            
            # ZCR confidence (human voice typically 0.01-0.35)
            if 0.01 <= zcr <= 0.35:
                confidence_factors.append(1.5)
            elif 0.005 <= zcr <= 0.5:
                confidence_factors.append(1.0)
            else:
                confidence_factors.append(0.3)
            
            # Energy distribution confidence
            if 0.1 <= energy_ratio <= 0.8:
                confidence_factors.append(1.2)
            else:
                confidence_factors.append(0.8)
            
            # Calculate overall confidence
            confidence = sum(confidence_factors) / len(confidence_factors)
            
            # SPEECH FIX: Decision logic with hysteresis
            if self.speech_started:
                # Once speech started, lower threshold to continue
                is_speech = confidence > 0.8
            else:
                # Higher threshold to start speech detection
                is_speech = confidence > 1.2
            
            return is_speech, rms, confidence
            
        except Exception as e:
            logger.warning(f"Advanced speech detection error: {e}")
            return False, 0.0, 0.0

    async def _process_speech_buffer(self, reason: str = "unknown"):
        if not self.speech_buffer:
            return
        
        chunk_to_process = bytes(self.speech_buffer)
        self.speech_buffer = bytearray()
        self.speech_started = False
        self.chunk_count += 1
        
        # SPEECH FIX: Enhanced audio file creation with metadata
        audio_path = os.path.join(self.temp_dir, f"speech_{self.chunk_count}_{reason}.wav")
        await self._create_professional_wav_file(chunk_to_process, audio_path)
        
        duration = len(chunk_to_process) / (16000 * 2)
        logger.info(f"PROCESSING SPEECH #{self.chunk_count}: {duration:.2f}s ({reason})")
        
        transcribed_text = await self._professional_transcribe_audio(audio_path, duration, reason)
        if not transcribed_text:
            logger.warning(f"No transcription for {duration:.2f}s audio ({reason})")
            return
        
        logger.info(f"TRANSCRIPTION SUCCESS: '{transcribed_text}' (from {duration:.2f}s audio)")
        
        ai_response = await self._get_chatgpt_response(transcribed_text)
        
        await manager.notify_session(self.session_id, {
            "type": "text_response",
            "user_text": transcribed_text,
            "ai_response": ai_response,
            "audio_duration": f"{duration:.2f}s",
            "processing_reason": reason
        })
        
        tts_audio_path = await self._create_tts_audio(ai_response)
        if tts_audio_path:
            await self._initiate_video_generation(transcribed_text, ai_response, tts_audio_path)

    async def _create_professional_wav_file(self, data: bytes, path: str):
        """SPEECH FIX: Professional-grade WAV file creation with validation"""
        try:
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # 16kHz
                wf.setcomptype('NONE', 'not compressed')
                wf.writeframes(data)
            
            # SPEECH FIX: Validate created file
            if os.path.exists(path):
                file_size = os.path.getsize(path)
                if file_size > 44:  # More than just WAV header
                    # Test if file can be read
                    try:
                        with wave.open(path, 'rb') as test_wf:
                            frames = test_wf.getnframes()
                            sample_rate = test_wf.getframerate()
                            duration = frames / sample_rate
                        logger.info(f"WAV file validated: {file_size} bytes, {duration:.2f}s")
                    except Exception as e:
                        logger.error(f"WAV file validation failed: {e}")
                        raise
                else:
                    raise ValueError(f"WAV file too small: {file_size} bytes")
            else:
                raise FileNotFoundError("WAV file not created")
                
        except Exception as e:
            logger.error(f"Professional WAV creation error: {e}")
            raise

    async def _professional_transcribe_audio(self, path: str, duration: float, reason: str) -> Optional[str]:
        """SPEECH FIX: Professional transcription with extensive validation and context"""
        if not os.path.exists(path):
            logger.error(f"Audio file not found: {path}")
            return None
        
        file_size = os.path.getsize(path)
        if file_size <= 44:
            logger.error(f"Audio file too small: {file_size} bytes")
            return None
        
        headers = {'Authorization': f'Bearer {settings.openai_api_key}'}
        
        # SPEECH FIX: Enhanced context prompts based on processing reason
        context_prompts = {
            "long_speech": "This is a complete English sentence or question about technology, programming, science, or general topics.",
            "normal_speech": "This is clear English speech, likely a question or statement about technology or general topics.",
            "short_speech": "This is a brief English word or phrase, possibly a question like 'What is Python' or a short command.",
            "buffer_full": "This is continuous English speech that may contain complete thoughts about technology, programming, or general questions."
        }
        
        prompt = context_prompts.get(reason, "This is clear English speech about technology or general topics.")
        
        files = {
            'file': (os.path.basename(path), open(path, 'rb'), 'audio/wav'),
            'model': (None, 'whisper-1'),
            'language': (None, 'en'),
            'prompt': (None, prompt),
            'temperature': (None, '0.0'),  # Most deterministic
            'response_format': (None, 'verbose_json')  # Get confidence scores
        }
        
        try:
            logger.info(f"TRANSCRIBING: {file_size} bytes, {duration:.2f}s, reason: {reason}")
            
            resp = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                files=files,
                headers=headers,
                timeout=40  # Longer timeout for better processing
            )
            
            files['file'][1].close()
            
            if resp.ok:
                result = resp.json()
                transcribed_text = result.get('text', '').strip()
                
                # SPEECH FIX: Enhanced text processing and validation
                if transcribed_text and len(transcribed_text) > 0:
                    # Clean up transcription
                    transcribed_text = ' '.join(transcribed_text.split())
                    
                    # Remove common transcription artifacts
                    artifacts = ['[BLANK_AUDIO]', '[NOISE]', '[SILENCE]', '...', 'uh', 'um', 'ah']
                    for artifact in artifacts:
                        transcribed_text = transcribed_text.replace(artifact, '').strip()
                    
                    if len(transcribed_text) > 0:
                        # Log additional info if available
                        if 'segments' in result and result['segments']:
                            confidence = result['segments'][0].get('avg_logprob', 'unknown')
                            logger.info(f"TRANSCRIPTION: '{transcribed_text}' (confidence: {confidence})")
                        else:
                            logger.info(f"TRANSCRIPTION: '{transcribed_text}'")
                        
                        return transcribed_text
                    else:
                        logger.warning("Transcription cleaned to empty string")
                        return None
                else:
                    logger.warning("Empty or invalid transcription result")
                    return None
            else:
                logger.error(f"Transcription API error: {resp.status_code} - {resp.text}")
                return None
                
        except Exception as e:
            logger.error(f"Professional transcription failed: {e}")
            try:
                files['file'][1].close()
            except:
                pass
            return None

    async def _initiate_video_generation(self, user_text: str, ai_response: str, audio_path: str):
        job_id = str(uuid.uuid4())
        video_generation_jobs[job_id] = {
            "status": "started",
            "session_id": self.session_id,
            "user_text": user_text,
            "ai_response": ai_response,
            "audio_path": audio_path,
            "created_at": time.time(),
            "progress": 0,
            "speech_optimized": True
        }
        
        await manager.notify_session(self.session_id, {
            "type": "video_generation_started",
            "job_id": job_id,
            "speech_optimized": True
        })
        
        asyncio.create_task(self._generate_ultra_low_memory_video(job_id, audio_path))

    async def _generate_ultra_low_memory_video(self, job_id: str, audio_path: str):
        job_info = video_generation_jobs[job_id]
        job_info.update({"status": "generating", "progress": 10})
        
        await manager.notify_session(self.session_id, {
            "type": "video_generation_progress",
            "job_id": job_id,
            "progress": 10,
            "message": "Speech-optimized ultra-low memory processing"
        })
        
        safe_temp_dir = None
        permanent_audio_path = None
        
        try:
            # Ultra-low memory: Aggressive cache clearing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            avatar_path = settings.avatar_video_path
            safe_temp_dir = file_manager.create_safe_temp_dir(f"video_{job_id[:8]}_")
            video_output_path = os.path.join(safe_temp_dir, f"output.mp4")
            final_video_path = os.path.join(video_storage_dir, f"{job_id}.mp4")
            
            permanent_audio_path = os.path.join(safe_temp_dir, f"audio.wav")
            
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            
            if not file_manager.safe_file_copy(audio_path, permanent_audio_path):
                raise Exception("Audio copy failed")
            
            file_manager.protect_file(permanent_audio_path)
            
            if not os.path.exists(permanent_audio_path) or os.path.getsize(permanent_audio_path) == 0:
                raise FileNotFoundError(f"Audio copy invalid: {permanent_audio_path}")
            
            from argparse import Namespace
            
            # Ultra-low memory settings with speech optimization
            args = Namespace(
                checkpoint_path=settings.wav2lip_checkpoint,
                face=avatar_path,
                audio=permanent_audio_path,
                outfile=video_output_path,
                static=False,
                fps=20.0,  # Keep low FPS for memory
                pads=[0, 10, 0, 0],
                face_det_batch_size=1,
                wav2lip_batch_size=1,  # Keep minimal for memory
                resize_factor=2,  # Keep image resizing for memory
                crop=[0, -1, 0, -1],
                rotate=False,
                nosmooth=False
            )
            
            job_info["progress"] = 30
            await manager.notify_session(self.session_id, {
                "type": "video_generation_progress",
                "job_id": job_id,
                "progress": 30,
                "message": "Processing with speech-optimized settings"
            })
            
            # Process video
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, wav2lip_infer, args)
            
            # Immediate cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            job_info["progress"] = 90
            await asyncio.sleep(0.2)
            
            if os.path.exists(video_output_path):
                video_size = os.path.getsize(video_output_path)
                if video_size > 0:
                    file_manager.force_unlock_file(video_output_path)
                    
                    if file_manager.safe_file_copy(video_output_path, final_video_path):
                        job_info.update({
                            "status": "completed",
                            "progress": 100,
                            "video_path": final_video_path,
                            "video_size": video_size,
                            "speech_optimized": True
                        })
                        
                        await manager.notify_session(self.session_id, {
                            "type": "video_generation_completed",
                            "job_id": job_id,
                            "download_url": f"/api/video/{job_id}",
                            "video_size": video_size,
                            "user_text": job_info.get("user_text", ""),
                            "ai_response": job_info.get("ai_response", ""),
                            "speech_optimized": True
                        })
                        
                        logger.info(f"Speech-optimized video completed: {video_size} bytes")
                    else:
                        raise Exception("Video copy failed")
                else:
                    raise Exception("Empty video file")
            else:
                raise Exception(f"Video not found: {video_output_path}")
        
        except Exception as e:
            logger.error(f"Speech-optimized video failed: {e}")
            job_info.update({"status": "error", "error": str(e)})
            
            await manager.notify_session(self.session_id, {
                "type": "video_generation_failed",
                "job_id": job_id,
                "error": str(e)
            })
        
        finally:
            if permanent_audio_path:
                file_manager.unprotect_file(permanent_audio_path)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            if safe_temp_dir:
                asyncio.create_task(self._delayed_cleanup(safe_temp_dir))

    async def _delayed_cleanup(self, temp_dir: str):
        try:
            await asyncio.sleep(0.5)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    async def _get_chatgpt_response(self, text: str) -> str:
        headers = {
            'Authorization': f'Bearer {settings.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [
                {'role': 'system', 'content': 'You are an AI assistant providing comprehensive responses of 80-150 words.'},
                {'role': 'user', 'content': text}
            ],
            'max_tokens': 300,
            'temperature': 0.7
        }
        
        try:
            resp = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=25
            )
            return resp.json()['choices'][0]['message']['content'].strip() if resp.ok else "AI service error."
        except Exception as e:
            logger.error(f"ChatGPT failed: {e}")
            return "Error processing request."

    async def _create_tts_audio(self, text: str) -> Optional[str]:
        headers = {
            'Authorization': f'Bearer {settings.openai_api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'tts-1',
            'input': text,
            'voice': 'alloy',
            'response_format': 'wav',
            'speed': 1.0
        }
        
        path = os.path.join(self.temp_dir, f"tts_{self.chunk_count}.wav")
        
        try:
            resp = requests.post(
                'https://api.openai.com/v1/audio/speech',
                headers=headers,
                json=data,
                timeout=25
            )
            
            if resp.ok:
                with open(path, 'wb') as f:
                    f.write(resp.content)
                return path
            return None
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return None

    def cleanup(self):
        pass

@app.websocket("/ws/audio-stream/{session_id}")
async def audio_stream_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    processor = SpeechOptimizedAudioProcessor(session_id)
    
    try:
        await manager.notify_session(session_id, {
            "type": "connection_established",
            "session_id": session_id,
            "message": "Speech-optimized ultra-low memory AI Avatar system ready",
            "speech_recognition": "Advanced multi-factor detection with adaptive noise floor",
            "gpu_available": torch.cuda.is_available()
        })
        
        while manager.is_connected(session_id):
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=12.0)
                
                if data.get("type") == "websocket.disconnect":
                    break
                
                if "bytes" in data:
                    chunk_data = data["bytes"]
                    if len(chunk_data) == 0:
                        break
                    await processor.process_audio_chunk(chunk_data)
                
                elif "text" in data:
                    try:
                        text_data = json.loads(data["text"])
                        if text_data.get("type") == "client_keepalive":
                            await manager.notify_session(session_id, {
                                "type": "server_pong",
                                "timestamp": time.time(),
                                "message": "Speech-optimized server active"
                            })
                    except json.JSONDecodeError:
                        pass
                
            except asyncio.TimeoutError:
                if not manager.is_connected(session_id):
                    break
                continue
            except (WebSocketDisconnect, RuntimeError):
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
        
    except Exception as e:
        logger.error(f"Fatal WebSocket error: {e}")
    
    finally:
        try:
            processor.cleanup()
        except:
            pass
        manager.disconnect(session_id)

@app.get("/api/video/{job_id}")
async def download_video(job_id: str):
    job = video_generation_jobs.get(job_id)
    if not job or job["status"] != "completed" or not job.get("video_path"):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        job["video_path"],
        media_type="video/mp4",
        filename=f"speech_optimized_{job_id}.mp4"
    )

@app.get("/api/video/{job_id}/status")
async def get_video_status(job_id: str):
    if job_id not in video_generation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return video_generation_jobs[job_id]

@app.get("/api/config")
async def get_config():
    return {
        "version": "24.0.0-Speech-Optimized-Ultra-Low-Memory",
        "gpu_available": torch.cuda.is_available(),
        "status": "configured",
        "speech_recognition": "Advanced multi-factor with adaptive noise floor",
        "memory_optimization": "Ultra-low memory with speech priority",
        "audio_processing": "3.5s buffers with intelligent silence detection"
    }

@app.get("/")
def index():
    return HTMLResponse("""
    <html>
        <head><title>AI Avatar Speech-Optimized Ultra-Low Memory System v24.0</title></head>
        <body style="background:#1a1a1a;color:#e0e0e0;font-family:'Segoe UI',sans-serif;text-align:center;padding:20px;">
            <h1>AI Avatar Speech-Optimized Ultra-Low Memory System v24.0</h1>
            <div style="margin:20px;padding:20px;background:#2a2a2a;border-radius:8px;">
                <h3>Speech Recognition Optimizations</h3>
                <ul style="text-align:left;max-width:800px;margin:0 auto;color:#ccc;">
                    <li><strong>Advanced Speech Detection:</strong> Multi-factor analysis with confidence scoring</li>
                    <li><strong>Adaptive Noise Floor:</strong> Automatically adjusts to environment</li>
                    <li><strong>Optimal Buffer Sizes:</strong> 3.5 seconds for complete phrase capture</li>
                    <li><strong>Intelligent Silence Detection:</strong> Context-aware processing triggers</li>
                    <li><strong>Professional WAV Creation:</strong> Validated audio files for transcription</li>
                    <li><strong>Enhanced Context Prompts:</strong> Reason-specific transcription hints</li>
                    <li><strong>Verbose JSON Response:</strong> Confidence scores for quality monitoring</li>
                    <li><strong>Ultra-Low Memory:</strong> 40% GPU limit with resize_factor=2</li>
                </ul>
            </div>
            <p><strong>Status:</strong> Optimized for accurate "What is Python" recognition while maintaining memory efficiency</p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    logger.info("Starting Speech-Optimized Ultra-Low Memory AI Avatar System v24.0")
    
    if torch.cuda.is_available():
        logger.info(f"Speech-Optimized Memory: {torch.cuda.get_device_name(0)}")
        logger.info("Advanced speech recognition with ultra-low memory optimization")
    
    api_key = settings.openai_api_key
    if not api_key:
        logger.error("OpenAI API key not found")
        exit(1)

    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False, log_level="info")
