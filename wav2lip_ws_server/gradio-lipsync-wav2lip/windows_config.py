import os
import sys
import tempfile
import subprocess
import logging
import psutil
import time
import gc

logger = logging.getLogger(__name__)

def configure_windows_environment():
    """Configure Windows environment for optimal video processing and file handling"""
    
    try:
        logger.info("🔧 Configuring Windows environment for AI Avatar system...")

        # Set custom FFmpeg path first for priority
        custom_ffmpeg_path = r"D:\realtime_avatar\ffmpeg-7.0.2-full_build\bin"
        if os.path.exists(custom_ffmpeg_path):
            current_path = os.environ.get('PATH', '')
            if custom_ffmpeg_path not in current_path:
                os.environ['PATH'] = custom_ffmpeg_path + os.pathsep + current_path
                logger.info(f"✅ Added custom FFmpeg path: {custom_ffmpeg_path}")
        else:
            logger.warning(f"⚠️ Custom FFmpeg path not found: {custom_ffmpeg_path}")
        
        # Set temporary directory to avoid long path issues
        temp_dir = os.path.join(os.getcwd(), "temp_video")
        os.makedirs(temp_dir, exist_ok=True)
        os.environ['TEMP'] = temp_dir
        os.environ['TMP'] = temp_dir
        logger.info(f"✅ Set Windows temp directory: {temp_dir}")
        
        # Set FFmpeg path if not found by custom path
        ffmpeg_paths = [
            r"C:\ffmpeg\bin", r"C:\Program Files\ffmpeg\bin",
            os.path.join(os.getcwd(), "ffmpeg", "bin")
        ]
        
        ffmpeg_found = custom_ffmpeg_path in os.environ.get('PATH', '')
        if not ffmpeg_found:
            for path in ffmpeg_paths:
                if os.path.exists(path):
                    current_path = os.environ.get('PATH', '')
                    if path not in current_path:
                        os.environ['PATH'] = path + os.pathsep + current_path
                        logger.info(f"✅ Added FFmpeg path: {path}")
                        ffmpeg_found = True
                    break
        
        if not ffmpeg_found:
            logger.warning("⚠️ FFmpeg not found in common locations")
        
        # Configure Windows-specific environment variables for video processing
        windows_env_vars = {
            'OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS': '0',
            'OPENCV_VIDEOIO_PRIORITY_MSMF': '0',
            'OPENCV_LOG_LEVEL': 'ERROR',
            'PYTHONUNBUFFERED': '1',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        }
        
        for var_name, var_value in windows_env_vars.items():
            os.environ[var_name] = var_value
            logger.info(f"✅ Set Windows env var: {var_name}={var_value}")
        
        # Set process priority
        try:
            current_process = psutil.Process()
            current_process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            logger.info("✅ Set process priority to ABOVE_NORMAL")
        except Exception as e:
            logger.warning(f"⚠️ Could not set process priority: {e}")
        
        logger.info("✅ Windows environment configuration completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Windows environment configuration failed: {e}")
        raise

def test_windows_codecs():
    """Test Windows audio/video codec availability"""
    try:
        import cv2
        logger.info("🧪 Testing Windows OpenCV codecs...")
        
        test_path = os.path.join(tempfile.gettempdir(), "test_codec.mp4")
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(test_path, codec, 25.0, (10, 10))
        if writer.isOpened():
            logger.info("✅ MP4V codec is available.")
            writer.release()
            if os.path.exists(test_path):
                os.remove(test_path)
        else:
            logger.warning("⚠️ MP4V codec might not be available.")

    except ImportError:
        logger.info("📦 OpenCV not available for codec testing")
    except Exception as e:
        logger.warning(f"⚠️ Windows codec test error: {e}")

def configure_windows_torch():
    """Configure PyTorch for Windows optimization"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("✅ Configured CUDA for Windows optimization")
        
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        logger.info(f"✅ Set PyTorch threads for Windows: {torch.get_num_threads()}")
    except ImportError:
        logger.info("📦 PyTorch not available for Windows optimization")
    except Exception as e:
        logger.warning(f"⚠️ PyTorch Windows configuration error: {e}")

def setup_windows_logging():
    """Setup Windows-optimized logging"""
    try:
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, "ai_avatar_windows.log")
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)
        logger.info(f"✅ Windows logging configured: {log_file}")
    except Exception as e:
        logger.warning(f"⚠️ Windows logging setup failed: {e}")

# Auto-configure when imported
if sys.platform == "win32":
    configure_windows_environment()
    configure_windows_torch()
    setup_windows_logging()
    test_windows_codecs()