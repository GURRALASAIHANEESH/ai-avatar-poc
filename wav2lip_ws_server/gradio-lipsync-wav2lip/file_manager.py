import os
import time
import psutil
import subprocess
import logging
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

class WindowsFileManager:
    """Enhanced Windows file management for video processing"""
    
    def __init__(self):
        self.locked_files = set()
        self.temp_dirs = []
    
    def force_unlock_file(self, file_path: str, max_attempts: int = 10) -> bool:
        """Force unlock a file on Windows"""
        try:
            if not os.path.exists(file_path):
                return True
            
            # Method 1: Wait and retry
            for attempt in range(max_attempts):
                try:
                    # Test if file is accessible
                    with open(file_path, 'r+b') as f:
                        pass
                    logger.info(f"✅ File unlocked: {file_path}")
                    return True
                except IOError:
                    if attempt < max_attempts - 1:
                        logger.info(f"🔄 Attempt {attempt + 1}/{max_attempts} to unlock: {file_path}")
                        time.sleep(1.0)  # Wait 1 second between attempts
                    else:
                        logger.warning(f"⚠️ Could not unlock file after {max_attempts} attempts: {file_path}")
            
            # Method 2: Kill processes holding the file
            try:
                for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                    try:
                        open_files = proc.info.get('open_files', []) or []
                        for of in open_files:
                            if of.path == file_path:
                                logger.warning(f"🔪 Terminating process {proc.info['name']} (PID: {proc.info['pid']}) holding file")
                                proc.terminate()
                                proc.wait(timeout=5)
                                time.sleep(1)
                                return True
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, psutil.TimeoutExpired):
                        continue
            except Exception as e:
                logger.warning(f"⚠️ Could not terminate processes: {e}")
            
            # Method 3: Windows handle utility
            try:
                result = subprocess.run([
                    'powershell', '-Command', 
                    f'Get-Process | Where-Object {{$_.ProcessName -ne "System"}} | Where-Object {{$_.Modules.FileName -like "*{os.path.basename(file_path)}*"}} | Stop-Process -Force'
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    time.sleep(2)
                    return True
            except Exception as e:
                logger.debug(f"PowerShell unlock failed: {e}")
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error in force_unlock_file: {e}")
            return False
    
    def create_safe_temp_dir(self, prefix: str = "ai_avatar_") -> str:
        """Create a safe temporary directory for Windows"""
        try:
            # Use a shorter path to avoid Windows path length limits
            base_temp = os.environ.get('TEMP', tempfile.gettempdir())
            safe_temp = os.path.join(base_temp, f"{prefix}{int(time.time())}")
            os.makedirs(safe_temp, exist_ok=True)
            self.temp_dirs.append(safe_temp)
            logger.info(f"✅ Created safe temp directory: {safe_temp}")
            return safe_temp
        except Exception as e:
            logger.error(f"❌ Failed to create safe temp directory: {e}")
            return tempfile.mkdtemp()
    
    def safe_file_move(self, src: str, dst: str, max_attempts: int = 5) -> bool:
        """Safely move a file with retry logic"""
        for attempt in range(max_attempts):
            try:
                if os.path.exists(src):
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    
                    # Try to move the file
                    shutil.move(src, dst)
                    logger.info(f"✅ Successfully moved: {src} -> {dst}")
                    return True
                else:
                    logger.warning(f"⚠️ Source file not found: {src}")
                    return False
                    
            except Exception as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"⚠️ Move attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(2)
                else:
                    logger.error(f"❌ Failed to move file after {max_attempts} attempts: {e}")
                    return False
        return False
    
    def cleanup_temp_dirs(self):
        """Clean up all temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    # First, unlock all files in the directory
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            self.force_unlock_file(file_path)
                    
                    # Wait a bit for Windows to release handles
                    time.sleep(2)
                    
                    # Remove the directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.info(f"🧹 Cleaned up temp directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"⚠️ Could not cleanup temp directory {temp_dir}: {e}")
        
        self.temp_dirs.clear()

# Global file manager instance
file_manager = WindowsFileManager()
