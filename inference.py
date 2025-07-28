import os
import sys
import platform
import time
import logging
import argparse
import subprocess
import traceback
import numpy as np
import cv2
from tqdm import tqdm
import torch

# Custom utility imports (must be present in your repo or PYTHONPATH)
import audio  # Audio utility for WAV/mel processing
from wav2lip_models import Wav2Lip  # Wav2Lip model definition
import face_detection  # Face detection utility

def parse_args():
    parser = argparse.ArgumentParser(description='Wav2Lip inference with detailed logging')
    parser.add_argument('--checkpoint_path', type=str, default="checkpoints/wav2lip_gan.pth",
                        help='Path to Wav2Lip checkpoint')
    parser.add_argument('--face', type=str, required=True,
                        help='Path to video/image with faces')
    parser.add_argument('--audio', type=str, required=True,
                        help='Path to audio file')
    parser.add_argument('--outfile', type=str, default='results/result_voice.mp4',
                        help='Output video path')
    parser.add_argument('--static', action='store_true',
                        help='Use only first frame for inference')
    parser.add_argument('--fps', type=float, default=25.,
                        help='FPS for static images (default: 25)')
    parser.add_argument('--pads', nargs=4, type=int, default=[0, 20, 0, 0],
                        help='Padding (top, bottom, left, right)')
    parser.add_argument('--face_det_batch_size', type=int, default=16,
                        help='Batch size for face detection')
    parser.add_argument('--wav2lip_batch_size', type=int, default=128,
                        help='Batch size for Wav2Lip')
    parser.add_argument('--resize_factor', type=int, default=4,
                        help='Reduce resolution by this factor (4 for quality)')
    parser.add_argument('--crop', nargs=4, type=int, default=[0, -1, 0, -1],
                        help='Crop video (top, bottom, left, right). -1 auto-infers')
    parser.add_argument('--rotate', action='store_true', default=False,
                        help='Flip video 90deg if true')
    parser.add_argument('--nosmooth', action='store_true', default=False,
                        help='Disable smoothing of face detections')
    parser.add_argument('--job_id', type=str, default='unknown',
                        help='Job ID for logging purposes')
    return parser.parse_args()

def setup_logging(job_id):
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Job {job_id}] %(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images, args, device, logger, job_temp_dir):
    try:
        detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                                flip_input=False, device=device)
        batch_size = args.face_det_batch_size
        predictions = []
        
        for i in range(0, len(images), batch_size):
            try:
                preds = detector.get_detections_for_batch(np.array(images[i:i + batch_size]))
                predictions.extend(preds)
            except RuntimeError as e:
                if batch_size == 1:
                    logger.error(f"Face detection failed even with batch_size=1: {e}")
                    raise RuntimeError('Image too big for GPU. Try larger --resize_factor')
                batch_size //= 2
                logger.warning(f"Reducing face detection batch size to {batch_size} due to OOM")
                del detector
                return face_detect(images, args, device, logger, job_temp_dir)
        
        results = []
        pady1, pady2, padx1, padx2 = args.pads
        
        for idx, (rect, image) in enumerate(zip(predictions, images)):
            if rect is None:
                logger.warning("No face detected in image %d, saving faulty frame", idx)
                cv2.imwrite(f'{job_temp_dir}/faulty_frame_{idx}.jpg', image)
                raise ValueError('Face not detected! Check video for a clear face or try different input.')
            
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])
        
        boxes = np.array(results)
        if not args.nosmooth:
            boxes = get_smoothened_boxes(boxes, T=5)
        
        out = [[image[y1:y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]
        del detector
        return out
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        traceback.print_exc()
        raise

def datagen(frames, mels, face_det_results, args, logger):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    
    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face_idx = idx if idx < len(face_det_results) else 0
        face, coords = face_det_results[face_idx]
        face = cv2.resize(face, (args.img_size, args.img_size))
        
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        
        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch_np, mel_batch_np = np.asarray(img_batch), np.asarray(mel_batch)
            img_masked = img_batch_np.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
            mel_batch_np = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])
            yield img_batch_np, mel_batch_np, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    
    if len(img_batch) > 0:
        img_batch_np, mel_batch_np = np.asarray(img_batch), np.asarray(mel_batch)
        img_masked = img_batch_np.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch_np = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
        mel_batch_np = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])
        yield img_batch_np, mel_batch_np, frame_batch, coords_batch

def _load(checkpoint_path, device, logger):
    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        logger.info("Checkpoint loaded successfully")
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        traceback.print_exc()
        raise

def load_model(path, device, logger):
    try:
        logger.info("Initializing Wav2Lip model")
        model = Wav2Lip()
        checkpoint = _load(path, device, logger)
        s = checkpoint["state_dict"]
        new_s = {k.replace('module.', ''): v for k, v in s.items()}
        model.load_state_dict(new_s)
        model = model.to(device).eval()
        logger.info("Model loaded and set to evaluation mode")
        return model
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        traceback.print_exc()
        raise

def convert_audio_to_wav(input_path, output_path, logger):
    """Convert any audio format to WAV using ffmpeg"""
    try:
        logger.info(f"Converting audio from {input_path} to {output_path}")
        
        # Use more robust ffmpeg command with error handling
        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-ar', '16000',  # Set sample rate to 16kHz
            '-ac', '1',      # Convert to mono
            '-c:a', 'pcm_s16le',  # Use PCM encoding
            output_path
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg conversion failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
        
        if not os.path.exists(output_path):
            raise RuntimeError(f"Converted audio file not created: {output_path}")
            
        logger.info(f"Audio conversion successful: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        traceback.print_exc()
        raise

def infer(args):
    try:
        args.img_size = 96
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger = setup_logging(getattr(args, "job_id", "unknown"))
        
        logger.info(f'Using {device} for inference')
        logger.info(f"Starting Wav2Lip inference with args: face={args.face}, audio={args.audio}")

        job_temp_dir = f'temp_{getattr(args, "job_id", "unknown")}'
        os.makedirs(job_temp_dir, exist_ok=True)

        # Static mode check
        if os.path.isfile(args.face) and args.face.lower().split('.')[-1] in ['jpg', 'png', 'jpeg']:
            args.static = True
            logger.info("Using static image mode")

        if not os.path.isfile(args.face):
            logger.error("Invalid face path: %s", args.face)
            raise ValueError('--face must be a valid video/image path')

        if not os.path.isfile(args.audio):
            logger.error("Invalid audio path: %s", args.audio)
            raise ValueError('--audio must be a valid audio file path')

        # Load all frames
        logger.info("Loading video frames")
        if args.face.lower().split('.')[-1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(args.face)]
            fps = args.fps
            logger.info("Loaded static image as single frame")
        else:
            video_stream = cv2.VideoCapture(args.face)
            fps = video_stream.get(cv2.CAP_PROP_FPS) or args.fps
            full_frames = []
            
            while video_stream.isOpened():
                still_reading, frame = video_stream.read()
                if not still_reading:
                    break
                if args.resize_factor > 1:
                    frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))
                if args.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                y1, y2, x1, x2 = args.crop
                if x2 == -1:
                    x2 = frame.shape[1]
                if y2 == -1:
                    y2 = frame.shape[0]
                frame = frame[y1:y2, x1:x2]
                full_frames.append(frame)
            
            video_stream.release()
            logger.info(f"Loaded {len(full_frames)} frames at {fps} FPS")

        if not full_frames:
            raise ValueError("No frames loaded from input!")

        # Pre-compute face detection
        logger.info("Running face detection")
        face_det_results = face_detect(full_frames.copy(), args, device, logger, job_temp_dir)
        logger.info(f"Face detection completed for {len(face_det_results)} frames")

        # Handle audio conversion - FIXED for WebM/OGG support
        logger.info("Processing audio input")
        temp_audio_path = f'{job_temp_dir}/temp_audio.wav'
        
        # Convert any audio format to WAV
        if not args.audio.lower().endswith('.wav'):
            audio_file = convert_audio_to_wav(args.audio, temp_audio_path, logger)
        else:
            audio_file = args.audio

        # Load audio and compute mel-spectrogram
        logger.info("Computing mel-spectrogram")
        try:
            wav = audio.load_wav(audio_file, 16000)
            mel = audio.melspectrogram(wav)
            
            if np.isnan(mel.reshape(-1)).sum() > 0:
                logger.error("Mel-spectrogram contains NaN values")
                raise ValueError('Mel contains nan! Audio may be corrupted or too short.')
                
            logger.info(f"Mel-spectrogram computed successfully, shape: {mel.shape}")
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            traceback.print_exc()
            raise

        # Compute mel chunks
        logger.info("Generating mel chunks")
        mel_chunks = []
        mel_idx_multiplier = 80. / fps
        mel_step_size = 16
        i = 0
        
        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
            i += 1

        logger.info(f"Generated {len(mel_chunks)} mel chunks")

        # Ensure the number of frames and mel chunks match
        full_frames = full_frames[:len(mel_chunks)]
        logger.info(f"Aligned {len(full_frames)} frames with {len(mel_chunks)} mel chunks")

        # Create output directory
        os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

        # Data loader/generator
        gen = datagen(full_frames, mel_chunks, face_det_results, args, logger)
        model = None
        out = None
        temp_result_path = f'{job_temp_dir}/result.avi'
        
        logger.info("Starting lip-sync processing")
        
        for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(
            gen, total=int(np.ceil(len(mel_chunks)/args.wav2lip_batch_size)), 
            desc=f"Lip-syncing Job {getattr(args, 'job_id', 'unknown')}")):
            
            if i == 0:
                model = load_model(args.checkpoint_path, device, logger)
                frame_h, frame_w = frames[0].shape[:-1]
                out = cv2.VideoWriter(temp_result_path,
                                      cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
                logger.info(f"Initialized video writer: {frame_w}x{frame_h} at {fps} FPS")
            
            img_batch_torch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
            mel_batch_torch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
            
            with torch.no_grad():
                pred = model(mel_batch_torch, img_batch_torch)
            
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            
            for j, (p, f, c) in enumerate(zip(pred, frames, coords)):
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
                out.write(f)
        
        out.release()
        logger.info("Video processing completed")

        # Combine video and original audio
        logger.info("Combining video with audio")
        try:
            command = [
                'ffmpeg', '-y',
                '-i', audio_file,
                '-i', temp_result_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-strict', '-2',
                '-q:v', '1',
                args.outfile
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg combining failed: {result.stderr}")
                raise RuntimeError(f"FFmpeg combining failed: {result.stderr}")
                
            logger.info(f"Final output saved to {args.outfile}")
            
        except Exception as e:
            logger.error(f"Audio-video combining error: {e}")
            traceback.print_exc()
            raise

        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(job_temp_dir)
            logger.info(f"Cleaned up temporary directory: {job_temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory: {e}")

        logger.info(f"Wav2Lip inference completed successfully. Output: {args.outfile}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        traceback.print_exc()
        raise

# ---- FOR CLI USE ONLY ----
if __name__ == '__main__':
    args = parse_args()
    infer(args)
