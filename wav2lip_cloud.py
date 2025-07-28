import time, os, shutil
from datetime import datetime
from gradio_client import Client, handle_file

def generate_lipsync_video(
    video_path,
    audio_path,
    output_dir,
    space_name="banao-tech/gradio-lipsync-wav2lip",
    hf_token=None,
    max_retries=5,
    backoff=15
):
    last_exc = None
    for i in range(max_retries):
        try:
            client = Client(space_name, hf_token=hf_token)
            break
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (1.5 ** i))
    else:
        raise RuntimeError(f"Could not connect to Wav2Lip space: {last_exc}")

    result = client.predict(
        video=handle_file(video_path),
        audio=handle_file(audio_path),
        checkpoint="wav2lip_gan",
        api_name="/submit_job"
    )
    job_id = result[1] if isinstance(result, (tuple, list)) and len(result) >= 2 else None
    timeout = time.time() + 60*20
    while job_id and time.time() < timeout:
        status = client.predict(job_id=job_id, api_name="/check_job_status")
        if (isinstance(status, (tuple, list)) and len(status) > 1 
            and isinstance(status[1], dict) and "video" in status[1]):
            video_file = status[1]["video"]
            break
        time.sleep(20)
    else:
        raise TimeoutError("Wav2Lip job did not finish in time.")

    os.makedirs(output_dir, exist_ok=True)
    dst = os.path.join(output_dir, f"lipsync_{datetime.now():%Y%m%d_%H%M%S}.mp4")
    shutil.copy2(video_file, dst)
    return dst
