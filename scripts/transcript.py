# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "yt-dlp",
#     "transformers>=4.40.0",
#     "torch>=2.0.0",
#     "litellm>=1.40.0",
#     "accelerate",
#     "python-dotenv",
#     "tqdm",
#     "librosa",
# ]
# ///

import argparse
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import litellm
import torch
import yt_dlp
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.utils import logging as transformers_logging

load_dotenv()

os.environ["LITELLM_LOG"] = "ERROR"

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

TOTAL_STEPS = 5


def print_progress(step: int, message: str) -> None:
    print(f"\n[{step}/{TOTAL_STEPS}] {message}", flush=True)


DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

CLEANUP_PROMPT = """You are a transcript editor. Your task is to clean up this raw speech-to-text transcript.

CRITICAL REQUIREMENTS:
1. The output MUST be approximately the same length as the input
2. Do NOT summarize, condense, or shorten the content
3. Keep EVERY sentence and idea from the original

What to fix:
- Punctuation and capitalization
- Obvious transcription errors (wrong words that sound similar)
- Remove filler words (um, uh, you know) only when they add nothing
- Add paragraph breaks at topic changes

What to preserve:
- Every single idea and point made by the speaker
- The speaker's voice, style, and personality
- All examples, stories, and details
- The original language

Output ONLY the cleaned transcript, nothing else."""

TRANSLATION_PROMPT = """Translate this text to {target_language}.

CRITICAL: Do NOT summarize. Preserve every sentence, idea, example, and detail.
Output ONLY the translation.
"""

MAX_CHUNK_CHARS = 1000000
LLM_TIMEOUT = 1200


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def download_audio(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / "audio.mp3"

    if audio_path.exists():
        print("  Audio file already exists, skipping download", flush=True)
        return audio_path

    output_template = str(output_dir / "audio.%(ext)s")

    progress_bar = None

    def progress_hook(d: dict) -> None:
        nonlocal progress_bar
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes", 0)
            if total and progress_bar is None:
                progress_bar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc="  Downloading",
                    leave=False,
                )
            if progress_bar:
                progress_bar.update(downloaded - progress_bar.n)
        elif d["status"] == "finished":
            if progress_bar:
                progress_bar.close()

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_path = output_dir / "audio.mp3"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found at {audio_path}")

    print("  Download complete!", flush=True)
    return audio_path


def trim_audio(
    audio_path: Path,
    start_time: float | None,
    end_time: float | None,
) -> Path:
    if start_time is None and end_time is None:
        return audio_path

    trimmed_path = audio_path.parent / "audio_trimmed.mp3"

    cmd = ["ffmpeg", "-y", "-i", str(audio_path)]

    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])

    if end_time is not None:
        if start_time is not None:
            duration = end_time - start_time
            cmd.extend(["-t", str(duration)])
        else:
            cmd.extend(["-t", str(end_time)])

    cmd.extend(["-c", "copy", str(trimmed_path)])

    subprocess.run(cmd, capture_output=True, check=True)

    trimmed_start = start_time or 0
    trimmed_end = end_time or "end"
    print(f"  Trimmed audio: {trimmed_start}s - {trimmed_end}", flush=True)

    return trimmed_path


def transcribe_audio(
    audio_path: Path, model_name: str, device: str, language: str | None
) -> str:
    import librosa
    import os

    num_threads = os.cpu_count()
    torch.set_num_threads(num_threads)

    print(f"  Using {num_threads} CPU threads for inference", flush=True)
    print("  Loading model...", flush=True)

    torch_dtype = torch.float16 if device != "cpu" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_name)

    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        torch_dtype=torch_dtype,
        device=device,
        generate_kwargs=generate_kwargs,
    )

    print(f"  Model loaded on {device}", flush=True)
    print("  Loading audio...", flush=True)

    audio_array, sampling_rate = librosa.load(str(audio_path), sr=16000)
    duration_seconds = len(audio_array) / sampling_rate
    duration_minutes = duration_seconds / 60

    print(f"  Audio duration: {duration_minutes:.1f} minutes", flush=True)

    chunk_duration_s = 30
    stride_duration_s = 5
    effective_chunk_s = chunk_duration_s - stride_duration_s
    num_chunks = max(1, int((duration_seconds - stride_duration_s) / effective_chunk_s) + 1)

    print(f"  Processing approximately {num_chunks} chunk(s)...", flush=True)

    with tqdm(total=num_chunks, desc="  Transcribing", leave=False) as pbar:
        result = pipe(audio_array, return_timestamps=False)
        pbar.update(num_chunks)

    transcript = result["text"].strip()

    print(f"  Transcription complete ({len(transcript)} characters)", flush=True)
    return transcript


def split_into_chunks(text: str, max_chars: int) -> list[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    if not chunks:
        for i in range(0, len(text), max_chars):
            chunks.append(text[i : i + max_chars])

    return chunks


def cleanup_transcript(transcript: str) -> str:
    litellm.suppress_debug_info = True

    if len(transcript) <= MAX_CHUNK_CHARS:
        print("  Sending to Claude...", flush=True)
        response = litellm.completion(
            model=CLAUDE_MODEL,
            messages=[
                {"role": "system", "content": CLEANUP_PROMPT},
                {"role": "user", "content": transcript},
            ],
            timeout=LLM_TIMEOUT,
        )
        cleaned = response.choices[0].message.content
        print(f"  Cleanup complete ({len(cleaned)} characters)", flush=True)
        return cleaned

    chunks = split_into_chunks(transcript, MAX_CHUNK_CHARS)
    print(f"  Processing {len(chunks)} chunks...", flush=True)

    cleaned_chunks = []
    for chunk in tqdm(chunks, desc="  Cleaning", leave=False):
        response = litellm.completion(
            model=CLAUDE_MODEL,
            messages=[
                {"role": "system", "content": CLEANUP_PROMPT},
                {"role": "user", "content": chunk},
            ],
            timeout=LLM_TIMEOUT,
        )
        cleaned_chunks.append(response.choices[0].message.content)

    cleaned = "\n\n".join(cleaned_chunks)
    print(f"  Cleanup complete ({len(cleaned)} characters)", flush=True)
    return cleaned


def translate_transcript(transcript: str, target_language: str) -> str:
    print("  Sending to Claude...", flush=True)

    litellm.suppress_debug_info = True
    prompt = TRANSLATION_PROMPT.format(target_language=target_language)

    response = litellm.completion(
        model=CLAUDE_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ],
        timeout=LLM_TIMEOUT,
    )

    translated = response.choices[0].message.content
    print(f"  Translation complete ({len(translated)} characters)", flush=True)
    return translated


def cleanup_files(audio_path: Path, trimmed_path: Path | None) -> None:
    if audio_path.exists():
        audio_path.unlink()
    if trimmed_path and trimmed_path.exists() and trimmed_path != audio_path:
        trimmed_path.unlink()
    print("  Removed intermediate audio file(s)", flush=True)


def main(
    url: str,
    output_dir: Path,
    whisper_model: str,
    skip_cleanup: bool,
    device: str | None,
    language: str | None = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> int:
    if device is None:
        device = get_device()

    print(f"\nUsing device: {device}", flush=True)

    print_progress(1, "Downloading audio from YouTube...")
    audio_path = download_audio(url, output_dir)

    trimmed_path = None
    if start_time is not None or end_time is not None:
        print("  Trimming audio with ffmpeg...", flush=True)
        trimmed_path = trim_audio(audio_path, start_time, end_time)
        transcribe_path = trimmed_path
    else:
        transcribe_path = audio_path

    raw_path = output_dir / "raw_transcript.txt"
    if raw_path.exists():
        print_progress(2, "Raw transcript exists, skipping transcription...")
        raw_transcript = raw_path.read_text(encoding="utf-8")
    else:
        print_progress(2, "Transcribing audio with Whisper...")
        raw_transcript = transcribe_audio(transcribe_path, whisper_model, device, language)
        raw_path.write_text(raw_transcript, encoding="utf-8")

    cleaned_path = output_dir / "cleaned_transcript.txt"
    if cleaned_path.exists():
        print_progress(3, "Cleaned transcript exists, skipping cleanup...")
        cleaned_transcript = cleaned_path.read_text(encoding="utf-8")
    else:
        print_progress(3, "Cleaning up transcript with Claude...")
        cleaned_transcript = cleanup_transcript(raw_transcript)
        cleaned_path.write_text(cleaned_transcript, encoding="utf-8")

    romanian_path = output_dir / "transcript_romanian.txt"
    if romanian_path.exists():
        print_progress(4, "Romanian transcript exists, skipping translation...")
    else:
        print_progress(4, "Translating to Romanian with Claude...")
        romanian_transcript = translate_transcript(cleaned_transcript, "Romanian")
        romanian_path.write_text(romanian_transcript, encoding="utf-8")

    print_progress(5, "Finalizing...")
    if not skip_cleanup:
        cleanup_files(audio_path, trimmed_path)

    print(f"\n{'=' * 50}")
    print("Pipeline complete!")
    print(f"{'=' * 50}")
    print(f"  Raw transcript:     {raw_path}")
    print(f"  Cleaned transcript: {cleaned_path}")
    print(f"  Romanian:           {romanian_path}")
    print(f"{'=' * 50}\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, transcribe, clean, and translate YouTube videos"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"Whisper model name (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Keep intermediate files (audio.mp3)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Force specific device (default: auto-detect)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code for transcription (e.g., 'ro' for Romanian, 'en' for English). If not set, Whisper auto-detects.",
    )
    parser.add_argument(
        "--from",
        type=float,
        dest="start_time",
        default=None,
        help="Start time in seconds (trim audio from this point)",
    )
    parser.add_argument(
        "--to",
        type=float,
        dest="end_time",
        default=None,
        help="End time in seconds (trim audio to this point)",
    )

    args = parser.parse_args()

    sys.exit(
        main(
            args.url,
            args.output_dir,
            args.whisper_model,
            args.skip_cleanup,
            args.device,
            args.language,
            args.start_time,
            args.end_time,
        )
    )
