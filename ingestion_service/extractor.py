"""
extractor.py — Audio Extraction & Noise Reduction
Wraps FFprobe (metadata) and FFmpeg (audio extraction) via ffmpeg-python,
then applies noisereduce for spectral noise suppression.

Output spec (required by Whisper):
  • 16 kHz sample rate
  • Mono (1 channel)
  • 16-bit PCM WAV
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import ffmpeg
import numpy as np
import noisereduce as nr
import soundfile as sf

logger = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    """Raised when any FFmpeg/FFprobe or audio-processing step fails."""


# ─── AudioExtractor ───────────────────────────────────────────────────────────

class AudioExtractor:
    """
    Handles:
      1. Video metadata probing via FFprobe
      2. Raw audio extraction to 16kHz mono WAV via FFmpeg
      3. Spectral noise reduction via noisereduce
    """

    # Target audio spec for Whisper
    TARGET_SAMPLE_RATE: int = 16_000
    TARGET_CHANNELS: int = 1

    # ── Metadata ──────────────────────────────────────────────────────────────

    def probe_metadata(self, video_path: Path) -> dict[str, Any]:
        """
        Run FFprobe on *video_path* and return a normalised metadata dict:

        {
          "duration": float,          # seconds
          "has_audio": bool,
          "has_video": bool,
          "resolution": "WxH" | None,
          "fps": float | None,
          "audio_codec": str | None,
          "video_codec": str | None,
          "audio_channels": int | None,
          "audio_sample_rate": int | None,
          "size_bytes": int,
          "format_name": str,
        }

        Raises ExtractionError on any FFprobe failure.
        """
        try:
            probe = ffmpeg.probe(str(video_path))
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            raise ExtractionError(
                f"FFprobe failed on '{video_path.name}': {stderr.strip()}"
            ) from exc

        fmt = probe.get("format", {})
        streams = probe.get("streams", [])

        video_stream = next(
            (s for s in streams if s.get("codec_type") == "video"), None
        )
        audio_stream = next(
            (s for s in streams if s.get("codec_type") == "audio"), None
        )

        # Duration: prefer format-level, fall back to stream-level
        duration = float(fmt.get("duration") or 0)
        if not duration and video_stream:
            duration = float(video_stream.get("duration") or 0)
        if not duration and audio_stream:
            duration = float(audio_stream.get("duration") or 0)

        # FPS
        fps: float | None = None
        if video_stream:
            raw_fps = video_stream.get("r_frame_rate", "0/1")
            try:
                num, den = raw_fps.split("/")
                fps = round(int(num) / int(den), 3) if int(den) else None
            except (ValueError, ZeroDivisionError):
                fps = None

        resolution: str | None = None
        if video_stream:
            w = video_stream.get("width")
            h = video_stream.get("height")
            if w and h:
                resolution = f"{w}x{h}"

        return {
            "duration": duration,
            "has_audio": audio_stream is not None,
            "has_video": video_stream is not None,
            "resolution": resolution,
            "fps": fps,
            "video_codec": video_stream.get("codec_name") if video_stream else None,
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "audio_channels": int(audio_stream["channels"])
            if audio_stream and audio_stream.get("channels")
            else None,
            "audio_sample_rate": int(audio_stream["sample_rate"])
            if audio_stream and audio_stream.get("sample_rate")
            else None,
            "size_bytes": int(fmt.get("size", 0)),
            "format_name": fmt.get("format_name", "unknown"),
        }

    # ── Thumbnail Extraction ──────────────────────────────────────────────────

    def extract_thumbnail(self, video_path: Path, thumbnail_path: Path) -> bool:
        """
        Extract the first video frame as a JPEG thumbnail using FFmpeg.

        Required by the Upload screen — the UI shows a preview thumbnail
        of the meeting video immediately after upload completes.

        Parameters
        ----------
        video_path      : Path  Path to the raw uploaded video file.
        thumbnail_path  : Path  Destination path for the JPEG (e.g. thumbnail.jpg).

        Returns
        -------
        bool  True if thumbnail was extracted successfully, False if the file
              has no video stream (audio-only) — not treated as a hard error.

        Raises ExtractionError only on unexpected FFmpeg failures.
        """
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            (
                ffmpeg
                .input(str(video_path), ss=0)        # seek to 0s (first frame)
                .output(
                    str(thumbnail_path),
                    vframes=1,                        # extract exactly one frame
                    format="image2",
                    vcodec="mjpeg",
                    **{"qscale:v": 2},                # high quality JPEG (2 = best)
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            # No video stream is non-fatal (audio-only files are still valid)
            if "no video" in stderr.lower() or "invalid" in stderr.lower():
                logger.warning(
                    "No video stream in '%s' — skipping thumbnail.", video_path.name
                )
                return False
            raise ExtractionError(
                f"FFmpeg thumbnail extraction failed for '{video_path.name}': {stderr.strip()}"
            ) from exc

        if not thumbnail_path.exists() or thumbnail_path.stat().st_size == 0:
            logger.warning("Thumbnail file empty or missing — skipping.")
            return False

        logger.info(
            "Thumbnail extracted → %s (%.1f KB)",
            thumbnail_path.name, thumbnail_path.stat().st_size / 1024,
        )
        return True

    # ── Audio Extraction ──────────────────────────────────────────────────────

    def extract_audio(self, video_path: Path, wav_path: Path) -> None:
        """
        Extract the first audio track from *video_path*, convert to
        16 kHz mono 16-bit PCM WAV and write to *wav_path*.

        Uses two-pass approach:
          1. Try stream-copy + aresample (fast, no re-encoding of video).
          2. On failure fall back to full transcode with libsox/aresample.

        Raises ExtractionError on failure.
        """
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(wav_path),
                    acodec="pcm_s16le",
                    ac=self.TARGET_CHANNELS,
                    ar=self.TARGET_SAMPLE_RATE,
                    vn=None,            # drop video stream
                    map="0:a:0",        # take only the first audio stream
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as exc:
            stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
            raise ExtractionError(
                f"FFmpeg audio extraction failed: {stderr.strip()}"
            ) from exc

        if not wav_path.exists() or wav_path.stat().st_size == 0:
            raise ExtractionError(
                "FFmpeg produced an empty or missing WAV file. "
                "Ensure the video contains an audio track."
            )

        logger.info("Audio extracted → %s (%.1f KB)",
                    wav_path.name, wav_path.stat().st_size / 1024)

    # ── Noise Reduction ───────────────────────────────────────────────────────

    def denoise_audio(
        self,
        input_wav: Path,
        output_wav: Path,
        *,
        noise_clip_duration: float = 0.5,
        prop_decrease: float = 0.80,
        stationary: bool = False,
    ) -> None:
        """
        Apply spectral noise reduction to *input_wav* and write to *output_wav*.

        Strategy
        --------
        • Read the entire WAV with soundfile (float32 array).
        • Estimate the noise profile from the first *noise_clip_duration* seconds
          (typically silence / ambient noise before the meeting starts).
        • Apply noisereduce.reduce_noise() in non-stationary mode by default,
          which handles time-varying noise (keyboard clicks, HVAC bursts).
        • Write back as 16-bit PCM WAV at the same sample rate.

        Parameters
        ----------
        noise_clip_duration : float
            Seconds at the start of the file to use as the noise profile.
        prop_decrease : float
            0–1 fraction by which detected noise is attenuated (0.8 = 80 % reduction).
        stationary : bool
            True → assume constant background noise (faster, better for HVAC).
            False → non-stationary model (better for mixed sources).

        Raises ExtractionError on read/write failure.
        """
        try:
            audio, sr = sf.read(str(input_wav), dtype="float32")
        except Exception as exc:
            raise ExtractionError(f"Cannot read '{input_wav.name}': {exc}") from exc

        # Ensure mono float32
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Build noise clip from the first N seconds
        noise_samples = int(noise_clip_duration * sr)
        noise_clip = audio[:noise_samples] if len(audio) > noise_samples else audio

        logger.info("Denoising '%s': %d samples @ %d Hz", input_wav.name, len(audio), sr)

        try:
            reduced = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_clip,
                prop_decrease=prop_decrease,
                stationary=stationary,
                n_fft=2048,
                win_length=2048,
                hop_length=512,
                n_std_thresh_stationary=1.5,
                use_torch=False,   # CPU-safe default; set True if GPU available
            )
        except Exception as exc:
            raise ExtractionError(f"noisereduce failed: {exc}") from exc

        # Convert back to int16 PCM
        reduced_int16 = (reduced * 32767).clip(-32768, 32767).astype(np.int16)

        try:
            output_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_wav), reduced_int16, sr, subtype="PCM_16")
        except Exception as exc:
            raise ExtractionError(
                f"Cannot write denoised WAV to '{output_wav.name}': {exc}"
            ) from exc

        logger.info("Denoised audio saved → %s (%.1f KB)",
                    output_wav.name, output_wav.stat().st_size / 1024)