#!/usr/bin/env python3

import argparse
import csv
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite import Interpreter

from microwakeword.audio.audio_utils import generate_features_for_clip


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".opus"}
TARGET_SAMPLE_RATE = 16000


class MWWModel:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # For the streaming model this is normally the number of
        # 10 ms feature slices consumed per inference.
        self.input_feature_slices = int(
            self.input_details[0]["shape"][1]
        )

        self.stride = self.input_feature_slices

        self.input_is_quantized = np.issubdtype(
            self.input_details[0]["dtype"],
            np.integer,
        )

        self.reset_state()

    def reset_state(self):
        for detail in self.input_details:
            self.interpreter.set_tensor(
                detail["index"],
                np.zeros(
                    detail["shape"],
                    dtype=detail["dtype"],
                ),
            )

    @staticmethod
    def quantize(data, detail):
        scale, zero_point = detail["quantization"]

        if not scale:
            return data.astype(detail["dtype"])

        result = np.round(data / scale + zero_point)

        info = np.iinfo(detail["dtype"])

        return np.clip(
            result,
            info.min,
            info.max,
        ).astype(detail["dtype"])

    @staticmethod
    def dequantize(value, detail):
        scale, zero_point = detail["quantization"]

        if not scale:
            return float(value)

        return float(
            (float(value) - zero_point) * scale
        )

    def predict_audio(self, pcm16, feature_step_ms=10):
        self.reset_state()

        spectrogram = generate_features_for_clip(
            pcm16,
            step_ms=feature_step_ms,
        )

        # Match microWakeWord feature scaling.
        if np.issubdtype(spectrogram.dtype, np.uint16):
            spectrogram = (
                spectrogram.astype(np.float32) * 0.0390625
            )

        elif spectrogram.dtype == np.float64:
            spectrogram = spectrogram.astype(np.float32)

        predictions = []

        for last_index in range(
            self.input_feature_slices,
            len(spectrogram) + 1,
            self.stride,
        ):
            chunk = spectrogram[
                last_index - self.input_feature_slices:
                last_index
            ]

            if len(chunk) != self.input_feature_slices:
                continue

            if (
                self.input_is_quantized
                and not np.issubdtype(chunk.dtype, np.integer)
            ):
                chunk = self.quantize(
                    chunk,
                    self.input_details[0],
                )

            self.interpreter.set_tensor(
                self.input_details[0]["index"],
                np.reshape(
                    chunk,
                    self.input_details[0]["shape"],
                ),
            )

            self.interpreter.invoke()

            output = self.interpreter.get_tensor(
                self.output_details[0]["index"]
            )[0][0]

            if np.issubdtype(
                self.output_details[0]["dtype"],
                np.integer,
            ):
                output = self.dequantize(
                    output,
                    self.output_details[0],
                )
            else:
                output = float(output)

            predictions.append(output)

        return np.asarray(
            predictions,
            dtype=np.float32,
        )


def _read_up_to(stream, byte_count):
    """
    Read up to byte_count bytes from a pipe.

    Pipe reads are allowed to return fewer bytes than requested, so keep
    reading until the requested amount is available or EOF is reached.
    """
    data = bytearray()

    while len(data) < byte_count:
        block = stream.read(byte_count - len(data))

        if not block:
            break

        data.extend(block)

    return bytes(data)


def stream_audio_chunks(
    path,
    sample_rate,
    chunk_seconds,
    overlap_seconds,
):
    """
    Decode WAV/Opus through ffmpeg and yield overlapping mono 16 kHz PCM16
    chunks without loading the entire source recording into RAM.

    Yields:
        (chunk_start_seconds, pcm16_array)
    """
    chunk_samples = int(chunk_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)

    if chunk_samples <= 0:
        raise ValueError("chunk_seconds must be greater than 0")

    if overlap_samples < 0:
        raise ValueError("overlap_seconds cannot be negative")

    if overlap_samples >= chunk_samples:
        raise ValueError(
            "chunk overlap must be smaller than chunk size"
        )

    command = [
        "ffmpeg",
        "-v", "error",
        "-nostdin",
        "-i", str(path),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(sample_rate),
        "pipe:1",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )

    if process.stdout is None or process.stderr is None:
        process.kill()
        raise RuntimeError("Could not open ffmpeg pipes")

    try:
        first_raw = _read_up_to(
            process.stdout,
            chunk_samples * 2,
        )

        if not first_raw:
            stderr = process.stderr.read().decode(
                "utf-8",
                errors="replace",
            )
            process.wait()
            raise RuntimeError(
                f"ffmpeg produced no audio for {path}\n{stderr}"
            )

        pcm16 = np.frombuffer(
            first_raw,
            dtype="<i2",
        ).copy()

        chunk_start_seconds = 0.0
        yield chunk_start_seconds, pcm16

        if len(pcm16) < chunk_samples:
            # Source ended within the first chunk.
            return

        advance_samples = chunk_samples - overlap_samples
        advance_seconds = advance_samples / sample_rate

        while True:
            tail = pcm16[-overlap_samples:].copy()

            raw = _read_up_to(
                process.stdout,
                advance_samples * 2,
            )

            if not raw:
                break

            new_pcm = np.frombuffer(
                raw,
                dtype="<i2",
            ).copy()

            pcm16 = np.concatenate(
                (tail, new_pcm)
            )

            chunk_start_seconds += advance_seconds

            yield chunk_start_seconds, pcm16

            if len(new_pcm) < advance_samples:
                break

    finally:
        # Make sure ffmpeg is reaped even when processing raises.
        if process.stdout:
            process.stdout.close()

        stderr = b""

        if process.stderr:
            stderr = process.stderr.read()
            process.stderr.close()

        return_code = process.wait()

        if return_code != 0:
            message = stderr.decode(
                "utf-8",
                errors="replace",
            ).strip()

            raise RuntimeError(
                f"ffmpeg failed for {path}"
                + (f":\n{message}" if message else "")
            )


def iter_with_last(iterable):
    """
    Yield (item, is_last) with one-item lookahead.
    """
    iterator = iter(iterable)

    try:
        previous = next(iterator)
    except StopIteration:
        return

    for current in iterator:
        yield previous, False
        previous = current

    yield previous, True


def find_hits(
    predictions,
    threshold,
    window,
    prediction_period,
):
    if len(predictions) < window:
        return []

    rolling_average = np.convolve(
        predictions,
        np.ones(
            window,
            dtype=np.float32,
        ) / window,
        mode="valid",
    )

    hits = []

    for index, average in enumerate(
        rolling_average
    ):
        if average < threshold:
            continue

        end_prediction = (
            index + window - 1
        )

        trigger_time = (
            end_prediction + 1
        ) * prediction_period

        peak = float(
            np.max(
                predictions[
                    index:
                    end_prediction + 1
                ]
            )
        )

        hits.append({
            "time": trigger_time,
            "average": float(average),
            "peak": peak,
        })

    return hits


def deduplicate_hits(hits, cooldown, time_key="time"):
    if not hits:
        return []

    hits = sorted(
        hits,
        key=lambda hit: hit[time_key],
    )

    groups = []
    current = [hits[0]]

    for hit in hits[1:]:
        if (
            hit[time_key]
            - current[-1][time_key]
            <= cooldown
        ):
            current.append(hit)

        else:
            groups.append(current)
            current = [hit]

    groups.append(current)

    # Keep strongest event from each group.
    return [
        max(
            group,
            key=lambda item: item["average"],
        )
        for group in groups
    ]


def extract_clip_pcm16(
    pcm16,
    trigger_time,
    sample_rate,
    pre_seconds,
    post_seconds,
):
    total_samples = int(
        (pre_seconds + post_seconds)
        * sample_rate
    )

    start = int(
        (trigger_time - pre_seconds)
        * sample_rate
    )

    end = start + total_samples

    left_pad = max(0, -start)
    right_pad = max(
        0,
        end - len(pcm16),
    )

    start = max(0, start)
    end = min(len(pcm16), end)

    clip = pcm16[start:end]

    if left_pad or right_pad:
        clip = np.pad(
            clip,
            (left_pad, right_pad),
        )

    # Guarantee exact duration.
    if len(clip) < total_samples:
        clip = np.pad(
            clip,
            (
                0,
                total_samples - len(clip),
            ),
        )

    elif len(clip) > total_samples:
        clip = clip[:total_samples]

    return clip.astype(
        np.int16,
        copy=False,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="/data/hard_negative_candidates",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--feature-step-ms",
        type=int,
        default=10,
    )

    # Avoid exporting 20 clips from one false trigger.
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1.5,
    )

    # Save 2 sec before trigger + 1 sec after.
    parser.add_argument(
        "--pre",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--post",
        type=float,
        default=1.0,
    )

    # Decode long source files in bounded chunks instead of loading the
    # entire podcast/recording into memory.
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=300.0,
        help="Audio decoded per inference chunk (default: 300 seconds).",
    )

    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=5.0,
        help=(
            "Overlap between inference chunks (default: 5 seconds). "
            "The script may increase this automatically so candidate "
            "clips have enough context."
        ),
    )

    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg was not found in PATH. "
            "Install ffmpeg before running this script."
        )

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    audio_files = sorted(
        path
        for path in input_dir.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_AUDIO_EXTENSIONS
        )
    )

    if not audio_files:
        supported = ", ".join(
            sorted(SUPPORTED_AUDIO_EXTENSIONS)
        )

        raise SystemExit(
            f"No supported audio files found in {input_dir}. "
            f"Supported: {supported}"
        )

    # We need enough overlap to provide both pre-trigger context and
    # post-trigger audio around a chunk boundary. Add 1 second of margin.
    minimum_overlap = (
        args.pre
        + args.post
        + 1.0
    )

    effective_overlap = max(
        args.chunk_overlap,
        minimum_overlap,
    )

    if effective_overlap >= args.chunk_seconds:
        raise SystemExit(
            "--chunk-seconds must be greater than the effective overlap "
            f"({effective_overlap:.1f}s)."
        )

    probe_model = MWWModel(
        args.model
    )

    # Example:
    # stride 3 * 10 ms = 30 ms/inference
    prediction_period = (
        probe_model.stride
        * args.feature_step_ms
        / 1000.0
    )

    rows = []

    print(
        f"Scanning {len(audio_files)} audio files "
        f"({', '.join(sorted(SUPPORTED_AUDIO_EXTENSIONS))})"
    )

    print(
        f"Model stride: "
        f"{probe_model.stride}"
    )

    print(
        f"Prediction interval: "
        f"{prediction_period:.3f}s"
    )

    print(
        f"Detection threshold: "
        f"{args.threshold}"
    )

    print(
        f"Sliding window: "
        f"{args.window}"
    )

    print(
        f"Decode chunk: "
        f"{args.chunk_seconds:.1f}s"
    )

    print(
        f"Chunk overlap: "
        f"{effective_overlap:.1f}s"
    )

    for number, audio_path in enumerate(
        audio_files,
        start=1,
    ):
        print(
            f"[{number}/{len(audio_files)}] "
            f"{audio_path}"
        )

        file_candidates = []

        chunks = stream_audio_chunks(
            audio_path,
            sample_rate=TARGET_SAMPLE_RATE,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=effective_overlap,
        )

        try:
            for (
                (chunk_start, pcm16),
                is_last,
            ) in iter_with_last(chunks):
                chunk_duration = (
                    len(pcm16)
                    / TARGET_SAMPLE_RATE
                )

                # New model instance for each chunk. The overlapped prefix
                # gives the model time to warm up before we accept hits from
                # the unique region of this chunk.
                model = MWWModel(
                    args.model
                )

                predictions = model.predict_audio(
                    pcm16,
                    feature_step_ms=args.feature_step_ms,
                )

                hits = find_hits(
                    predictions,
                    args.threshold,
                    args.window,
                    prediction_period,
                )

                # Adjacent chunks overlap. Split responsibility for the
                # boundary so the same false trigger is not exported twice.
                #
                # For non-first chunks, accept boundary hits from
                # overlap - post onward. That lets the new chunk provide
                # real post-trigger audio instead of padding silence.
                if chunk_start == 0.0:
                    lower_bound = 0.0
                else:
                    lower_bound = max(
                        0.0,
                        effective_overlap - args.post,
                    )

                if is_last:
                    upper_bound = chunk_duration + 1e-9
                else:
                    upper_bound = max(
                        lower_bound,
                        chunk_duration - args.post,
                    )

                hits = [
                    hit
                    for hit in hits
                    if (
                        hit["time"] >= lower_bound
                        and hit["time"] < upper_bound
                    )
                ]

                hits = deduplicate_hits(
                    hits,
                    args.cooldown,
                )

                for hit in hits:
                    clip = extract_clip_pcm16(
                        pcm16,
                        hit["time"],
                        TARGET_SAMPLE_RATE,
                        args.pre,
                        args.post,
                    )

                    file_candidates.append({
                        "absolute_time":
                            chunk_start + hit["time"],

                        "average":
                            hit["average"],

                        "peak":
                            hit["peak"],

                        "clip":
                            clip,
                    })

        except RuntimeError as error:
            print(
                f"  ERROR decoding {audio_path}: "
                f"{error}"
            )
            continue

        # One final dedupe in absolute file time handles any event whose
        # threshold window happens to straddle a chunk boundary.
        file_candidates = deduplicate_hits(
            file_candidates,
            args.cooldown,
            time_key="absolute_time",
        )

        for hit in file_candidates:
            candidate_id = str(
                uuid.uuid4()
            )

            output_file = (
                output_dir
                / f"{candidate_id}.wav"
            )

            sf.write(
                output_file,
                hit["clip"],
                TARGET_SAMPLE_RATE,
                subtype="PCM_16",
            )

            rows.append({
                "candidate":
                    output_file.name,

                "source_file":
                    str(audio_path),

                "trigger_time_sec":
                    f'{hit["absolute_time"]:.3f}',

                "rolling_average":
                    f'{hit["average"]:.6f}',

                "peak_probability":
                    f'{hit["peak"]:.6f}',

                "threshold":
                    args.threshold,

                "sliding_window":
                    args.window,
            })

            print(
                "  FALSE CANDIDATE "
                f"time={hit['absolute_time']:.2f}s "
                f"avg={hit['average']:.4f} "
                f"peak={hit['peak']:.4f}"
            )

    manifest = (
        output_dir
        / "manifest.csv"
    )

    with manifest.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        fieldnames = [
            "candidate",
            "source_file",
            "trigger_time_sec",
            "rolling_average",
            "peak_probability",
            "threshold",
            "sliding_window",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Found {len(rows)} "
        "hard-negative candidates."
    )

    print(
        f"Saved to: {output_dir}"
    )

    print(
        f"Manifest: {manifest}"
    )


if __name__ == "__main__":
    main()
