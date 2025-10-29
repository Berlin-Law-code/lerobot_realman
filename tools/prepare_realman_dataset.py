#!/usr/bin/env python
"""
Convert the raw RealMan data collected under ``data_raw`` into a LeRobot v3.0 dataset
where the action is defined as the next-step joint position.

The resulting dataset is tailored for fine-tuning PI0-style policies that expect:
    - proprioceptive state under ``observation.state``
    - torque history in ``observation.joint_torques``
    - high resolution stereo images
    - joint-position actions in ``action``
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

# Ensure local lerobot source tree is importable without package installation
CURRENT_DIR = Path(__file__).resolve().parent
LEROBOT_SRC = CURRENT_DIR / "lerobot" / "src"
if LEROBOT_SRC.is_dir() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

if "" in sys.path:
    sys.path.remove("")
current_dir_str = str(CURRENT_DIR)
if current_dir_str in sys.path:
    sys.path.remove(current_dir_str)

from lerobot.datasets.lerobot_dataset import LeRobotDataset


JOINT_NAMES = [f"joint_{i}" for i in range(7)]
POSE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw"]
FORCE_NAMES = ["fx", "fy", "fz", "tx", "ty", "tz"]


def consolidate_parquet_chunks(dataset_root: Path, relative_dir: str) -> bool:
    """Merge per-chunk parquet files into a single data.parquet per chunk.

    Returns True if any chunk was rewritten.
    """
    base_dir = dataset_root / relative_dir
    if not base_dir.exists():
        return False

    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        pa = None  # type: ignore
        pq = None  # type: ignore
        missing_exc = exc
    else:
        missing_exc = None

    changed = False
    for chunk_dir in sorted(p for p in base_dir.glob("chunk-*") if p.is_dir()):
        source_files = sorted(f for f in chunk_dir.glob("*.parquet") if f.name != "data.parquet")
        if not source_files:
            continue

        target = chunk_dir / "data.parquet"
        if len(source_files) == 1:
            src = source_files[0]
            if src == target:
                continue
            try:
                if target.exists():
                    target.unlink()
                src.rename(target)
                changed = True
            except Exception as exc:
                print(f"[WARN] failed to rename {src} to {target}: {exc}")
            continue

        if pq is None or pa is None:
            print(f"[WARN] pyarrow unavailable, cannot merge multiple parquet files in {chunk_dir}: {missing_exc}")
            continue

        tables = []
        for parquet_file in source_files:
            try:
                tables.append(pq.read_table(parquet_file))
            except Exception as exc:
                print(f"[WARN] failed to read {parquet_file}: {exc}")
        if not tables:
            continue

        try:
            merged = pa.concat_tables(tables, promote=True)
        except Exception as exc:
            print(f"[WARN] failed to concatenate tables under {chunk_dir}: {exc}")
            continue

        try:
            pq.write_table(merged, target)
        except Exception as exc:
            print(f"[WARN] failed to write merged parquet {target}: {exc}")
            continue

        changed = True
        for parquet_file in source_files:
            try:
                parquet_file.unlink(missing_ok=True)
            except Exception:
                pass

    return changed


def consolidate_videos_per_chunk(dataset_root: Path, video_keys: list[str], fps: int) -> bool:
    """Merge per-episode MP4 files into data.mp4 per chunk and video key."""
    videos_root = dataset_root / "videos"
    if not videos_root.exists():
        return False

    any_written = False
    av_module = None
    av_exc: Exception | None = None

    for key in video_keys:
        key_root = videos_root / key
        if not key_root.is_dir():
            continue
        for chunk_dir in sorted(p for p in key_root.glob("chunk-*") if p.is_dir()):
            source_files = sorted(f for f in chunk_dir.glob("*.mp4") if f.name != "data.mp4")
            if not source_files:
                continue

            out_path = chunk_dir / "data.mp4"

            if len(source_files) == 1:
                src = source_files[0]
                if src == out_path:
                    continue
                try:
                    if out_path.exists():
                        out_path.unlink()
                    src.rename(out_path)
                    any_written = True
                except Exception as exc:
                    print(f"[WARN] failed to rename {src} to {out_path}: {exc}")
                continue

            if av_module is None:
                try:
                    import av  # type: ignore
                except Exception as exc:
                    av_exc = exc
                    av_module = None
                else:
                    av_module = av  # type: ignore

            if av_module is None:
                print(f"[WARN] PyAV unavailable, cannot merge videos in {chunk_dir}: {av_exc}")
                continue

            codec_name = "mpeg4"
            width = height = None
            pix_fmt = "yuv420p"
            try:
                with av_module.open(str(source_files[0]), mode="r") as ic:
                    vstream = next(s for s in ic.streams if s.type == "video")
                    codec_name = vstream.codec_context.name or codec_name
                    width = vstream.width
                    height = vstream.height
                    if vstream.format and isinstance(vstream.format.name, str):
                        pix_fmt = vstream.format.name
            except Exception:
                pass

            try:
                with av_module.open(str(out_path), mode="w") as oc:
                    stream = oc.add_stream(codec_name, rate=fps)
                    if width and height:
                        stream.width = width
                        stream.height = height
                    stream.pix_fmt = pix_fmt

                    for mp4_file in source_files:
                        try:
                            with av_module.open(str(mp4_file), mode="r") as ic:
                                vstream = next(s for s in ic.streams if s.type == "video")
                                for frame in ic.decode(vstream):
                                    frame = frame.reformat(width=stream.width, height=stream.height, format=stream.pix_fmt)
                                    for packet in stream.encode(frame):
                                        oc.mux(packet)
                        except Exception as exc:
                            print(f"[WARN] failed to process {mp4_file}: {exc}")

                    for packet in stream.encode(None):
                        oc.mux(packet)

                any_written = True
                for mp4_file in source_files:
                    try:
                        mp4_file.unlink(missing_ok=True)
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[WARN] failed to write merged video {out_path}: {exc}")

    return any_written


def update_info_paths(info_path: Path, data_path: str | None = None, video_path: str | None = None) -> None:
    if not info_path.is_file():
        return
    try:
        info_data = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] unable to update info.json: {exc}")
        return

    changed = False
    if data_path is not None and info_data.get("data_path") != data_path:
        info_data["data_path"] = data_path
        changed = True
    if video_path is not None and info_data.get("video_path") != video_path:
        info_data["video_path"] = video_path
        changed = True
    if changed:
        info_path.write_text(json.dumps(info_data, ensure_ascii=False, indent=4), encoding="utf-8")


def find_episode_dirs(data_root: Path) -> list[Path]:
    """Return all episode directories that expose observation/leftImg/rightImg folders."""

    episodes: list[Path] = []
    for collect_dir in data_root.rglob("collect_data"):
        for candidate in sorted(collect_dir.iterdir()):
            if not candidate.is_dir():
                continue
            if not (candidate / "observation").is_dir():
                continue
            episodes.append(candidate)
    return sorted(episodes)


def load_image(path: Path) -> np.ndarray:
    """Load RGB image as uint8 numpy array (H, W, 3)."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def create_dataset(
    output_dir: Path,
    repo_id: str,
    fps: int,
    image_shape: tuple[int, int],
    use_videos: bool,
    image_writer_processes: int,
    image_writer_threads: int,
) -> LeRobotDataset:
    """Instantiate an empty LeRobot dataset ready to record episodes."""

    height, width = image_shape
    image_dtype = "video" if use_videos else "image"
    features = {
        "observation.images.left": {
            "dtype": image_dtype,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.images.right": {
            "dtype": image_dtype,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "observation.joint_positions": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "observation.end_effector_pose": {
            "dtype": "float32",
            "shape": (len(POSE_NAMES),),
            "names": POSE_NAMES,
        },
        "observation.force_readings": {
            "dtype": "float32",
            "shape": (len(FORCE_NAMES),),
            "names": FORCE_NAMES,
        },
        "observation.joint_currents": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "observation.joint_torques": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
        "observation.action_target_pose": {
            "dtype": "float32",
            "shape": (len(POSE_NAMES),),
            "names": POSE_NAMES,
        },
        "observation.button_pressed": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["pressed"],
        },
        "observation.command_success": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["success"],
        },
        "observation.force_exceeded": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["exceeded"],
        },
        "next_torque": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },        
        "action": {
            "dtype": "float32",
            "shape": (len(JOINT_NAMES),),
            "names": JOINT_NAMES,
        },
    }

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        features=features,
        robot_type="realman",
        use_videos=use_videos,
        batch_encoding_size=1,
        image_writer_processes=max(0, image_writer_processes),
        image_writer_threads=max(1, image_writer_threads),
    )


def convert_episode(dataset: LeRobotDataset, episode_dir: Path) -> None:
    """Write a single episode into the dataset."""
    obs_dir = episode_dir / "observation"
    left_dir = episode_dir / "leftImg"
    right_dir = episode_dir / "rightImg"

    obs_paths = sorted(obs_dir.glob("*.npz"), key=lambda p: int(p.stem))
    if len(obs_paths) < 2:
        return

    # Prepare lazy loaders for images
    left_paths = sorted(left_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    right_paths = sorted(right_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    if len(left_paths) != len(obs_paths) or len(right_paths) != len(obs_paths):
        raise RuntimeError(f"Image/observation count mismatch in {episode_dir}")

    next_obs = load_npz(obs_paths[0])
    task = f"realman_episode_{episode_dir.name}"

    for idx in range(len(obs_paths) - 1):
        curr_obs = next_obs
        next_obs = load_npz(obs_paths[idx + 1])

        joint_positions = curr_obs["joint_positions"].astype(np.float32)
        next_joint_positions = next_obs["joint_positions"].astype(np.float32)

        frame = {
            "task": task,
            "observation.images.left": load_image(left_paths[idx]),
            "observation.images.right": load_image(right_paths[idx]),
            "observation.state": joint_positions,
            "observation.joint_positions": joint_positions,
            "observation.end_effector_pose": curr_obs["end_effector_pose"].astype(np.float32),
            "observation.force_readings": curr_obs["force_readings"].astype(np.float32),
            "observation.joint_currents": curr_obs["joint_currents"].astype(np.float32),
            "observation.joint_torques": curr_obs["joint_torques"].astype(np.float32),
            "observation.action_target_pose": curr_obs["action_target_pose"].astype(np.float32),
            "observation.button_pressed": np.asarray([curr_obs["button_pressed"]], dtype=np.float32),
            "observation.command_success": np.asarray([curr_obs["command_success"]], dtype=np.float32),
            "observation.force_exceeded": np.asarray([curr_obs["force_exceeded"]], dtype=np.float32),
            "next_torque": next_obs["joint_torques"].astype(np.float32),
            "action": next_joint_positions,
        }

        dataset.add_frame(frame)

    dataset.save_episode()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RealMan raw data into LeRobot v3 format with actions.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data_raw"),
        help="Root directory that contains the raw RealMan collections (default: ./data_raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lerobot_realman_dataset"),
        help="Destination directory for the generated LeRobot dataset.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="realman_joint_next_action",
        help="Repo identifier used inside the dataset metadata.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frame rate used to timestamp frames (default: 30).",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="Disable video export and keep image sequences.",
    )
    default_processes = max(0, (os.cpu_count() or 2) // 2)
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=default_processes,
        help=(
            "Number of worker processes to offload image serialization (default: half of available CPU cores). "
            "Set to 0 to disable multiprocessing."
        ),
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of threads per process for image serialization (default: 4).",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Raw data root {data_root} does not exist.")

    if output_dir.exists():
        raise FileExistsError(f"Output directory {output_dir} already exists. Please remove it or choose another path.")

    episodes = find_episode_dirs(data_root)
    if not episodes:
        raise RuntimeError(f"No episodes were found under {data_root}")

    # Infer image shape from the first episode
    sample_left = next(iter((episodes[0] / "leftImg").glob("*.jpg")))
    with Image.open(sample_left) as img:
        width, height = img.size

    use_videos = not args.no_videos
    dataset = create_dataset(
        output_dir,
        args.repo_id,
        args.fps,
        image_shape=(height, width),
        use_videos=use_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    for episode_dir in tqdm(episodes, desc="Converting episodes"):
        convert_episode(dataset, episode_dir)

    dataset.finalize()

    # Merge chunk files to adhere to LeRobot v3.0 guidance
    data_changed = consolidate_parquet_chunks(output_dir, "data")
    _ = consolidate_parquet_chunks(output_dir, "meta/episodes")
    video_changed = False
    if use_videos:
        video_keys = ["observation.images.left", "observation.images.right"]
        video_changed = consolidate_videos_per_chunk(output_dir, video_keys, fps=args.fps)

    info_path = output_dir / "meta" / "info.json"
    data_path_tpl = "data/chunk-{chunk_index:03d}/data.parquet" if data_changed else None
    video_path_tpl = None
    if use_videos and video_changed:
        video_path_tpl = "videos/{video_key}/chunk-{chunk_index:03d}/data.mp4"
    update_info_paths(info_path, data_path=data_path_tpl, video_path=video_path_tpl)

    print(f"Dataset successfully written to {output_dir}")


if __name__ == "__main__":
    main()
