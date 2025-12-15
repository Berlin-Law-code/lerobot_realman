#!/usr/bin/env python
"""
本地推理 + 机器人交互一体运行：不走 WebSocket，单进程完成采集-推理-下发。
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
import threading

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.policies.utils import populate_queues
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES

import cv2  # type: ignore

FILE_DIR = Path(__file__).resolve().parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))

from realman_data_collector import LeRobotDataCollector  # noqa: E402


def to_tensor(obj: Any) -> Any:
    if isinstance(obj, list):
        return torch.tensor(np.array(obj), dtype=torch.float32)
    if isinstance(obj, dict):
        return {k: to_tensor(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return torch.tensor(obj, dtype=torch.float32)
    return obj


def to_python(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


class RealmanRobot:
    """精简版：采集观测 + 下发动作，带日志和图片保存。"""

    def __init__(
        self,
        *,
        robot_ip: str,
        robot_port: int,
        camera_fps: int,
        robot_fps: int,
        max_actions_per_chunk: int,
        action_gain: float,
        deadband_deg: float,
        save_dir: Path,
    ) -> None:
        self._collector = LeRobotDataCollector(
            robot_ip=robot_ip,
            robot_port=robot_port,
            data_dir="/tmp/realman_client_unused",
            episode_length=50,
            fps=camera_fps,
            show_video=False,
            camera_fps=camera_fps,
            robot_fps=robot_fps,
            teleop_fps=robot_fps,
        )

        if not self._collector.initialize_robot():
            raise RuntimeError("Failed to initialize Realman robot connection")
        if not self._collector.initialize_cameras():
            raise RuntimeError("Failed to initialize RealSense cameras")

        self._collector.stop_flag = False
        self._collector.camera_thread_stop = False
        self._collector.threads = getattr(self._collector, "threads", [])
        self._command_interval = float(getattr(self._collector, "dt_robot", 0.001))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = Path(save_dir).expanduser()
        self._run_dir = base_dir / f"run_{timestamp}"
        self._image_dir = self._run_dir / "images"
        self._cmd_log_path = self._run_dir / "actions.csv"
        self._obs_log_path = self._run_dir / "observations.csv"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._step_idx = 0
        self._action_counter = 0
        self._last_obs_step = None
        self._action_history: list[list[float]] = []
        self._max_actions_per_chunk = int(max_actions_per_chunk)
        self._action_gain = float(action_gain)
        self._deadband_deg = float(deadband_deg)

        # 启动摄像头线程（最多两路，high + left_wrist）
        self._camera_threads = []
        for cam_idx in range(min(2, len(self._collector.camera_pipelines))):
            thread = threading.Thread(
                target=self._collector.run_thread_cam,
                args=(cam_idx,),
                name=f"CameraReader-{cam_idx}",
                daemon=True,
            )
            thread.start()
            self._camera_threads.append(thread)
            self._collector.threads.append(thread)

        time.sleep(1.0)  # 等待相机缓冲

    def _append_csv(self, path: Path, header: list[str], row: list[Any]) -> None:
        try:
            write_header = not path.exists() or path.stat().st_size == 0
            with path.open("a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
        except Exception as exc:
            logging.warning("Failed to write %s: %s", path, exc)

    def get_observation(self) -> dict[str, Any]:
        robot_obs = self._collector.get_robot_observation()
        if robot_obs is None:
            raise RuntimeError("Failed to retrieve robot observation")
        images = self._collector.get_camera_images()
        if not images:
            raise RuntimeError("Failed to retrieve camera images")

        cam_high = images[0].copy()
        cam_left = images[1].copy() if len(images) > 1 else cam_high.copy()
        logging.info("cam_high_shape %s cam_left_shape %s", cam_high.shape, cam_left.shape)
        step_idx = self._step_idx
        self._step_idx += 1
        self._last_obs_step = step_idx
        timestamp = time.time()
        cam_high_path = self._image_dir / f"{step_idx:06d}_cam_high.png"
        cam_left_path = self._image_dir / f"{step_idx:06d}_cam_left_wrist.png"
        try:
            cv2.imwrite(str(cam_high_path), cam_high)
            cv2.imwrite(str(cam_left_path), cam_left)
        except Exception as exc:
            logging.warning("Failed to save images for step %s: %s", step_idx, exc)
        pose = np.asarray(robot_obs["end_effector_pose"], dtype=np.float32).reshape(-1)
        obs_header = [
            "timestamp",
            "step",
            "cam_high_path",
            "cam_left_wrist_path",
            *[f"pose_{i}" for i in range(len(pose))],
        ]
        obs_row = [timestamp, step_idx, str(cam_high_path), str(cam_left_path), *pose.astype(float).tolist()]
        self._append_csv(self._obs_log_path, obs_header, obs_row)
        # obs = {
        #     "observation.images.cam_high": cam_high,
        #     "observation.images.cam_left_wrist": cam_left,
        #     "observation.state": pose,
        # }
        obs = {
            "observation.images.cam_left_wrist": cam_left,
            "observation.state": pose,
        }
        return obs

    def apply_action(self, actions: np.ndarray) -> None:
        arm = getattr(self._collector, "arm", None)
        if arm is None:
            logging.warning("Robot arm not initialised; dropping action.")
            return
        if actions.ndim == 1:
            act_seq = [actions]
        else:
            act_seq = list(actions)
        limit = len(act_seq) if self._max_actions_per_chunk <= 0 else min(len(act_seq), self._max_actions_per_chunk)
        block_start = time.perf_counter()
        for idx, act in enumerate(act_seq[:limit]):
            target = np.asarray(act[:6], dtype=np.float32)
            joint = target
            cmd_list = joint.tolist()
            logging.info("Action (movel, deg): %s", cmd_list)
            action_start = time.perf_counter()
            ret = None
            try:
                ret = arm.rm_movep_canfd(cmd_list, True, 2, 50)
                # ret = arm.rm_movej_p(cmd_list, 20, 0, 0, 1)
                time.sleep(0.03)
            finally:
                elapsed = time.perf_counter() - action_start
                logging.info("rm_movel ret=%s, action_idx=%s, elapsed=%.3fs", ret, idx, elapsed)
        logging.info("Finished %s actions in %.3fs", limit, time.perf_counter() - block_start)

    def close(self) -> None:
        try:
            self._collector.stop_collection()
        except Exception:
            pass
        try:
            self._collector.cleanup()
        except Exception:
            pass


def _prepare_payload(obs: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for k, v in obs.items():
        if k.startswith("observation.images."):
            arr = np.asarray(v, dtype=np.float32) / 255.0
            if arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1))
            payload[k] = arr
        else:
            payload[k] = np.asarray(v, dtype=np.float32)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local DiffusionPolicy runner (no server-client).")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            "/home/belrin/Berlin/lerobot/outputs/train/realman_data_single_peg_pose_20251210_v21_single_view/checkpoints/last/pretrained_model"
        ),
        help="Path to pretrained_model directory.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-latency", action="store_true")
    parser.add_argument("--robot-ip", default="192.168.10.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument("--robot-fps", type=int, default=15)
    parser.add_argument(
        "--max-actions-per-chunk",
        type=int,
        default=0,
        help="0=执行完整序列，正数表示每个chunk只执行前N步（调试闭环可设1）。",
    )
    parser.add_argument(
        "--action-gain",
        type=float,
        default=1.0,
        help=">1 放大(目标-当前)的差值，<1 缩小，=1 直接使用策略输出。",
    )
    parser.add_argument(
        "--deadband-deg",
        type=float,
        default=0.2,
        help="偏差低于该角度（度）时跳过下发，减少抖动/点头。",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("/home/belrin/Berlin/lerobot/tmp/realman_pose_viewer_local"),
        help="保存输出动作和观测图片的目录（会自动创建时间戳子目录）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 加载策略
    cfg = PreTrainedConfig.from_pretrained(args.model_dir)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.model_dir, config=cfg)
    policy.to(args.device)
    policy.eval()
    policy.reset()

    overrides = {"device_processor": {"device": args.device}}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.model_dir, config_filename="policy_preprocessor.json", overrides=overrides
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.model_dir, config_filename="policy_postprocessor.json", overrides=overrides
    )

    robot = RealmanRobot(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        camera_fps=args.camera_fps,
        robot_fps=args.robot_fps,
        max_actions_per_chunk=args.max_actions_per_chunk,
        action_gain=args.action_gain,
        deadband_deg=args.deadband_deg,
        save_dir=args.save_dir,
    )

    try:
        while True:
            start = time.perf_counter()
            obs = robot.get_observation()
            payload = _prepare_payload(obs)
            obs_tensor = to_tensor(payload)
            med = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=args.device, dtype=torch.float16):
                if getattr(policy, "_queues", None) is None:
                    policy.reset()
                norm_obs = preprocessor(obs_tensor)
                batch = {k: v for k, v in norm_obs.items() if v is not None}
                if not batch:
                    raise ValueError("Empty observation after preprocessing.")
                if policy.config.image_features:
                    batch = dict(batch)
                    imgs = [batch.get(key) for key in policy.config.image_features]
                    if any(img is None for img in imgs):
                        raise ValueError("Missing image in observation batch.")
                    batch[OBS_IMAGES] = torch.stack(imgs, dim=-4)
                policy._queues = populate_queues(policy._queues, batch, exclude_keys=[])
                actions = policy.predict_action_chunk(batch)  # (B, horizon, act_dim)
                policy._queues[ACTION].clear()
                policy._queues[ACTION].extend(actions.transpose(0, 1))
                action_chunk = actions[0]
                action_dict = postprocessor({ACTION: action_chunk})
                chunk = action_dict.get(ACTION, action_dict) if isinstance(action_dict, dict) else action_dict
                chunk_py = to_python(chunk)
                # target_pose = [0.578506, 0.0, 0.060504, float(np.pi), 0.0, float(np.pi)]
                # chunk_py = [target_pose for _ in range(max(1, 8))]
                first_py = chunk_py[0] if isinstance(chunk_py, list) and chunk_py else chunk_py
            latency_ms = (time.perf_counter() - start) * 1000.0
            latency_ms_2 = (time.perf_counter() - med) * 1000.0
            logging.info("latency_ms %.2f ms", latency_ms)
            logging.info("latency_ms_2 %.2f ms", latency_ms_2)
            if args.log_latency:
                logging.info("infer latency: %.2f ms | first_action=%s", latency_ms, first_py)
            arr = np.asarray(chunk_py, dtype=np.float32)
            apply_t0 = time.perf_counter()
            robot.apply_action(arr)
            apply_ms = (time.perf_counter() - apply_t0) * 1000.0
            if args.log_latency:
                logging.info("apply_action took %.2f ms", apply_ms)
    except KeyboardInterrupt:
        logging.info("Received KeyboardInterrupt, shutting down.")
    finally:
        robot.close()


if __name__ == "__main__":
    main()
