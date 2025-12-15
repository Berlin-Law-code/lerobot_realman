#!/usr/bin/env python
"""
基于 realman_client.py 的思路，从真机采集观测并通过 WebSocket 调用 serve_diffusion.py。
LeRobotDataCollector 已随本目录打包（deploy_in_realman/realman_data_collector.py），无需再引用外部脚本。
"""

import argparse
import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import websockets
import threading

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

FILE_DIR = Path(__file__).resolve().parent
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))

from realman_data_collector import LeRobotDataCollector


class RealmanRobot:
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
        self._plot_enabled = plt is not None
        self._warned_plot_missing = False
        logging.info("Saving images and actions to %s", self._run_dir)
        # 0 表示不限制（执行完整序列），否则执行前 N 个
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
        cam_left = images[1].copy()
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
        obs = {
            "observation.images.cam_high": cam_high,
            "observation.images.cam_left_wrist": cam_left,
            # 服务器期望 6 维末端位姿
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
            # RM SDK rm_movej 按官方文档使用「角度制」输入（单位：度）
            target = np.asarray(act[:6], dtype=np.float32)
            joint = target
            cmd_list = joint.tolist()
            logging.info("Action: %s", cmd_list) 
            action_start = time.perf_counter()
            ret = None
            try:
                # ret = arm.rm_movep_canfd(cmd_list, True, 2, 50)
                ret = arm.rm_movel(cmd_list, 20, 0, 0, 1)
            finally:
                elapsed = time.perf_counter() - action_start
                logging.info("rm_movel ret=%s, action_idx=%s, elapsed=%.3fs", ret, idx, elapsed)
                # action_header = [
                #     "timestamp",
                #     "action_global_idx",
                #     "action_idx_in_chunk",
                #     "obs_step",
                #     "elapsed_sec",
                #     "rm_ret",
                #     *[f"joint_{i}" for i in range(len(cmd_list))],
                # ]
                # obs_step = self._last_obs_step if self._last_obs_step is not None else -1
                # row = [
                #     time.time(),
                #     self._action_counter,
                #     idx,
                #     obs_step,
                #     elapsed,
                #     ret if ret is not None else "rm_movel_error",
                #     *cmd_list,
                # ]
                # self._append_csv(self._cmd_log_path, action_header, row)
                # try:
                #     self._action_history.append([float(x) for x in cmd_list])
                #     self._update_action_plot()
                # except Exception as exc:
                #     logging.warning("Failed to update action plot: %s", exc)
                # self._action_counter += 1
            # ret = arm.rm_movep_canfd(cmd_list, True, 1, 30)
            # ret = arm.rm_movej(cmd_list, 20, 0, 0, 1)
            # time.sleep(self._command_interval)
            
            # try:
            #     joint_state = None
            #     try:
            #         status, state_data = arm.rm_get_current_arm_state()
            #         if status == 0:
            #             current_joint = np.asarray(state_data.get("joint", [0] * 7)[:7], dtype=np.float32)
            #             joint_state = current_joint
            #             delta = target - current_joint
            #             logging.info("Current joints: %s", current_joint.tolist())
            #             logging.info("Delta (cmd-current): %s", [round(x, 3) for x in delta.tolist()])
            #             # 死区：如果偏差很小就不下发，避免“点头”抖动
            #             if np.all(np.abs(delta) < self._deadband_deg):
            #                logging.info(
            #                     "Delta within deadband (%.3f deg); skipping action_idx=%s", self._deadband_deg, idx
            #                 ) 
            #                 continue
            #             if self._action_gain != 1.0:
            #                 joint = current_joint + self._action_gain * delta
            #                 logging.info(
            #                     "Applying action_gain=%.2f -> commanded joints: %s",
            #                     self._action_gain,
            #                     [round(x, 3) for x in joint.tolist()],
            #                 )
            #         else:
            #             logging.warning("rm_get_current_arm_state status=%s, using raw target", status)
            #     except Exception:
            #         pass
            #     cmd_list = joint.tolist()
            #     logging.info("Serving Diffusion Policy action: %s", cmd_list)
            #     try:
            #         write_header = not self._cmd_log_path.exists() or self._cmd_log_path.stat().st_size == 0
            #         with self._cmd_log_path.open("a", newline="") as f:
            #             writer = csv.writer(f)
            #             if write_header:
            #                 header = ["timestamp", "action_idx"] + [f"joint_{i}" for i in range(len(cmd_list))]
            #                 writer.writerow(header)
            #             writer.writerow([time.time(), idx, *cmd_list])
            #     except Exception as exc:
            #         logging.warning("Failed to write cmd_list to %s: %s", self._cmd_log_path, exc)
            #     logging.info("rm_movej sending cmd_list: %s", cmd_list)
            #     ret = arm.rm_movej(cmd_list, 20, 0, 0, 1)
            #     logging.info("rm_movej ret=%s (action_idx=%s)", ret, idx)
            #     if ret != 0:
            #         logging.error("rm_movej returned error code %s (action_idx=%s)", ret, idx)
            #         # 出错就不继续后面的动作，避免重复触发保护
            #         break
            # except Exception as exc:  # pragma: no cover - 硬件调用
            #     logging.error("Failed to send joint command (action_idx=%s): %s", idx, exc)
            # time.sleep(self._command_interval)
        logging.info("Finished %s actions in %.3fs", limit, time.perf_counter() - block_start)

    def _update_action_plot(self) -> None:
        if not self._action_history:
            return
        data = np.asarray(self._action_history, dtype=np.float32)
        steps = np.arange(data.shape[0])
        fig, ax = plt.subplots(figsize=(8, 4))
        for j in range(data.shape[1]):
            ax.plot(steps, data[:, j], label=f"joint_{j}")
        ax.set_xlabel("Action index")
        ax.set_ylabel("Commanded joint angle (deg)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        plot_path = self._run_dir / "actions_plot.png"
        fig.savefig(plot_path)
        plt.close(fig)

    def close(self) -> None:
        try:
            self._collector.stop_collection()
        except Exception:
            pass
        try:
            self._collector.cleanup()
        except Exception:
            pass


def to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


def _prepare_payload(obs: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for k, v in obs.items():
        if k.startswith("observation.images."):
            arr = np.asarray(v, dtype=np.float32) / 255.0  # 转 0-1
            if arr.shape[-1] == 3:  # HWC -> CHW
                arr = np.transpose(arr, (2, 0, 1))
            payload[k] = arr
        else:
            payload[k] = np.asarray(v, dtype=np.float32)
    return payload


async def run_loop(args: argparse.Namespace) -> None:
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
    uri = f"ws://{args.host}:{args.port}"
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            while True:
                obs = robot.get_observation()
                payload = _prepare_payload(obs)
                loop_t0 = time.perf_counter()
                await ws.send(json.dumps(to_python(payload)))
                resp_raw = await ws.recv()
                resp = json.loads(resp_raw)
                t_after_recv = time.perf_counter()
                action_arr = resp["actions"] if "actions" in resp else resp.get("action")
                if action_arr is None:
                    logging.warning("No action in response keys=%s", resp.keys())
                    continue
                arr = np.asarray(action_arr, dtype=np.float32)
                logging.info(
                    "Received action array shape: %s | ws roundtrip=%.2f ms",
                    arr.shape,
                    (t_after_recv - loop_t0) * 1000.0,
                )
                apply_t0 = time.perf_counter()
                robot.apply_action(arr)
                robot_apply_t1 = time.perf_counter()
                logging.info(
                    "Action applied | recv->apply=%.2f ms | apply->robot=%.2f ms | loop total=%.2f ms",
                    (apply_t0 - t_after_recv) * 1000.0,
                    (robot_apply_t1 - apply_t0) * 1000.0,
                    (time.perf_counter() - loop_t0) * 1000.0,
                )
    except asyncio.CancelledError:
        pass
    finally:
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realman client for serve_diffusion.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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
        default=Path("/home/belrin/Berlin/lerobot/deploy_in_realman/tmp/realman_pose_viewer"),
        help="保存输出动作和观测图片的目录（会自动创建时间戳子目录）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(run_loop(args))


if __name__ == "__main__":
    main()
