#!/usr/bin/env python
"""
基于 realman_client.py 的思路，从真机采集观测并通过 WebSocket 调用 serve_diffusion.py。
注意：需要你已有的 LeRobotDataCollector（参考 TA-VLA/examples/simple_client/5_collect_data_asynchronous.py）。
"""

import argparse
import asyncio
import importlib.util
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


def _load_collector_class(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"LeRobot data collector module not found at {path}")
    module_name = "realman_data_collector"
    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load module spec from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    if not hasattr(module, "LeRobotDataCollector"):
        raise AttributeError("Loaded module does not define LeRobotDataCollector")
    return getattr(module, "LeRobotDataCollector")


class RealmanRobot:
    def __init__(
        self,
        *,
        collector_path: Path,
        robot_ip: str,
        robot_port: int,
        camera_fps: int,
        robot_fps: int,
        max_actions_per_chunk: int,
        action_gain: float,
        deadband_deg: float,
    ) -> None:
        LeRobotDataCollector = _load_collector_class(collector_path)
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
        self._cmd_log_path = Path("/home/belrin/Berlin/lerobot/tmp/realman_cmd_list.csv")
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

    def get_observation(self) -> dict[str, Any]:
        robot_obs = self._collector.get_robot_observation()
        if robot_obs is None:
            raise RuntimeError("Failed to retrieve robot observation")
        images = self._collector.get_camera_images()
        if not images:
            raise RuntimeError("Failed to retrieve camera images")

        cam_high = images[1].copy()
        cv2.imwrite('./cam1.png', images[1])
        cam_left = images[0].copy()
        cv2.imwrite('./cam0.png', images[0])
        logging.info("cam_high_shape %s cam_left_shape %s", cam_high.shape, cam_left.shape)
        joint_deg = np.asarray(robot_obs["joint_positions"], dtype=np.float32)
        joint_rad = np.deg2rad(joint_deg)
        obs = {
            "observation.images.cam_high": cam_high,
            "observation.images.cam_left_wrist": cam_left,
            # 服务器期望弧度制
            "observation.state": joint_rad,
        }
        return obs

    def apply_action(self, actions: np.ndarray) -> None:
        arm = getattr(self._collector, "arm", None)
        if arm is None:
            logging.warning("Robot arm not initialised; dropping action.")
            return
        # 支持一段动作序列：逐条下发，每步取前 7 维为关节角
        # 服务器返回弧度制，真机 SDK 使用角度制 -> 转为度再下发
        act_array = np.rad2deg(np.asarray(actions, dtype=np.float32))
        if act_array.ndim == 1:
            act_seq = [act_array]
        else:
            act_seq = list(act_array)
        limit = len(act_seq) if self._max_actions_per_chunk <= 0 else min(len(act_seq), self._max_actions_per_chunk)
        for idx, act in enumerate(act_seq[:limit]):
            # RM SDK rm_movej 按官方文档使用「角度制」输入（单位：度）
            target = np.asarray(act[:7], dtype=np.float32)
            joint = target
            cmd_list = joint.tolist()
            ret = arm.rm_movej(cmd_list, 20, 0, 0, 1)
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
            #                 logging.info(
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
        collector_path=args.collector_path,
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        camera_fps=args.camera_fps,
        robot_fps=args.robot_fps,
        max_actions_per_chunk=args.max_actions_per_chunk,
        action_gain=args.action_gain,
        deadband_deg=args.deadband_deg,
    )
    uri = f"ws://{args.host}:{args.port}"
    try:
        async with websockets.connect(uri, max_size=None) as ws:
            while True:
                obs = robot.get_observation()
                payload = _prepare_payload(obs)
                await ws.send(json.dumps(to_python(payload)))
                resp = json.loads(await ws.recv())
                action_arr = resp["actions"] if "actions" in resp else resp.get("action")
                if action_arr is None:
                    logging.warning("No action in response keys=%s", resp.keys())
                    continue
                arr = np.asarray(action_arr, dtype=np.float32)
                logging.info("Received action array shape: %s", arr.shape)
                robot.apply_action(arr)
    except asyncio.CancelledError:
        pass
    finally:
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realman client for serve_diffusion.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--collector-path",
        type=Path,
        default=Path("/home/belrin/Berlin/TA-VLA/examples/simple_client/5_collect_data_asynchronous.py"),
        help="Path to LeRobotDataCollector module.",
    )
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
        "--benchmark-apply",
        action="store_true",
        help="只测试 robot.apply_action 的耗时，不连接 WebSocket。",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="apply_action 重复次数。",
    )
    parser.add_argument(
        "--benchmark-action",
        type=float,
        nargs="+",
        default=[0, 0, 0, 0, 0, 0, 0],
        help="测试用关节角（弧度，长度>=6/7）。默认全0。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.benchmark_apply:
        # 基准测试：仅测 apply_action，不做通信
        robot = RealmanRobot(
            collector_path=args.collector_path,
            robot_ip=args.robot_ip,
            robot_port=args.robot_port,
            camera_fps=args.camera_fps,
            robot_fps=args.robot_fps,
            max_actions_per_chunk=args.max_actions_per_chunk,
            action_gain=args.action_gain,
            deadband_deg=args.deadband_deg,
        )
        try:
            action = np.asarray(args.benchmark_action, dtype=np.float32)
            if action.ndim == 1 and action.size >= 6:
                pass
            elif action.ndim == 2:
                pass
            else:
                raise ValueError("benchmark_action 需要长度>=6的一维或二维数组")
            timings = []
            for i in range(max(1, args.benchmark_runs)):
                t0 = time.perf_counter()
                robot.apply_action(action)
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)
                logging.info("apply_action run %d took %.2f ms", i + 1, timings[-1])
            logging.info(
                "apply_action benchmark finished | runs=%d avg=%.2f ms min=%.2f ms max=%.2f ms",
                len(timings),
                float(np.mean(timings)),
                float(np.min(timings)),
                float(np.max(timings)),
            )
        finally:
            robot.close()
    else:
        asyncio.run(run_loop(args))


if __name__ == "__main__":
    main()
