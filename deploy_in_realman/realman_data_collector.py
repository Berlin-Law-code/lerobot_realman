#!/usr/bin/env python3
"""
基于 /home/belrin/Berlin/Collerct_data/3Dmouse/3_collect_data_synchronous.py 的本地化精简版。
仅保留 serve_diffusion 客户端所需：机器人连接、状态读取、双相机采集与清理。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyrealsense2 as rs

from Robotic_Arm.rm_robot_interface import *

logger = logging.getLogger(__name__)


@dataclass
class AssetsConfig:
    """保持接口兼容的占位配置。"""

    asset_id: str = "realman"


class LeRobotDataCollector:
    """Realman + 双 RealSense 最小采集器（同步采集思路）。"""

    def __init__(
        self,
        robot_ip: str = "192.168.10.18",
        robot_port: int = 8080,
        data_dir: str | os.PathLike[str] = "/tmp/realman_data",
        episode_length: int = 1000,
        fps: int = 30,
        show_video: bool = True,
        video_scale: float = 0.8,
        camera_fps: int | None = None,
        robot_fps: int = 120,
        **_: Any,
    ) -> None:

        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.data_dir = Path(data_dir)
        self.episode_length = episode_length
        self.fps = fps
        self.dt = 1.0 / max(1, fps)
        self.show_video = show_video
        self.video_scale = video_scale

        self.camera_fps = camera_fps if camera_fps is not None else fps
        self.robot_fps = robot_fps
        self.dt_cam = 1.0 / max(1, self.camera_fps)
        self.dt_robot = 1.0 / max(1, self.robot_fps)

        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 运行时状态
        self.arm: RoboticArm | None = None
        self.callback_ptr: Any | None = None
        self.current_robot_state: dict[str, Any] | None = None
        self.force_threshold: float = 25.0
        self.force_exceeded: bool = False

        self.camera_pipelines: list[rs.pipeline] = []
        self.img_list: list[np.ndarray] = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]

        self.stop_flag = False
        self.camera_thread_stop = False
        self.threads: list[threading.Thread] = []

        self.state_lock = threading.Lock()
        self.current_to_torque_coeffs = np.ones(7, dtype=float)

    # --------------------------------------------------------------------- Robot init
    def initialize_robot(self) -> bool:
        """连接机器人并注册状态回调。"""
        try:
            self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            handle = self.arm.rm_create_robot_arm(self.robot_ip, self.robot_port)
            logger.info("机器人连接成功，ID: %s", getattr(handle, "id", "unknown"))

            custom_config = rm_udp_custom_config_t()
            custom_config.aloha_state = 0
            custom_config.joint_speed = 0
            custom_config.lift_state = 0
            custom_config.expand_state = 0
            custom_config.arm_current_status = 0
            custom_config.hand_state = 0

            period_ms = max(1, int(round(1000.0 / self.robot_fps)))
            udp_config = rm_realtime_push_config_t(
                period_ms,
                True,
                8089,
                0,
                "192.168.10.50",
                custom_config,
            )
            result = self.arm.rm_set_realtime_push(udp_config)
            if result != 0:
                logger.error("UDP配置设置失败，错误码: %s", result)
                return False

            self.callback_ptr = rm_realtime_arm_state_callback_ptr(self._robot_state_callback)
            self.arm.rm_realtime_arm_state_call_back(self.callback_ptr)
            logger.info("机器人状态回调注册成功")

            time.sleep(1.0)  # 等待回调填充
            return True

        except Exception as exc:  # pragma: no cover - 硬件相关
            logger.error("机器人初始化失败: %s", exc)
            return False

    # ----------------------------------------------------------------- Camera support
    def initialize_cameras(self) -> bool:
        """初始化两路 RealSense 相机。"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) < 2:
                logger.error("找到 %d 个 RealSense 设备，需要至少 2 个", len(devices))
                return False

            for idx, device in enumerate(devices[:2]):
                pipeline = rs.pipeline()
                config = rs.config()

                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                logger.info("初始化相机 %d: %s (序列号: %s)", idx + 1, name, serial)

                for sensor in device.query_sensors():
                    try:
                        if sensor.supports(rs.option.frames_queue_size):
                            sensor.set_option(rs.option.frames_queue_size, 1)
                    except Exception as exc:
                        logger.debug("设置 frames_queue_size 失败(相机%d): %s", idx + 1, exc)

                config.enable_device(serial)
                config.enable_stream(
                    rs.stream.color,
                    640,
                    480,
                    rs.format.bgr8,
                    int(self.camera_fps),
                )

                pipeline.start(config)
                # 预热，确保有帧输出
                for _ in range(30):
                    try:
                        frames = pipeline.wait_for_frames(timeout_ms=3000)
                        if frames and frames.get_color_frame():
                            break
                    except Exception:
                        pass

                self.camera_pipelines.append(pipeline)
                logger.info("相机 %d 初始化完成", idx + 1)

            logger.info("相机初始化完成，共 %d 路", len(self.camera_pipelines))
            return True

        except Exception as exc:  # pragma: no cover - 硬件相关
            logger.error("相机初始化失败: %s", exc)
            return False

    def run_thread_cam(self, cam_idx: int) -> None:
        """持续更新指定相机的图像缓存。"""
        assert 0 <= cam_idx < len(self.camera_pipelines)
        logger.info("相机线程 %d 启动", cam_idx)

        pipeline = self.camera_pipelines[cam_idx]
        while not self.stop_flag and not self.camera_thread_stop:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame() if frames else None
                if color_frame:
                    image = np.asanyarray(color_frame.get_data())
                    self.img_list[cam_idx] = image.copy()
                else:
                    time.sleep(0.01)
            except Exception as exc:
                logger.warning("相机线程 %d 获取帧失败: %s", cam_idx, exc)
                time.sleep(0.05)

        logger.info("相机线程 %d 结束", cam_idx)

    def get_camera_images(self) -> list[np.ndarray]:
        """返回当前缓存的相机图像（BGR）。"""
        return [img.copy() for img in self.img_list]

    # ---------------------------------------------------------------- Robot polling
    def _robot_state_callback(self, data: Any) -> None:  # pragma: no cover - 硬件回调
        try:
            with self.state_lock:
                force_data = getattr(data, "force_sensor", None)
                if force_data is not None:
                    force_dict = force_data.to_dict()
                    zero_force = force_dict.get("zero_force", [0, 0, 0, 0, 0, 0])
                    force_z = float(zero_force[2]) if len(zero_force) >= 3 else 0.0
                    self.force_exceeded = abs(force_z) > self.force_threshold
                else:
                    force_dict = {"zero_force": [0, 0, 0, 0, 0, 0]}
                    force_z = 0.0
                    self.force_exceeded = False

                self.current_robot_state = {
                    "timestamp": time.time(),
                    "force_sensor": force_dict,
                    "force_z": force_z,
                    "force_exceeded": self.force_exceeded,
                }
        except Exception as exc:
            logger.error("机器人状态回调错误: %s", exc)

    def get_robot_observation(self) -> dict[str, np.ndarray] | None:
        """读取最新的机器人观测。"""
        if self.arm is None:
            return None

        try:
            status_arm, state_data = self.arm.rm_get_current_arm_state()
            if status_arm != 0:
                logger.warning("获取机器人状态失败")
                return None

            joint_positions = np.array(state_data.get("joint", [0] * 7)[:7], dtype=float)
            end_effector_pose = np.array(state_data.get("pose", [0] * 6)[:6], dtype=float)

            with self.state_lock:
                force_state = self.current_robot_state or {
                    "force_sensor": {"zero_force": [0] * 6},
                    "force_z": 0.0,
                    "force_exceeded": False,
                }

            try:
                status_current, current_data = self.arm.rm_get_current_joint_current()
                joint_currents = np.array(current_data[:7] if status_current == 0 else [0] * 7, dtype=float)
            except Exception:
                joint_currents = np.zeros(7, dtype=float)

            try:
                coeffs = np.asarray(self.current_to_torque_coeffs, dtype=float).flatten()
                if coeffs.size < 7:
                    tmp = np.ones(7, dtype=float)
                    tmp[: coeffs.size] = coeffs
                    coeffs = tmp
                joint_torques = joint_currents * coeffs[:7]
            except Exception:
                joint_torques = joint_currents

            return {
                "joint_positions": joint_positions,
                "end_effector_pose": end_effector_pose,
                "force_readings": np.array(force_state["force_sensor"].get("zero_force", [0] * 6), dtype=float),
                "joint_currents": joint_currents,
                "joint_torques": joint_torques,
                "force_exceeded": bool(force_state["force_exceeded"]),
            }
        except Exception as exc:
            logger.error("获取机器人观测失败: %s", exc)
            return None

    # ---------------------------------------------------------------- Shutdown logic
    def stop_collection(self) -> None:
        """标记停止并等待线程退出。"""
        self.stop_flag = True
        self.camera_thread_stop = True
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        self.threads.clear()

    def cleanup(self) -> None:  # pragma: no cover - 硬件相关
        """释放硬件资源。"""
        self.stop_collection()

        for pipeline in self.camera_pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        self.camera_pipelines.clear()

        if self.arm is not None:
            try:
                udp_config = rm_realtime_push_config_t(
                    100,
                    False,
                    8089,
                    0,
                    "192.168.10.50",
                    rm_udp_custom_config_t(),
                )
                self.arm.rm_set_realtime_push(udp_config)
            except Exception:
                pass

            try:
                self.arm.rm_delete_robot_arm()
            except Exception:
                pass

            self.arm = None

        self.callback_ptr = None


def main() -> None:  # pragma: no cover - 手动测试入口
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    collector = LeRobotDataCollector()
    if not collector.initialize_robot():
        return
    if not collector.initialize_cameras():
        collector.cleanup()
        return

    for idx in range(len(collector.camera_pipelines)):
        thread = threading.Thread(target=collector.run_thread_cam, args=(idx,), daemon=True)
        collector.threads.append(thread)
        thread.start()

    logger.info("开始轮询观察数据，按 Ctrl+C 退出")
    try:
        while True:
            obs = collector.get_robot_observation()
            if obs:
                logger.info("Joint positions: %s", np.array2string(obs["joint_positions"], precision=3))
            time.sleep(0.2)
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    finally:
        collector.cleanup()


if __name__ == "__main__":  # pragma: no cover
    main()
