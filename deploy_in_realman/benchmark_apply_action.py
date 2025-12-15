#!/usr/bin/env python
"""
单独测试 rm_movel 下发速度（6D 笛卡尔位姿），不包含相机/网络/线程。

默认发送固定 6D 位姿多次，统计耗时。请在确保安全的环境下运行。
"""

import argparse
import logging
import time
from typing import Sequence

import numpy as np

from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Realman rm_movel latency (6D pose).")
    parser.add_argument("--robot-ip", default="192.168.10.18")
    parser.add_argument("--robot-port", type=int, default=8080)
    parser.add_argument(
        "--pose",
        type=float,
        nargs="+",
        default=[0.578506, 0.0, 0.060504, float(np.pi), 1.0, float(np.pi)],
        help="测试用 6D 位姿 [x,y,z,roll,pitch,yaw]。默认使用目标位姿。",
    )
    parser.add_argument("--runs", type=int, default=1000, help="重复次数。")
    return parser.parse_args()


def validate_pose(pose: Sequence[float]) -> list[float]:
    arr = np.asarray(pose, dtype=np.float32).flatten()
    if arr.size < 6:
        raise ValueError("pose 需要至少 6 个数 [x,y,z,roll,pitch,yaw]")
    return arr[:6].tolist()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pose_cmd = validate_pose(args.pose)

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    try:
        arm.rm_create_robot_arm(args.robot_ip, args.robot_port)
        timings = []
        for i in range(max(1, args.runs)):
            t0 = time.perf_counter()
            ret = arm.rm_movep_canfd([0.578506, 0.0, 0.060504, float(np.pi), 1.0, float(np.pi)], True, 2, 50)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            timings.append(elapsed_ms)
            logging.info("run %d ret=%s elapsed=%.2f ms", i + 1, ret, elapsed_ms)
        logging.info(
            "rm_movel benchmark finished | runs=%d avg=%.2f ms min=%.2f ms max=%.2f ms",
            len(timings),
            float(np.mean(timings)),
            float(np.min(timings)),
            float(np.max(timings)),
        )
    finally:
        try:
            arm.rm_delete_robot_arm()
        except Exception:
            pass


if __name__ == "__main__":
    main()
