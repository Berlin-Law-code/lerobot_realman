#!/usr/bin/env python
import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import websockets
from websockets.server import WebSocketServerProtocol

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.policies.utils import populate_queues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Diffusion Policy over WebSocket.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/belrin/Berlin/lerobot/outputs/train/realman_data_single_peg_pose_v21/checkpoints/last/pretrained_model"),
        help="Path to pretrained_model directory (contains config.json, model.safetensors, etc.).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-latency", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=8, help="伪造动作序列的长度。")
    return parser.parse_args()


def to_tensor(obj: Any) -> Any:
    if isinstance(obj, list):
        return torch.tensor(np.array(obj), dtype=torch.float32)
    if isinstance(obj, dict):
        return {k: to_tensor(v) for k, v in obj.items()}
    return obj


def to_python(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj


async def handle(
    ws: WebSocketServerProtocol,
    path: str | None,
    policy,
    preprocessor: PolicyProcessorPipeline,
    postprocessor: PolicyProcessorPipeline,
    log_latency: bool,
    chunk_size: int,
) -> None:
    async for msg in ws:
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            await ws.send(json.dumps({"error": "invalid json"}))
            continue

        try:
            # 返回伪造的动作序列（仅一帧），更像真实输出用 list 包裹
            # 6D 位姿：[x, y, z, roll, pitch, yaw]
            target_pose = [0.578506, 0.0, 0.060504, float(np.pi), 0.0, float(np.pi)]
            chunk_py = [target_pose for _ in range(max(1, chunk_size))]  # 形如 (horizon, act_dim)
            first_py = chunk_py[0]
            action = {"actions": chunk_py, "action": first_py}
            await ws.send(json.dumps(action))
        except Exception as exc:  # pragma: no cover - serve-time errors
            logging.exception("Inference failed")
            await ws.send(json.dumps({"error": str(exc)}))
            await ws.close(code=1011, reason=str(exc))
            break


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load config and policy
    cfg = PreTrainedConfig.from_pretrained(args.model_dir)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.model_dir, config=cfg)
    policy.to(args.device)
    policy.eval()
    policy.reset()

    # Load pre/post processors
    overrides = {"device_processor": {"device": args.device}}
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        args.model_dir, config_filename="policy_preprocessor.json", overrides=overrides
    )
    postprocessor = PolicyProcessorPipeline.from_pretrained(
        args.model_dir, config_filename="policy_postprocessor.json", overrides=overrides
    )

    async def _run_server():
        server = await websockets.serve(
            lambda ws: handle(ws, None, policy, preprocessor, postprocessor, args.log_latency, args.chunk_size),
            args.host,
            args.port,
            max_size=None,
        )
        logging.info("Serving Diffusion Policy on ws://%s:%d", args.host, args.port)
        await asyncio.Future()  # run forever

    asyncio.run(_run_server())


if __name__ == "__main__":
    main()
