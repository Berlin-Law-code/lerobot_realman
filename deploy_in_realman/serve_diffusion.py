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
        default=Path("/home/belrin/Berlin/lerobot/outputs/train/realman_data_single_peg_pose_20251210_v21/checkpoints/last/pretrained_model"),
        help="Path to pretrained_model directory (contains config.json, model.safetensors, etc.).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-latency", action="store_true")
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
) -> None:
    device = policy.config.device
    async for msg in ws:
        try:
            payload = json.loads(msg)
        except json.JSONDecodeError:
            await ws.send(json.dumps({"error": "invalid json"}))
            continue

        try:
            obs = to_tensor(payload)
            start = time.perf_counter()
            with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16):
                if getattr(policy, "_queues", None) is None:
                    policy.reset()
                norm_obs = preprocessor(obs)

                # Replicate DiffusionPolicy.select_action logic but return full chunk
                batch = {k: v for k, v in norm_obs.items() if v is not None}
                if not batch:
                    raise ValueError("Empty observation after preprocessing.")
                if policy.config.image_features:
                    batch = dict(batch)
                    imgs = [batch.get(key) for key in policy.config.image_features]
                    if any(img is None for img in imgs):
                        raise ValueError("Missing image in observation batch.")
                    # 与训练保持一致：将相机维度叠在倒数第4维，得到 (B, num_cams, C, H, W)
                    batch[OBS_IMAGES] = torch.stack(imgs, dim=-4)
                # Populate queues (handles initial padding)
                policy._queues = populate_queues(policy._queues, batch, exclude_keys=[])

                actions = policy.predict_action_chunk(batch)  # (B, horizon, act_dim)
                # Update action queue for potential future use
                policy._queues[ACTION].clear()
                policy._queues[ACTION].extend(actions.transpose(0, 1))

                action_chunk = actions[0]  # remove batch dim -> (horizon, act_dim)
                action_dict = postprocessor({ACTION: action_chunk})

                if isinstance(action_dict, dict):
                    chunk = action_dict.get(ACTION, action_dict)
                else:
                    chunk = action_dict

                chunk_py = to_python(chunk)
                # 兼容老客户端：既返回完整序列，也返回首步
                first_py = chunk_py[0] if isinstance(chunk_py, list) and chunk_py else chunk_py
                action = {"actions": chunk_py, "action": first_py}
            latency_ms = (time.perf_counter() - start) * 1000.0
            if log_latency:
                logging.info("infer latency: %.2f ms", latency_ms)
            await ws.send(json.dumps(to_python(action)))
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
            lambda ws: handle(ws, None, policy, preprocessor, postprocessor, args.log_latency),
            args.host,
            args.port,
            max_size=None,
        )
        logging.info("Serving Diffusion Policy on ws://%s:%d", args.host, args.port)
        await asyncio.Future()  # run forever

    asyncio.run(_run_server())


if __name__ == "__main__":
    main()
