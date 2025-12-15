conda activate DP
lerobot-train --config_path=examples/training/diffusion_realman_train_pose_single_view.json

conda activate DP
python serve_diffusion.py --port 8001

conda activate vla
python /home/belrin/Berlin/lerobot/serve_diffusion_realman_client.py --port 8001
python /home/belrin/Berlin/lerobot/deploy_in_realman/serve_diffusion_realman_client_pose_viewer_all.py --port 8001

(lerobot) belrin@belrin-PRC-Desktop:~/Berlin$ python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py   --repo-id realman_data_single_peg_pose_20251210_v21   --root /home/belrin/Berlin   --push-to-hub false   --force-conversion   --video-file-size-in-mb 1
(tavla) belrin@belrin-PRC-Desktop:~/Berlin$ python TA-VLA/convert_realman_to_lerobot_v21.py --overwrite
(lerobot) belrin@belrin-PRC-Desktop:~/Berlin$ /home/belrin/Berlin/TA-VLA/compute_video_stats.sh

conda activate DPlocal
python /home/belrin/Berlin/lerobot/deploy_in_realman/run_diffusion_realman_local_single_view.py

conda activate vla
python /home/belrin/Berlin/Collerct_data/3Dmouse/1_goto_initial_position.py
python /home/belrin/Berlin/Collerct_data/3Dmouse/2_force_sensor_set_zeros.py