lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so100_follower \
  --robot.cameras='{
    "secondary_0": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30},
    "main": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=gamepad \
  --display_data=true \
  --dataset.single_task="Pick cuboid" \
  --dataset.repo_id=${HF_USER}/lerobot_testing \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=true

export HF_USER=Sri-Ram-A


rm -rf ~/.cache/huggingface/lerobot/Sri-Ram-A/lerobot_testing

lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so100_follower \
  --robot.cameras='{
    "secondary_0": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30},
    "main": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}
  }' \
  --teleop.type=gamepad \
  --display_data=true \
  --dataset.single_task="Pick cuboid" \
  --dataset.repo_id=${HF_USER}/lerobot_testing_1 \
  --dataset.num_episodes=5 \
  --dataset.episode_time_s=10 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=true

```bash
(lerobot) boston@boston:~/lerobot/src$ \
python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8081
INFO 2026-03-31 13:11:40 y_server.py:420 {'fps': 30,
 'host': '0.0.0.0',
 'inference_latency': 0.03333333333333333,
 'obs_queue_timeout': 2,
 'port': 8080}
INFO 2026-03-31 13:11:40 y_server.py:430 PolicyServer started on 0.0.0.0:8080


hostname -I
# 172.16.2.131 10.44.44.1 10.44.44.129 10.200.0.1 10.201.0.1 
# Spin up a client with:

python -m lerobot.async_inference.robot_client \
  --server_address=172.16.2.131:8081 \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so100_follower \
  --robot.cameras='{
    "main": {
      "type":"opencv",
      "index_or_path":"/dev/video2",
      "width":640,
      "height":480,
      "fps":30
    },
    "secondary_0": {
      "type":"opencv",
      "index_or_path":"/dev/video4",
      "width":640,
      "height":480,
      "fps":30
    }
  }' \
  --task="Pick cuboid" \
  --policy_type=act \
  --pretrained_name_or_path=Sri-Ram-A/act_final_third \
  --policy_device=cpu \
  --actions_per_chunk=58 \
  --chunk_size_threshold=0.1 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True
  
```


-------------------------------------------
-------------------------------------------
NAME            |    MIN |    POS |    MAX
shoulder_pan    |   1131 |   2082 |   2953
shoulder_lift   |    716 |    735 |   3175
elbow_flex      |    801 |   3106 |   3173
wrist_flex      |    679 |   2980 |   3227
gripper         |   1986 |   2061 |   3570

```json
// Calibration saved to /home/srirama/.cache/huggingface/lerobot/calibration/robots/so_follower/so100_follower.json
{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -234,
        "range_min": 1131,
        "range_max": 2953
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": 1127,
        "range_min": 716,
        "range_max": 3175
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -1013,
        "range_min": 801,
        "range_max": 3173
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": -10,
        "range_min": 679,
        "range_max": 3227
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 1709,
        "range_min": 0,
        "range_max": 4095
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": -867,
        "range_min": 1986,
        "range_max": 3570
    }
}
```