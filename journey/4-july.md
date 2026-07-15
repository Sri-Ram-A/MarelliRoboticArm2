```bash
lerobot-calibrate \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so100_follower

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


{
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": -312,
        "range_min": 787,
        "range_max": 3354
    },
    "shoulder_lift": {
        "id": 2,
        "drive_mode": 0,
        "homing_offset": 491,
        "range_min": 771,
        "range_max": 3378
    },
    "elbow_flex": {
        "id": 3,
        "drive_mode": 0,
        "homing_offset": -212,
        "range_min": 810,
        "range_max": 3078
    },
    "wrist_flex": {
        "id": 4,
        "drive_mode": 0,
        "homing_offset": 1181,
        "range_min": 725,
        "range_max": 3255
    },
    "wrist_roll": {
        "id": 5,
        "drive_mode": 0,
        "homing_offset": 1813,
        "range_min": 0,
        "range_max": 4095
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": -808,
        "range_min": 2011,
        "range_max": 3548
    }
}