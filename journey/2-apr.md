
# April 1st - Marelli didnt visit

## Playing with robot
- Looking at the motor labels I have - Feetech
- I am able to control motor using only degrees (tests/1_motor_testing.py)
- I was playing around with : ~/RoboticArm/lerobot/src/lerobot/robots/so_follower/__init__.py 
- Clearly understood working of motors with degress :
(".pos" must be appended with the motor_ids in the action dictionary for motor to move using any degree). Understood How SO100Follower is implemented using dracas library for dataclasses
- Now,plaing with the recorded dataset to understand some stats 

## Playing with dataset
- Dataset is Sri-Ram-A/pnp1 from hugging face
- Running in Interactive terminal of VSCode : ~/RoboticArm/tests/2_dataset.py
- ![Lerobot Visualizer 6 Actions](actions.png)
- Below conclusions and outputs are in : ~/RoboticArm/tests/2_dataset.py
  - Actions in dataset are in radians 
  - Both The Camera Images are normalised 
  - Look at 
![Action Stats](action-stats.png)
- The model outputs Normalised actions , which must be denormalised and then converted to degrees 
- Camera footage is internally normalised
  
# Trying to run inference
- https://www.mintlify.com/huggingface/lerobot/installation
pip install lerobot[damiao]
sudo apt install v4l-utils

(lerobot) ~/Documents/sr_proj/RoboticArm$ v4l2-ctl -d /dev/video2 --get-fmt-video
Format Video Capture:
        Width/Height      : 1920/1080
        Pixel Format      : 'YUYV' (YUYV 4:2:2)
        Field             : None
        Bytes per Line    : 3840
        Size Image        : 4147200
        Colorspace        : sRGB
        Transfer Function : Rec. 709
        YCbCr/HSV Encoding: ITU-R 601
        Quantization      : Default (maps to Limited Range)
        Flags             : 

# Camera backend issue
- https://discuss.huggingface.co/t/lerobot-camera-backend-issues/173200?utm_source=chatgpt.com
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
Do this:
micromamba remove opencv -y
micromamba install -c conda-forge opencv=4.12 -y
Then verify:
python -c "import cv2; print(cv2.__version__)" # 4.12.0