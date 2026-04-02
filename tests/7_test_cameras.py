import cv2

def test_camera_resolution(device_path, width, height):
    cap = cv2.VideoCapture(device_path)
    if not cap.isOpened():
        print(f"❌ {device_path}: Cannot open")
        return False
    
    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Read actual resolution
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Try to read a frame
    ret, frame = cap.read()
    
    cap.release()
    
    if ret and actual_w == width and actual_h == height:
        print(f"✅ {device_path}: Supports {width}×{height}")
        return True
    else:
        print(f"⚠️  {device_path}: Actual {actual_w}×{actual_h} (requested {width}×{height})")
        return False

# Test each camera
test_camera_resolution(0, 320, 240)   # /dev/video0
test_camera_resolution(2, 320, 240)   # /dev/video2
test_camera_resolution(4, 320, 240)   # /dev/video4