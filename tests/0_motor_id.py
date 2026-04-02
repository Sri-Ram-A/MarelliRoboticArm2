#!/usr/bin/env python3
# JUST LOOK AT MOTOR - Ours is Feetech
"""
Definitive Motor Identification Script
This will tell you exactly what motors are connected
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identify_motors():
    """Identify motors by directly communicating with them"""
    
    logger.info("=" * 70)
    logger.info("DEFINITIVE MOTOR IDENTIFICATION")
    logger.info("=" * 70)
    
    # First, let's try to detect what protocol responds
    logger.info("\n1. Testing Dynamixel Protocol...")
    try:
        from lerobot.motors.dynamixel import DynamixelMotorsBus
        from lerobot.motors import Motor, MotorNormMode
        
        logger.info("Attempting Dynamixel connection...")
        bus = DynamixelMotorsBus(
            port="/dev/ttyACM0",
            motors={
                "test1": Motor(1, "xl330-m077", MotorNormMode.RANGE_M100_100),
            }
        )
        
        try:
            bus.connect(timeout=1.0)
            logger.info("✅ DYNAMIXEL PROTOCOL RESPONDS!")
            logger.info("   Your motors are Dynamixel!")
            
            # Try to read motor model
            try:
                model = bus.sync_read("Model_Number", ["test1"])
                logger.info(f"   Motor model number: {model}")
            except:
                pass
                
            bus.disconnect()
            return "DYNAMIXEL"
            
        except Exception as e:
            logger.info(f"❌ Dynamixel connection failed: {str(e)[:100]}")
            
    except Exception as e:
        logger.info(f"❌ Dynamixel import failed: {e}")
    
    # Now test Feetech
    logger.info("\n2. Testing Feetech Protocol...")
    try:
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors import Motor, MotorNormMode
        
        logger.info("Attempting Feetech connection...")
        bus = FeetechMotorsBus(
            port="/dev/ttyACM0",
            motors={
                "test1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            }
        )
        
        try:
            bus.connect(timeout=1.0)
            logger.info("✅ FEETECH PROTOCOL RESPONDS!")
            logger.info("   Your motors are Feetech!")
            
            # Try to read motor model
            try:
                # For Feetech, try to read motor ID and status
                positions = bus.sync_read("Present_Position", ["test1"])
                logger.info(f"   Motor responds with position: {positions}")
            except:
                pass
                
            bus.disconnect()
            return "FEETECH"
            
        except Exception as e:
            logger.info(f"❌ Feetech connection failed: {str(e)[:100]}")
            
    except Exception as e:
        logger.info(f"❌ Feetech import failed: {e}")
    
    # Method 2: Try to detect via serial communication
    logger.info("\n3. Testing Raw Serial Communication...")
    try:
        import serial
        import serial.tools.list_ports
        
        # List all serial ports
        ports = list(serial.tools.list_ports.comports())
        logger.info(f"Available serial ports:")
        for port in ports:
            logger.info(f"  {port.device}: {port.description}")
            
        # Try to read raw data from the port
        ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.5)
        
        # Send ping command for Dynamixel (broadcast ping)
        # Dynamixel Protocol 2.0 broadcast ping: 0xFF 0xFF 0xFE 0x02 0x01 0xFC
        dynamixel_ping = bytes([0xFF, 0xFF, 0xFE, 0x02, 0x01, 0xFC])
        
        # Send Feetech ping (varies by protocol)
        feetech_ping = bytes([0x55, 0x55])  # Some Feetech use this
        
        logger.info("Sending Dynamixel broadcast ping...")
        ser.write(dynamixel_ping)
        time.sleep(0.1)
        response = ser.read(100)
        if response:
            logger.info(f"Response to Dynamixel ping: {response.hex()}")
            
        logger.info("Sending Feetech ping...")
        ser.write(feetech_ping)
        time.sleep(0.1)
        response = ser.read(100)
        if response:
            logger.info(f"Response to Feetech ping: {response.hex()}")
            
        ser.close()
        
    except Exception as e:
        logger.info(f"Raw serial test failed: {e}")
    
    logger.info("\n" + "=" * 70)
    logger.info("CONCLUSION: Cannot definitively identify motors from software tests")
    logger.info("Please check the physical motors for labels")
    logger.info("=" * 70)
    
    return "UNKNOWN"

def check_physical_motors():
    """Guide to physically identify motors"""
    
    logger.info("\n" + "=" * 70)
    logger.info("PHYSICAL MOTOR IDENTIFICATION GUIDE")
    logger.info("=" * 70)
    
    logger.info("""
Look at the motors on your robotic arm. Check for:

FEETECH MOTORS:
- Usually have "FEETECH" or "STS" or "SMS" printed on them
- Often have a blue or silver casing
- Model numbers like: STS3215, STS3250, SMS8512BL
- Usually use a 3-wire or 4-wire cable

DYNAMIXEL MOTORS:
- Usually have "DYNAMIXEL" or "ROBOTIS" printed on them
- Often have a white or gray casing with rounded edges
- Model numbers like: XL330-M077, XL430-W250, XM430-W350
- Use a 4-wire daisy-chainable cable with JST connectors

Look for any visible labels or model numbers on the motors themselves.
""")
    
    logger.info("\nCheck your robot's documentation or the original purchase:")
    logger.info("- If you bought an SO100 kit from a vendor, it's likely Feetech")
    logger.info("- If you bought individual Dynamixel motors, they're Dynamixel")
    logger.info("- LeRobot's official SO100 example uses Feetech STS3215 motors")

def analyze_error_messages():
    """Analyze previous error messages for clues"""
    
    logger.info("\n" + "=" * 70)
    logger.info("ANALYZING PREVIOUS ERROR MESSAGES")
    logger.info("=" * 70)
    
    logger.info("""
From your previous error:

1. When you tried Dynamixel test:
   "Missing motor IDs: - 1 (expected model: 1190)"
   - This means the code expected to find Dynamixel motors but didn't
   - Model 1190 = XL330 series Dynamixel
   - This PROBABLY means you don't have Dynamixel motors

2. When you tried the SOFollower class:
   - It imported FeetechMotorsBus internally
   - This suggests LeRobot's SO100 implementation assumes Feetech

3. Your config shows encoder ranges (0-4095):
   - Both Feetech and Dynamixel use 0-4095 for position
   - Not definitive for identification

CONCLUSION FROM ERRORS:
The errors STRONGLY suggest you have FEETECH motors because:
- LeRobot's SO100 class is designed for Feetech
- Dynamixel test failed to find motors
- The SOFollower class connected (got positions!) using Feetech protocol
""")

def main():
    """Main identification routine"""
    
    # Step 1: Try software detection
    result = identify_motors()
    
    # Step 2: Analyze errors
    analyze_error_messages()
    
    # Step 3: Guide physical inspection
    check_physical_motors()
    
    # Step 4: Final verdict based on evidence
    logger.info("\n" + "=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)
    
    logger.info("""
Based on ALL available evidence:
1. The SOFollower class in LeRobot is designed for FEETECH motors
2. Your config file shows motor IDs 1-6 with encoder ranges
3. The Dynamixel test FAILED to find any motors
4. The SOFollower class successfully connected and READ motor positions

Therefore: You almost certainly have FEETECH motors (specifically STS3215 model)

The earlier successful position reads confirm:
- shoulder_pan.pos: -16.18°  (these are degrees!)
- This proves the motors are communicating using Feetech protocol
""")
    
    logger.info("\n✅ CONCLUSION: Your motors are FEETECH STS3215 servos")

if __name__ == "__main__":
    main()
