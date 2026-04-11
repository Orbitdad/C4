import time
from jarvis.vision.manager import VisionManager

def main():
    print("Testing VisionManager gesture tracking...")
    vm = VisionManager()
    if vm.fallback_mode:
        print("VisionManager initialized in fallback mode! MediaPipe may be failing.")
    
    vm.start()
    print("Capturing 10 seconds of camera input. Make gestures in front of the camera...")
    
    try:
        for _ in range(10):
            time.sleep(1)
            status = vm.get_status()
            print(f"Status: {status} | Detections: {vm.last_detection}")
    except KeyboardInterrupt:
        print("Stopping earlier...")
    finally:
        vm.stop()

if __name__ == "__main__":
    main()
