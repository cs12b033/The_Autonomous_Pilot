
from LaneDetector import LaneDetector


if __name__ == '__main__':
    ld = LaneDetector()
    ld.imageLaneDetect("test_image.jpg")
    ld.videoLaneDetect("test2.mp4")
