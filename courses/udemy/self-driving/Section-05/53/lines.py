
from LaneDetector import LaneDetector


if __name__ == '__main__':
    ldIm = LaneDetector()
    ldIm.detectLane("test_image.jpg")
    ldVd = LaneDetector()
    ldVd.detectLane("test2.mp4")
    help(LaneDetector)
