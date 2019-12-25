"""
# Author: Ajay Pratap Singh
# Course: Udemy - Self Driving Course
#
"""
import cv2
import numpy as np


class LaneDetector:
    def __init__(self):
        print("LaneDetector obj created")

    def make_points(self, image, line):
        """Summary

        Args:
            image (TYPE): Description
            line (TYPE): Description

        Returns:
            TYPE: Description
        """
        slope, intercept = line
        y1 = int(image.shape[0])  # bottom of the image
        y2 = int(y1 * 3 / 5)      # slightly lower than the middle
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [[x1, y1, x2, y2]]

    def average_slope_intercept(self, image, lines):
        """Summary

        Args:
            image (TYPE): Description
            lines (TYPE): Description

        Returns:
            TYPE: Description
        """
        left_fit = []
        right_fit = []
        if lines is None:
            return None
        for line in lines:
            for x1, y1, x2, y2 in line:
                fit = np.polyfit((x1, x2), (y1, y2), 1)
                slope = fit[0]
                intercept = fit[1]
                if slope < 0:  # y is reversed in image
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
        # add more weight to longer lines
        left_fit_average = np.average(left_fit, axis=0)
        right_fit_average = np.average(right_fit, axis=0)
        left_line = self.make_points(image, left_fit_average)
        right_line = self.make_points(image, right_fit_average)
        averaged_lines = [left_line, right_line]
        return averaged_lines

    def canny(self, img):
        """Summary

        Args:
            img (TYPE): Description

        Returns:
            TYPE: Description
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # kernel = 5
        # blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        canny = cv2.Canny(gray, 50, 150)
        return canny

    def display_lines(self, img, lines):
        """Summary

        Args:
            img (TYPE): Description
            lines (TYPE): Description

        Returns:
            TYPE: Description
        """
        line_image = np.zeros_like(img)
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image

    def region_of_interest(self, img):
        """Summary

        Args:
            img (TYPE): Description

        Returns:
            TYPE: Description
        """
        height = img.shape[0]
        # width = img.shape[1]
        mask = np.zeros_like(img)

        triangle = np.array([[
            (200, height),
            (550, 250),
            (1100, height), ]], np.int32)

        cv2.fillPoly(mask, triangle, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def putLane(self, frame):
        """Summary

        Args:
            frame (TYPE): Description

        Returns:
            TYPE: Description
        """
        canny_image = self.canny(frame)
        cropped_canny = self.region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = self.average_slope_intercept(frame, lines)
        line_image = self.display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        return combo_image

    def imageLaneDetect(self, imageFile):
        """Summary

        Returns:
            TYPE: Description
        """
        image = cv2.imread(imageFile)
        cv2.imshow("image result", self.putLane(np.copy(image)))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return

    def videoLaneDetect(self, videoFile):
        """Summary

        Returns:
            TYPE: Description
        """
        cap = cv2.VideoCapture(videoFile)
        while(cap.isOpened()):
            cv2.imshow("video result", self.putLane(cap.read()[1]))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        return
