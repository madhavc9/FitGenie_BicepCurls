import cv2
import numpy as np
import PoseModule as pm 
class BicepCurlCounter:
    def __init__(self):
        self.detector = pm.poseDetector()
        self.count = 0
        self.direction = 0

    def process_frame(self, img):
        maxh, maxw, _ = img.shape

        # Process frame with pose estimation
        img = self.detector.findPose(img, False)
        lmList = self.detector.findPosition(img, False)

        # Check if landmarks are detected
        if len(lmList) != 0:
            # Calculate angle
            angle = self.detector.findAngle(img, 12, 14, 16)

            # Interpolate for percentage and bar level
            per = np.interp(angle, (55, 145), (100, 0))
            bar = np.interp(angle, (55, 145), (400, 50))

            # Check for the dumbbell curls
            color = (255, 0, 255)
            if per == 100:
                color = (0, 255, 0)
                if self.direction == 0:
                    self.count += 0.5
                    self.direction = 1
            if per == 0:
                color = (0, 255, 0)
                if self.direction == 1:
                    self.count += 0.5
                    self.direction = 0

            # Draw bar
            cv2.rectangle(img, (maxw-100, maxh-400), (maxw - 50, maxh - 50), color, 3)
            cv2.rectangle(img, (maxw - 100, maxh - int(bar)), (maxw - 50, maxh - 50), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (maxw - 105, maxh - 430), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

            # Draw Curl Count
            cv2.rectangle(img, (0, 0), (150, 100), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(self.count)), (40, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

        return img

def main():
    # Initialize video capture from webcam
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    counter = BicepCurlCounter()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Process the frame and display the results
        frame = counter.process_frame(frame)
        cv2.imshow("Bicep Curl Counter", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()