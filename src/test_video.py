import detection_models
import cv2
import time

if __name__ == '__main__':
    ssh_model = detection_models.SSDModel()

    video_file = '20200519105054.avi'
    capture = cv2.VideoCapture(video_file)

    while(capture.isOpened()):
        ret, frame = capture.read()
        if not ret:
            break

        detections = ssh_model.detect(frame)
        ssh_model.annotate(detections, frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.1)

capture.release()
cv2.destroyAllWindows()
