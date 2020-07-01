# Test SSD and YOLO models for object detection

# import the necessary packages
import cv2
import logging
import numpy as np
import os
import time

import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)
logger.info('Module is initialized')
logger.debug('Module is initialized')

class SSDModel():
    def __init__(self):
        self.confidence = 0.2
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
        "sofa", "train", "tvmonitor"]
        np.random.seed(42)
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        proto_text = 'MobileNetSSD_deploy.prototxt.txt'
        ssd_model = 'MobileNetSSD_deploy.caffemodel'
        self.net = cv2.dnn.readNetFromCaffe(proto_text, ssd_model)

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        logger.debug("Computing object detections...")
        self.net.setInput(blob)
        start_time = time.time()
        detections = self.net.forward()
        end_time = time.time()
        logger.debug("Computing took {:.2f} seconds".format(end_time - start_time))

        return detections

    def annotate(self, detections, image):
        (h, w) = image.shape[:2]
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence:
                # extract the index of the class label from the `detections`,
                # then compute the (x, y)-coordinates of the bounding box for
                # the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction
                label = "{}: {:.2f}%".format(self.CLASSES[idx], confidence * 100)
                logger.info("Found {}".format(label))
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    self.COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

        return image


class YOLOModel():
    def __init__(self):
        self.confidence = 0.5
        self.threshold = 0.3

        yolo_directory = 'yolo-coco'
        labelsPath = os.path.sep.join([yolo_directory, "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
            dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([yolo_directory, "yolov3.weights"])
        configPath = os.path.sep.join([yolo_directory, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        logger.info("Loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
        self.net.setInput(blob)
        logger.debug("Computing object detections...")
        start_time = time.time()
        layerOutputs = self.net.forward(self.ln)
        end_time = time.time()
        logger.debug("Computing took {:.2f} seconds".format(end_time - start_time))

        return layerOutputs


    def annotate(self, detections, image):
        (H, W) = image.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        # In YOLO model, detections are actually output of the last layer
        layerOutputs = detections
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence,
            self.threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        
        return image

if __name__ == '__main__':
    # Create SSD model
    ssd_model = SSDModel()
    yolo_model = YOLOModel()

    # Load image
    test_image = 'example_06.jpg'
    image_org = cv2.imread(test_image)

    for model in [yolo_model, ssd_model]:
        image = image_org.copy()

        detections = model.detect(image)
        image = model.annotate(detections, image)

        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)