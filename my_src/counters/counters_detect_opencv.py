import cv2
import numpy as np
from trvo_utils.imutils import imreadRGB, rgb2bgr, imshowWait
from trvo_utils.timer import timeit


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(img, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(img, (left, top), (right, bottom), (255, 178, 50), 3)

    label = f'{classId} {conf}:.2f'

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(img, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                  (255, 255, 255), cv2.FILLED)
    cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(frame, outs, confThreshold, nmsThreshold, img):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(img, classIds[i], confidences[i], left, top, left + width, top + height)


def main():
    confThreshold = 0.5  # Confidence threshold
    nmsThreshold = 0.4  # Non-maximum suppression threshold
    inpWidth = 416  # Width of network's input image
    inpHeight = 416  # Height of network's input image
    modelConfiguration = "data/yolov3.cfg"

    modelWeights = "test_counter.weights"
    image_file = 'data/counters/4_from_phone/P_20200517_141635.jpg'

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    frame = imreadRGB(image_file)

    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    with timeit():
        outs = net.forward(getOutputsNames(net))
    with timeit():
        outs = net.forward(getOutputsNames(net))
    with timeit():
        outs = net.forward(getOutputsNames(net))
    with timeit():
        outs = net.forward(getOutputsNames(net))
    with timeit():
        postprocess(frame, outs, confThreshold, nmsThreshold, frame)

    imshowWait(rgb2bgr(frame))


main()
