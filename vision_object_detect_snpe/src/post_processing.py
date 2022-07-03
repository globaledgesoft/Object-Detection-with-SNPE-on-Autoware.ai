#!/usr/bin/env python3.6
from __future__ import print_function
import rospy, sys
from cv_bridge import CvBridge
from vision_object_detect.srv import webcam,webcamResponse
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
import numpy as np
import qcsnpe as qc
import itertools

CPU = 0
GPU = 1
DSP = 2
out_layers = np.array(["Postprocessor/BatchMultiClassNonMaxSuppression", "add_6"])
model = qc.qcsnpe("/autoware.ai/ros_workspace/src/vision_object_detect_snpe/assets/mobilenet_ssd.dlc", out_layers, CPU)
def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def decode_class_names(classes_path):
    with open(classes_path, 'r') as f:
        lines = f.readlines()
        classes = []
        for line in lines:
            line = line.strip()
            if line:
                classes.append(line)
    return classes

def load_image_pixels(image, shape):
    width, height = image.shape[1], image.shape[0]

    image = cv2.resize(image, shape)
    # scale pixel values to [0, 1]
    # image = image.astype('float32')
    # image /= 255.0

    # image = np.expand_dims(image, 0)
    return image

def postprocess(out, video_height, video_width):
    boxes = out["Postprocessor/BatchMultiClassNonMaxSuppression_boxes"]
    #print("boxes", boxes)
    scores = out["Postprocessor/BatchMultiClassNonMaxSuppression_scores"]
    # classes = out["Postprocessor/BatchMultiClassNonMaxSuppression_classes"]
    classes = out["detection_classes:0"]
    found, labels, score = [], [], []
    class_names = decode_class_names("/autoware.ai/ros_workspace/src/vision_object_detect_snpe/assets/classes.txt")

    for cur in range(len(scores)):
        probability = scores[cur]
        class_index = int(classes[cur])
        if probability < 0.25:
            continue

        labels.append(class_names[class_index])  
        score.append(scores[cur])

        y1 = int(boxes[4 * cur] * video_height)
        x1 = int(boxes[4 * cur + 1] * video_width)
        y2 = int(boxes[4 * cur + 2] * video_height)
        x2 = int(boxes[4 * cur + 3] * video_width)

        found.append([(x1, y1), (x2, y2)])
    
    #print(found)
    print(len(labels))

    return found, labels, score

def draw_boxes(img, v_boxes, v_labels, v_scores, image_h, image_w):

    filename = cv2.resize(img, (image_h, image_w))
    for i in range(len(v_boxes)):

        x , y = v_boxes[i]
        print(x, y)
        x1, y1 = x
        x2, y2 = y

        cv2.rectangle(filename, (x1, y1), (x2, y2) , (0, 255, 0), 2)

        # draw the box
        #ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        cv2.putText(filename, label, (x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2,cv2.LINE_AA)

        #pyplot.text(x1, y1, label, color='white')
        print("Saving img..")
    
        cv2.imwrite("/out_obj_detect.png", filename)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # cv2.imshow("frame...", filename)
    return filename

def image_callback(req):
    cv_img = imgmsg_to_cv2(req.in_img)
    rospy.loginfo("reading image")
    w , h, c = cv_img.shape
    input_w, input_h = 300, 300
    shaped_img = load_image_pixels(cv_img, (input_w, input_h))
    data_array = model.predict(shaped_img)
    #print(data_array)
    v_boxes, v_labels, v_scores = postprocess(data_array, w, h)
    #print(len(v_boxes))
    for i in range(len(v_boxes)):
        print("Object Detected: " +v_labels[i]+ " Score: ", v_scores[i])
    if len(v_boxes) == 0:
        print("No object detected !!!")
        #return
    final_img = draw_boxes(cv_img, v_boxes, v_labels, v_scores, h, w)
    out_img = cv2_to_imgmsg(final_img)
    # return webcamResponse(out_img)
    return webcamResponse(out_img)



if __name__ == "__main__":
    rospy.init_node('my_inference')
    rospy.loginfo("service node started")
    rospy.Service('/webcam', webcam, image_callback)
    rospy.spin()
