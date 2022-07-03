#!/usr/bin/env python3.6
from __future__ import print_function
#import qcsnpe as qc
import numpy as np
import cv2
from sensor_msgs.msg import Image
import rospy
from sys import exit
from vision_object_detect_snpe.srv import *

CPU = 0
GPU = 1
DSP = 2

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


# def postprocess(out, video_height, video_width):
#     boxes = out["Postprocessor/BatchMultiClassNonMaxSuppression_boxes"]
#     scores = out["Postprocessor/BatchMultiClassNonMaxSuppression_scores"]
#     #classes = out["Postprocessor/BatchMultiClassNonMaxSuppression_classes"]
#     classes = out["detection_classes:0"]
#     found = []

#     for cur in range(len(scores)):
#         probability = scores[cur]
#         class_index = int(classes[cur])
#         #if class_index != 1 or probability < 0.5:
#         if probability < 0.2:
#             continue

#         y1 = int(boxes[4 * cur] * video_height)
#         x1 = int(boxes[4 * cur + 1] * video_width)
#         y2 = int(boxes[4 * cur + 2] * video_height)
#         x2 = int(boxes[4 * cur + 3] * video_width)
#         found.append([(x1, y1), (x2, y2)])

#     return found

def capture():
    # pub = rospy.Publisher('/webcam', Image, queue_size=1)
    # rospy.init_node('image', anonymous=False)
    # rate = rospy.Rate(10)  # 10hz 
    rospy.wait_for_service('/webcam')
    #cap = cv2.VideoCapture(0)
    start_capture = rospy.ServiceProxy('/webcam', webcam)
    try:
        while not rospy.is_shutdown() :
            #-- capture single frame to send and process.
           # ret,img = cap.read()
            img = cv2.imread("/autoware.ai/ros_workspace/src/vision_object_detect/images/image.png")
            #if not ret:
               # break
            rosImg =  cv2_to_imgmsg(img)
            
            ret = start_capture(rosImg)
            #print(ret)
            out_img = imgmsg_to_cv2(ret.out_img)
            # imgmsg_to_cv2(ret.out_img)
            cv2.imwrite("/client_out_img.png", out_img)
            #if cv2.waitKey(0) & 0xFF == ord('q'):
             #   break
            
        #pub.publish(img_msg)
       # cap.release()
    except rospy.ServiceException as e:
       # if rospy.is_shutdown():
            #cap.release()
        print("Service call failed: %s"%e)
    except KeyboardInterrupt:
        print("releasing the cap..")
        #cap.release()


if __name__ == "__main__":
    try:
        capture()
    except rospy.ROSInterruptException:
        pass



    
# import qcsnpe as qc
# import numpy as np
# import cv2

# CPU = 0
# GPU = 1
# DSP = 2

# def postprocess(out, video_height, video_width):
#     boxes = out["Postprocessor/BatchMultiClassNonMaxSuppression_boxes"]
#     scores = out["Postprocessor/BatchMultiClassNonMaxSuppression_scores"]
#     #classes = out["Postprocessor/BatchMultiClassNonMaxSuppression_classes"]
#     classes = out["detection_classes:0"]
#     found = []

#     for cur in range(len(scores)):
#         probability = scores[cur]
#         class_index = int(classes[cur])
#         #if class_index != 1 or probability < 0.5:
#         if probability < 0.2:
#             continue

#         y1 = int(boxes[4 * cur] * video_height)
#         x1 = int(boxes[4 * cur + 1] * video_width)
#         y2 = int(boxes[4 * cur + 2] * video_height)
#         x2 = int(boxes[4 * cur + 3] * video_width)
#         found.append([(x1, y1), (x2, y2)])

#     return found


# out_layers = np.array(["Postprocessor/BatchMultiClassNonMaxSuppression", "add_6"])
# # out_layers = np.array(["Postprocessor/BatchMultiClassNonMaxSuppression"])

# model = qc.qcsnpe("./mobilenet_ssd.dlc", out_layers, CPU)

# #cap = cv2.VideoCapture(2)

# # while(cap.isOpened()):
# while(1):
#     print("R")
#     #ret, image = cap.read()
#     image = cv2.imread("/autoware.ai/ros_workspace/src/vision_object_detect/images/img1.jpg")
#     w, h, c= image.shape
#     img = cv2.resize(image, (300,300))
#     out = model.predict(img)
#     print(type(out))
#     #print(out.keys())
#     res = postprocess(out, w, h)
#     for box in res:
#             cv2.rectangle(image, box[0], box[1], (255,0,0), 2)
    
#     cv2.imwrite("/out_img.png", image)

#     ## image variable is the your output frame which can be displayed or written as video
#     ## use res variable of bounding box for the writing business logic over the detected object.
