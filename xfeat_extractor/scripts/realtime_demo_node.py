'''
xfeat demo overrided by ros
by lvwenzhen 2025.04.04 
'''
#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
from pynput import keyboard
from time import time, sleep
from modules.xfeat import XFeat

class Args:
    def __init__(self):
        self.image_topic = rospy.get_param("/realtime_demo_node/image_topic")
        self.method = rospy.get_param("/realtime_demo_node/method")
        self.max_kpts = rospy.get_param("/realtime_demo_node/max_kpts")

class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(torch.tensor(x).permute(2,0,1).float()[None])[0]
        

class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher

class MatchDemoNode:
    def __init__(self, args):
        rospy.init_node('match_demo_node')
        self.args = args
        self.cv_bridge = CvBridge()

        # 成员变量
        self.width = None
        self.height = None
        self.ref_frame = None
        self.current_frame = None
        if args.method in ["ORB", "SIFT"]:
            self.ref_precomp = [[], []]
        else:
            # current['keypoints'], current['descriptors']
            # self.ref_frame = [{"keypoints": []}, {"descriptors": []}]
            self.ref_frame = {"keypoints": torch.empty(0, 2),
                              "descriptors": torch.empty(0, 64)}

        self.method = self.init_Method()
        self.first_msg_flag = True
        self.keyboard_call_flag = False
        self.listener = keyboard.Listener(on_press=self.waiting_keyboard)
        self.listener.start()

        # 存储四个角点的列表
        if "gray" in self.args.image_topic:
            self.border_len = 50
        else:
            self.border_len = 400
        self.corners = None
        
        # 单应矩阵参数
        self.H = None
        self.min_inliners = 50
        self.ransac_thr = 4.0

        # FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 30

        # 文本字体设置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        if "gray" in self.args.image_topic:
            self.line_thickness = 3
            self.thickness = 1
        else:
            self.line_thickness = 20
            self.thickness = 14
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0,255,0)
        
        self.sub = rospy.Subscriber(args.image_topic, Image, self.image_callback)
        self.pub = rospy.Publisher(f"/extractor/{self.args.method}_method", Image, queue_size=10)

    def init_Method(self):
        if self.args.method == "ORB":
            descriptor = cv2.ORB_create(self.args.max_kpts, fastThreshold=10)
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.args.method == "SIFT":
            descriptor = cv2.SIFT_create(self.args.max_kpts, contrastThreshold=-1, edgeThreshold=1000)
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif self.args.method == "XFeat":
            descriptor = CVWrapper(XFeat(top_k=self.args.max_kpts))
            matcher = XFeat()
        else:
            raise RuntimeError("Invalid method !!!")
        return Method(descriptor, matcher)

    def waiting_keyboard(self, key):
        if key.char == "r":
            self.keyboard_call_flag = True

    def image_callback(self, msg: Image):
        # image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)
        if "gray" in self.args.image_topic:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if self.width is None or self.height is None:
            self.width = image.shape[1]
            self.height = image.shape[0]
            self.corners = [[self.border_len, self.border_len], 
                            [self.width - self.border_len, self.border_len], 
                            [self.width - self.border_len, self.height - self.border_len],
                            [self.border_len, self.height - self.border_len]]

        if self.first_msg_flag or self.keyboard_call_flag:
            self.ref_frame = image.copy()
            self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None)
            self.first_msg_flag = False
            self.keyboard_call_flag = False

        self.current_frame = image
        top_frame_canvas = self.create_top_frame()
        t0 = time()

        matched_frame, good_matches = self.match(self.ref_frame, self.current_frame)
        self.time_list.append(time() - t0)
        if len(self.time_list) > self.max_cnt:
            self.time_list.pop(0)
        self.FPS = 1.0 / np.array(self.time_list).mean()

        bottom_frame = self.draw_matches(matched_frame, good_matches)

        if self.H is not None and len(self.corners) > 1:
            self.draw_quad(top_frame_canvas, self.warp_points(self.corners, self.H, self.width))

        canvas = np.vstack((top_frame_canvas, bottom_frame))
        canvas_msg = self.cv_bridge.cv2_to_imgmsg(canvas, encoding="rgb8")
        canvas_msg.header = msg.header
        self.pub.publish(canvas_msg)

    def draw_quad(self, frame, point_list):
        if len(self.corners) > 1:
            # 循环执行三次，在最后一次循环是闭合矩形
            for i in range(len(self.corners) - 1):
                # tuple将列表元素转成元组
                cv2.line(frame, tuple(point_list[i]), tuple(point_list[i + 1]), self.line_color, self.line_thickness, lineType = self.line_type)
            if len(self.corners) == 4:  # Close the quadrilateral if 4 corners are defined
                cv2.line(frame, tuple(point_list[3]), tuple(point_list[0]), self.line_color, self.line_thickness, lineType = self.line_type)

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # Draw the border
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=borderColor, thickness=thickness+2, lineType=lineType)
        # Draw the text
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=textColor, thickness=thickness, lineType=lineType)

    def warp_points(self, points, H, x_offset = 0):
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)

        warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
        warped_points_np[:, 0] += x_offset
        warped_points = warped_points_np.astype(int).tolist()
        
        return warped_points

    def create_top_frame(self):
        top_frame_canvas = np.zeros((self.height, self.width * 2, 3), dtype=np.uint8)
        top_frame = np.hstack((self.ref_frame, self.current_frame))
        color = (3, 186, 252)
        cv2.rectangle(top_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)  # Orange color line as a separator
        top_frame_canvas[0:self.height, 0:self.width*2] = top_frame
        
        # Adding captions on the top frame canvas
        self.putText(canvas=top_frame_canvas, text="Reference Frame:", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=self.thickness, lineType=self.line_type)

        self.putText(canvas=top_frame_canvas, text="Target Frame:", org=(650, 30), fontFace=self.font, 
                    fontScale=self.font_scale,  textColor=(0,0,0), borderColor=color, thickness=self.thickness, lineType=self.line_type)
        
        self.draw_quad(top_frame_canvas, self.corners)
        
        return top_frame_canvas
    
    def match(self, ref_frame, current_frame):

        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        # Detect and compute features
        if self.args.method in ['SIFT', 'ORB']:
            # 理论上 kp2:List[cv2.KeyPoint] des2:np.ndarray
            # 经测试 kp2:Tuple[cv2.KeyPoint] des2:np.ndarray
            kp1, des1 = self.ref_precomp
            kp2, des2 = self.method.descriptor.detectAndCompute(current_frame, None)
        else:
            """
            Compute sparse keypoints & descriptors. Supports batched mode.

			input:
				x -> torch.Tensor(B, C, H, W): grayscale or rgb image
				top_k -> int: keep best k features
			return:
				List[Dict]: 
					'keypoints'    ->   torch.Tensor(N, 2): keypoints (x,y)
					'scores'       ->   torch.Tensor(N,): keypoint scores
					'descriptors'  ->   torch.Tensor(N, 64): local features
            """
            current = self.method.descriptor.detectAndCompute(current_frame)
            kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
            kpts2, descs2 = current['keypoints'], current['descriptors']
            # kpts1, descs1 = self.ref_precomp[0], self.ref_precomp[1]
            # kpts2, descs2 = current[0], current[1]
            # 使用了两个描述符做点积的经典match方式，基于余弦相似度的匹配
            idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)

            points1 = kpts1[idx0].cpu().numpy()
            points2 = kpts2[idx1].cpu().numpy()

        if len(kp1) > 10 and len(kp2) > 10 and self.args.method in ['SIFT', 'ORB']:
            # Match descriptors
            # matches:List[cv2.DMatch]
            matches = self.method.matcher.match(des1, des2)

            if len(matches) > 10:
                points1 = np.zeros((len(matches), 2), dtype=np.float32) # [匹配个数， 2]
                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                for i, match in enumerate(matches):
                    points1[i, :] = kp1[match.queryIdx].pt
                    points2[i, :] = kp2[match.trainIdx].pt

        if len(points1) > 10 and len(points2) > 10:
            # Find homography
            # 返回H矩阵, 和一个形状为 (N, 1) 的二值数组(内点)
            self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
            # 将 (N,1) 变成 (N,) 的布尔数组，True 表示是 内点
            inliers = inliers.flatten() > 0 

            if inliers.sum() < self.min_inliners:
                self.H = None

            if self.args.method in ["SIFT", "ORB"]:
                good_matches = [m for i,m in enumerate(matches) if inliers[i]]
            else:
                kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
                kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
                good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

            # Draw matches
            # 将内点的goodmatch可视化
            matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
            
        else:
            matched_frame = np.hstack([ref_frame, current_frame])
        return matched_frame, good_matches
    
    def draw_matches(self, matched_frame, good_matches):
        color = (240, 89, 169)

        # Add a colored rectangle to separate from the top frame
        cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)

        # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="%s Matches: %d"%(self.args.method, len(good_matches)), org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=self.thickness, lineType=self.line_type)
        
                # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="FPS (registration): {:.1f}".format(self.FPS), org=(650, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=self.thickness, lineType=self.line_type)

        return matched_frame
    
    def run(self):
        rospy.loginfo("MatchDemoNode started, listening on topic: %s" % self.args.image_topic)
        rospy.spin()
        
if __name__ == "__main__" :
    args = Args()
    demo_node = MatchDemoNode(args=args)
    print("Available param: ")
    print("image_topic: ", rospy.get_param('/realtime_demo_node/image_topic', 'default_value'))
    print("method: ", rospy.get_param('/realtime_demo_node/method', 'default_value'))
    print("max_kpts: ", rospy.get_param('/realtime_demo_node/max_kpts', 'default_value'))
    demo_node.run()
