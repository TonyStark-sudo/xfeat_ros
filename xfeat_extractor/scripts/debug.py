import cv2
import numpy as np

# 读取图像
img_path = "/home/mark50/Pictures/query.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# 选择 ORB 或 SIFT
orb = cv2.ORB_create(500)  # 500 个特征点
sift = cv2.SIFT_create(500)

# 使用 ORB 进行特征点检测
kp2, des2 = orb.detectAndCompute(img, None)
print("-------ORB-------")
# 检查类型
print(type(kp2))  # <class 'list'>
print(isinstance(kp2, list))
print(type(kp2[0]))  # <class 'cv2.KeyPoint'>
print(type(des2))  # <class 'numpy.ndarray'>
print(des2.shape)  # (N, 32)

# 使用 SIFT 进行特征点检测
kp2_sift, des2_sift = sift.detectAndCompute(img, None)
print("-------SIFT-------")
print(type(kp2_sift))
print(type(kp2_sift[0]))
print(type(des2_sift))
print(des2_sift.shape)  # (N, 128)
