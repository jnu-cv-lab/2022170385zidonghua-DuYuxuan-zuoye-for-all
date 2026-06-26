import cv2
import numpy as np

# 任务一：生成测试图
def create_white_test_image():
    # 建立一张 500x500 的白色画布
    img = np.ones((500, 500), dtype=np.uint8) * 255
    
    # 1. 画平行线 (左上角)
    cv2.line(img, (50, 50), (200, 50), 0, 2)
    cv2.line(img, (50, 80), (200, 80), 0, 2)
    
    # 2. 画垂直线 (右上角，构成一个直角)
    cv2.line(img, (350, 50), (350, 150), 0, 2)
    cv2.line(img, (350, 150), (450, 150), 0, 2)
    
    # 3. 画矩形 (左下角)
    cv2.rectangle(img, (50, 300), (200, 400), 0, 2)
    
    # 4. 画圆 (右下角)
    cv2.circle(img, (380, 350), 60, 0, 2)
    
    return img

# 任务二：施加三种变换
def apply_transformations(img):
    out_width, out_height = 800, 800
    
    # 1. 相似变换 
    theta = np.radians(30)
    scale = 0.8
    tx, ty = 200, 100  
    M_sim = np.float32([
        [scale * np.cos(theta), -scale * np.sin(theta), tx],
        [scale * np.sin(theta),  scale * np.cos(theta), ty]
    ])
    img_sim = cv2.warpAffine(img, M_sim, (out_width, out_height), borderValue=255)
    cv2.imwrite('1_similarity.jpg', img_sim)

    # 2. 仿射变换
    M_aff = np.float32([
        [1.0, 0.5, 50], 
        [0.0, 1.0, 50]  
    ])
    img_aff = cv2.warpAffine(img, M_aff, (out_width, out_height), borderValue=255)
    cv2.imwrite('2_affine.jpg', img_aff)

    # 3. 透视变换
    M_per = np.float32([
        [1.0, 0.0, 0],
        [0.0, 1.0, 0],
        [0.001, 0.002, 1.0] 
    ])
    img_per = cv2.warpPerspective(img, M_per, (out_width, out_height), borderValue=255)
    cv2.imwrite('3_perspective.jpg', img_per)