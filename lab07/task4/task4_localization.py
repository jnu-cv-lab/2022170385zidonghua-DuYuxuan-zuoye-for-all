import cv2
import numpy as np

def main():
    # 读取图像：处理过程用灰度图
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)
    
    # 💡 小技巧：为了让画出来的框更醒目，我们在彩色版本的场景图上画框
    img2_color = cv2.imread('box_in_scene.png')

    if img1 is None or img2 is None:
        print("❌ 找不到图片！请确认已将图片复制到 task4 文件夹中。")
        return

    # [前置工作整合] 提取、匹配、RANSAC计算矩阵
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
    
    # 获取单应矩阵 M
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # ==========================================
    # 核心任务四：目标定位 (Object Localization)
    # ==========================================
    if M is not None:
        # 要求 1: 获取 box.png 的四个角点
        h, w = img1.shape
        # 顺序：左上，左下，右下，右上 (围成一个封闭四边形)
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

        # 要求 2: 使用 cv2.perspectiveTransform() 进行角点投影
        # 魔法发生的地方：将原图的四个直角坐标，通过矩阵 M 映射到场景图里扭曲的位置
        dst = cv2.perspectiveTransform(pts, M)

        # 要求 3: 使用 cv2.polylines() 在场景图中画出四边形边框
        # True 表示闭合图形，(0, 255, 0) 是鲜艳的绿色，4 是线条宽度
        img2_with_box = cv2.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 4, cv2.LINE_AA)

        # 要求 4: 保存/显示最终目标定位结果
        cv2.imwrite('localized_box.png', img2_with_box)

        print("\n" + "="*50)
        print("🏆 任务四：目标定位结果 🏆")
        print("="*50)
        print("✅ 目标边框已成功绘制并保存！请查看 'localized_box.png'")
        print("="*50)
    else:
        print("❌ 单应矩阵计算失败，无法进行目标定位！")

if __name__ == "__main__":
    main()