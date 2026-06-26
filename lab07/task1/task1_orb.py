import cv2
import numpy as np

def main():
    # 读取图像 (灰度图用于特征检测效果更好，彩色图用于最后画图更直观)
    img1_gray = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)
    
    img1_color = cv2.imread('box.png')
    img2_color = cv2.imread('box_in_scene.png')

    # 安全检查：防止图片没放对位置报错
    if img1_gray is None or img2_gray is None:
        print("❌ 找不到图片！请确认 box.png 和 box_in_scene.png 已经放在 lab07 文件夹中。")
        return

    # 要求 1 & 2: 使用 cv2.ORB_create() 创建检测器，并设置 nfeatures=1000
    orb = cv2.ORB_create(nfeatures=1000)

    # 要求 3: 使用 detectAndCompute() 得到关键点 (keypoints) 和描述子 (descriptors)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    # 要求 4: 使用 cv2.drawKeypoints() 可视化关键点
    # DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS 会画出带有大小和方向的圆圈，非常漂亮和专业
    img1_result = cv2.drawKeypoints(img1_color, kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_result = cv2.drawKeypoints(img2_color, kp2, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 将可视化的结果保存下来，方便你提交实验报告
    cv2.imwrite('box_keypoints.png', img1_result)
    cv2.imwrite('box_in_scene_keypoints.png', img2_result)

    # 要求 5 & 6: 输出关键点数量和描述子的维度
    print("\n" + "="*40)
    print("🏆 任务一：ORB 特征检测结果 🏆")
    print("="*40)
    
    print(f"👉 [box.png] 的关键点数量: {len(kp1)} 个")
    if des1 is not None:
        print(f"👉 [box.png] 的描述子维度: {des1.shape} (即 {des1.shape[0]} 个特征，每个特征由 {des1.shape[1]} 维向量表示)")
    
    print("-" * 40)
    
    print(f"👉 [box_in_scene.png] 的关键点数量: {len(kp2)} 个")
    if des2 is not None:
        print(f"👉 [box_in_scene.png] 的描述子维度: {des2.shape}")
    
    print("="*40)
    print("✅ 可视化图片已成功保存至当前目录！")

if __name__ == "__main__":
    main()