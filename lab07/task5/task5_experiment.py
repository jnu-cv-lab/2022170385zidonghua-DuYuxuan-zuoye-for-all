import cv2
import numpy as np

def main():
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("❌ 找不到图片！")
        return

    nfeatures_list = [500, 1000, 2000]

    print("\n" + "="*70)
    print("🏆 任务五：ORB nfeatures 参数对比实验 🏆")
    print("="*70)
    
    # 打印表头
    print(f"nfeatures | 模板关键点 | 场景关键点 | 匹配数量 | RANSAC内点 | 内点比例 | 是否成功定位")
    print("-" * 70)

    for n in nfeatures_list:
        # 1. 检测
        orb = cv2.ORB_create(nfeatures=n)
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            continue

        # 2. 匹配
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # 3. RANSAC 剔除
        if len(matches) >= 4:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            inliers = int(np.sum(mask)) if mask is not None else 0
            ratio = inliers / len(matches) if len(matches) > 0 else 0
            
            # 简单判断是否成功：如果算出了矩阵且内点足够支撑一个稳健的几何变换（比如大于15个）
            success = "成功" if M is not None and inliers > 15 else "失败"
        else:
            inliers = 0
            ratio = 0
            success = "失败"

        # 打印这一行的数据
        print(f"{n:<9} | {len(kp1):<10} | {len(kp2):<10} | {len(matches):<8} | {inliers:<10} | {ratio*100:.2f}%   | {success}")

    print("="*70)

if __name__ == "__main__":
    main()