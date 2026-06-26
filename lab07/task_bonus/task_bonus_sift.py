import cv2
import numpy as np
import time

def main():
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)
    img2_color_sift = cv2.imread('box_in_scene.png') # 用于画 SIFT 定位框

    if img1 is None or img2 is None:
        print("❌ 找不到图片！")
        return

    print("\n" + "="*80)
    print("🏆 选做任务：ORB vs SIFT 特征匹配综合对比实验 🏆")
    print("="*80)

    # ==========================================
    # 1. 运行 ORB 流程 (设定为 1000 个特征以供基准对比)
    # ==========================================
    start_time_orb = time.time()
    
    orb = cv2.ORB_create(nfeatures=1000)
    kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
    kp2_orb, des2_orb = orb.detectAndCompute(img2, None)
    
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_orb = bf_orb.match(des1_orb, des2_orb)
    
    src_pts_orb = np.float32([ kp1_orb[m.queryIdx].pt for m in matches_orb ]).reshape(-1, 1, 2)
    dst_pts_orb = np.float32([ kp2_orb[m.trainIdx].pt for m in matches_orb ]).reshape(-1, 1, 2)
    M_orb, mask_orb = cv2.findHomography(src_pts_orb, dst_pts_orb, cv2.RANSAC, 5.0)
    
    time_orb = time.time() - start_time_orb
    
    matches_orb_count = len(matches_orb)
    inliers_orb = int(np.sum(mask_orb)) if mask_orb is not None else 0
    ratio_orb = inliers_orb / matches_orb_count if matches_orb_count > 0 else 0
    success_orb = "成功" if M_orb is not None and inliers_orb > 10 else "失败"

    # ==========================================
    # 2. 运行 SIFT 流程 (本次选做任务的核心)
    # ==========================================
    start_time_sift = time.time()
    
    # 要求 1: 使用 cv2.SIFT_create()
    sift = cv2.SIFT_create()
    kp1_sift, des1_sift = sift.detectAndCompute(img1, None)
    kp2_sift, des2_sift = sift.detectAndCompute(img2, None)
    
    # 要求 2: 使用 cv2.NORM_L2 进行匹配 (SIFT使用的是浮点数描述子，不能用Hamming)
    # 注意：KNN匹配时不要用 crossCheck=True
    bf_sift = cv2.BFMatcher(cv2.NORM_L2)
    
    # 要求 3: 使用 KNN matching (找到最近的2个邻居，k=2)
    knn_matches = bf_sift.knnMatch(des1_sift, des2_sift, k=2)
    
    # 要求 4: 使用 Lowe ratio test 筛选匹配
    good_matches_sift = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good_matches_sift.append(m)
            
    matches_sift_count = len(good_matches_sift)
    
    # 要求 5: 使用 RANSAC + Homography 完成目标定位
    if matches_sift_count >= 4:
        src_pts_sift = np.float32([ kp1_sift[m.queryIdx].pt for m in good_matches_sift ]).reshape(-1, 1, 2)
        dst_pts_sift = np.float32([ kp2_sift[m.trainIdx].pt for m in good_matches_sift ]).reshape(-1, 1, 2)
        M_sift, mask_sift = cv2.findHomography(src_pts_sift, dst_pts_sift, cv2.RANSAC, 5.0)
        
        inliers_sift = int(np.sum(mask_sift)) if mask_sift is not None else 0
        ratio_sift = inliers_sift / matches_sift_count if matches_sift_count > 0 else 0
        success_sift = "成功" if M_sift is not None and inliers_sift > 10 else "失败"
        
        # 画出 SIFT 的定位框用于提交
        if M_sift is not None:
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M_sift)
            img2_color_sift = cv2.polylines(img2_color_sift, [np.int32(dst)], True, (0, 0, 255), 4, cv2.LINE_AA) # SIFT 用红色框
            cv2.imwrite('sift_localized_box.png', img2_color_sift)
            
    else:
        inliers_sift, ratio_sift, success_sift = 0, 0, "失败"
        M_sift = None

    time_sift = time.time() - start_time_sift

    # ==========================================
    # 3. 打印对比表格
    # ==========================================
    print(f"{'方法':<6} | {'匹配数量':<8} | {'RANSAC内点数':<12} | {'内点比例':<8} | {'是否成功定位':<10} | {'运行耗时 (客观客观)'} | {'运行速度主观评价'}")
    print("-" * 85)
    print(f"{'ORB':<8} | {matches_orb_count:<12} | {inliers_orb:<18} | {ratio_orb*100:.2f}%    | {success_orb:<16} | {time_orb:.4f} 秒         | 极快，适合实时视频处理")
    print(f"{'SIFT':<8} | {matches_sift_count:<12} | {inliers_sift:<18} | {ratio_sift*100:.2f}%    | {success_sift:<16} | {time_sift:.4f} 秒         | 较慢，有明显肉眼可见的延迟")
    print("="*85)
    print("✅ SIFT 定位结果图已保存为 'sift_localized_box.png' (红色边框)")

if __name__ == "__main__":
    main()