import cv2
import numpy as np

def main():
    # 读取图像
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("❌ 找不到图片！请确认已将图片复制到 task3 文件夹中。")
        return

    # [前置工作] ORB 检测与 BF 匹配
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    total_matches = len(matches)

    # ==========================================
    # 核心任务三：RANSAC 单应矩阵估计与错误剔除
    # ==========================================
    
    # 要求 1: 从匹配结果中提取两幅图像中的对应点坐标
    # queryIdx 对应 img1 的关键点索引，trainIdx 对应 img2 的关键点索引
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)

    # 要求 2, 3, 4: 使用 findHomography，选择 RANSAC 方法，设置阈值为 5.0
    # M 是计算出的 3x3 单应矩阵，mask 是一个标记数组（1 表示内点，0 表示外点/错误匹配）
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 将 mask 转换为列表，方便后面画图和计算
    matchesMask = mask.ravel().tolist()

    # 计算内点数量
    inliers_count = matchesMask.count(1)
    
    # 要求 6: 计算内点比例 inlier_ratio = number_of_inliers / number_of_matches
    inlier_ratio = inliers_count / total_matches

    # 打印需要提交的数据
    print("\n" + "="*50)
    print("🏆 任务三：RANSAC 剔除错误匹配结果 🏆")
    print("="*50)
    print(f"👉 总匹配数量 (Total Matches) : {total_matches}")
    print(f"👉 RANSAC 内点数量 (Inliers)  : {inliers_count}")
    print(f"👉 内点比例 (Inlier Ratio)    : {inlier_ratio:.4f} (即 {inlier_ratio*100:.2f}%)")
    print("-" * 50)
    print("👉 Homography 矩阵 (3x3):")
    print(np.round(M, 4)) # 保留4位小数让输出更美观
    print("="*50)

    # 要求 5: 根据返回的 mask 显示 RANSAC 后的内点匹配
    # 设置 draw_params，只画 mask 中值为 1 的线（即绿色的内点连线）
    draw_params = dict(matchColor=(0, 255, 0), # 正确匹配画绿色
                       singlePointColor=None,
                       matchesMask=matchesMask, # 这里的 mask 起到了过滤作用
                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_ransac_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)

    # 保存图片用于提交
    cv2.imwrite('ransac_matches.png', img_ransac_matches)
    print("\n✅ RANSAC 匹配图片已成功保存至当前目录！请查看 'ransac_matches.png'")

if __name__ == "__main__":
    main()