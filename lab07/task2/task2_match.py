import cv2
import numpy as np

def main():
    # 读取图像 (灰度图即可)
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("❌ 找不到图片！请确认已将图片复制到 task2 文件夹中。")
        return

    # [前置工作] 初始化 ORB 并在两幅图像中检测关键点和描述子
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # ==========================================
    # 核心任务二开始
    # ==========================================
    
    # 要求 1, 2, 3: 创建暴力匹配器 BFMatcher
    # NORM_HAMMING 专用于二进制描述子(如 ORB)
    # crossCheck=True 表示互相匹配才算数，能过滤掉很多错误匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 进行特征匹配
    matches = bf.match(des1, des2)

    # 要求 4: 按照匹配距离从小到大排序 (distance 越小，说明两个特征越相似)
    matches = sorted(matches, key=lambda x: x.distance)

    # 要求 6: 输出总匹配数量
    print("\n" + "="*40)
    print("🏆 任务二：ORB 特征匹配结果 🏆")
    print("="*40)
    print(f"👉 经过 crossCheck 筛选后，总匹配数量为: {len(matches)} 个")
    print("="*40)

    # 需要提交 1：ORB 初始匹配图 (这里我们画出所有找到的 matches)
    # flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS 表示不画没有匹配上的孤立点
    img_all_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('all_matches.png', img_all_matches)

    # 要求 5 & 需要提交 3：显示前 50 个匹配结果
    top_n = 50
    img_top_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:top_n], None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite('top_50_matches.png', img_top_matches)

    print("\n✅ 匹配图片已成功保存至当前目录！")
    print("👉 请查看 'all_matches.png' (初始匹配图)")
    print("👉 请查看 'top_50_matches.png' (前 50 个最佳匹配图)")

if __name__ == "__main__":
    main()