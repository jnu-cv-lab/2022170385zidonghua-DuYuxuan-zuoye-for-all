import cv2
import numpy as np

def scan_document(image_path):
    # 1. 读取并预处理图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 找不到图片: {image_path}")
        return
    
    orig = img.copy()
    # 缩小图片处理（加快检测速度）
    ratio = img.shape[0] / 500.0
    copy = cv2.resize(img, (int(img.shape[1] / ratio), 500))
    
    # 2. 边缘检测与轮廓寻找
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    
    screen_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screen_cnt = approx
            break
            
    if screen_cnt is None:
        print("⚠️ 未检测到纸张边界，将使用预设区域...")
        h_t, w_t = copy.shape[:2]
        screen_cnt = np.array([[[int(w_t*0.1), int(h_t*0.1)]], [[int(w_t*0.9), int(h_t*0.1)]], 
                               [[int(w_t*0.9), int(h_t*0.9)]], [[int(w_t*0.1), int(h_t*0.9)]]])

    # 3. 还原坐标并执行透视变换
    pts = screen_cnt.reshape(4, 2) * ratio
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    (tl, tr, br, bl) = rect
    width = max(int(np.sqrt(((br[0]-bl[0])**2) + ((br[1]-bl[1])**2))), 
                int(np.sqrt(((tr[0]-tl[0])**2) + ((tr[1]-tl[1])**2))))
    height = max(int(np.sqrt(((tr[0]-br[0])**2) + ((tr[1]-br[1])**2))), 
                 int(np.sqrt(((tl[0]-bl[0])**2) + ((tl[1]-bl[1])**2))))
    
    dst = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (width, height))
    
    # 4. 彩色扫描质感增强（关键滤镜）
    enhanced = cv2.convertScaleAbs(warped, alpha=1.2, beta=10) # 调对比度
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) # 锐化
    final = cv2.filter2D(enhanced, -1, sharpen_kernel)
    
    # 5. 保存结果
    cv2.imwrite('final_scan_result.jpg', final)
    # 除错图：看看程序选中的范围
    cv2.drawContours(copy, [screen_cnt], -1, (0, 255, 0), 2)
    cv2.imwrite('debug_auto_detect.jpg', copy)
    
    print("✅ 扫描完成！")
    print("👉 请查看 final_scan_result.jpg")

if __name__ == "__main__":
    scan_document('a4_paper.jpg')