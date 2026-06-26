import cv2
import numpy as np
import os

print(" 第四次作业: 抗混叠滤波与自适应下采样 ")

# 确保 build 文件夹存在
os.makedirs('build', exist_ok=True)

# 第一部分：生成测试图、下采样与频域分析
print("\n 第一部分：混叠现象观察 ")

# 1. 生成棋盘格 (Checkerboard)
def generate_checkerboard(size, block_size):
    img = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // block_size + j // block_size) % 2 == 0:
                img[i, j] = 255
    return img

# 2. 生成 Chirp 测视图 (同心圆波纹，频率从中心向外逐渐变高)
def generate_chirp(size):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    # sin函数生成波纹，频率随半径R的平方增加
    img = np.sin(50 * np.pi * R**2) 
    img = ((img + 1) * 127.5).astype(np.uint8)
    return img

size = 512
checkerboard = generate_checkerboard(size, 8)
chirp = generate_chirp(size)

cv2.imwrite("build/1_checkerboard.jpg", checkerboard)
cv2.imwrite("build/1_chirp.jpg", chirp)
print("✅ 成功生成原始棋盘格和 Chirp 测试图！")

# 下采样设置：缩小 4 倍 (M=4)
M = 4

# 方法 A：直接下采样 (每隔 M 个像素硬抽样一个点)
down_direct_checker = checkerboard[::M, ::M]
down_direct_chirp = chirp[::M, ::M]

cv2.imwrite("build/1_down_direct_checker.jpg", down_direct_checker)
cv2.imwrite("build/1_down_direct_chirp.jpg", down_direct_chirp)
print(" 直接下采样完成！(注意观察 build 文件夹里图片产生的奇怪水波纹或马赛克)")

# 方法 B：加高斯滤波再下采样 (抗混叠)
# 根据最佳 sigma 约为 0.45 * M = 1.8
sigma = 1.8
ksize = int(2 * np.ceil(2 * sigma) + 1) # 自动计算合适的核大小
blur_checker = cv2.GaussianBlur(checkerboard, (ksize, ksize), sigma)
blur_chirp = cv2.GaussianBlur(chirp, (ksize, ksize), sigma)

down_gaussian_checker = blur_checker[::M, ::M]
down_gaussian_chirp = blur_chirp[::M, ::M]

cv2.imwrite("build/1_down_gaussian_checker.jpg", down_gaussian_checker)
cv2.imwrite("build/1_down_gaussian_chirp.jpg", down_gaussian_chirp)
print("✅ 高斯滤波后下采样完成！(画面变模糊了，但奇怪的伪影消失了)")

# 画 FFT 频谱 (沿用上周的魔法)
def get_fft_spectrum(img):
    f = np.fft.fft2(np.float32(img))
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imwrite("build/1_fft_direct_chirp.jpg", get_fft_spectrum(down_direct_chirp))
cv2.imwrite("build/1_fft_gaussian_chirp.jpg", get_fft_spectrum(down_gaussian_chirp))
print(" FFT 频谱图生成完毕！(准备好在频域抓捕混叠元凶吧)")

# 第二部分：验证 sigma 公式
print("\n--- 第二部分：验证 sigma 公式 ---")

# 固定下采样倍数
M_val = 4
# 老师要求测试的 sigma 列表
sigmas_to_test = [0.5, 1.0, 2.0, 4.0]

for s in sigmas_to_test:
    # 自动计算合适的高斯核大小 (公式: 2 * ceil(2 * sigma) + 1)
    ksize = int(2 * np.ceil(2 * s) + 1)
    
    # 1. 对 Chirp 图进行高斯滤波
    blur_chirp_test = cv2.GaussianBlur(chirp, (ksize, ksize), s)
    
    # 2. 下采样 (缩小 M 倍)
    down_chirp_test = blur_chirp_test[::M_val, ::M_val]
    
    # 3. 保存图片，文件名标明当前的 sigma 值
    cv2.imwrite(f"build/2_sigma_{s}.jpg", down_chirp_test)

print(" 第二部分完成！不同 sigma 值的图片已保存至 build 文件夹。")

# 第三部分：自适应下采样
print("\n 第三部分：自适应下采样 ")

# 我们继续使用 chirp 图来进行自适应测试
img_test = chirp 

# 1. 梯度分析：计算 Sobel 梯度幅值
grad_x = cv2.Sobel(img_test, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img_test, cv2.CV_64F, 0, 1, ksize=3)
grad_mag = cv2.magnitude(grad_x, grad_y)

# 将梯度归一化到 0~1 之间，作为权重掩码 (Mask)
# 梯度大的地方趋近1（高频易混叠区），梯度小的地方趋近0（低频平滑区）
mask = cv2.normalize(grad_mag, None, 0, 1, cv2.NORM_MINMAX)

# 2. 准备两组不同强度的滤波图
# 低频区：用较小的 sigma (保留一点锐度)
blur_low = cv2.GaussianBlur(img_test, (5, 5), 0.5)
# 高频区：用理论最佳 sigma 1.8 (强力抗混叠)
blur_high = cv2.GaussianBlur(img_test, (9, 9), 1.8)

# 3. 核心：自适应融合！
# 公式：当前像素值 = 权重 * 强滤波 + (1-权重) * 弱滤波
adaptive_blur = (mask * blur_high + (1 - mask) * blur_low).astype(np.uint8)

# 4. 执行下采样 (M=4)
down_adaptive = adaptive_blur[::4, ::4]
# 全图统一滤波下采样 (作为对比基准)
down_uniform = blur_high[::4, ::4]

# 5. 计算误差图并拉伸对比度以便观察
error_map = cv2.absdiff(down_adaptive, down_uniform)
error_map_display = cv2.normalize(error_map, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite("build/3_down_adaptive.jpg", down_adaptive)
cv2.imwrite("build/3_error_map.jpg", error_map_display)

print(" 第三部分完成！自适应下采样结果与误差图已生成！")