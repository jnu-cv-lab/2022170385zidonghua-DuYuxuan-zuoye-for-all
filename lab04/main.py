import cv2
import numpy as np
import os

print("🚀 Lab 04 (Python版): 图像缩小、恢复与频域分析 启动！")

# 确保 build 文件夹存在 (防止万一没建好报错)
os.makedirs('build', exist_ok=True)

# ==========================================
# 任务1：读入一幅灰度图像
# ==========================================
# cv2.IMREAD_GRAYSCALE (或 0) 会自动将图片读取为单通道灰度图
img_path = 'test.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("❌ 错误：找不到 test.jpg！请检查图片是否在 lab04 目录下。")
    exit()

# img.shape 返回 (高度, 宽度)，所以索引 1 是宽，0 是高
print(f"✅ 任务1完成：成功读取灰度图，原始尺寸: {img.shape[1]}x{img.shape[0]}")

# ==========================================
# 任务2：下采样 (将原图缩小为 1/2)
# ==========================================
# 方法 A：不做预滤波，直接缩小 (使用最近邻插值 cv2.INTER_NEAREST，容易产生锯齿)
down_direct = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

# 方法 B：先进行高斯平滑 (抗混叠滤波)，然后再缩小
# (5, 5) 是高斯核大小，1.5 是标准差
blurred_img = cv2.GaussianBlur(img, (5, 5), 1.5)
down_gaussian = cv2.resize(blurred_img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

print(f"✅ 任务2完成：已生成两种下采样图像 (尺寸: {down_direct.shape[1]}x{down_direct.shape[0]})")

# 将结果保存到 build 文件夹中s
cv2.imwrite("build/down_direct.jpg", down_direct)
cv2.imwrite("build/down_gaussian.jpg", down_gaussian)
print("   -> 缩小后的图片已保存至 build 文件夹")

# ==========================================
# 任务3：图像恢复 (将缩小后的图放大回原始尺寸)
# ==========================================
# 获取原始图像的尺寸 (宽, 高)
target_size = (img.shape[1], img.shape[0])

# 这里我们拿直接缩小的图 (down_direct) 来做恢复测试
# 1. 最近邻内插 (INTER_NEAREST)
restore_nearest = cv2.resize(down_direct, target_size, interpolation=cv2.INTER_NEAREST)
# 2. 双线性内插 (INTER_LINEAR)
restore_linear = cv2.resize(down_direct, target_size, interpolation=cv2.INTER_LINEAR)
# 3. 双三次内插 (INTER_CUBIC)
restore_cubic = cv2.resize(down_direct, target_size, interpolation=cv2.INTER_CUBIC)

# 把恢复出来的三张图也存进 build 文件夹
cv2.imwrite("build/restore_nearest.jpg", restore_nearest)
cv2.imwrite("build/restore_linear.jpg", restore_linear)
cv2.imwrite("build/restore_cubic.jpg", restore_cubic)

print("✅ 任务3完成：已使用三种插值方法将图像放大恢复至原尺寸！")

# ==========================================
# 任务4：空间域比较 (计算 MSE 和 PSNR)
# ==========================================
print("\n--- 任务4：空间域误差评估 ---")

def evaluate_error(original, restored, method_name):
    # 将图像转为 float64 类型，防止像素相减时产生溢出报错
    diff = np.float64(original) - np.float64(restored)
    # 计算均方误差 (MSE)
    mse = np.mean(diff ** 2)
    
    if mse == 0:
        psnr = float('inf') # 如果一模一样，PSNR是无穷大
    else:
        # 计算峰值信噪比 (PSNR)
        psnr = 10 * np.log10((255 ** 2) / mse)
        
    print(f"[{method_name}] MSE: {mse:.2f} \t| PSNR: {psnr:.2f} dB")

# 调用函数，计算三种方法恢复的图与原图的差距
evaluate_error(img, restore_nearest, "最近邻内插")
evaluate_error(img, restore_linear, "双线性内插")
evaluate_error(img, restore_cubic, "双三次内插")

print("✅ 任务4完成：误差计算完毕！(MSE越小越好，PSNR越大越好)")

# ==========================================
# 任务5：傅里叶变换 (FFT) 分析
# ==========================================
print("\n--- 任务5：频域分析 (傅里叶变换) ---")

def get_fourier_spectrum(image):
    # 1. 傅里叶变换前，需要把图像数据转为 float32 格式
    f = np.float32(image)
    
    # 2. 执行二维离散傅里叶变换
    f_transform = np.fft.fft2(f)
    
    # 3. 将零频率分量（最亮的低频中心）移动到图像正中心
    f_shift = np.fft.fftshift(f_transform)
    
    # 4. 取绝对值求幅度，并进行对数变换以便人眼观察 (20 * log(1 + |F|))
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # 5. 将数值归一化到 0-255，方便存成 jpg 图片
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return magnitude_spectrum

# 分别计算：原图、缩小图(down_direct)、双线性恢复图(restore_linear) 的频谱
fft_original = get_fourier_spectrum(img)
fft_down = get_fourier_spectrum(down_direct)
fft_restored = get_fourier_spectrum(restore_linear)

# 把三张酷炫的频谱图存进 build 文件夹
cv2.imwrite("build/fft_original.jpg", fft_original)
cv2.imwrite("build/fft_down.jpg", fft_down)
cv2.imwrite("build/fft_restored.jpg", fft_restored)

print("✅ 任务5完成：原图、缩小图、恢复图的傅里叶频谱图已生成并保存！")

# ==========================================
# 任务6：离散余弦变换 (DCT) 分析
# ==========================================
print("\n--- 任务6：离散余弦变换 (DCT) 分析 ---")

def perform_dct_and_energy(image, method_name):
    # 1. OpenCV 的 DCT 要求图像长宽必须是偶数，且类型为 float32
    h, w = image.shape
    h_even, w_even = h - (h % 2), w - (w % 2)
    img_even = image[0:h_even, 0:w_even]
    
    f = np.float32(img_even)
    
    # 2. 执行 DCT 变换
    dct_result = cv2.dct(f)
    
    # 3. 将 DCT 系数转为对数形式，以便生成可视化图片
    dct_display = 20 * np.log(np.abs(dct_result) + 1)
    dct_display = cv2.normalize(dct_display, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 根据英文名生成对应的图片文件名
    filename = method_name.replace(" ", "_")
    cv2.imwrite(f"build/dct_{filename}.jpg", dct_display)
    
    # 4. 能量统计：计算左上角低频区域能量占比
    # 总能量：所有系数的平方和
    total_energy = np.sum(dct_result ** 2)
    
    # 设定左上角低频区域的大小 (例如宽和高的 1/8)
    roi_h, roi_w = h_even // 8, w_even // 8
    # 取出左上角区域并计算其能量
    low_freq_energy = np.sum(dct_result[0:roi_h, 0:roi_w] ** 2)
    
    # 计算百分比
    energy_ratio = (low_freq_energy / total_energy) * 100
    print(f"[{method_name}] 左上角低频能量占比: {energy_ratio:.4f}%")

# 分别对原图、最近邻恢复图、双线性恢复图做 DCT 分析
perform_dct_and_energy(img, "Original")
perform_dct_and_energy(restore_nearest, "Restore Nearest")
perform_dct_and_energy(restore_linear, "Restore Linear")

print("✅ 任务6完成：所有 DCT 频谱图已生成，能量占比已输出！")