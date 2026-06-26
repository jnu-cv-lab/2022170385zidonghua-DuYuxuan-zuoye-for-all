#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // 任务1：读取一张测试图片

    string img_path = "test.jpg";
    Mat img = imread(img_path);

    // 检查图片是否成功读取
    if (img.empty()) {
        cout << "❌ 错误：找不到图片，请检查 'test.jpg' 是否在同级目录下！" << endl;
        return -1;
    }
    cout << "✅ 任务1完成：成功读取图片！" << endl;

    // 任务2：输出图像基本信息
    cout << "\n--- 任务2：图像基本信息 ---" << endl;
    cout << "图像宽度 (Width): " << img.cols << " 像素" << endl;
    cout << "图像高度 (Height): " << img.rows << " 像素" << endl;
    cout << "图像通道数 (Channels): " << img.channels() << endl;

    // 任务4：转换为灰度图
    // (我们先转换，等下和原图一起显示)
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    cout << "✅ 任务4完成：彩色图已成功转换为灰度图。" << endl;

    // 任务5：保存处理结果
    // 生成的文件存到 build 文件夹里
    imwrite("build/gray_result.jpg", gray_img);
    cout << "✅ 任务5完成：灰度图已成功保存为 'build/gray_result.jpg'" << endl;

    // 任务6：替代 NumPy 做一个简单操作 -> 图像裁剪
    // C++ 没有 NumPy，所以我们用 OpenCV 的 Rect(x, y, 宽, 高) 来裁剪左上角
    Rect crop_region(0, 0, 300, 300); // 裁剪左上角 300x300 的区域
    Mat cropped_img = img(crop_region);
    imwrite("build/cropped_result.jpg", cropped_img);
    cout << "✅ 任务6完成：左上角裁剪区域已成功保存为 'build/cropped_result.jpg'" << endl;

    // 任务3：显示原图和灰度图（已针对 WSL 显示问题进行优化）
    cout << "\n正在弹出图片窗口... (请在弹出的窗口上按任意键关闭并结束程序)" << endl;
    
    // 1. 显式创建窗口（有助于 WSL 窗口初始化）
    namedWindow("Task 3: Original Image", WINDOW_AUTOSIZE);
    namedWindow("Task 4: Grayscale Image", WINDOW_AUTOSIZE);
    
    // 2. 将图片放入窗口
    imshow("Task 3: Original Image", img);
    imshow("Task 4: Grayscale Image", gray_img);
    
    // 3. 核心修复：强制刷新 UI 队列，防止窗口显示灰色
    pollKey(); 
    
    // 4. 等待按键（如果窗口还是灰的，可以多点两下窗口试试）
    waitKey(0); 

    return 0;
}