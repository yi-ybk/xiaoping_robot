// #include <ros/ros.h>
// #include <sensor_msgs/LaserScan.h>
// #include <geometry_msgs/Twist.h>
// #include <cv_bridge/cv_bridge.h>
// #include <opencv2/imgcodecs/imgcodecs.hpp>
// #include <opencv2/highgui/highgui.hpp>

// // 全局变量：存储当前HSV阈值（H范围0-179，S和V范围0-255）
// int h_min = 0, h_max = 179;
// int s_min = 0, s_max = 255;
// int v_min = 0, v_max = 255;

// // 原始图像和HSV格式图像（全局变量，方便滑动条回调函数访问）
// cv::Mat img_bgr, img_hsv;

// /**
//  * @brief 滑动条回调函数（空实现，仅用于触发图像更新）
//  * 由于OpenCV滑动条必须绑定回调函数，这里仅作为占位符
//  */
// void onTrackbarChange(int, void*) {}

// /**
//  * @brief 生成并显示当前阈值对应的掩膜
//  */
// void updateMask() {
//     // 根据当前滑动条值生成HSV阈值范围
//     cv::Scalar lower(h_min, s_min, v_min);
//     cv::Scalar upper(h_max, s_max, v_max);
    
//     // 生成掩膜（二值图像：目标区域为白色，背景为黑色）
//     cv::Mat mask;
//     cv::inRange(img_hsv, lower, upper, mask);
    
//     // 显示原始图像和掩膜
//     cv::imshow("Original Image", img_bgr);
//     cv::imshow("HSV Mask", mask);
    
//     // 打印当前阈值（方便用户记录）
//     std::cout << "\rCurrent HSV Threshold: "
//               << "H[" << h_min << "," << h_max << "] "
//               << "S[" << s_min << "," << s_max << "] "
//               << "V[" << v_min << "," << v_max << "] "
//               << std::flush; // 实时刷新同一行，避免打印刷屏
// }

// int main(int argc, char *argv[])
// {
//     setlocale(LC_ALL,"");
//     ros::init(argc,argv,"hsv_calibrator");
//     // 1. 读取输入图像（支持命令行传参或手动输入路径）
//     std::string img_path;
//     // if (argc > 1) {
//     //     img_path = argv[1]; // 从命令行参数获取图像路径
//     // } else {
//     //     std::cout << "请输入图像路径: ";
//     //     std::cin >> img_path; // 手动输入图像路径
//     // }
//     img_path = "/home/yi/桌面/1.png";

//     // 加载图像（BGR格式，OpenCV默认读取格式）
//     img_bgr = cv::imread(img_path);
//     if (img_bgr.empty()) {
//         std::cerr << "错误：无法加载图像！路径可能无效: " << img_path << std::endl;
//         return -1;
//     }

//     // 转换为HSV颜色空间（便于颜色阈值分割）
//     cv::cvtColor(img_bgr, img_hsv, cv::COLOR_BGR2HSV);

//     // 2. 创建窗口
//     cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
//     cv::namedWindow("HSV Mask", cv::WINDOW_AUTOSIZE);
//     cv::namedWindow("HSV Controls", cv::WINDOW_AUTOSIZE); // 滑动条控制窗口

//     // 3. 创建滑动条（H/S/V的最小值和最大值）
//     // 参数：滑动条名称、窗口名称、初始值、最大值、回调函数
//     cv::createTrackbar("H Min", "HSV Controls", &h_min, 179, onTrackbarChange);
//     cv::createTrackbar("H Max", "HSV Controls", &h_max, 179, onTrackbarChange);
//     cv::createTrackbar("S Min", "HSV Controls", &s_min, 255, onTrackbarChange);
//     cv::createTrackbar("S Max", "HSV Controls", &s_max, 255, onTrackbarChange);
//     cv::createTrackbar("V Min", "HSV Controls", &v_min, 255, onTrackbarChange);
//     cv::createTrackbar("V Max", "HSV Controls", &v_max, 255, onTrackbarChange);

//     // 4. 初始显示
//     updateMask();

//     // 5. 主循环：等待用户调整滑动条，按ESC键退出
//     std::cout << "使用滑动条调整HSV阈值，按ESC键退出并输出最终阈值..." << std::endl;
//     while (true) {
//         // 检测按键（等待1ms，避免程序卡顿）
//         char key = cv::waitKey(1);
//         if (key == 27) { // ESC键的ASCII码为27
//             break;
//         }
//         // 每次循环更新掩膜（滑动条变动后实时刷新）
//         updateMask();
//     }

//     // 6. 退出时输出最终阈值（便于复制到目标程序中）
//     std::cout << "\n\n最终HSV阈值：" << std::endl;
//     std::cout << "lower_hsv = cv::Scalar(" << h_min << ", " << s_min << ", " << v_min << ");" << std::endl;
//     std::cout << "upper_hsv = cv::Scalar(" << h_max << ", " << s_max << ", " << v_max << ");" << std::endl;

//     // 释放资源
//     cv::destroyAllWindows();
//     return 0;
// }

#include <opencv2/opencv.hpp>
#include <iostream>

// 全局变量：二值化阈值（0-255）和反转标志（0表示不反转，1表示反转）
int threshold_value = 127;
int threshold_type = 0;  // 0: THRESH_BINARY, 1: THRESH_BINARY_INV

// 全局图像变量
cv::Mat img_bgr, img_gray, img_binary;

/**
 * @brief 滑动条回调函数（空实现，用于触发图像更新）
 */
void onTrackbarChange(int, void*) {}

/**
 * @brief 根据当前阈值和类型更新二值化结果并显示
 */
void updateBinaryImage() {
    // 选择二值化类型（是否反转）
    int type = (threshold_type == 0) ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV;
    
    // 执行二值化操作
    cv::threshold(img_gray, img_binary, threshold_value, 255, type);
    
    // 显示图像
    cv::imshow("Original Image", img_bgr);
    cv::imshow("Grayscale Image", img_gray);
    cv::imshow("Binary Image", img_binary);
    
    // 实时显示当前参数
    std::cout << "\r当前阈值: " << threshold_value 
              << " | 反转模式: " << (threshold_type ? "是" : "否") 
              << std::flush;
}

int main(int argc, char* argv[]) {
    // 1. 读取输入图像
    std::string img_path;
    if (argc > 1) {
        img_path = argv[1];  // 从命令行参数获取图像路径
    } else {
        std::cout << "请输入图像路径: ";
        std::cin >> img_path;  // 手动输入图像路径
    }

    // 加载图像（BGR格式）
    img_bgr = cv::imread(img_path);
    if (img_bgr.empty()) {
        std::cerr << "错误：无法加载图像！路径可能无效: " << img_path << std::endl;
        return -1;
    }

    // 转换为灰度图
    cv::cvtColor(img_bgr, img_gray, cv::COLOR_BGR2GRAY);

    // 2. 创建显示窗口
    cv::namedWindow("Original Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Grayscale Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Binary Image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Binary Controls", cv::WINDOW_AUTOSIZE);  // 控制窗口

    // 3. 创建滑动条
    cv::createTrackbar("Threshold Value", "Binary Controls", 
                      &threshold_value, 255, onTrackbarChange);
    cv::createTrackbar("Inverse (0:No,1:Yes)", "Binary Controls", 
                      &threshold_type, 1, onTrackbarChange);

    // 4. 初始显示
    updateBinaryImage();

    // 5. 主循环：等待用户操作
    std::cout << "使用滑动条调整二值化参数，按ESC键退出并输出最终参数..." << std::endl;
    while (true) {
        char key = cv::waitKey(1);
        if (key == 27) {  // ESC键退出
            break;
        }
        updateBinaryImage();  // 实时更新
    }

    // 6. 输出最终参数
    std::cout << "\n\n最终二值化参数：" << std::endl;
    std::cout << "阈值: " << threshold_value << std::endl;
    std::cout << "二值化类型: " << (threshold_type == 0 ? "THRESH_BINARY" : "THRESH_BINARY_INV") << std::endl;
    std::cout << "对应代码: cv::threshold(gray_img, binary_img, " 
              << threshold_value << ", 255, cv::" 
              << (threshold_type == 0 ? "THRESH_BINARY" : "THRESH_BINARY_INV") 
              << ");" << std::endl;

    // 释放资源
    cv::destroyAllWindows();
    return 0;
}