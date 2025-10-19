#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <numeric>


// 角点排序函数：将四边形角点按顺时针排序（左上->右上->右下->左下）
#include <numeric>   // 加这一行

void sortCorners(std::vector<cv::Point>& corners) {
    if (corners.size() != 4) return;

    std::vector<size_t> idx(4);
    std::iota(idx.begin(), idx.end(), 0);

    // 按 x+y 升序 → 左上 vs 右下
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) {
                  return (corners[a].x + corners[a].y) < (corners[b].x + corners[b].y);
              });
    cv::Point top_left     = corners[idx[0]];
    cv::Point bottom_right = corners[idx[3]];

    // 按 x-y 降序 → 右上 vs 左下
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) {
                  return (corners[a].x - corners[a].y) > (corners[b].x - corners[b].y);
              });
    cv::Point top_right    = corners[idx[0]];
    cv::Point bottom_left  = corners[idx[3]];

    corners[0] = top_left;
    corners[1] = top_right;
    corners[2] = bottom_right;
    corners[3] = bottom_left;
}

void Cam_RGB_Callback(const sensor_msgs::Image msg){
    //将ROS图像消息转换为OpenCV的Mat格式
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        /* code */
        cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::BGR8);
    }
    catch(const std::exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s",e.what());
        return;
    }

    //注：不是RGB
    cv::Mat img_BGR888 = cv_ptr->image; // 提取转换后的OpenCV图像（BGR888）

    // 将图像从BGR转换为HSV颜色空间（便于颜色分割）
    // HSV空间中颜色信息与亮度分离，比BGR更适合基于颜色的掩膜提取
    cv::Mat img_hsv;
    cv::cvtColor(img_BGR888, img_hsv, cv::COLOR_BGR2HSV);  // 转换颜色空间

    //掩膜提取数字所在的区域(二值化)
    cv::Mat mask;  // 存储掩膜结果（白色为目标区域，黑色为背景）
    //将HSV图像中落在阈值范围内的像素设为白色（255），其余为黑色（0）
    cv::inRange(img_hsv, 
                cv::Scalar(148, 3, 102),      // 低阈值（H:0-180, S:0-255, V:0-255）
                cv::Scalar(164, 20, 122), // 高阈值（H范围0-180，S允许轻微饱和度，V高亮度）
                mask);

    //查找轮廓：从掩膜中提取数字所在面的轮廓（轮廓是目标区域的边界）
    std::vector<std::vector<cv::Point>> contours;  // 存储所有轮廓（每个轮廓是点的集合）
    std::vector<cv::Vec4i> hierarchy;             // 存储轮廓层级关系（用于筛选轮廓）
    // findContours函数：从二值图像（掩膜）中提取轮廓
    // 参数说明：
    // - mask：输入二值图像
    // - contours：输出轮廓集合
    // - hierarchy：输出轮廓层级
    // - RETR_EXTERNAL：只提取最外层轮廓（忽略嵌套轮廓）
    // - CHAIN_APPROX_SIMPLE：简化轮廓（如矩形只保留四个角点）
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);            

    // 初始化识别结果变量
    int recognized_digit = -1;  // 识别到的数字（-1表示未识别）
    std::vector<cv::Point> corner_points;  // 存储面的四个角点坐标

    // 筛选目标轮廓（假设数字所在的面是最大的轮廓）
    if (!contours.empty()) {  // 若存在轮廓（点集合非空,即二值化后识别到了图像，若集合为空，可能是二值化阈值设置不合理）
        size_t max_contour_idx = 0;  // 最大轮廓的索引
        double max_area = 0;         // 最大轮廓的面积

        // 遍历所有轮廓，找到面积最大的轮廓（假设为目标面）
        // contours.size()获取容器长度
        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);  // 计算轮廓面积
            if (area > max_area) {
                max_area = area;
                max_contour_idx = i;
            }
        }

        // 获取最大轮廓
        std::vector<cv::Point> target_contour = contours[max_contour_idx];

        //轮廓近似：将轮廓简化为四边形（提取四个角点）
        std::vector<cv::Point> approx_contour;  // 存储近似后的轮廓
        double epsilon = cv::arcLength(cv::Mat(target_contour), true) * 0.02;  //计算轮廓周长，生成近似参数，定义了原始轮廓上的点到近似后多边形的最大允许距离
        // approxPolyDP函数：用多边形近似轮廓（减少点数量，用更简单的多边形来逼近原始轮廓）
        cv::approxPolyDP(cv::Mat(target_contour), approx_contour, epsilon, true);

        // 检查近似后的轮廓是否为四边形（4个角点）
        if (approx_contour.size() == 4) {
            corner_points = approx_contour;  // 保存四个角点

            // 角点排序
            sortCorners(corner_points);

            // 可视化：在原图上绘制目标轮廓（绿色，线宽2）
            cv::drawContours(img_BGR888, contours, max_contour_idx, cv::Scalar(0, 255, 0), 2);

            // 可视化：在原图上标记四个角点（红色圆点，半径5，填充）
            for (size_t i = 0; i < corner_points.size(); i++) {
                cv::circle(img_BGR888, corner_points[i], 5, cv::Scalar(0, 0, 255), -1);
                // 标记角点顺序（便于调试）
                cv::putText(img_BGR888, std::to_string(i), corner_points[i] + cv::Point(5, 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2);
            }

            
            // 透视变换参数设置
            int target_width = 50;   // 矫正后图像宽度（根据模板尺寸调整）
            int target_height = 50;  // 矫正后图像高度
            std::vector<cv::Point2f> src_pts;
            std::vector<cv::Point2f> dst_pts;

            // 源点：排序后的四边形角点
            for (auto& p : corner_points) {
                src_pts.emplace_back((float)p.x, (float)p.y);
            }

            // 目标点：正矩形的四个角点（左上->右上->右下->左下）
            dst_pts.emplace_back(0, 0);
            dst_pts.emplace_back(target_width - 1, 0);
            dst_pts.emplace_back(target_width - 1, target_height - 1);
            dst_pts.emplace_back(0, target_height - 1);

            // 计算透视变换矩阵
            cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_pts, dst_pts);

            // 执行透视变换，得到矫正后的正视图
            cv::Mat corrected_roi;
            cv::warpPerspective(img_BGR888, corrected_roi, perspective_matrix, 
                               cv::Size(target_width, target_height));

            // 显示矫正后的图像
            cv::imshow("Corrected ROI", corrected_roi);


            // // 提取数字所在的ROI（识别区域）并识别数字
            // // 计算轮廓的外接矩形（包含住轮廓的最小矩形）
            // cv::Rect roi_rect = cv::boundingRect(target_contour);
            // // 从原图中裁剪出ROI（数字所在的区域）
            // cv::Mat roi_bgr = img_BGR888(roi_rect);


            // 数字识别：基于模板匹配（蒙版识别模板）
            // 获取功能包“cv_pkg”的绝对路径
            std::string pkg_path = ros::package::getPath("cv_pkg");
            // 拼接模板文件夹路径(不依赖启动节点时的终端目录)
            std::string template_dir = pkg_path + "/template/";  // 模板文件夹路径

            int best_match = -1;         // 最佳匹配的数字
            double max_match_val = 0.1;  // 匹配阈值（大于该值视为匹配成功）

            // 遍历所有模板，寻找与ROI最相似的图片
            for (int digit = 1; digit <= 27; digit++) {
                // 模板图像路径（如template/1.png）
                std::string template_path = template_dir + std::to_string(digit) + ".png";
                cv::Mat templ = cv::imread(template_path);  // 读取模板图像

                if (templ.empty()) {  // 若模板不存在，打印警告并跳过
                    printf("未找到模板图像: %s\n", template_path.c_str());
                    continue;
                }

                // 调整模板大小与ROI一致（确保匹配时尺寸相同）
                cv::Mat resized_templ;
                cv::resize(templ, resized_templ, corrected_roi.size());

                // 模板匹配：计算ROI与模板的相似度
                cv::Mat match_result;  // 存储匹配结果（每个像素表示相似度）
                // TM_CCOEFF_NORMED：归一化相关系数法（结果范围[-1,1]，1表示完全匹配）
                cv::matchTemplate(corrected_roi, resized_templ, match_result, cv::TM_CCOEFF_NORMED);

                // 提取匹配结果中的最大值（最高相似度）
                double min_val, max_val;
                cv::minMaxLoc(match_result, &min_val, &max_val);

                // 若当前模板匹配度更高，则更新最佳匹配
                if (max_val > max_match_val) {
                    max_match_val = max_val;
                    best_match = digit;
                }
            } 

            recognized_digit = best_match;  // 保存识别结果       

            // 若识别失败，输出结果
            if (recognized_digit == -1) {
                printf("未匹配到任何数字（匹配度低于阈值）\n");
            } 
            
        }
        else {
            printf("目标轮廓不是四边形（角点数量: %lu)\n", approx_contour.size());
        }

    }
    else {
        printf("未从掩膜中找到任何轮廓（可能是掩膜阈值设置不当）\n");
    }

    
    cv::imshow("RGB_Result",img_BGR888);   //窗口显示识别结果
    cv::waitKey(1);                 // 等待1ms（确保窗口响应，避免卡顿）

    //输出最终结果（数字+四角坐标）
    if (recognized_digit != -1 && !corner_points.empty()) {
        printf("最终结果: 数字=%d, 四角坐标=((%d,%d), (%d,%d), (%d,%d), (%d,%d))\n",
                 recognized_digit,
                 corner_points[0].x, corner_points[0].y,
                 corner_points[1].x, corner_points[1].y,
                 corner_points[2].x, corner_points[2].y,
                 corner_points[3].x, corner_points[3].y);
    }
}

int main(int argc, char** argv){

    setlocale(LC_ALL,"zh_CN.UTF-8");

    ros::init(argc, argv, "cv_image_node");
    ros::NodeHandle nh;
    // ros::Subscriber rgb_sub = nh.subscribe("/kinect2/qhd/image_color_rect",1,Cam_RGB_Callback);
    ros::Subscriber rgb_sub = nh.subscribe("/camera/color/image_raw",1,Cam_RGB_Callback);
    
    // 创建显示窗口（用于展示识别结果）
    cv::namedWindow("RGB_Result");      // 最终结果窗口
    cv::namedWindow("Corrcected ROI");  // 彩色ROI窗口
    ros::spin();

return 0;
}