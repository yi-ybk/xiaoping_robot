#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <numeric>

//HSV阈值结构体类型
struct HSVColor {
    cv::Scalar low_hsv;        // HSV低阈值
    cv::Scalar high_hsv;       // HSV高阈值
};

std::vector<HSVColor> hsv_colors = {
    //   HSV低阈值                    HSV高阈值          
    {cv::Scalar(148, 3, 102),  cv::Scalar(164, 20, 122), }, // 普通方块
    {cv::Scalar(0, 0, 85)   ,  cv::Scalar(159, 3, 112), },  // 放置区域
    {cv::Scalar(0, 3, 94)   ,  cv::Scalar(179, 43, 102), }, // 普通方块角度2
    {cv::Scalar(0, 0, 102)  ,  cv::Scalar(20, 30, 130), }   // 数字6
};

// 角点排序函数：将四边形角点按顺时针排序（左上->右上->右下->左下）
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

void Cam_RGB_Callback(const sensor_msgs::Image::ConstPtr& msg){
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
    cv::Mat img_hsv;
    cv::cvtColor(img_BGR888, img_hsv, cv::COLOR_BGR2HSV);  // 转换颜色空间

    //掩膜提取数字所在的区域(二值化)
    cv::Mat mask;  // 存储掩膜结果（白色为目标区域，黑色为背景）
    //将HSV图像中落在阈值范围内的像素设为白色（255），其余为黑色（0）
    //处理多HSV阈值
    for(int i = 0;i < hsv_colors.size(); ++i){
        cv::inRange(img_hsv, 
                    hsv_colors[i].low_hsv,
                    hsv_colors[i].high_hsv,
                    mask);

        //查找轮廓：从掩膜中提取数字所在面的轮廓（轮廓是目标区域的边界）
        std::vector<std::vector<cv::Point>> contours;  // 存储所有轮廓（每个轮廓是点的集合）
        std::vector<cv::Vec4i> hierarchy;             // 存储轮廓层级关系（用于筛选轮廓）
        // findContours函数：从二值图像（掩膜）中提取轮廓
        cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);            

        // 无轮廓
        if (contours.empty()) { 
            printf("未从掩膜中找到任何轮廓（可能是掩膜阈值设置不当）\n");
            cv::imshow("RGB_Result", img_BGR888);
            cv::waitKey(1);
            return;
        }

        // 若存在轮廓（点集合非空,即二值化后识别到了图像，若集合为空，可能是HSV二值化阈值设置不合理）
        // 初始化当前帧的目标ID（每帧从1开始递增，确保同帧ID唯一）
        int current_obj_id = 1;
        // 遍历所有轮廓，处理每个有效目标
        for (size_t contour_idx = 0; contour_idx < contours.size(); contour_idx++) {

            std::vector<cv::Point> target_contour = contours[contour_idx];
            double contour_area = cv::contourArea(target_contour);

            // 筛选条件1：过滤面积过小的轮廓（排除噪声，根据目标大小调整阈值）
            const double MIN_CONTOUR_AREA = 5000;  // 最小轮廓面积阈值
            if (contour_area < MIN_CONTOUR_AREA) {
                continue;
            }

            //轮廓近似：将轮廓简化为四边形（提取四个角点）
            std::vector<cv::Point> approx_contour;  // 存储近似后的轮廓
            double epsilon = cv::arcLength(cv::Mat(target_contour), true) * 0.02;  //计算轮廓周长，生成近似参数，定义了原始轮廓上的点到近似后多边形的最大允许距离
            // approxPolyDP函数：用多边形近似轮廓（减少点数量，用更简单的多边形来逼近原始轮廓）
            cv::approxPolyDP(cv::Mat(target_contour), approx_contour, epsilon, true);
            // 检查近似后的轮廓是否为四边形（4个角点）
            if (approx_contour.size() != 4) {
                // printf("目标轮廓不是四边形（角点数量: %lu)\n", approx_contour.size());
                continue;
            }


            /*************对每个目标进行处理************* */
            int recognized_digit = -1;  // 初始化识别结果变量（-1表示未识别）
            char recognized_char = 'N';  // 初始化识别结果变量（N表示未识别）
            std::vector<cv::Point> corner_points = approx_contour;  // 保存四个角点
            // 角点排序
            sortCorners(corner_points);

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

            // 数字识别：基于模板匹配（蒙版识别模板）
            // 获取功能包“cv_pkg”的绝对路径
            std::string pkg_path = ros::package::getPath("cv_pkg");
            // 拼接模板文件夹路径(不依赖启动节点时的终端目录)
            std::string template_dir = pkg_path + "/template/";  // 模板文件夹路径
            int best_match = -1;         // 最佳匹配的图片索引
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

            //printf("图片为第 %d 张\n",best_match);

            //识别结果转化
            if(best_match >= 1 && best_match <= 4){
                recognized_digit = 1;
            }
            else if(best_match >= 5 && best_match <= 8){
                recognized_digit = 2;
            }
            else if(best_match >= 9 && best_match <= 12){
                recognized_digit = 3;
            }
            else if(best_match >= 13 && best_match <= 16){
                recognized_digit = 4;
            }
            else if(best_match >= 17 && best_match <= 20){
                recognized_digit = 5;
            }
            else if(best_match >= 21 && best_match <= 24){
                recognized_digit = 6;
            }
            else if(best_match  == 25){
                recognized_char = 'B';
            }
            else if(best_match  == 26){
                recognized_char = 'O';
            }
            else if(best_match  == 27){
                recognized_char = 'X';
            }
            else{
                recognized_digit = best_match;
            }           
            // 若识别失败，输出结果
            if (recognized_digit == -1 && recognized_char == 'N') {
                printf("未匹配到任何数字（匹配度低于阈值）\n");
            } 


            /********************可视化与结果输出*********************/
            // 绘制目标轮廓（绿色，线宽2）
            cv::drawContours(img_BGR888, contours, contour_idx, cv::Scalar(0, 255, 0), 2);

            // 绘制角点（红色圆点）+ 标注“ID-角点序号”（如“1-0”）
            for (size_t j = 0; j < corner_points.size(); j++) {
                cv::circle(img_BGR888, corner_points[j], 5, cv::Scalar(0, 0, 255), -1);
            }

            // 标注目标ID和识别结果（黄色文字，位于左上点上方，避免遮挡）
            cv::Point text_pos = corner_points[0] + cv::Point(0, -10);
            if (recognized_digit != -1) {
                std::string obj_label = "ID:" + std::to_string(current_obj_id) + 
                                        " Digit:" + std::to_string(recognized_digit);
                cv::putText(img_BGR888, obj_label, text_pos,
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);                        
            }
            else if(recognized_char != 'N'){
                std::string obj_label = "ID:" + std::to_string(current_obj_id) + 
                                        " Char:" + " " + recognized_char;
                cv::putText(img_BGR888, obj_label, text_pos,
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2); 
            }


            // 显示矫正后的ROI（窗口将保留最后一个目标的ROI，便于单独查看）
            cv::imshow("Corrected ROI", corrected_roi);

            //输出最终结果（数字+四角坐标）
            if (recognized_digit != -1) {
                printf("最终结果: 数字=%d, 四角坐标=((%d,%d), (%d,%d), (%d,%d), (%d,%d))\n",
                         recognized_digit,
                         corner_points[0].x, corner_points[0].y,
                         corner_points[1].x, corner_points[1].y,
                         corner_points[2].x, corner_points[2].y,
                         corner_points[3].x, corner_points[3].y);
            }
            else if (recognized_char != 'N'){
                printf("最终结果: 字母=%c, 四角坐标=((%d,%d), (%d,%d), (%d,%d), (%d,%d))\n",
                        recognized_char,
                        corner_points[0].x, corner_points[0].y,
                        corner_points[1].x, corner_points[1].y,
                        corner_points[2].x, corner_points[2].y,
                        corner_points[3].x, corner_points[3].y);
            }

            // 当前目标处理完成，ID自增
            current_obj_id++;
        }
    }

    cv::imshow("RGB_Result",img_BGR888);   //窗口显示识别结果
    cv::waitKey(1);  // 等待1ms（确保窗口响应，避免卡顿）    
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
