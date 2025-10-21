#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>

class CubeFaceRecognizer
{
private:
    /* data */
public:
    // HSV阈值结构体类型
    struct HSVColor {
        cv::Scalar low_hsv;        // HSV低阈值
        cv::Scalar high_hsv;       // HSV高阈值
    };

    cv::Mat rosImageToCvMat(const sensor_msgs::Image::ConstPtr& msg);
    cv::Mat extractMask(const cv::Mat& img_hsv, const CubeFaceRecognizer::HSVColor & hsv_color);
    std::vector<std::vector<cv::Point>> findAndFilterContours(const cv::Mat& mask, double min_area);
    std::vector<cv::Point> approximateQuadrilateral(const std::vector<cv::Point>& contour);
    void sortCorners(std::vector<cv::Point>& corners);
    cv::Mat perspectiveTransformRoi(const cv::Mat& src_img, 
                                    const std::vector<cv::Point>& corners, 
                                    const cv::Size& target_size);
    int matchTemplateDigit(const cv::Mat& roi, const std::string& template_dir);
    void convertMatchResult(int best_match, int& digit, char& character);
    void visualizeResults(cv::Mat& img, 
                          const std::vector<cv::Point>& contour, 
                          const std::vector<cv::Point>& corners, 
                          int obj_id, 
                          int digit, 
                          char character);
    void printResults(int digit, char character, const std::vector<cv::Point>& corners);
};

//创建全局 识别功能 对象
CubeFaceRecognizer Re;

// 动态数组容器（还不太熟悉）
// 全局HSV阈值配置
std::vector<CubeFaceRecognizer::HSVColor> hsv_colors = {
    {cv::Scalar(148, 3, 102),  cv::Scalar(164, 20, 122)}, // 普通方块
    {cv::Scalar(0, 0, 85)   ,  cv::Scalar(159, 3, 112)},  // 放置区域
    {cv::Scalar(0, 3, 94)   ,  cv::Scalar(179, 43, 102)}, // 普通方块角度2
    {cv::Scalar(0, 0, 102)  ,  cv::Scalar(20, 30, 130)}   // 数字6
};


void Cam_RGB_Callback(const sensor_msgs::Image::ConstPtr& msg) {
    // ROS图像转OpenCV
    cv::Mat img_BGR888 = Re.rosImageToCvMat(msg);
    if (img_BGR888.empty()){
        return;
    } 

    // 转换为HSV颜色空间
    cv::Mat img_hsv;
    cv::cvtColor(img_BGR888, img_hsv, cv::COLOR_BGR2HSV);

    // 过滤杂波与小方块
    const double MIN_CONTOUR_AREA = 5000;  // 最小轮廓面积
    const cv::Size ROI_SIZE(50, 50);       // 设置矫正后ROI尺寸，与模版大小相同，便于匹配

    //获取模版文件的绝对路径（不受节点的运行目录影响）
    std::string template_dir = ros::package::getPath("cv_pkg") + "/template/";

    int current_obj_id = 1;  // 设置目标成功识别的ID计数器

    //auto 关键字遍历HSV阈值容器（还不太熟悉）
    for (const auto& hsv : hsv_colors) {
        // 提取掩膜（将识别到的HSV图像二值化）
        cv::Mat mask = Re.extractMask(img_hsv, hsv);

        // 查找并筛选轮廓
        // std::vector<cv::Point> 表示单个轮廓的点集合
        // std::vector<std::vector<cv::Point>> 表示多个轮廓的集合 
        std::vector<std::vector<cv::Point>> contours = Re.findAndFilterContours(mask, MIN_CONTOUR_AREA);
        if (contours.empty()){
            continue;
        }

        // 处理每个轮廓
        for (const auto& contour : contours) {
            // 轮廓近似为四边形
            std::vector<cv::Point> corners = Re.approximateQuadrilateral(contour);
            if (corners.empty()){
                continue;
            }

            // 角点排序,透视变换（cv::getPerspectiveTransform）需要源四边形与目标四边形顶点一一对应
            Re.sortCorners(corners);

            // 透视变换获取矫正ROI
            cv::Mat corrected_roi = Re.perspectiveTransformRoi(img_BGR888, corners, ROI_SIZE);

            // 模板匹配识别
            int best_match = Re.matchTemplateDigit(corrected_roi, template_dir);
            int recognized_digit;
            char recognized_char;
            Re.convertMatchResult(best_match, recognized_digit, recognized_char);

            // 可视化结果
            Re.visualizeResults(img_BGR888, contour, corners, current_obj_id, recognized_digit, recognized_char);

            // 显示矫正后的ROI
            cv::imshow("Corrected ROI", corrected_roi);

            // 打印结果
            Re.printResults(recognized_digit, recognized_char, corners);

            current_obj_id++;  // ID自增
        }
    }

    // 显示最终结果图像
    cv::imshow("RGB_Result", img_BGR888);
    cv::waitKey(1);
}

int main(int argc, char**argv) {
    setlocale(LC_ALL, "zh_CN.UTF-8");
    ros::init(argc, argv, "cv_image_node");
    ros::NodeHandle nh;

    // 订阅图像话题
    ros::Subscriber rgb_sub = nh.subscribe("/camera/color/image_raw", 1, Cam_RGB_Callback);

    // 创建显示窗口
    cv::namedWindow("RGB_Result");
    cv::namedWindow("Corrected ROI");

    ros::spin();
    return 0;
}

// 角点排序函数：将四边形角点按顺时针排序（左上->右上->右下->左下）
void CubeFaceRecognizer::sortCorners(std::vector<cv::Point>& corners) {
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

/**
 * @brief 将ROS图像消息转换为OpenCV Mat
 * @param msg ROS图像消息指针
 * @return 转换后的OpenCV图像（BGR格式），空Mat表示转换失败
 */
cv::Mat CubeFaceRecognizer::rosImageToCvMat(const sensor_msgs::Image::ConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        return cv_ptr->image;
    } catch (const std::exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return cv::Mat();
    }
}

/**
 * @brief 提取HSV掩膜
 * @param img_hsv HSV格式图像
 * @param hsv_color HSV阈值配置
 * @return 二值化掩膜（目标区域为白色）
 */
cv::Mat CubeFaceRecognizer::extractMask(const cv::Mat& img_hsv, const CubeFaceRecognizer::HSVColor & hsv_color) {
    cv::Mat mask;
    cv::inRange(img_hsv, hsv_color.low_hsv, hsv_color.high_hsv, mask);
    return mask;
}

/**
 * @brief 查找并筛选轮廓（过滤小面积噪声）
 * @param mask 二值化掩膜
 * @param min_area 最小轮廓面积阈值
 * @return 筛选后的轮廓列表
 */
std::vector<std::vector<cv::Point>> CubeFaceRecognizer::findAndFilterContours(const cv::Mat& mask, double min_area) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy; //满足调用findContours()的需求，后续无作用
    //cv::RETR_EXTERNAL 只取最外层轮廓
    //cv::CHAIN_APPROX_SIMPLE 把直线段压成两个端点
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> valid_contours;
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) >= min_area) {
            valid_contours.push_back(contour);
        }
    }
    return valid_contours;
}

/**
 * @brief 将轮廓近似为四边形
 * @param contour 原始轮廓
 * @return 近似后的四边形角点（空表示不是四边形）
 */
std::vector<cv::Point> CubeFaceRecognizer::approximateQuadrilateral(const std::vector<cv::Point>& contour) {
    double epsilon = cv::arcLength(cv::Mat(contour), true) * 0.02;  //若两点之间的角度偏差小于该值，认为在同一条直线上
    std::vector<cv::Point> approx;
    cv::approxPolyDP(cv::Mat(contour), approx, epsilon, true);  //把曲线段压缩成直线段,并保留各线段端点
    return (approx.size() == 4) ? approx : std::vector<cv::Point>();    //判断压缩后的端点数量,若为4个,认为是矩形
}

/**
 * @brief 透视变换获取矫正后的ROI
 * @param src_img 原始图像
 * @param corners 排序后的四边形角点
 * @param target_size 目标ROI尺寸
 * @return 矫正后的ROI图像
 */
cv::Mat CubeFaceRecognizer::perspectiveTransformRoi(const cv::Mat& src_img, 
                                           const std::vector<cv::Point>& corners, 
                                           const cv::Size& target_size) {
    std::vector<cv::Point2f> src_pts;
    for (const auto& p : corners) {
        src_pts.emplace_back((float)p.x, (float)p.y);
    }

    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(0, 0),
        cv::Point2f(target_size.width - 1, 0),
        cv::Point2f(target_size.width - 1, target_size.height - 1),
        cv::Point2f(0, target_size.height - 1)
    };

    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_pts, dst_pts);  //根据原角点坐标和目标变换坐标,生成变换矩阵
    cv::Mat corrected_roi;
    cv::warpPerspective(src_img, corrected_roi, perspective_matrix, target_size); //根据变换矩阵,把任意四边形区域映射成矩形（或任意目标四边形），实现视角矫正
    return corrected_roi;
}

/**
 * @brief 模板匹配识别数字/字母
 * @param roi 待识别的ROI图像
 * @param template_dir 模板文件夹路径
 * @return 最佳匹配的模板索引（-1表示未匹配）
 */
int CubeFaceRecognizer::matchTemplateDigit(const cv::Mat& roi, const std::string& template_dir) {
    int best_match = -1;
    double max_match_val = 0.1;  // 匹配阈值

    for (int digit = 1; digit <= 27; digit++) {
        std::string template_path = template_dir + std::to_string(digit) + ".png";
        cv::Mat templ = cv::imread(template_path);
        if (templ.empty()) {
            ROS_WARN("未找到模板图像: %s", template_path.c_str());
            continue;
        }

        cv::Mat resized_templ;
        cv::resize(templ, resized_templ, roi.size());  //缩放模版,用于匹配

        cv::Mat match_result;
        cv::matchTemplate(roi, resized_templ, match_result, cv::TM_CCOEFF_NORMED);
        
        double min_val, max_val;
        cv::minMaxLoc(match_result, &min_val, &max_val);

        if (max_val > max_match_val) {
            max_match_val = max_val;
            best_match = digit;
        }
    }
    return best_match;
}

/**
 * @brief 转换模板匹配结果为数字/字母
 * @param best_match 模板匹配索引
 * @param[out] digit 识别的数字（-1表示非数字）
 * @param[out] character 识别的字母（'N'表示非字母）
 */
void CubeFaceRecognizer::convertMatchResult(int best_match, int& digit, char& character) {
    digit = -1;
    character = 'N';

    if (best_match >= 1 && best_match <= 4)      digit = 1;
    else if (best_match >= 5 && best_match <= 8) digit = 2;
    else if (best_match >= 9 && best_match <= 12)digit = 3;
    else if (best_match >= 13 && best_match <= 16)digit = 4;
    else if (best_match >= 17 && best_match <= 20)digit = 5;
    else if (best_match >= 21 && best_match <= 24)digit = 6;
    else if (best_match == 25)                   character = 'B';
    else if (best_match == 26)                   character = 'O';
    else if (best_match == 27)                   character = 'X';
}

/**
 * @brief 可视化识别结果（绘制轮廓、角点、标签）
 * @param img 待绘制的图像
 * @param contour 目标轮廓
 * @param corners 排序后的角点
 * @param obj_id 目标ID
 * @param digit 识别的数字
 * @param character 识别的字母
 */
void CubeFaceRecognizer::visualizeResults(cv::Mat& img, 
                     const std::vector<cv::Point>& contour, 
                     const std::vector<cv::Point>& corners, 
                     int obj_id, 
                     int digit, 
                     char character) {
    // 绘制轮廓
    cv::drawContours(img, std::vector<std::vector<cv::Point>>{contour}, 0, cv::Scalar(0, 255, 0), 2);

    // 绘制角点
    for (const auto& p : corners) {
        cv::circle(img, p, 5, cv::Scalar(0, 0, 255), -1);
    }

    // 绘制标签
    cv::Point text_pos = corners[0] + cv::Point(0, -10);
    std::string obj_label;
    if (digit != -1) {
        obj_label = "ID:" + std::to_string(obj_id) + " Digit:" + std::to_string(digit);
    } else if (character != 'N') {
        obj_label = "ID:" + std::to_string(obj_id) + " Char:" + character;
    } else {
        obj_label = "ID:" + std::to_string(obj_id) + " Unknown";
    }
    cv::putText(img, obj_label, text_pos,
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
}

/**
 * @brief 打印识别结果信息
 * @param digit 识别的数字
 * @param character 识别的字母
 * @param corners 排序后的角点
 */
void CubeFaceRecognizer::printResults(int digit, char character, const std::vector<cv::Point>& corners) {
    if (digit != -1) {
        printf("最终结果: 数字=%d, 四角坐标=((%d,%d), (%d,%d), (%d,%d), (%d,%d))\n",
               digit,
               corners[0].x, corners[0].y,
               corners[1].x, corners[1].y,
               corners[2].x, corners[2].y,
               corners[3].x, corners[3].y);
    } else if (character != 'N') {
        printf("最终结果: 字母=%c, 四角坐标=((%d,%d), (%d,%d), (%d,%d), (%d,%d))\n",
               character,
               corners[0].x, corners[0].y,
               corners[1].x, corners[1].y,
               corners[2].x, corners[2].y,
               corners[3].x, corners[3].y);
    } else {
        printf("未匹配到有效目标（匹配度低于阈值）\n");
    }
}