#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <numeric>
#include <opencv2/calib3d.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

class CubeFaceRecognizer
{
private:
    /* data */
    // 状态锁定相关变量
    std::vector<cv::Point> lastValidCorners;  // 上一帧的有效角点
    const int CORNER_DIFF_THRESH = 8;         // 角点差异阈值（像素），小于此值则锁定

    // 滑动平均相关变量
    std::deque<std::vector<cv::Point>> cornerHistory;  // 角点历史队列
    const int HISTORY_LEN = 5;                         // 历史帧数（建议3-5）

    std::deque<cv::Mat> rvecHistory;  // 旋转向量历史队列
    const int RVEC_HISTORY_LEN = 5;   // 旋转向量历史帧数（与角点保持一致）
    const float WEIGHT_DECAY = 0.8f;  // 加权平均衰减系数（近期帧权重更高）

    void convertToBaseLink(cv::Mat& rotationMatrix, cv::Mat& tvec, cv::Mat& T_target_base);
    cv::Point3d getPositionFromT(cv::Mat& T_target_base,bool& success);
    cv::Vec3d getOrientationFromT(cv::Mat& T_target_base, bool& success);
    std::vector<cv::Point> processCorners(std::vector<cv::Point> rawCorners);
    cv::Mat filterRvec(const cv::Mat& rawRvec);

public:
    //以下数值通过 /camera/color/camera_info 话题获得
    //相机内参矩阵
    const cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 
            617.305419921875, 0,                424.0,  // fx, 0, cx
            0,                617.305419921875, 240.0,  // 0, fy, cy
            0,                0,                1       // 0, 0 , 1
        );
    cv::Mat cached_depth_img;       // 缓存的深度图像（CV_16UC1，单位：mm）
    std::mutex depth_mutex;         // 保护深度图像和内参的互斥锁
    bool depth_initialized = false; // 标记深度图像和内参是否已初始化

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
    void Aim_TF_Pub(std::vector<cv::Point>& corners);
};

// 创建 TF 订阅节点
tf2_ros::Buffer buffer;

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
    ros::Time img_stamp = msg->header.stamp;  // 记录当前RGB图像时间戳

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
    const cv::Size ROI_SIZE(50, 50);       // 矫正后ROI尺寸

    // 获取模版文件的绝对路径
    std::string template_dir = ros::package::getPath("cv_pkg") + "/template/";

    int current_obj_id = 1;  // 目标ID计数器

    // 存储已处理物体的中心坐标（用于去重）
    std::vector<cv::Point2f> processed_centers;
    // 距离阈值（平方，避免开方运算，单位：像素²，根据实际物体大小调整）
    const double DISTANCE_THRESHOLD_SQ = 3000;  // 例如：20像素的距离阈值


    for (const auto& hsv : hsv_colors) {
        // 提取掩膜
        cv::Mat mask = Re.extractMask(img_hsv, hsv);
        // 形态学闭运算填充小空洞，开运算去除小噪点
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

        // 查找并筛选轮廓
        std::vector<std::vector<cv::Point>> contours = Re.findAndFilterContours(mask, MIN_CONTOUR_AREA);
        if (contours.empty()){
            continue;
        }

        // 处理每个轮廓
        for (const auto& contour : contours) {
            // 计算轮廓的矩（用于求中心）
            cv::Moments contour_moments = cv::moments(contour);
            if (contour_moments.m00 < 1e-6) {  
                continue;
            }
            cv::Point2f current_center(
                contour_moments.m10 / contour_moments.m00,  
                contour_moments.m01 / contour_moments.m00   
            );

            // 动态计算距离阈值（基于轮廓大小，避免固定阈值的局限性）
            cv::Rect contour_rect = cv::boundingRect(contour);
            double contour_diag = sqrt(contour_rect.width*contour_rect.width + contour_rect.height*contour_rect.height);
            const double DISTANCE_THRESHOLD_SQ = pow(contour_diag * 0.2, 2);  // 阈值为对角线的20%

            // 检查重复
            bool is_duplicate = false;
            for (const auto& processed_center : processed_centers) {
                double dx = current_center.x - processed_center.x;
                double dy = current_center.y - processed_center.y;
                double distance_sq = dx * dx + dy * dy;  
                if (distance_sq < DISTANCE_THRESHOLD_SQ) {
                    is_duplicate = true;
                    break;
                }
            }
            if (is_duplicate) {
                continue;  
            }

            // 轮廓近似为四边形
            std::vector<cv::Point> corners = Re.approximateQuadrilateral(contour);
            if (corners.empty()){
                continue;
            }

            // 角点排序
            Re.sortCorners(corners);

            // 透视变换获取矫正ROI
            cv::Mat corrected_roi = Re.perspectiveTransformRoi(img_BGR888, corners, ROI_SIZE);

            // 模板匹配识别
            int best_match = Re.matchTemplateDigit(corrected_roi, template_dir);
            int recognized_digit;
            char recognized_char;
            Re.convertMatchResult(best_match, recognized_digit, recognized_char);

            Re.Aim_TF_Pub(corners);

            // 可视化结果
            Re.visualizeResults(img_BGR888, contour, corners, current_obj_id, recognized_digit, recognized_char);

            // 显示矫正后的ROI
            cv::imshow("Corrected ROI", corrected_roi);

            // 打印结果
            Re.printResults(recognized_digit, recognized_char, corners);

            // 记录当前物体中心（用于后续去重）
            processed_centers.push_back(current_center);
            current_obj_id++;  // ID自增
        }
    }

    // 显示最终结果图像
    cv::imshow("RGB_Result", img_BGR888);
    cv::waitKey(1);
}

/**
 * @brief 深度图像回调函数：缓存深度图像
 */
void depthImageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    std::lock_guard<std::mutex> lock(Re.depth_mutex); // 线程安全
    try {
        // 关键修改：将解析格式改为32FC1（与Python一致）
        Re.cached_depth_img = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
        Re.depth_initialized = true;
    } catch (cv_bridge::Exception& e) {
        printf("深度图像转换失败: %s\n", e.what());
        Re.depth_initialized = false;
    }
}


int main(int argc, char**argv) {
    setlocale(LC_ALL, "zh_CN.UTF-8");
    ros::init(argc, argv, "cv_image_node");
    ros::NodeHandle nh;

    // 订阅图像话题
    ros::Subscriber rgb_sub = nh.subscribe("/camera/color/image_raw", 1, Cam_RGB_Callback);
    ros::Subscriber depth_sub = nh.subscribe("/camera/aligned_depth_to_color/image_raw",10,depthImageCallback);

    tf2_ros::TransformListener listener(buffer);

    // 创建显示窗口
    cv::namedWindow("RGB_Result");
    cv::namedWindow("Corrected ROI");

    ros::spin();
    return 0;
}

// 角点排序函数：将四边形角点按顺时针排序（左上->右上->右下->左下）
void CubeFaceRecognizer::sortCorners(std::vector<cv::Point>& corners) {
    if (corners.size() != 4) {
        ROS_WARN("角点数量不为4,无法排序");
        return;
    }

    // 步骤1：计算四边形中心坐标（用于象限划分）
    cv::Point2f center(0, 0);
    for (const auto& p : corners) {
        center.x += p.x;
        center.y += p.y;
    }
    center.x /= 4.0f;
    center.y /= 4.0f;

    // 步骤2：按“相对于中心的象限”排序（确保唯一顺序）
    std::vector<cv::Point> sorted(4);
    int idx = 0;

    // 左上象限：x < 中心x 且 y < 中心y（优先级最高）
    for (const auto& p : corners) {
        if (p.x < center.x - 1e-3 && p.y < center.y - 1e-3) {  // 减微小值避免浮点误差
            sorted[idx++] = p;
            break;
        }
    }

    // 右上象限：x > 中心x 且 y < 中心y
    for (const auto& p : corners) {
        if (p.x > center.x + 1e-3 && p.y < center.y - 1e-3 && 
            std::find(sorted.begin(), sorted.begin() + idx, p) == sorted.begin() + idx) {
            sorted[idx++] = p;
            break;
        }
    }

    // 右下象限：x > 中心x 且 y > 中心y
    for (const auto& p : corners) {
        if (p.x > center.x + 1e-3 && p.y > center.y + 1e-3 && 
            std::find(sorted.begin(), sorted.begin() + idx, p) == sorted.begin() + idx) {
            sorted[idx++] = p;
            break;
        }
    }

    // 左下象限：x < 中心x 且 y > 中心y（剩余的最后一个点）
    for (const auto& p : corners) {
        if (std::find(sorted.begin(), sorted.begin() + idx, p) == sorted.begin() + idx) {
            sorted[idx++] = p;
            break;
        }
    }

    // 替换原角点（若排序成功）
    if (idx == 4) {
        corners = sorted;
    } else {
        ROS_WARN("角点排序失败，使用备用方案");
        // 备用方案：按“x+y”之和排序（左上和最小，右下和最大）
        std::sort(corners.begin(), corners.end(), [](const cv::Point& a, const cv::Point& b) {
            return (a.x + a.y) < (b.x + b.y);
        });
    }
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
    double max_match_val = 0.15;  // 匹配阈值

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

/* 旋转向量加权平滑函数 */
cv::Mat CubeFaceRecognizer::filterRvec(const cv::Mat& rawRvec) {
    // 初始化历史队列（第一帧直接返回原始值）
    if (rvecHistory.empty()) {
        rvecHistory.push_back(rawRvec.clone());
        return rawRvec;
    }

    // 加入当前帧旋转向量，保持队列长度
    rvecHistory.push_back(rawRvec.clone());
    while (rvecHistory.size() > RVEC_HISTORY_LEN) {
        rvecHistory.pop_front();
    }

    // 加权平均平滑（与角点平滑逻辑一致）
    cv::Mat filteredRvec = cv::Mat::zeros(3, 1, CV_64F);
    float totalWeight = 0.0f;
    float currentWeight = 1.0f;

    // 倒序遍历，最新帧权重最高
    for (auto it = rvecHistory.rbegin(); it != rvecHistory.rend(); ++it) {
        filteredRvec += (*it) * currentWeight;
        totalWeight += currentWeight;
        currentWeight *= WEIGHT_DECAY;
    }

    // 归一化权重
    filteredRvec /= totalWeight;
    return filteredRvec;
}

// 将目标在相机光学坐标系下的位姿转换到/base_link坐标系
void CubeFaceRecognizer::convertToBaseLink(cv::Mat& rotationMatrix, cv::Mat& tvec, cv::Mat& T_target_base) {
    // 定义相机光学系到相机彩色系的变换（固定旋转，Realsense标准约定）
    // camera_color_optical_frame → camera_color_frame
    // 旋转矩阵：光学系到彩色系
    cv::Mat R_optical_to_color = (cv::Mat_<double>(3, 3) <<
        0, 0 , 1,  //z
        -1, 0 , 0, //x
        0, -1, 0); //y
    // 平移向量：光学系与彩色系原点重合（无平移）
    cv::Mat t_optical_to_color = (cv::Mat_<double>(3, 1) << 0, 0, 0);

    // 构造4x4变换矩阵 T_optical_to_color
    cv::Mat T_optical_to_color = cv::Mat::eye(4, 4, CV_64F);  //创建单位矩阵(对角线元素全为1)
    R_optical_to_color.copyTo(T_optical_to_color(cv::Rect(0, 0, 3, 3)));
    t_optical_to_color.copyTo(T_optical_to_color(cv::Rect(3, 0, 1, 3)));

    // 2. camera_color_frame → camera_link 的变换（通过相对坐标计算）
    // camera_color_frame 相对于 base_link：(0.165, 0.0325, 0.005)
    // camera_link 相对于 base_link：(0.165, 0.0175, 0.005)
    // 因此：camera_color_frame 相对于 camera_link 的平移 = (0.165-0.165, 0.0325-0.0175, 0.005-0.005) = (0, 0.015, 0)
    cv::Mat R_color_to_cam = cv::Mat::eye(3, 3, CV_64F);  // 无旋转（数据未提及旋转差异）
    cv::Mat t_color_to_cam = (cv::Mat_<double>(3, 1) << 0, 0.015, 0);  // 相对平移

    // 构造 T_color_to_cam（4x4齐次矩阵）
    cv::Mat T_color_to_cam = cv::Mat::eye(4, 4, CV_64F);
    R_color_to_cam.copyTo(T_color_to_cam(cv::Rect(0, 0, 3, 3)));
    t_color_to_cam.copyTo(T_color_to_cam(cv::Rect(3, 0, 1, 3)));


    // 3. camera_link → base_link 的变换（直接使用已知参数）
    // camera_link 相对于 base_link 的平移：(0.165, 0.0175, 0.005)（数据提供）
    // 旋转：无额外旋转（数据未提及，默认与base_link姿态一致）
    cv::Mat R_cam_to_base = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_cam_to_base = (cv::Mat_<double>(3, 1) << 0.165, 0.0175, 0.005);  // 绝对平移

    // 构造 T_cam_to_base（4x4齐次矩阵）
    cv::Mat T_cam_to_base = cv::Mat::eye(4, 4, CV_64F);
    R_cam_to_base.copyTo(T_cam_to_base(cv::Rect(0, 0, 3, 3)));
    t_cam_to_base.copyTo(T_cam_to_base(cv::Rect(3, 0, 1, 3)));


    // 4. 目标 → 相机光学系的变换（PnP解算结果）
    cv::Mat T_target_to_optical = cv::Mat::eye(4, 4, CV_64F);
    rotationMatrix.copyTo(T_target_to_optical(cv::Rect(0, 0, 3, 3)));  // 旋转
    tvec.copyTo(T_target_to_optical(cv::Rect(3, 0, 1, 3)));  // 平移


    // 5. 计算目标 → base_link 的总变换（严格按层级链）
    // 目标 → 光学系 → 彩色系 → camera_link → base_link
    cv::Mat T_target_color = T_optical_to_color * T_target_to_optical;  // 目标→彩色系
    cv::Mat T_target_cam = T_color_to_cam * T_target_color;  // 目标→camera_link
    T_target_base = T_cam_to_base * T_target_cam;  // 目标→base_link（最终结果）
}

// 从T_target_base中提取三维位置（x:前，y:左，z:上）
// 输出参数success：true表示提取成功，false表示矩阵不符合要求
cv::Point3d CubeFaceRecognizer::getPositionFromT(cv::Mat& T_target_base, bool& success) {
    // 先默认设置为失败状态
    success = false;
    // 检查矩阵是否为4x4
    if (T_target_base.rows != 4 || T_target_base.cols != 4) {
        // 返回含NaN的点表示错误（NaN可通过isnan()判断）
        return cv::Point3d(NAN, NAN, NAN);
    }
    // 矩阵符合要求，提取位置并设置成功状态
    double x = T_target_base.at<double>(0, 3); // X轴：小车前进方向
    double y = T_target_base.at<double>(1, 3); // Y轴：小车左侧方向
    double z = T_target_base.at<double>(2, 3); // Z轴：小车上方（垂直地面）
    success = true;
    return cv::Point3d(x, y, z);
}

// 功能：处理原始角点，依次进行状态锁定→滑动平均，输出稳定的角点
// 输入：检测到的角点
// 输出：稳定后的角点
std::vector<cv::Point> CubeFaceRecognizer::processCorners(std::vector<cv::Point> rawCorners) {
    if (rawCorners.empty() || rawCorners.size() != 4) {
        ROS_ERROR("原始角点数量无效，返回空");
        return {};
    }

    // 状态锁定（沿用原有逻辑，避免高频跳变）
    std::vector<cv::Point> lockedCorners;
    if (!lastValidCorners.empty()) {
        double avgDiff = 0.0;
        for (int i = 0; i < 4; i++) {
            avgDiff += cv::norm(rawCorners[i] - lastValidCorners[i]);
        }
        avgDiff /= 4.0;
        if (avgDiff < CORNER_DIFF_THRESH) {
            lockedCorners = lastValidCorners;
        } else {
            lockedCorners = rawCorners;
            lastValidCorners = rawCorners;
        }
    } else {
        lockedCorners = rawCorners;
        lastValidCorners = rawCorners;
    }

    // 优化为【加权滑动平均】（核心修改）
    cornerHistory.push_back(lockedCorners);
    while (cornerHistory.size() > HISTORY_LEN) {
        cornerHistory.pop_front();
    }

    std::vector<cv::Point> smoothedCorners(4, cv::Point(0, 0));
    float totalWeight = 0.0f;
    float currentWeight = 1.0f;  // 最新帧权重最高

    // 倒序遍历历史（从最新帧到最旧帧），应用权重衰减
    for (auto it = cornerHistory.rbegin(); it != cornerHistory.rend(); ++it) {
        for (int i = 0; i < 4; i++) {
            smoothedCorners[i].x += (*it)[i].x * currentWeight;
            smoothedCorners[i].y += (*it)[i].y * currentWeight;
        }
        totalWeight += currentWeight;
        currentWeight *= WEIGHT_DECAY;  // 权重衰减
    }

    // 加权平均归一化（整数化）
    for (auto& p : smoothedCorners) {
        p.x = round(p.x / totalWeight);
        p.y = round(p.y / totalWeight);
    }

    return smoothedCorners;
}



void CubeFaceRecognizer::Aim_TF_Pub(std::vector<cv::Point>& corners){
    static cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
    const static double squareSize = 0.05; 
    std::vector<cv::Point> stableCorners = processCorners(corners);
    if (stableCorners.empty()) {
        printf("角点处理失败,跳过TF发布\n");
        return;
    }

    std::vector<cv::Point2f> imagePoints;
    for (const auto& corner : stableCorners) {
        imagePoints.push_back(cv::Point2f(corner.x, corner.y));
    }

    // 3D目标点（顺序与角点排序一致：左上→右上→右下→左下）
    std::vector<cv::Point3f> object_points = {
        cv::Point3f(0, 0, 0),
        cv::Point3f(0, -1.5*squareSize, 0),
        cv::Point3f(0, -1.5*squareSize, -1.5*squareSize),
        cv::Point3f(0, 0, -1.5*squareSize)
    };

    // 1. 原始位姿解算（沿用原有逻辑）
    cv::Mat rvec, tvec;
    cv::solvePnP(
        object_points,
        imagePoints,
        cameraMatrix,
        distCoeffs,
        rvec, tvec,
        false,
        cv::SOLVEPNP_ITERATIVE
    );

    // 2. 新增：旋转向量平滑（核心优化）
    cv::Mat filteredRvec = this->filterRvec(rvec);

    // 3. 用平滑后的旋转向量转换为旋转矩阵（替换原rvec）
    cv::Mat rotationMatrix;
    cv::Rodrigues(filteredRvec, rotationMatrix);  // 此处改为filteredRvec

    // 后续坐标转换、TF发布逻辑保持不变
    cv::Mat T_target_base;
    this->convertToBaseLink(rotationMatrix,tvec,T_target_base);
    bool success;
    cv::Point3d position = this->getPositionFromT(T_target_base,success);
    if(success == false){
        return;
    }

    cv::Mat R_target_base = T_target_base(cv::Rect(0, 0, 3, 3)).clone();

    tf2::Matrix3x3 tf_R;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            tf_R[i][j] = R_target_base.at<double>(i, j);
        }
    }
    tf2::Quaternion qtn;
    tf_R.getRotation(qtn);
    qtn.normalize();

    static tf2_ros::TransformBroadcaster pub;
    geometry_msgs::TransformStamped ts;
    ts.header.frame_id = "base_link";
    ts.header.stamp = ros::Time::now();
    ts.child_frame_id = "Aim";
    ts.transform.translation.x = position.x;
    ts.transform.translation.y = position.y;
    ts.transform.translation.z = position.z;
    ts.transform.rotation.x = qtn.getX();
    ts.transform.rotation.y = qtn.getY();
    ts.transform.rotation.z = qtn.getZ();
    ts.transform.rotation.w = qtn.getW();
    pub.sendTransform(ts);
}