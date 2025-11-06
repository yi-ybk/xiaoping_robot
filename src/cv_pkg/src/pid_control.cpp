#include <ros/ros.h>
#include <tf/tf.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

//运行误差
#define Allowable_error 0.02

// 定义PID参数结构体（用于存储本地参数）
struct PIDParams {
    float Target, Actual, Out;
    float kp, ki, kd;
    float Error0, Error1, ErrorInt;
    float OutMax;
};

PIDParams X_pid;  
PIDParams Y_pid;  
PIDParams Yaw_pid;  

double quat_to_yaw(const geometry_msgs::Quaternion& quat);

int is_AimPoint = 0;
int Aim = 0;
bool Get_Aim = false;
geometry_msgs::Twist twist;
bool is_grip = false;
geometry_msgs::TransformStamped tfs;

int lose_Aim = 0;
float last_diff_x = 0;
float last_diff_y = 0;
float last_diff_z = 0;


//更新pid参数
void PIDparamCallback(const ros::TimerEvent&){
    static ros::NodeHandle nh2;

    nh2.getParam("/Xkp", X_pid.kp);
    nh2.getParam("/Xki", X_pid.ki);
    nh2.getParam("/Xkd", X_pid.kd);
    nh2.getParam("/Ykp", Y_pid.kp);
    nh2.getParam("/Yki", Y_pid.ki);
    nh2.getParam("/Ykd", Y_pid.kd);
    nh2.getParam("/Rkp", Yaw_pid.kp);
    nh2.getParam("/Rki", Yaw_pid.ki);
    nh2.getParam("/Rkd", Yaw_pid.kd);

    nh2.getParam("/is_grip", is_grip);  //更新是否抓取到方块的状态
    nh2.getParam("/Aim_Num", Aim);  //更新是否抓取到方块的状态
}

void PID_Init(void){
    X_pid.Target = 0.35;
    X_pid.OutMax = 0.5;
    Y_pid.Target = 0.0;
    Y_pid.OutMax = 0.5;
    Yaw_pid.Target = 0.0;
    Yaw_pid.OutMax = 0.3;
}

float PID_Update(PIDParams& p)
{
	p.Error1 = p.Error0;
	p.Error0 = p.Target - p.Actual;
	
	if (p.ki != 0)
	{
		p.ErrorInt += p.Error0;
	}
	else
	{
		p.ErrorInt = 0;
	}

    if (p.ErrorInt > 0.5) {p.ErrorInt = 0.5;}
	if (p.ErrorInt < -0.5) {p.ErrorInt = -0.5;}
	
	p.Out =  p.kp * p.Error0
		   + p.ki * p.ErrorInt
		   + p.kd * (p.Error0 - p.Error1);

    if(p.Out > p.OutMax){
        p.Out = p.OutMax;
    }

    if(fabs(p.Error0) <= Allowable_error){
        p.Out = 0;
    }

    return p.Out;       
}

int main(int argc,char *argv[]){
    ros::init(argc,argv,"pid_control");

    ros::NodeHandle nh;
    ros::Publisher vel_pub = nh.advertise<geometry_msgs::Twist>("cmd_vel", 10);
    ros::Publisher arm_pub = nh.advertise<geometry_msgs::Pose>("/arm_position", 10);
    ros::Publisher arm_pub2 = nh.advertise<geometry_msgs::Point>("/arm_gripper", 10);
    // 注册参数变化（监听指定参数）
    ros::Timer param_check_timer = nh.createTimer(ros::Duration(0.1), 
                                           &PIDparamCallback);                                     
                                                    
    // 创建 TF 订阅节点
    tf2_ros::Buffer buffer;
    tf2_ros::TransformListener listener(buffer);

    PID_Init();

    
    ros::Rate r(10);
    while(ros::ok()){

        try
        {
            tfs = buffer.lookupTransform("base_link","Aim",ros::Time(0));
            if(Aim != 0){
                Get_Aim = true;
            }
            if(is_grip == true){       //更新是否抓取到方块的状态
                if(Get_Aim == true){
                    twist.angular.z = 0;
                    twist.linear.y  = 0;
                    twist.linear.x  = 0;
                }
                Get_Aim = false;
            }
        }
        catch(const std::exception& e)
        {
            printf("异常信息:%s\n",e.what());
            twist.angular.z = 0;
            twist.linear.y  = 0;
            twist.linear.x  = 0;

            if(Get_Aim == true){
                vel_pub.publish(twist);
            }
            Get_Aim = false;
        }
    
        if(Get_Aim == true){
            X_pid.Actual = tfs.transform.translation.x;
            Y_pid.Actual = tfs.transform.translation.y;
            //四元数转化为欧拉角
            Yaw_pid.Actual = quat_to_yaw(tfs.transform.rotation);

            twist.angular.z = PID_Update(Yaw_pid);
            twist.linear.y = PID_Update(Y_pid);
            twist.linear.x = PID_Update(X_pid);
            //防止太近识别失败
            if(twist.linear.y != 0){
                twist.linear.x = 0;
            }

            // 丢失目标,原地旋转寻找目标
            if(last_diff_x == X_pid.Actual && last_diff_y == Y_pid.Actual && last_diff_z == Yaw_pid.Actual){
                lose_Aim++;
                if(lose_Aim >= 5){
                    lose_Aim = 0;
                    if(twist.linear.y != 0 && twist.angular.z != 0){
                        twist.linear.y = 0;
                        twist.angular.z = -twist.angular.z;
                    }
                }
 
            }
            last_diff_x = X_pid.Actual;
            last_diff_y = Y_pid.Actual;
            last_diff_z = Yaw_pid.Actual;


            vel_pub.publish(twist);

            //调试用
            printf("current_diff_x = %f, current_diff_y = %f,diff_yaw = %f\n",X_pid.Actual,Y_pid.Actual,Yaw_pid.Actual);
            printf("p = %f, i = %f,d = %f\n",Y_pid.kp,Y_pid.ki,Y_pid.kd);
            printf("vel_x = %f, vel_y = %f,vel_yaw = %f\n",twist.linear.x,twist.linear.y,twist.angular.z);

            //3个速度都为零，代表到达目标位置
            if(twist.angular.z == 0 && twist.linear.y == 0 && twist.linear.x == 0){
                is_AimPoint++;
            }
            else{
                is_AimPoint = 0;
            }
            
            //连续5s判定到达目标位置，进入抓取流程
            if(is_AimPoint >= 50){
                is_AimPoint = 0;
                nh.setParam("Aim_Num",0);  //复位目标方块

                // 创建Pose消息对象
                geometry_msgs::Pose target_pose;
                target_pose.position.x = 0.22;   
                target_pose.position.y = -0.08;   //放下机械臂
                target_pose.position.z = 0.0;   
                target_pose.orientation.x = 0.0;
                target_pose.orientation.y = 0.0;
                target_pose.orientation.z = 0.0;
                target_pose.orientation.w = 0.0;  
                arm_pub.publish(target_pose);

                geometry_msgs::Point Is_gripper;
                Is_gripper.x = 1;
                Is_gripper.y = 0;
                Is_gripper.z = 0;
                arm_pub2.publish(Is_gripper);

                ros::Duration(2).sleep();  // 延时2秒，等待机械臂到达目标位置

                target_pose.position.y = 0.05;   //收起机械臂
                arm_pub.publish(target_pose);

                nh.setParam("is_grip",true); //更新是否抓取到方块的状态
                is_grip = true;
            }
        }

        r.sleep();
        ros::spinOnce();
    }

    return 0;
}

// 四元数转化为欧拉角
double quat_to_yaw(const geometry_msgs::Quaternion& quat) {
  tf::Quaternion q(quat.x, quat.y, quat.z, quat.w);
  tf::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  return yaw;
}