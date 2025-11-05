#include <ros/ros.h>

int main(int argc,char *argv[]){
     ros::init(argc,argv,"pub_param");

    ros::NodeHandle nh;
    nh.setParam("kp",0.0); //浮点型
    nh.setParam("ki",0.0); //浮点型
    nh.setParam("kd",0.0); //浮点型

    return 0;
}