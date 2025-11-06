#include <ros/ros.h>

int main(int argc,char *argv[]){
    ros::init(argc,argv,"pub_param");

    ros::NodeHandle nh;
    nh.setParam("Xkp",-1.3);  //浮点型  x-> -1.3 
    nh.setParam("Xki",-0.2);  //浮点型  x-> -0.2     
    nh.setParam("Xkd",0.0);   //浮点型
    nh.setParam("Ykp",-1.3);  //浮点型  y->-1.3
    nh.setParam("Yki",-0.2);  //浮点型  y->-0.2
    nh.setParam("Ykd",0.0);   //浮点型
    nh.setParam("Rkp",-0.3);  //浮点型  -0.3
    nh.setParam("Rki",-0.2);  //浮点型  -0.2
    nh.setParam("Rkd",0.3);   //浮点型   0.3
    nh.setParam("Aim_Num",0); //整型

    return 0;
}