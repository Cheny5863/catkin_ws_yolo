# catkin_ws_yolo
这个ROS工作区有一个名为object_detect的程序和一个模拟包，旨在使用yolov3检测ROS视频流中的常见对象
# 用法
0. 下载[yolov3.weights](https://www.kaggle.com/datasets/shivam316/yolov3-weights) 到**src/object_detect/src/yolov3/**
1. `git clone https://github.com/Cheny5863/catkin_ws_yolo.git`
2. `cd catkin_ws_yolo/`
3. `catkin_make`
4. 启动仿真环境
`source ./devel/setup.bash`
`roslauch six_wheel_robot openGazebo.launch`
5. 运行目标检测程序
`source ./devel/setup.bash`
`rosrun object_detect object_detect_node `

然后你就能看见，如下图所示的效果

# 结果
![Screenshot from 2023-01-19 08-34-32](https://user-images.githubusercontent.com/40204259/213339628-d48c3fc4-7ba9-4c21-a253-17699a2404b5.png)
