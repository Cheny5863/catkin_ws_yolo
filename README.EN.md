# catkin_ws_yolo
this ROS workspace is have a program called object_detect and a simulation package intend to detect common object from ROS video stream using yolov3

# usage
1. `git clone https://github.com/Cheny5863/catkin_ws_yolo.git`
2. `cd catkin_ws_yolo/`
3. `catkin_make`
4. launch the simulation environment 
`source ./devel/setup.bash`
`roslauch six_wheel_robot openGazebo.launch`
5. run the object_detect program
`rosrun object_detect object_detect_node `

Then you can see the effect, as shown in the image below

# result
![Screenshot from 2023-01-19 08-34-32](https://user-images.githubusercontent.com/40204259/213338223-ae986808-5ba9-4913-8436-4f0d1b65288c.png)
