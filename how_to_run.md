# How to Run the RMUA Base

This guide assumes the RMUA simulator is already running and ROS topics are visible.

## 1. Start the simulator first

In the simulator repo:

```bash
cd ~/jacksit/rmua/IntelligentUAVChampionshipSimulator

docker rm -f sim01 2>/dev/null

docker run -d \
  --net host \
  --gpus '"device=0"' \
  -e Seed=123 \
  --name sim01 \
  simulator01
```

Check:

```bash
docker ps | grep sim01
source /opt/ros/noetic/setup.zsh
rostopic list
```

You should see topics such as:

```text
/airsim_node/drone_1/gps
/airsim_node/drone_1/imu/imu
/airsim_node/drone_1/vel_body_cmd
/airsim_node/end_goal
```

## 2. Build the Base image

Go to the Base repo:

```bash
cd ~/jacksit/rmua/IntelligentUAVChampionshipBase/basic_dev
```

Build:

```bash
docker build --network=host -t basic_dev .
```

or rebuild:

```bash
docker build --network=host --no-cache -t basic_dev .
```

## 3. Run Base

```bash
chmod +x run_basic_dev.sh
./run_basic_dev.sh
```

This should start the Base / algorithm container and connect it to the simulator through ROS.

## 4. Verify connection inside Base

Inside the Base container:

```bash
source /opt/ros/noetic/setup.bash
source /basic_dev/devel/setup.bash
rostopic list
```

Check sensor data:

```bash
rostopic echo -n 1 /airsim_node/end_goal
rostopic echo -n 1 /airsim_node/drone_1/gps
rostopic hz /airsim_node/drone_1/imu/imu
```

If these work, Base is connected to the simulator.

## 5. Test takeoff

Inside the Base container:

```bash
rosservice call /airsim_node/drone_1/takeoff "{}"
```

## how to stop docker

```bash
docker ps

docker rm -f basic_dev
```


If the drone takes off in the simulator GUI, the Base setup works.

## 6. Useful commands

Check velocity command message type:

```bash
rostopic type /airsim_node/drone_1/vel_body_cmd
rostopic type /airsim_node/drone_1/vel_body_cmd | xargs rosmsg show
```

Check available AirSim services:

```bash
rosservice list | grep airsim
```

Stop simulator:

```bash
docker rm -f sim01
```

rerun docker:

```bash
docker build --network=host -t basic_dev .
docker restart sim01
# docker restart sim_gui
./run_basic_dev.sh
```

rqt:

```bash
source /opt/ros/noetic/setup.zsh
rqt
```

how to record yolo
```bash
docker run --rm --net host --name yolo_recorder \
  -v /home/rl/jacksit/rmua1/IntelligentUAVChampionshipBase/drone_ws/src/navigation_vision/yolov8/images/train:/record \
  -v /home/rl/jacksit/rmua1/IntelligentUAVChampionshipBase/drone_ws/src/navigation_vision/scripts/get_photo.py:/tmp/get_photo.py:ro \
  --entrypoint bash basic_dev -lc '
    source /opt/ros/noetic/setup.bash
    python3 /tmp/get_photo.py \
      _display:=false \
      _autosave:=true \
      _save_interval:=0.2 \
      _max_images:=10000 \
      _save_dir:=/record
  '
```


## 7. Next step

After takeoff works, write the first controller:

```text
read /airsim_node/drone_1/gps
read /airsim_node/end_goal
publish /airsim_node/drone_1/vel_body_cmd
```

Goal: make the drone fly toward the end goal.
