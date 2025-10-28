from Robotic_Arm.rm_robot_interface import *
import time
import numpy as np

robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = robot.rm_create_robot_arm("192.168.10.18", 8080)
print("机械臂ID：", handle.id)

software_info = robot.rm_get_arm_software_info()
if software_info[0] == 0:
    print("\n================== Arm Software Information ==================")
    print("Arm Model: ", software_info[1]['product_version'])
    print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
    print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
    print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
    print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
    print("==============================================================\n")
else:
    print("\nFailed to get arm software information, Error code: ", software_info[0], "\n")

def main():
    try:
        # 1. 初始化机械臂实例（使用三线程模式）
        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)

        # 2. 创建机械臂连接（替换为实际IP和端口）
        handle = arm.rm_create_robot_arm("192.168.10.18", 8080)
        print(f"机械臂连接ID: {handle.id}")

        # 3. 设置目标关节角度 [0, -90, 0, 0, 0, 0]（单位：度）
        target_joints = [ 0.578506, 0, 0.060504, np.pi, 0, np.pi]
        ret = arm.rm_movel(target_joints, 25, 0, 0, 1)

        if ret == 0:
            print("指令发送成功，机械臂正在运动...")

            # 5. 可选：等待运动完成（阻塞模式下会自动等待）
            time.sleep(1)  # 额外等待1秒确保稳定

            # 6. 验证当前状态
            state = arm.rm_get_current_arm_state()
            if state[0] == 0:
                print(f"当前实际关节角: {state[1]['joint']}")
            else:
                print("状态查询失败")
        else:
            print(f"运动指令失败，错误码: {ret}")

    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        # 7. 断开连接
        if 'arm' in locals():
            arm.rm_delete_robot_arm()
            print("机械臂连接已释放")

if __name__ == "__main__":
    main()







