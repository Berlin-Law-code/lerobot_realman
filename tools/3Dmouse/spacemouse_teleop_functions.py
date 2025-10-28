"""
SpaceMouse遥操作函数
用于通过SpaceMouse控制机械臂的6D姿态
"""

import numpy as np
import time
from typing import Tuple, Optional, Union
from multiprocessing.managers import SharedMemoryManager
from transforms3d import euler, quaternions, affines
from spacemouse_shared_memory import Spacemouse


class SpaceMouseTeleoperator:
    """SpaceMouse遥操作器类，用于处理机械臂的实时遥操作"""
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        position_scale: float = 0.1,
        orientation_scale: float = 0.3,
        deadzone: float = 0.01,
        max_value: int = 500,
        frequency: int = 200,
        pose_format: str = "matrix"  # "matrix", "position_euler", "position_quaternion"
    ):
        """
        初始化SpaceMouse遥操作器
        
        Args:
            shm_manager: 共享内存管理器
            position_scale: 位置变化的缩放因子 (m/s)
            orientation_scale: 旋转变化的缩放因子 (rad/s)
            deadzone: 死区阈值，小于此值的输入将被忽略
            max_value: SpaceMouse的最大值 (300为有线版本，500为无线版本)
            frequency: 采样频率
            pose_format: 姿态表示格式
        """
        self.position_scale = position_scale
        self.orientation_scale = orientation_scale
        self.pose_format = pose_format
        
        # 创建SpaceMouse实例
        self.spacemouse = Spacemouse(
            shm_manager=shm_manager,
            deadzone=deadzone,
            max_value=max_value,
            frequency=frequency
        )
        
        # 记录上次更新时间，用于计算dt
        self.last_time = None
        
    def start(self):
        """启动SpaceMouse"""
        self.spacemouse.start()
        self.last_time = time.monotonic()
        
    def stop(self):
        """停止SpaceMouse"""
        self.spacemouse.stop()
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    def _parse_pose_input(self, current_pose) -> Tuple[np.ndarray, np.ndarray]:
        """
        解析输入的姿态格式，返回位置和旋转矩阵
        
        Args:
            current_pose: 当前姿态，支持多种格式
            
        Returns:
            position: (3,) 位置向量
            rotation_matrix: (3,3) 旋转矩阵
        """
        if isinstance(current_pose, np.ndarray):
            if current_pose.shape == (4, 4):
                # 4x4变换矩阵
                position = current_pose[:3, 3]
                rotation_matrix = current_pose[:3, :3]
            elif current_pose.shape == (6,):
                # [x, y, z, roll, pitch, yaw]
                position = current_pose[:3]
                rotation_matrix = euler.euler2mat(
                    current_pose[3], current_pose[4], current_pose[5]
                )
            elif current_pose.shape == (7,):
                # [x, y, z, qw, qx, qy, qz]
                position = current_pose[:3]
                rotation_matrix = quaternions.quat2mat(current_pose[3:])
            else:
                raise ValueError(f"不支持的姿态数组形状: {current_pose.shape}")
        else:
            raise ValueError(f"不支持的姿态类型: {type(current_pose)}")
            
        return position.copy(), rotation_matrix.copy()
        
    def _format_output_pose(
        self, 
        position: np.ndarray, 
        rotation_matrix: np.ndarray
    ) -> Union[np.ndarray]:
        """
        根据指定格式输出姿态
        
        Args:
            position: (3,) 位置向量
            rotation_matrix: (3,3) 旋转矩阵
            
        Returns:
            格式化后的姿态
        """
        if self.pose_format == "matrix":
            # 返回4x4变换矩阵
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = rotation_matrix
            pose_matrix[:3, 3] = position
            return pose_matrix
        elif self.pose_format == "position_euler":
            # 返回[x, y, z, roll, pitch, yaw]
            euler_angles = euler.mat2euler(rotation_matrix)
            return np.concatenate([position, euler_angles])
        elif self.pose_format == "position_quaternion":
            # 返回[x, y, z, qw, qx, qy, qz]
            quaternion = quaternions.mat2quat(rotation_matrix)
            return np.concatenate([position, quaternion])
        else:
            raise ValueError(f"不支持的输出格式: {self.pose_format}")
            
    def teleop_space(self, current_pose, dt: Optional[float] = None):
        """
        使用SpaceMouse进行遥操作，返回更新后的姿态
        
        Args:
            current_pose: 当前6D姿态，支持多种格式:
                        - 4x4变换矩阵
                        - [x, y, z, roll, pitch, yaw] (6元素数组)
                        - [x, y, z, qw, qx, qy, qz] (7元素数组，四元数)
            dt: 时间步长，如果为None则自动计算
            
        Returns:
            next_pose: 更新后的6D姿态，格式与输入相同
        """
        # 计算时间步长
        current_time = time.monotonic()
        if dt is None:
            if self.last_time is not None:
                dt = current_time - self.last_time
            else:
                dt = 1.0 / 100.0  # 默认100Hz
        self.last_time = current_time
        
        # 获取SpaceMouse输入
        spacemouse_state = self.spacemouse.get_motion_state_transformed()
        
        # 解析当前姿态
        position, rotation_matrix = self._parse_pose_input(current_pose)
        
        # 计算位置增量 (在全局坐标系中)
        position_delta = spacemouse_state[:3] * self.position_scale * dt
        new_position = position + position_delta
        
        # 计算旋转增量 (在当前末端执行器坐标系中)
        rotation_delta = spacemouse_state[3:] * self.orientation_scale * dt
        
        # 将旋转增量转换为旋转矩阵 (小角度近似)
        angle = np.linalg.norm(rotation_delta)
        if angle > 1e-6:
            axis = rotation_delta / angle
            delta_rotation_matrix = euler.axangle2mat(axis, angle)
            # 在当前旋转的基础上叠加增量旋转
            new_rotation_matrix = rotation_matrix @ delta_rotation_matrix
        else:
            new_rotation_matrix = rotation_matrix
            
        # 格式化输出
        return self._format_output_pose(new_position, new_rotation_matrix)
        
    def get_button_states(self) -> Tuple[bool, bool]:
        """
        获取SpaceMouse按钮状态
        
        Returns:
            (left_button, right_button): 左右按钮的按下状态
        """
        return (
            self.spacemouse.is_button_pressed(0),
            self.spacemouse.is_button_pressed(1)
        )
        
    def get_raw_input(self) -> np.ndarray:
        """
        获取原始SpaceMouse输入 (归一化后的6DOF数据)
        
        Returns:
            6DOF输入数据: [tx, ty, tz, rx, ry, rz]
        """
        return self.spacemouse.get_motion_state_transformed()


# 全局变量，用于保存遥操作器实例
_global_teleoperator = None
_global_shm_manager = None


def teleop_space(current_pose, **kwargs):
    """
    简化的遥操作函数，使用全局SpaceMouse实例
    
    Args:
        current_pose: 当前6D姿态
        **kwargs: 其他参数，如dt、position_scale、orientation_scale等
        
    Returns:
        next_pose: 更新后的6D姿态
        
    使用示例:
        # 初始化
        init_spacemouse_teleop()
        
        # 在循环中使用
        current_pose = np.eye(4)  # 或其他格式的姿态
        for i in range(1000):
            current_pose = teleop_space(current_pose)
            time.sleep(0.01)
            
        # 清理
        cleanup_spacemouse_teleop()
    """
    global _global_teleoperator
    
    if _global_teleoperator is None:
        raise RuntimeError("请先调用 init_spacemouse_teleop() 初始化遥操作器")
        
    return _global_teleoperator.teleop_space(current_pose, **kwargs)


def init_spacemouse_teleop(
    position_scale: float = 0.1,
    orientation_scale: float = 0.3,
    deadzone: float = 0.3,
    max_value: int = 500,
    pose_format: str = "matrix"
):
    """
    初始化全局SpaceMouse遥操作器
    
    Args:
        position_scale: 位置变化的缩放因子 (m/s)
        orientation_scale: 旋转变化的缩放因子 (rad/s)
        deadzone: 死区阈值
        max_value: SpaceMouse最大值
        pose_format: 姿态格式 ("matrix", "position_euler", "position_quaternion")
    """
    global _global_teleoperator, _global_shm_manager
    
    if _global_teleoperator is not None:
        print("遥操作器已经初始化，先清理再重新初始化")
        cleanup_spacemouse_teleop()
        
    _global_shm_manager = SharedMemoryManager()
    _global_shm_manager.start()
    
    _global_teleoperator = SpaceMouseTeleoperator(
        shm_manager=_global_shm_manager,
        position_scale=position_scale,
        orientation_scale=orientation_scale,
        deadzone=deadzone,
        max_value=max_value,
        pose_format=pose_format
    )
    
    _global_teleoperator.start()
    print("SpaceMouse遥操作器初始化完成")


def cleanup_spacemouse_teleop():
    """清理全局SpaceMouse遥操作器"""
    global _global_teleoperator, _global_shm_manager
    
    if _global_teleoperator is not None:
        _global_teleoperator.stop()
        _global_teleoperator = None
        
    if _global_shm_manager is not None:
        _global_shm_manager.shutdown()
        _global_shm_manager = None
        
    print("SpaceMouse遥操作器已清理")


def get_spacemouse_buttons() -> Tuple[bool, bool]:
    """
    获取SpaceMouse按钮状态 (使用全局实例)
    
    Returns:
        (left_button, right_button): 左右按钮的按下状态
    """
    global _global_teleoperator
    
    if _global_teleoperator is None:
        raise RuntimeError("请先调用 init_spacemouse_teleop() 初始化遥操作器")
        
    return _global_teleoperator.get_button_states()


def get_spacemouse_raw_input() -> np.ndarray:
    """
    获取原始SpaceMouse输入 (使用全局实例)
    
    Returns:
        6DOF输入数据: [tx, ty, tz, rx, ry, rz]
    """
    global _global_teleoperator
    
    if _global_teleoperator is None:
        raise RuntimeError("请先调用 init_spacemouse_teleop() 初始化遥操作器")
        
    return _global_teleoperator.get_raw_input()


if __name__ == "__main__":
    """测试代码"""
    
    # 测试1: 使用类方式
    print("测试1: 使用SpaceMouseTeleoperator类")
    with SharedMemoryManager() as shm_manager:
        with SpaceMouseTeleoperator(
            shm_manager=shm_manager,
            position_scale=0.05,
            orientation_scale=0.2,
            pose_format="position_euler"
        ) as teleop:
            
            # 初始姿态
            current_pose = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0])  # [x,y,z,roll,pitch,yaw]
            
            print(f"初始姿态: {current_pose}")
            
            for i in range(5000):
                # 获取更新后的姿态
                current_pose = teleop.teleop_space(current_pose)
                
                # 获取按钮状态
                left_btn, right_btn = teleop.get_button_states()
                
                # 获取原始输入
                raw_input = teleop.get_raw_input()
                
                print(f"Step {i:3d}: Pose=[{current_pose[0]:.3f}, {current_pose[1]:.3f}, {current_pose[2]:.3f}, {current_pose[3]:.3f}, {current_pose[4]:.3f}, {current_pose[5]:.3f}], "
                        f"Buttons=({left_btn}, {right_btn}), "
                        f"Raw=[{raw_input[0]:.3f}, {raw_input[1]:.3f}, {raw_input[2]:.3f}]")
            
                time.sleep(0.01)
    
    print("\n" + "="*50 + "\n")
    
    # 测试2: 使用全局函数方式
    print("测试2: 使用全局函数")
    try:
        # 初始化
        init_spacemouse_teleop(
            position_scale=0.05,
            orientation_scale=0.2,
            pose_format="matrix"
        )
        
        # 初始姿态 (4x4矩阵)
        current_pose = np.eye(4)
        current_pose[:3, 3] = [0.3, 0.0, 0.2]
        
        print(f"初始姿态: {current_pose[:3, 3]}")
        
        for i in range(2000):
            # 更新姿态
            current_pose = teleop_space(current_pose)
            
            if i % 20 == 0:
                print(f"Step {i:3d}: Position=[{current_pose[0, 3]:.3f}, {current_pose[1, 3]:.3f}, {current_pose[2, 3]:.3f}]")
                
            time.sleep(0.01)
            
    finally:
        # 清理
        cleanup_spacemouse_teleop()
        
    print("测试完成!")