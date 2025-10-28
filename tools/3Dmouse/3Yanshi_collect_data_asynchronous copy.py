#!/usr/bin/env python3
"""
机器人操作数据采集程序 - LeRobot格式
整合遥操作、视觉采集、机器人状态读取，并保存为LeRobot兼容格式
"""

import time
import threading
import numpy as np
import cv2
import os
import json
import h5py
from datetime import datetime
from multiprocessing.managers import SharedMemoryManager
from collections import defaultdict
import queue
import logging
import gc
import ctypes

# 导入相关模块
from Robotic_Arm.rm_robot_interface import *
from spacemouse_teleop_functions import SpaceMouseTeleoperator
import pyrealsense2 as rs

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LeRobotDataCollector:
    """LeRobot格式的机器人数据采集器"""
    
    def __init__(self, 
                 robot_ip="192.168.10.18",
                 robot_port=8080,
                 data_dir="/media/zyf/WD_BLACK/realman_data/lerobot_data",
                 episode_length=1000,
                 fps=30,
                 show_video=True,
                 video_scale=0.8,
                 # 新增：分别设置相机、机器人与遥操作频率（默认相机30Hz、机器人/遥操作180Hz）
                 camera_fps: int | None = None,
                 robot_fps: int = 120,
                 teleop_fps: int = 120):
        
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.data_dir = data_dir
        self.episode_length = episode_length
        # 保持向后兼容：原来的 fps 作为相机帧率的默认值
        self.fps = fps  # 仅用于兼容旧日志/输出
        # 采样频率配置
        self.camera_fps = camera_fps if camera_fps is not None else fps
        self.robot_fps = robot_fps
        self.teleop_fps = teleop_fps
        # 各自dt
        self.dt_cam = 1.0 / max(1, self.camera_fps)
        self.dt_robot = 1.0 / max(1, self.robot_fps)
        self.dt_teleop = 1.0 / max(1, self.teleop_fps)
        # 同步线程以相机帧为节拍
        self.dt_sync = self.dt_cam
        self.show_video = show_video
        self.video_scale = video_scale
        
        # 创建数据目录
        os.makedirs(data_dir, exist_ok=True)
        
        # 初始化组件
        self.arm = None
        self.camera_pipelines = []
        self.teleoperator = None
        self.shm_manager = None
        
        # 控制标志
        self.is_collecting = False
        self.stop_flag = False
        self.video_paused = False
        
        # 线程和队列
        self.data_queue = queue.Queue()
        self.threads = []
        
        # 多线程数据采集相关
        # 动态调整队列大小：容纳约1秒的数据，避免队列溢出
        self.robot_data_queue = queue.Queue(maxsize=max(60, int(self.robot_fps * 1.5)))
        self.camera_data_queue = queue.Queue(maxsize=max(30, int(self.camera_fps * 1.5)))
        self.teleop_data_queue = queue.Queue(maxsize=max(60, int(self.teleop_fps * 1.5)))
        self.sync_data_queue = queue.Queue(maxsize=max(60, int(self.camera_fps * 2)))
        
        # I/O 异步写盘队列
        # 提高队列容量以承载 180Hz 机器人 + 180Hz 遥操作 + 30Hz 相机写入
        self.save_queue = queue.Queue(maxsize=10000)
        self.writer_thread_stop = False
        
        # 线程控制标志
        self.robot_thread_stop = False
        self.camera_thread_stop = False
        self.teleop_thread_stop = False        # 新增：命令下发队列与发送线程（避免 teleop 线程被同步发送阻塞）
        self.command_queue = queue.Queue(maxsize=256)
        self.command_thread_stop = False
        self.command_thread = None
        self.command_lock = threading.Lock()
        self.command_status = {'last_send_time': None, 'last_duration': None, 'last_success': None}
        
        # 数据缓存锁
        self.data_lock = threading.Lock()
        
        # 机器人状态
        self.current_robot_state = None
        self.force_threshold = 25.0
        self.force_exceeded = False
        self.state_lock = threading.Lock()  # 添加线程锁
        self.callback_ptr = None  # 保持回调函数引用防止被GC回收
        
        # 视频显示相关
        self.latest_images = [np.zeros((480, 640, 3), dtype=np.uint8)] * 2
        self.latest_robot_data = None
        self.frame_count = 0
        # 新增：保存各线程最新样本，供同步线程读取，避免高频时抽干队列
        self.latest_robot_sample = None
        self.latest_teleop_sample = None
        # 保护 latest_* 的锁，避免读写竞争
        self.latest_sample_lock = threading.Lock()
        # 同步统计（可用于诊断）
        self.sync_stats = {'total': 0, 'matched': 0, 'mismatch': 0}
        
        # 初始位姿
        self.initial_pose = np.array([0.578506, 0, 0.060504, np.pi, 0, np.pi])
        self.last_safe_pose = self.initial_pose.copy()
        
        # 参照 run_control.py 的存储结构（会话/临时存放）
        self.recording = False
        self.last_button_state = False
        self.current_idx = 0
        self.collect_base_dir = os.path.join(self.data_dir, "collect_data")
        os.makedirs(self.collect_base_dir, exist_ok=True)
        self.session_dir = None
        self.left_dir = None
        self.right_dir = None
        self.obs_dir = None
        # 新增：独立日志路径占位
        self.robot_log_path = None
        self.teleop_log_path = None
        self.camera_ts_path = None
        
        # 录制切换防抖（避免按钮抖动导致频繁开关）
        self.toggle_cooldown = 1  # 秒
        self._last_toggle_ts = 0.0
        
        # 新增：同步空队列告警节流时间戳
        self._last_sync_empty_warn = 0.0
        
        # 视频显示画布（参照 run_control.py 的 show_canvas 风格，双相机并排）
        self.show_canvas = np.zeros((480, 640 * 2, 3), dtype=np.uint8)
        
        # 参照 run_control.py 的相机共享缓冲（双相机版本）
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
        self.npy_list = [np.zeros(480*640*3, dtype=np.uint8), np.zeros(480*640*3, dtype=np.uint8)]
        self.npy_len_list = [0, 0]
        self.img_list = [np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8)]

        # 限制 OpenCV 线程，避免 CPU 过度竞争
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # 新增：电流->力矩系数（7个），默认全1，可通过 set_current_to_torque_coeffs 设置
        self.current_to_torque_coeffs = np.ones(7, dtype=float)
    
    def initialize_robot(self):
        """初始化机器人连接"""
        try:
            self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
            handle = self.arm.rm_create_robot_arm(self.robot_ip, self.robot_port)
            logger.info(f"机器人连接成功，ID: {handle.id}")
            
            # 配置UDP实时数据推送
            custom_config = rm_udp_custom_config_t()
            custom_config.aloha_state = 0
            custom_config.joint_speed = 0
            custom_config.lift_state = 0
            custom_config.expand_state = 0
            custom_config.arm_current_status = 0
            custom_config.hand_state = 0

            # 使用 robot_fps 计算上报周期（ms）
            period_ms = max(1, int(round(1000.0 / self.robot_fps)))
            udp_config = rm_realtime_push_config_t(
                period_ms,  # 上报周期(ms)
                True,       # 使能上报
                8089,       # 上报端口
                0,          # 力数据坐标系: 0-传感器坐标系
                "192.168.10.50",  # 目标IP
                custom_config
            )

            result = self.arm.rm_set_realtime_push(udp_config)
            if result == 0:
                logger.info(f"UDP配置设置成功, 周期: {period_ms} ms (~{1000.0/period_ms:.1f} Hz)")
            else:
                logger.error(f"UDP配置设置失败，错误码: {result}")
                return False
                
            # 注册回调函数 - 保持引用防止被垃圾回收
            self.callback_ptr = rm_realtime_arm_state_callback_ptr(self._robot_state_callback)
            self.arm.rm_realtime_arm_state_call_back(self.callback_ptr)
            logger.info("机器人状态回调注册成功")
            
            # 防止回调函数被垃圾回收
            gc.disable()  # 临时禁用垃圾回收
            time.sleep(1)  # 等待回调初始化
            gc.enable()   # 重新启用垃圾回收
            
            return True
            
        except Exception as e:
            logger.error(f"机器人初始化失败: {e}")
            return False
    
    def _robot_state_callback(self, data):
        """机器人状态回调函数 - 线程安全版本"""
        try:
            # 使用线程锁保护共享数据
            with self.state_lock:
                # 安全地提取力传感器数据
                if hasattr(data, 'force_sensor') and data.force_sensor is not None:
                    try:
                        force_data_dict = data.force_sensor.to_dict()
                        force_value = force_data_dict.get('zero_force', [0, 0, 0, 0, 0, 0])
                        
                        # 确保索引安全
                        if len(force_value) >= 3:
                            force_z = float(force_value[2])  # 显式转换为Python float
                        else:
                            force_z = 0.0
                            
                        # 检查力是否超过阈值
                        if abs(force_z) > self.force_threshold:
                            self.force_exceeded = True
                            if hasattr(self, 'logger'):
                                logger.warning(f"力传感器Z轴数据超过阈值: {force_z} N")
                        else:
                            self.force_exceeded = False
                            
                        # 更新当前机器人状态
                        self.current_robot_state = {
                            'timestamp': time.time(),
                            'force_sensor': force_data_dict,
                            'force_z': force_z,
                            'force_exceeded': self.force_exceeded
                        }
                        
                    except (AttributeError, IndexError, TypeError, ValueError) as e:
                        # 如果数据格式有问题，使用默认值
                        logger.error(f"解析力传感器数据失败: {e}")
                        self.current_robot_state = {
                            'timestamp': time.time(),
                            'force_sensor': {'zero_force': [0, 0, 0, 0, 0, 0]},
                            'force_z': 0.0,
                            'force_exceeded': False
                        }
                        self.force_exceeded = False
                else:
                    # 如果没有力传感器数据，使用默认值
                    self.current_robot_state = {
                        'timestamp': time.time(),
                        'force_sensor': {'zero_force': [0, 0, 0, 0, 0, 0]},
                        'force_z': 0.0,
                        'force_exceeded': False
                    }
                    self.force_exceeded = False
                    
        except Exception as e:
            # 捕获所有异常，避免回调函数崩溃
            logger.error(f"机器人状态回调严重错误: {e}")
            # 设置安全的默认状态
            try:
                with self.state_lock:
                    self.current_robot_state = {
                        'timestamp': time.time(),
                        'force_sensor': {'zero_force': [0, 0, 0, 0, 0, 0]},
                        'force_z': 0.0,
                        'force_exceeded': False
                    }
                    self.force_exceeded = False
            except:
                pass  # 如果连锁锁都获取不了，就忽略这次更新
    
    def initialize_cameras(self):
        """初始化双相机 - 改进版本"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) < 2:
                logger.error(f"找到 {len(devices)} 个RealSense设备，需要至少2个")
                return False
                
            logger.info(f"找到 {len(devices)} 个RealSense设备")
            
            # 初始化两个相机管道
            for i in range(2):
                pipeline = rs.pipeline()
                config = rs.config()
                
                device = devices[i]
                serial = device.get_info(rs.camera_info.serial_number)
                name = device.get_info(rs.camera_info.name)
                
                logger.info(f"初始化相机 {i+1}: {name} (序列号: {serial})")
                
                # 将每个传感器队列长度设为1，降低缓存延迟
                try:
                    for sensor in device.query_sensors():
                        try:
                            if sensor.supports(rs.option.frames_queue_size):
                                sensor.set_option(rs.option.frames_queue_size, 1)
                        except Exception as e:
                            logger.debug(f"设置 frames_queue_size 失败(相机{i+1}): {e}")
                except Exception as e:
                    logger.debug(f"查询传感器失败(相机{i+1}): {e}")
                
                # 配置相机流，使用 camera_fps
                config.enable_device(serial)
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, int(self.camera_fps))
                
                try:
                    # 启动管道
                    profile = pipeline.start(config)
                    
                    # 等待更多预热帧以稳定相机
                    logger.info(f"相机 {i+1} 启动成功，等待稳定...")
                    for _ in range(30):
                        try:
                            frames = pipeline.wait_for_frames(timeout_ms=3000)
                            if frames and frames.get_color_frame():
                                break
                        except Exception:
                            pass
                    
                    self.camera_pipelines.append(pipeline)
                    logger.info(f"相机 {i+1} 初始化完成")
                    
                except Exception as e:
                    logger.error(f"相机 {i+1} 启动失败: {e}")
                    return False
            
            logger.info("所有相机初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"相机初始化失败: {e}")
            return False
    
    def initialize_teleoperator(self):
        """初始化遥操作器"""
        try:
            self.shm_manager = SharedMemoryManager()
            self.shm_manager.start()
            
            self.teleoperator = SpaceMouseTeleoperator(
                shm_manager=self.shm_manager,
                position_scale=0.01,
                orientation_scale=0.05,
                pose_format="position_euler"
            )
            self.teleoperator.start()
            logger.info("遥操作器初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"遥操作器初始化失败: {e}")
            return False
    
    def set_current_to_torque_coeffs(self, coeffs):
        """设置关节电流到关节力矩的换算系数。
        coeffs: 长度为7的可迭代对象，对应每个关节的系数。
        """
        try:
            arr = np.asarray(coeffs, dtype=float).flatten()
            if arr.size < 7:
                tmp = np.ones(7, dtype=float)
                tmp[:arr.size] = arr
                arr = tmp
            elif arr.size > 7:
                arr = arr[:7]
            self.current_to_torque_coeffs = arr
            logger.info(f"已更新电流->力矩系数: {self.current_to_torque_coeffs}")
        except Exception as e:
            logger.error(f"设置力矩系数失败，保持默认: {e}")
    
    def get_robot_observation(self):
        """获取机器人观测数据 - 线程安全版本"""
        try:
            # 获取关节状态
            status_arm, state_data = self.arm.rm_get_current_arm_state()
            if status_arm != 0:
                logger.warning("获取机器人状态失败")
                return None
                
            joint_positions = np.array(state_data.get('joint', [0] * 7)[:7])
            end_effector_pose = np.array(state_data.get('pose', [0] * 6)[:6])
            
            # 线程安全地获取力传感器数据
            with self.state_lock:
                if self.current_robot_state is not None:
                    force_data = self.current_robot_state['force_sensor']
                    force_readings = np.array(force_data.get('zero_force', [0] * 6)[:6])
                    force_exceeded = self.current_robot_state['force_exceeded']
                else:
                    force_readings = np.zeros(6)
                    force_exceeded = False
            
            # 获取关节电流
            try:
                status_current, current_data = self.arm.rm_get_current_joint_current()
                joint_currents = np.array(current_data[:7] if status_current == 0 else [0] * 7, dtype=float)
            except Exception as e:
                logger.warning(f"获取关节电流失败: {e}")
                joint_currents = np.zeros(7, dtype=float)
            
            # 新增：根据电流与系数计算关节力矩
            try:
                coeffs = self.current_to_torque_coeffs
                if coeffs.shape[0] != 7:
                    tmp = np.ones(7, dtype=float)
                    n = min(7, coeffs.shape[0]) if hasattr(coeffs, 'shape') else 7
                    tmp[:n] = np.asarray(coeffs, dtype=float).flatten()[:n]
                    coeffs = tmp
                joint_torques = joint_currents * coeffs
            except Exception as e:
                logger.warning(f"计算关节力矩失败，置零: {e}")
                joint_torques = np.zeros(7, dtype=float)
            
            return {
                'joint_positions': joint_positions,
                'end_effector_pose': end_effector_pose,
                'force_readings': force_readings,
                'joint_currents': joint_currents,
                'joint_torques': joint_torques,
                'force_exceeded': force_exceeded
            }
            
        except Exception as e:
            logger.error(f"获取机器人观测失败: {e}")
            return None
    
    def run_thread_cam(self, which_cam: int):
        """参照 run_control.py 的相机线程：持续从对应 pipeline 取帧，更新 img_list 与 JPEG 缓冲"""
        assert which_cam in (0, 1)
        logger.info(f"相机取帧线程{which_cam}启动")
        frame_count = 0
        # 避免额外开销：不在采集线程做 JPEG 编码，仅更新 img_list
        while not self.stop_flag and not self.camera_thread_stop:
            try:
                pipeline = self.camera_pipelines[which_cam] if which_cam < len(self.camera_pipelines) else None
                if pipeline is None:
                    time.sleep(0.001)
                    continue

                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame() if frames else None
                if color_frame:
                    image_cam = np.asanyarray(color_frame.get_data())  # BGR
                    self.img_list[which_cam] = image_cam.copy()  # 使用copy确保线程安全
                    frame_count += 1
                    if frame_count % 100 == 0:
                        logger.debug(f"相机{which_cam}已获取{frame_count}帧")
                else:
                    # 占位
                    self.img_list[which_cam][:] = 0
            except Exception as e:
                logger.warning(f"相机线程{which_cam} 获取帧失败: {e}")
                time.sleep(0.001)
        logger.info(f"相机取帧线程{which_cam}结束")

    def get_camera_images(self):
        """获取相机图像（改为从线程缓冲读取，参照 run_control.py）"""
        try:
            # 直接从共享 img_list 读取拷贝
            img1 = self.img_list[0].copy() if len(self.img_list) > 0 else np.zeros((480, 640, 3), dtype=np.uint8)
            img2 = self.img_list[1].copy() if len(self.img_list) > 1 else np.zeros((480, 640, 3), dtype=np.uint8)
            self.latest_images = [img1, img2]
            return [img1, img2]
        except Exception as e:
            logger.error(f"获取相机图像失败: {e}")
            return [np.zeros((480, 640, 3), dtype=np.uint8)] * 2
    
    def get_teleop_action(self, current_pose):
        """获取遥操作动作"""
        try:
            if self.teleoperator is None:
                return current_pose, False
                
            # 检查按钮状态
            left_btn, right_btn = self.teleoperator.get_button_states()
            
            # 如果力超过阈值，使用安全位置
            if self.force_exceeded:
                return self.last_safe_pose, True
                
            # 获取遥操作输入
            new_pose = self.teleoperator.teleop_space(current_pose)
            self.last_safe_pose = new_pose.copy()
            
            return new_pose, left_btn or right_btn
            
        except Exception as e:
            logger.error(f"获取遥操作动作失败: {e}")
            return current_pose, False
    
    def send_robot_action(self, pose):
        """非阻塞：将命令放入下发队列，由单独线程实际发送。返回是否成功入队。"""
        try:
            try:
                self.command_queue.put_nowait(pose.copy())
                return True
            except queue.Full:
                logger.warning("命令队列已满，发送命令被丢弃")
                return False
        except Exception as e:
            logger.error(f"入队发送命令失败: {e}")
            return False

    def _command_sender_thread(self):
        """独立命令下发线程：从 command_queue 取出位姿并调用 rm_movep_canfd，不阻塞 teleop 线程。"""
        logger.info("命令下发线程启动")
        while not self.command_thread_stop and not self.stop_flag:
            try:
                try:
                    pose = self.command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                t0 = time.time()
                success = False
                try:
                    ret = self.arm.rm_movep_canfd(pose, True, 1, 30)
                    success = (ret == 0)
                    if not success:
                        logger.warning(f"命令下发失败，错误码: {ret}")
                except Exception as e:
                    logger.error(f"命令下发调用失败: {e}")
                    success = False
                duration = (time.time() - t0)

                # 更新状态
                with self.command_lock:
                    self.command_status['last_send_time'] = time.time()
                    self.command_status['last_duration'] = duration
                    self.command_status['last_success'] = bool(success)

                # 小延迟避免紧循环
                if duration < 0.001:
                    time.sleep(0.001)

            except Exception as e:
                logger.error(f"命令下发线程错误: {e}")
                time.sleep(0.01)

        logger.info("命令下发线程结束")
    
    def robot_data_thread(self):
        """机器人数据采集线程"""
        logger.info("机器人数据采集线程启动")
        next_t = time.time()
        data_sent = 0
        while not self.robot_thread_stop and not self.stop_flag:
            try:
                timestamp = time.time()
                robot_obs = self.get_robot_observation()
                if robot_obs is not None:
                    robot_data = {'timestamp': timestamp, 'type': 'robot', 'data': robot_obs}
                    # 更新最新样本缓存，供同步线程非阻塞读取（加锁保护）
                    try:
                        with self.latest_sample_lock:
                            self.latest_robot_sample = robot_data
                    except Exception:
                        # 如果锁出错，仍继续流程
                        self.latest_robot_sample = robot_data
                    try:
                        self.robot_data_queue.put_nowait(robot_data)
                        data_sent += 1
                        if data_sent % 200 == 0:
                            logger.info(f"机器人数据已发送{data_sent}条到队列")
                    except queue.Full:
                        try:
                            self.robot_data_queue.get_nowait(); self.robot_data_queue.put_nowait(robot_data)
                            data_sent += 1
                        except queue.Empty:
                            pass
                    # 新增：按读取速率独立存储机器人数据（JSONL，含时间戳）
                    if self.recording and self.session_dir is not None and self.robot_log_path is not None:
                        try:
                            payload = {
                                't': timestamp,
                                'q': robot_obs['joint_positions'].tolist(),
                                'pose': robot_obs['end_effector_pose'].tolist(),
                                'force': robot_obs['force_readings'].tolist(),
                                'curr': robot_obs['joint_currents'].tolist(),
                                'tau': robot_obs.get('joint_torques', np.zeros(7, dtype=float)).tolist(),
                                'force_exceeded': bool(robot_obs.get('force_exceeded', False))
                            }
                            self.save_queue.put_nowait(('jsonl', self.robot_log_path, payload))
                        except Exception as e:
                            logger.debug(f"机器人日志入队失败: {e}")
                # 基于节拍的调度（使用 dt_robot）
                next_t += self.dt_robot
                sleep_t = max(0, next_t - time.time())
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = time.time()
            except Exception as e:
                logger.error(f"机器人数据采集线程错误: {e}")
                time.sleep(0.002)
        logger.info("机器人数据采集线程结束")
    
    def camera_data_thread(self):
        """相机数据采集线程 - 仅打包当前最新帧，避免大拷贝与频繁分配"""
        logger.info("相机数据采集线程启动")
        consecutive_failures = 0
        next_t = time.time()
        frame_sent = 0
        while not self.camera_thread_stop and not self.stop_flag:
            try:
                now = time.time()
                # 从共享 img_list 读取引用并复制一次（尽量减少复制）
                img1 = self.img_list[0]
                img2 = self.img_list[1] if len(self.img_list) > 1 else None
                valid = (img1 is not None and img1.size > 0) and (img2 is not None and img2.size > 0)
                if valid:
                    consecutive_failures = 0
                    img1_copy = img1.copy()
                    img2_copy = img2.copy()
                    
                    # 更新 latest_images 用于视频显示
                    with self.data_lock:
                        self.latest_images = [img1_copy, img2_copy]
                    
                    camera_data = {
                        'timestamp': now,
                        'type': 'camera',
                        'data': {
                            'camera_1': img1_copy,
                            'camera_2': img2_copy
                        }
                    }
                    try:
                        self.camera_data_queue.put_nowait(camera_data)
                        frame_sent += 1
                        if frame_sent % 100 == 0:
                            logger.info(f"相机数据已发送{frame_sent}帧到队列")
                    except queue.Full:
                        try:
                            self.camera_data_queue.get_nowait()
                            self.camera_data_queue.put_nowait(camera_data)
                        except queue.Empty:
                            pass
                else:
                    consecutive_failures += 1
                    if consecutive_failures % 30 == 0:
                        logger.warning(f"相机图像无效，连续失败: {consecutive_failures}, img1 size: {img1.size if img1 is not None else 'None'}, img2 size: {img2.size if img2 is not None else 'None'}")
                # 基于节拍的调度（使用 dt_cam）
                next_t += self.dt_cam
                sleep_t = max(0, next_t - time.time())
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = time.time()
            except Exception as e:
                logger.error(f"相机数据采集线程错误: {e}")
                time.sleep(0.005)
        logger.info("相机数据采集线程结束")
    
    def teleop_data_thread(self):
        """遥操作数据采集线程"""
        logger.info("遥操作数据采集线程启动")
        
        current_pose = self.initial_pose.copy()
        next_t = time.time()
        data_sent = 0
        
        while not self.teleop_thread_stop and not self.stop_flag:
            try:
                timestamp = time.time()
                
                action_pose, button_pressed = self.get_teleop_action(current_pose)

                # 更新最新遥操作样本缓存
                teleop_data = {
                    'timestamp': timestamp,
                    'type': 'teleop',
                    'data': {
                        'target_pose': action_pose,
                        'button_pressed': button_pressed,
                        'command_success': None
                    }
                }

                # 录制开关逻辑（防抖）：按钮从未按下->按下，且距离上次切换超过 cooldown
                if button_pressed and not self.last_button_state:
                    if (timestamp - self._last_toggle_ts) > self.toggle_cooldown:
                        self.recording = not self.recording
                        self._last_toggle_ts = timestamp
                        if self.recording:
                            self.start_new_session()
                            logger.info("开始录制，会话目录: %s", self.session_dir)
                        else:
                            logger.info("停止录制")
                self.last_button_state = button_pressed

                action_pose[3] = np.pi
                action_pose[4] = 0       
                # 非阻塞入队发送，由发送线程实际下发
                queued = self.send_robot_action(action_pose)
                teleop_data['data']['command_success'] = queued
                current_pose = action_pose.copy()
                
                try:
                    self.teleop_data_queue.put_nowait(teleop_data)
                    data_sent += 1
                    if data_sent % 200 == 0:
                        logger.info(f"遥操作数据已发送{data_sent}条到队列")
                except queue.Full:
                    try:
                        self.teleop_data_queue.get_nowait()
                        self.teleop_data_queue.put_nowait(teleop_data)
                        data_sent += 1
                    except queue.Empty:
                        pass

                # 更新缓存（加锁保护）
                try:
                    with self.latest_sample_lock:
                        self.latest_teleop_sample = teleop_data
                except Exception:
                    self.latest_teleop_sample = teleop_data

                # 新增：按读取速率独立存储遥操作数据（JSONL，含时间戳）
                if self.recording and self.session_dir is not None and self.teleop_log_path is not None:
                    try:
                        payload = {
                            't': timestamp,
                            'target_pose': action_pose.tolist(),
                            'button_pressed': bool(button_pressed),
                            'command_success': bool(queued)
                        }
                        self.save_queue.put_nowait(('jsonl', self.teleop_log_path, payload))
                    except Exception as e:
                        logger.debug(f"遥操作日志入队失败: {e}")
                
                # 基于节拍的调度（使用 dt_teleop）
                next_t += self.dt_teleop
                sleep_t = max(0, next_t - time.time())
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = time.time()
                
            except Exception as e:
                logger.error(f"遥操作数据采集线程错误: {e}")
                time.sleep(0.05)
        
        logger.info("遥操作数据采集线程结束")

    def data_sync_thread(self):
        """数据同步线程 - 以相机帧为锚点，将最近(<=t_cam)或最接近的机器人/遥操作样本对齐到该相机帧"""
        logger.info("数据同步线程启动")
        # 自适应同步窗口：根据各线程采样频率动态调整
        # 策略：窗口应该能容纳足够多的周期，确保数据有时间进入队列
        
        # 为相机帧率变化预留足够的同步窗口
        # 当相机帧率提高时，需要更宽松的窗口来容纳其他线程的延迟
        cam_window_multiplier = 20.0  # 增加到20倍，更好地处理高帧率场景
        safety_margin = 0.1  # 增加到100ms安全余量
        
        # 为机器人和遥操作分别计算窗口
        # 使用各自的周期时间，而不是相机周期
        window_r = max(
            cam_window_multiplier * self.dt_cam,  # 基于相机周期
            50 * self.dt_robot  # 基于机器人周期，增加到50倍
        ) + safety_margin
        
        window_t = max(
            cam_window_multiplier * self.dt_cam,  # 基于相机周期
            50 * self.dt_teleop  # 基于遥操作周期，增加到50倍
        ) + safety_margin
        
        logger.info(f"同步窗口配置: 相机周期={self.dt_cam*1000:.1f}ms ({self.camera_fps}Hz), "
                   f"机器人周期={self.dt_robot*1000:.1f}ms ({self.robot_fps}Hz), "
                   f"遥操作周期={self.dt_teleop*1000:.1f}ms ({self.teleop_fps}Hz) | "
                   f"窗口: 机器人={window_r*1000:.1f}ms, 遥操作={window_t*1000:.1f}ms "
                   f"(倍数={cam_window_multiplier}x, 安全余量={safety_margin*1000:.0f}ms)")
        last_robot = None
        last_cam = None
        last_tel = None
        # 我们不再频繁抽干 robot/teleop 队列，而是使用最新缓存样本
        last_cam = None
        last_robot_leq = None
        last_tel_leq = None
        next_t = time.time()
        sync_count = 0
        iteration = 0
        while not self.stop_flag:
            try:
                iteration += 1
                cam_list = []
                # 仅抽干相机队列以获取最新帧；不要抽干 robot/teleop 队列
                while True:
                    try:
                        last_cam = self.camera_data_queue.get_nowait()
                        cam_list.append(last_cam)
                    except queue.Empty:
                        break

                # 每100次迭代输出一次队列状态（注意 robot/teleop 使用缓存长度为1）
                if iteration % 100 == 0:
                    logger.info(f"同步线程迭代{iteration}: 获取数据 - 机器人(缓存):{1 if self.latest_robot_sample is not None else 0}, 相机:{len(cam_list)}, 遥操作(缓存):{1 if self.latest_teleop_sample is not None else 0}, 已同步:{sync_count}帧")
                
                # 诊断：如果某个队列长时间为空，输出警告
                if iteration % 300 == 0:
                    if len(cam_list) == 0:
                        logger.warning("相机数据队列持续为空！检查camera_data_thread是否正常运行")

                # 仅在获得新的相机帧时进行一次对齐
                cam_sample = cam_list[-1] if cam_list else None
                if cam_sample is not None:
                    t_cam = cam_sample['timestamp']
                    # 使用缓存的最新样本作为候选（加锁读取）
                    with self.latest_sample_lock:
                        robot_candidate = self.latest_robot_sample
                        teleop_candidate = self.latest_teleop_sample

                    # 如果缓存为空，尝试从队列中取一个样本作为备选（但不要长期抽干）
                    if robot_candidate is None:
                        try:
                            robot_candidate = self.robot_data_queue.get_nowait()
                        except queue.Empty:
                            robot_candidate = None
                    if teleop_candidate is None:
                        try:
                            teleop_candidate = self.teleop_data_queue.get_nowait()
                        except queue.Empty:
                            teleop_candidate = None

                    if robot_candidate and teleop_candidate:
                        dt_r = abs(robot_candidate['timestamp'] - t_cam)
                        dt_t = abs(teleop_candidate['timestamp'] - t_cam)
                        # 放宽窗口：分别对机器人和遥操作使用更宽松阈值
                        if dt_r <= window_r and dt_t <= window_t:
                            sync_data = {
                                'timestamp': t_cam,
                                'robot_data': robot_candidate['data'],
                                'camera_data': cam_sample['data'],
                                'teleop_data': teleop_candidate['data']
                            }
                            try:
                                self.sync_data_queue.put_nowait(sync_data)
                                sync_count += 1
                                self.sync_stats['total'] += 1
                                self.sync_stats['matched'] += 1
                                if sync_count % 50 == 0:
                                    logger.info(f"已成功同步{sync_count}帧数据")
                            except queue.Full:
                                try:
                                    self.sync_data_queue.get_nowait(); self.sync_data_queue.put_nowait(sync_data)
                                    sync_count += 1
                                    self.sync_stats['total'] += 1
                                    self.sync_stats['matched'] += 1
                                except queue.Empty:
                                    pass
                        else:
                            # 记录未对齐原因（每50次输出一次，避免刷屏）
                            self.sync_stats['total'] += 1
                            self.sync_stats['mismatch'] += 1
                            if iteration % 50 == 0:
                                logger.warning(f"时间窗口不匹配 [相机{self.camera_fps}Hz]: "
                                             f"dt_robot={dt_r*1000:.1f}ms (阈值{window_r*1000:.1f}ms, "
                                             f"超出{(dt_r-window_r)*1000:.1f}ms), "
                                             f"dt_teleop={dt_t*1000:.1f}ms (阈值{window_t*1000:.1f}ms, "
                                             f"超出{(dt_t-window_t)*1000:.1f}ms) | "
                                             f"建议: 增加同步窗口或降低采样频率")
                    else:
                        # 记录缺失数据（每50次输出一次）
                        self.sync_stats['total'] += 1
                        self.sync_stats['mismatch'] += 1
                        if iteration % 50 == 0:
                            missing = []
                            if not robot_candidate:
                                missing.append("机器人")
                            if not teleop_candidate:
                                missing.append("遥操作")
                            logger.warning(f"缺少数据候选: {', '.join(missing)}")
                else:
                    # 没有新的相机帧，记录一次（避免刷屏）
                    if iteration % 100 == 0 and len(cam_list) == 0:
                        logger.debug("等待新的相机帧...")
                        
                # 节拍调度（以相机帧率为基准）
                next_t += self.dt_sync
                sleep_t = max(0, next_t - time.time())
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_t = time.time()
                    # 即使没有睡眠，也给CPU一点喘息时间
                    time.sleep(0.001)
            except Exception as e:
                logger.error(f"数据同步线程错误: {e}")
                time.sleep(0.005)
        logger.info("数据同步线程结束")
    
    def collect_single_step(self, step_idx):
        """采集单步数据 - 多线程版本"""
        try:
            # 从同步队列获取数据
            sync_data = self.sync_data_queue.get(timeout=1.0)  # 1秒超时
            
            timestamp = sync_data['timestamp']
            robot_obs = sync_data['robot_data']
            camera_data = sync_data['camera_data']
            teleop_data = sync_data['teleop_data']
            
            # 更新最新机器人数据用于显示
            with self.data_lock:
                self.latest_robot_data = {
                    'step': step_idx,
                    'joint_positions': robot_obs['joint_positions'],
                    'end_effector_pose': robot_obs['end_effector_pose'],
                    'force_readings': robot_obs['force_readings'],
                    'force_exceeded': robot_obs['force_exceeded'],
                    'button_pressed': teleop_data['button_pressed'],
                    'command_success': teleop_data['command_success'],
                    'timestamp': timestamp
                }
                
                # 更新最新图像
                self.latest_images = [camera_data['camera_1'], camera_data['camera_2']]
            
            # 构造数据包
            data_point = {
                'timestamp': timestamp,
                'step': step_idx,
                'observation': {
                    'joint_positions': robot_obs['joint_positions'],
                    'end_effector_pose': robot_obs['end_effector_pose'],
                    'force_readings': robot_obs['force_readings'],
                    'joint_currents': robot_obs['joint_currents'],
                    'joint_torques': robot_obs.get('joint_torques', np.zeros(7, dtype=float)),
                    'camera_1': camera_data['camera_1'],
                    'camera_2': camera_data['camera_2']
                },
                'action': {
                    'target_pose': teleop_data['target_pose'],
                    'button_pressed': teleop_data['button_pressed']
                },
                'info': {
                    'force_exceeded': robot_obs['force_exceeded'],
                    'command_success': teleop_data['command_success']
                }
            }
            
            # 若录制开关开启，按 run_control 方式保存到临时结构
            if self.recording and self.session_dir is not None:
                self.save_current_step(self.current_idx, data_point)
                self.current_idx += 1
            
            return data_point
            
        except queue.Empty:
            # 节流告警，避免刷屏
            now_ts = time.time()
            if (now_ts - getattr(self, '_last_sync_empty_warn', 0)) > 2.0:
                logger.warning("同步数据队列为空，跳过此步")
                self._last_sync_empty_warn = now_ts
            return None
        except Exception as e:
            logger.error(f"采集单步数据失败: {e}")
            return None
    
    def _mk_dir(self, path: str) -> bool:
        """仿 run_control 的 mk_dir: 目录不存在则创建并返回 True，否则 False"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            return True
        return False
    
    def start_new_session(self):
        """开始新的录制会话，创建 leftImg/rightImg/observation 目录"""
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        self.session_dir = os.path.join(self.collect_base_dir, ts)
        self.left_dir = os.path.join(self.session_dir, "leftImg")
        self.right_dir = os.path.join(self.session_dir, "rightImg")
        self.obs_dir = os.path.join(self.session_dir, "observation")
        created_left = self._mk_dir(self.left_dir)
        self._mk_dir(self.right_dir)
        self._mk_dir(self.obs_dir)
        # 新增：独立流日志文件路径
        self.robot_log_path = os.path.join(self.session_dir, "robot_log.jsonl")
        self.teleop_log_path = os.path.join(self.session_dir, "teleop_log.jsonl")
        self.camera_ts_path = os.path.join(self.session_dir, "camera_timestamps.csv")
        # 初始化文件（相机时间戳写入表头）
        try:
            # 确保会话目录存在
            os.makedirs(self.session_dir, exist_ok=True)
            # 写入表头，若文件不存在
            if not os.path.exists(self.camera_ts_path):
                with open(self.camera_ts_path, 'w', encoding='utf-8') as f:
                    f.write('idx,timestamp\n')
            # 确保JSONL文件创建
            if not os.path.exists(self.robot_log_path):
                open(self.robot_log_path, 'a', encoding='utf-8').close()
            if not os.path.exists(self.teleop_log_path):
                open(self.teleop_log_path, 'a', encoding='utf-8').close()
        except Exception as e:
            logger.warning(f"初始化会话日志文件失败: {e}")
        self.current_idx = 0 if created_left else self.current_idx
        logger.info(f"新录制会话开始: {self.session_dir}")
    
    def save_current_step(self, idx: int, data_point: dict):
        """按 run_control 风格保存单步数据：两路图片 + 观测/动作到 observation（异步写盘）"""
        try:
            # 降低写盘开销：JPEG质量=50，与 run_control 保持一致
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            img1 = data_point['observation']['camera_1']
            img2 = data_point['observation']['camera_2']
            # 将写盘任务放入队列，由独立线程处理
            self.save_queue.put_nowait(('image', os.path.join(self.left_dir, f"{idx}.jpg"), img1.copy(), encode_param))
            self.save_queue.put_nowait(('image', os.path.join(self.right_dir, f"{idx}.jpg"), img2.copy(), encode_param))
            # 新增：记录相机帧对应时间戳，便于与高频数据按时间对齐
            if self.camera_ts_path:
                cam_ts = data_point.get('timestamp', time.time())
                self.save_queue.put_nowait(('text', self.camera_ts_path, f"{idx},{cam_ts}\n"))
            obs = {
                'timestamp': data_point['timestamp'],
                'joint_positions': data_point['observation']['joint_positions'],
                'end_effector_pose': data_point['observation']['end_effector_pose'],
                'force_readings': data_point['observation']['force_readings'],
                'joint_currents': data_point['observation']['joint_currents'],
                'joint_torques': data_point['observation']['joint_torques'],
                'action_target_pose': data_point['action']['target_pose'],
                'button_pressed': data_point['action']['button_pressed'],
                'command_success': data_point['info']['command_success'],
                'force_exceeded': data_point['info']['force_exceeded'],
            }
            self.save_queue.put_nowait(('npz', os.path.join(self.obs_dir, f"{idx}.npz"), obs))
        except Exception as e:
            logger.error(f"保存单步数据入队失败(idx={idx}): {e}")
    
    def _writer_loop(self):
        """异步写盘线程：统一处理图片和npz写入，减小主路径IO阻塞"""
        logger.info("写盘线程启动")
        while not self.writer_thread_stop or not self.save_queue.empty():
            try:
                item = self.save_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                kind = item[0]
                if kind == 'image':
                    _, path, img, enc = item
                    cv2.imwrite(path, img, enc)
                elif kind == 'npz':
                    _, path, obs = item
                    np.savez(path, **obs)
                # 新增：JSONL 逐行追加
                elif kind == 'jsonl':
                    _, path, obj = item
                    with open(path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
                # 新增：文本逐行追加（用于相机时间戳CSV）
                elif kind == 'text':
                    _, path, line = item
                    with open(path, 'a', encoding='utf-8') as f:
                        f.write(line)
            except Exception as e:
                logger.error(f"写盘失败: {e}")
        logger.info("写盘线程结束")
    
    def create_video_display(self):
        """创建用于显示的图像（仅相机画面）- 多线程安全版本"""
        with self.data_lock:
            if not self.latest_images:
                return np.zeros((480, 1280, 3), dtype=np.uint8)
            
            # 获取相机图像的副本
            img1 = self.latest_images[0].copy()
            img2 = self.latest_images[1].copy()
        
        # 缩放图像
        if self.video_scale != 1.0:
            new_width = int(img1.shape[1] * self.video_scale)
            new_height = int(img1.shape[0] * self.video_scale)
            img1 = cv2.resize(img1, (new_width, new_height))
            img2 = cv2.resize(img2, (new_width, new_height))
        
        # 水平拼接两个相机画面
        combined_img = np.hstack([img1, img2])
        
        # 添加分割线
        mid_x = combined_img.shape[1] // 2
        cv2.line(combined_img, (mid_x, 0), (mid_x, combined_img.shape[0]), (0, 255, 0), 2)
        
        # 添加相机标签
        cv2.putText(combined_img, "Camera 1", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(combined_img, "Camera 2", (mid_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 添加多线程状态信息
        height, width = combined_img.shape[:2]
        
        # 队列状态
        queue_info = f"Queues - R:{self.robot_data_queue.qsize()} C:{self.camera_data_queue.qsize()} T:{self.teleop_data_queue.qsize()} S:{self.sync_data_queue.qsize()}"
        cv2.putText(combined_img, queue_info, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
        
        # Step信息
        with self.data_lock:
            if self.latest_robot_data:
                step_info = f"Step: {self.latest_robot_data['step']}"
                cv2.putText(combined_img, step_info, (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)
    
        # 添加控制说明
        cv2.putText(combined_img, "Controls: 'q'-Quit  'p'-Pause  's'-Screenshot  'r'-Reset", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        
        return combined_img
    
    def video_display_loop(self):
        """实时视频显示循环（基于 show_canvas 并排显示）"""
        logger.info("启动实时视频显示")
        
        cv2.namedWindow('Robot Data Collection - Camera View', cv2.WINDOW_AUTOSIZE)
        
        frame_displayed = 0
        try:
            while not self.stop_flag:
                if not self.video_paused:
                    # 拿到相机图像副本
                    with self.data_lock:
                        img1 = self.latest_images[0].copy() if len(self.latest_images) > 0 else np.zeros((480, 640, 3), dtype=np.uint8)
                        img2 = self.latest_images[1].copy() if len(self.latest_images) > 1 else np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    # 检查图像是否有效（不全是黑色）
                    img1_valid = img1 is not None and img1.size > 0 and np.any(img1 > 0)
                    img2_valid = img2 is not None and img2.size > 0 and np.any(img2 > 0)
                    
                    if not img1_valid or not img2_valid:
                        if frame_displayed % 30 == 0:
                            logger.debug(f"等待相机图像... img1_valid={img1_valid}, img2_valid={img2_valid}")
                        frame_displayed += 1
                        time.sleep(0.033)  # 30fps
                        continue
                    
                    # 可选缩放
                    if self.video_scale != 1.0:
                        new_w = int(img1.shape[1] * self.video_scale)
                        new_h = int(img1.shape[0] * self.video_scale)
                        img1 = cv2.resize(img1, (new_w, new_h))
                        img2 = cv2.resize(img2, (new_w, new_h))
                        canvas_h, canvas_w = new_h, new_w * 2
                        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
                        canvas[:, :new_w] = img1
                        canvas[:, new_w:new_w*2] = img2
                    else:
                        # 直接写入 show_canvas
                        self.show_canvas[:, :640] = np.asarray(img1, dtype=np.uint8)
                        self.show_canvas[:, 640:1280] = np.asarray(img2, dtype=np.uint8)
                        canvas = self.show_canvas
                    
                    # 叠加信息文本
                    mid_x = canvas.shape[1] // 2
                    cv2.line(canvas, (mid_x, 0), (mid_x, canvas.shape[0]), (0, 255, 0), 2)
                    cv2.putText(canvas, "Camera 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(canvas, "Camera 2", (mid_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    queue_info = f"Queues - R:{self.robot_data_queue.qsize()} C:{self.camera_data_queue.qsize()} T:{self.teleop_data_queue.qsize()} S:{self.sync_data_queue.qsize()}"
                    cv2.putText(canvas, queue_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    with self.data_lock:
                        if self.latest_robot_data:
                            step = self.latest_robot_data['step']
                            cv2.putText(canvas, f"Step:{step}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
                    cv2.putText(canvas, "Controls: 'q'-Quit  'p'-Pause  's'-Screenshot  'r'-Reset", (10, canvas.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    # 显示
                    cv2.imshow('Robot Data Collection - Camera View', canvas)
                    self.frame_count += 1
                    frame_displayed += 1
                    
                    # 每100帧输出一次日志
                    if frame_displayed % 100 == 0:
                        logger.info(f"视频显示已渲染{frame_displayed}帧")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("用户请求退出")
                    self.stop_flag = True
                    break
                elif key == ord('p'):
                    self.video_paused = not self.video_paused
                    logger.info(f"视频显示{'暂停' if self.video_paused else '继续'}")
                elif key == ord('s'):
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    screenshot_path = os.path.join(self.data_dir, f"screenshot_{timestamp}.png")
                    with self.data_lock:
                        img1 = self.latest_images[0].copy() if len(self.latest_images) > 0 else np.zeros((480, 640, 3), dtype=np.uint8)
                        img2 = self.latest_images[1].copy() if len(self.latest_images) > 1 else np.zeros((480, 640, 3), dtype=np.uint8)
                    self.show_canvas[:, :640] = np.asarray(img1, dtype=np.uint8)
                    self.show_canvas[:, 640:1280] = np.asarray(img2, dtype=np.uint8)
                    cv2.imwrite(screenshot_path, self.show_canvas)
                    logger.info(f"截图已保存到: {screenshot_path}")
                elif key == ord('r'):
                    logger.info("用户请求重置episode")
                
                time.sleep(1.0 / 60)
                
        except Exception as e:
            logger.error(f"视频显示循环错误: {e}")
        finally:
            cv2.destroyAllWindows()
            logger.info("视频显示已关闭")
    
    def collection_loop(self):
        """主数据采集循环 - 多线程版本"""
        logger.info(f"开始数据采集，Episode长度: {self.episode_length}, FPS: {self.fps}")
        
        step_count = 0
        
        start_time = time.time()
        last_progress_time = start_time
        
        # 新增：滑动窗口FPS计算（最近N帧的平均速度）
        fps_window_size = 60  # 使用最近60帧计算FPS
        fps_window_times = []  # 存储最近N帧的时间戳
        
        try:
            while not self.stop_flag:
                # 采集单步数据（从同步队列获取）
                data_point = self.collect_single_step(step_count)
                
                if data_point is not None:
                    step_count += 1
                    
                    # 更新滑动窗口
                    current_time = time.time()
                    fps_window_times.append(current_time)
                    if len(fps_window_times) > fps_window_size:
                        fps_window_times.pop(0)  # 移除最老的时间戳
                    
                    # 显示进度（每30步或每5秒显示一次）q
                    if step_count % 30 == 0 or (current_time - last_progress_time) > 5:
                        # 计算两种FPS：累积平均 + 实时滑动窗口
                        elapsed_total = current_time - start_time
                        fps_avg = step_count / elapsed_total if elapsed_total > 0 else 0
                        
                        # 滑动窗口FPS（更能反映当前性能）
                        if len(fps_window_times) >= 2:
                            window_duration = fps_window_times[-1] - fps_window_times[0]
                            fps_current = (len(fps_window_times) - 1) / window_duration if window_duration > 0 else 0
                        else:
                            fps_current = 0
                        
                        queue_sizes = {
                            'robot': self.robot_data_queue.qsize(),
                            'camera': self.camera_data_queue.qsize(),
                            'teleop': self.teleop_data_queue.qsize(),
                            'sync': self.sync_data_queue.qsize()
                        }
                        
                        # 显示两种FPS：当前实时FPS + 累积平均FPS
                        logger.info(f"Step {step_count}/{self.episode_length}, "
                                  f"FPS: {fps_current:.1f} (avg: {fps_avg:.1f}), "
                                  f"队列: {queue_sizes}")
                        last_progress_time = current_time
                
                # 不需要手动控制频率，因为数据来自队列
                # 但添加一个小的延迟防止CPU过载
                time.sleep(0.001)  # 1ms延迟
                    
        except KeyboardInterrupt:
            logger.info("收到中断信号，停止采集...")
        except Exception as e:
            logger.error(f"采集循环错误: {e}")
        finally:            
            logger.info("主采集循环结束")
    
    def start_collection(self):
        """开始数据采集 - 多线程版本"""
        if self.is_collecting:
            logger.warning("数据采集已在运行中")
            return False
        
        # 初始化所有组件
        if not self.initialize_robot():
            return False
        if not self.initialize_cameras():
            return False
        if not self.initialize_teleoperator():
            return False
        
        time.sleep(2)
        
        self.is_collecting = True
        self.stop_flag = False
        
        self.robot_thread_stop = False
        self.camera_thread_stop = False
        self.teleop_thread_stop = False
        
        logger.info("启动多线程数据采集...")
        
        # 相机取帧线程（与 run_control.py 对齐：每路相机一个线程，持续更新缓冲）
        for cam_idx in range(2):
            t = threading.Thread(target=self.run_thread_cam, args=(cam_idx,), name=f"CameraReader-{cam_idx}")
            t.daemon = True
            t.start()
            self.threads.append(t)
        
        # 启动异步写盘线程
        writer_thread = threading.Thread(target=self._writer_loop, name="WriterThread")
        writer_thread.daemon = True
        writer_thread.start()
        self.threads.append(writer_thread)
        
        # 机器人数据采集线程
        robot_thread = threading.Thread(target=self.robot_data_thread, name="RobotDataThread")
        robot_thread.daemon = True
        robot_thread.start()
        self.threads.append(robot_thread)
        
        # 相机数据打包线程（从 img_list 读，送入同步队列）
        camera_thread = threading.Thread(target=self.camera_data_thread, name="CameraDataThread")
        camera_thread.daemon = True
        camera_thread.start()
        self.threads.append(camera_thread)
        
        # 遥操作线程
        teleop_thread = threading.Thread(target=self.teleop_data_thread, name="TeleopDataThread")
        teleop_thread.daemon = True
        teleop_thread.start()
        self.threads.append(teleop_thread)
        
        # 同步线程
        sync_thread = threading.Thread(target=self.data_sync_thread, name="DataSyncThread")
        sync_thread.daemon = True
        sync_thread.start()
        self.threads.append(sync_thread)
        
        # 主采集线程
        collection_thread = threading.Thread(target=self.collection_loop, name="CollectionThread")
        collection_thread.daemon = True
        collection_thread.start()
        self.threads.append(collection_thread)
        
        # 实时显示
        if self.show_video:
            video_thread = threading.Thread(target=self.video_display_loop, name="VideoDisplayThread")
            video_thread.daemon = True
            video_thread.start()
            self.threads.append(video_thread)
            logger.info("实时视频显示已启动")
        
        logger.info(f"数据采集已启动，共启动 {len(self.threads)} 个线程")
        for thread in self.threads:
            logger.info(f"线程 {thread.name} 已启动")
        
        # 等待相机线程开始采集
        logger.info("等待相机线程启动...")
        time.sleep(2.0)
        
        # 检查相机是否开始采集
        with self.data_lock:
            img1_ready = self.latest_images[0] is not None and np.any(self.latest_images[0] > 0)
            img2_ready = self.latest_images[1] is not None and np.any(self.latest_images[1] > 0) if len(self.latest_images) > 1 else False
        
        if img1_ready and img2_ready:
            logger.info("相机已就绪，开始数据采集")
        else:
            logger.warning(f"相机可能未就绪: camera1={img1_ready}, camera2={img2_ready}")
        
        # 启动命令发送线程
        self.command_thread_stop = False
        command_thread = threading.Thread(target=self._command_sender_thread, name="CommandSenderThread")
        command_thread.daemon = True
        command_thread.start()
        self.command_thread = command_thread
        self.threads.append(command_thread)
        
        logger.info("命令发送线程已启动")
        
        return True
    
    def stop_collection(self):
        """停止数据采集 - 多线程版本"""
        logger.info("开始停止数据采集...")
        
        # 设置停止标志
        self.stop_flag = True
        self.is_collecting = False
        
        # 停止各个数据采集线程
        self.robot_thread_stop = True
        self.camera_thread_stop = True
        self.teleop_thread_stop = True
        # 停止命令下发线程
        self.command_thread_stop = True
        
        # 通知异步写盘线程在队列排空后退出
        self.writer_thread_stop = True
        
        # 等待线程结束
        logger.info("等待线程结束...")
        for thread in self.threads:
            logger.info(f"等待线程 {thread.name} 结束...")
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"线程 {thread.name} 在5秒内未结束")
            else:
                logger.info(f"线程 {thread.name} 已结束")

        # 确保命令线程已停止
        if self.command_thread is not None and self.command_thread.is_alive():
            self.command_thread.join(timeout=1.0)
            if self.command_thread.is_alive():
                logger.warning("命令下发线程未能及时退出")
            else:
                logger.info("命令下发线程已退出")
        
        # 清空队列
        logger.info("清空数据队列...")
        try:
            while not self.robot_data_queue.empty():
                self.robot_data_queue.get_nowait()
        except:
            pass
        
        try:
            while not self.camera_data_queue.empty():
                self.camera_data_queue.get_nowait()
        except:
            pass
        
        try:
            while not self.teleop_data_queue.empty():
                self.teleop_data_queue.get_nowait()
        except:
            pass
        
        try:
            while not self.sync_data_queue.empty():
                self.sync_data_queue.get_nowait()
        except:
            pass
            
        logger.info("数据采集已停止")
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止采集
            self.stop_collection()
            
            # 清理机器人连接和回调
            if self.arm:
                try:
                    # 停止实时数据推送
                    udp_config = rm_realtime_push_config_t(
                        100,  # 上报周期(ms)
                        False,  # 禁用上报
                        8089,  # 上报端口
                        0,  # 力数据坐标系
                        "192.168.10.50",  # 目标IP
                        rm_udp_custom_config_t()
                    )
                    self.arm.rm_set_realtime_push(udp_config)
                    
                    # 清理回调
                    if self.callback_ptr:
                        self.callback_ptr = None
                    
                    # 删除机器人连接
                    self.arm.rm_delete_robot_arm()
                    logger.info("机器人连接已清理")
                except Exception as e:
                    logger.error(f"清理机器人连接失败: {e}")
            
            # 清理遥操作器
            if self.teleoperator:
                self.teleoperator.stop()
            if self.shm_manager:
                self.shm_manager.shutdown()
                
            # 清理相机
            for pipeline in self.camera_pipelines:
                pipeline.stop()
            cv2.destroyAllWindows()
                
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='机器人数据采集程序 - LeRobot格式')
    parser.add_argument('--robot-ip', default="192.168.10.18", help='机器人IP地址')
    parser.add_argument('--robot-port', type=int, default=8080, help='机器人端口')
    parser.add_argument('--episode-length', type=int, default=3000, help='每个episode的步数')
    parser.add_argument('--fps', type=int, default=60, help='[兼容] 旧参数：作为相机帧率默认值')
    parser.add_argument('--camera-fps', type=int, help='相机采样频率，默认30')
    parser.add_argument('--robot-fps', type=int, default=60, help='机器人状态采样频率，默认180')
    parser.add_argument('--teleop-fps', type=int, default=60, help='遥操作采样/下发频率，默认')
    parser.add_argument('--no-video', action='store_true', help='关闭实时视频显示')
    parser.add_argument('--video-scale', type=float, default=0.8, help='视频显示缩放比例')
    parser.add_argument('--data-dir', type=str, default=f"./media/zyf/WD_BLACK/realman_data/lerobot_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}", help='数据保存目录')

    args = parser.parse_args()
    
    # 配置参数
    data_dir = args.data_dir

    show_video = not args.no_video
    
    with LeRobotDataCollector(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        data_dir=data_dir,
        episode_length=args.episode_length,
        fps=args.fps,
        show_video=show_video,
        video_scale=args.video_scale,
        camera_fps=args.camera_fps if args.camera_fps else None,
        robot_fps=args.robot_fps,
        teleop_fps=args.teleop_fps
    ) as collector:
        
        try:
            # 开始数据采集
            if collector.start_collection():
                print("\n" + "="*60)
                print("机器人数据采集程序已启动 (多线程版本)")
                print("="*60)
                print("- 使用多线程并行采集数据:")
                print("  * 机器人状态采集线程 @ {} Hz".format(collector.robot_fps))
                print("  * 相机图像采集线程 @ {} Hz".format(collector.camera_fps)) 
                print("  * 遥操作控制线程 @ {} Hz".format(collector.teleop_fps))
                print("  * 数据同步线程 (以相机帧为锚对齐)")
                print("  * 主数据处理线程")
                print("- 使用SpaceMouse进行遥操作")
                print("- 按SpaceMouse按钮记录关键动作")
                print("- 数据将保存为LeRobot兼容的HDF5格式")
                if show_video:
                    print("- 实时视频显示已启动")
                    print("  * 按 'q' 退出程序")
                    print("  * 按 'p' 暂停/继续视频")
                    print("  * 按 's' 保存截图")
                    print("  * 按 'r' 重置当前episode")
                    print("  * 显示队列状态和线程信息")
                else:
                    print("- 按Ctrl+C停止采集")
                print("="*60)
                
                # 主循环等待
                while collector.is_collecting and not collector.stop_flag:
                    time.sleep(0.1)
            else:
                logger.error("数据采集启动失败")
                
        except KeyboardInterrupt:
            print("\n收到停止信号，正在停止采集...")
            collector.stop_flag = True
        except Exception as e:
            logger.error(f"程序错误: {e}")
        finally:
            print(f"数据已保存到: {data_dir}")


if __name__ == "__main__":
    main()
