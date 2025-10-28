import socket
import json
import time

def send_clear_command():
    """
    发送六维力传感器清零指令
    """
    try:
        # 创建TCP连接
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)  # 设置超时时间为5秒
        s.connect(("192.168.10.18", 8080))

        # 构建并发送清零指令
        command = {"command": "clear_force_data"}
        s.sendall((json.dumps(command) + "\n").encode())

        # 接收响应
        response = s.recv(1024).decode()
        result = json.loads(response)

        if result.get("clear_state") == True:
            print("六维力传感器清零成功")
            return True
        else:
            print("六维力传感器清零失败")
            return False

    except socket.timeout:
        print("连接超时，请检查机械臂是否在线")
        return False
    except ConnectionRefusedError:
        print("连接被拒绝，请检查端口是否正确或机械臂是否已启动服务")
        return False
    except Exception as e:
        print(f"通信错误: {e}")
        return False
    finally:
        try:
            s.close()
        except:
            pass

def main():
    """
    主函数
    """
    print("开始执行六维力传感器清零程序...")
    print(f"目标机械臂: 192.168.10.18:8080")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    # 发送清零指令
    success = send_clear_command()

    print("-" * 50)
    if success:
        print("程序执行成功，六维力传感器已清零")
    else:
        print("程序执行失败，请检查连接和机械臂状态")

    print(f"程序结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()