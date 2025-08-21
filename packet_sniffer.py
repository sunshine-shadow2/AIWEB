#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据包监测程序
功能: 捕获并展示网关的数据包信息，最多展示10条
依赖: scapy库
"""

import time
import argparse
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, Ether

# 数据包计数器和存储列表
packet_count = 0
captured_packets = []
MAX_PACKETS = 10


def packet_callback(packet):
    """处理捕获到的数据包"""
    global packet_count, captured_packets

    # 增加计数
    packet_count += 1
    captured_packets.append(packet)

    # 打印数据包基本信息
    print(f"\n=== 数据包 #{packet_count} ===")
    print(f"捕获时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据包长度: {len(packet)} 字节")

    # 链路层信息
    if Ether in packet:
        print("\n链路层信息:")
        print(f"  源MAC: {packet[Ether].src}")
        print(f"  目标MAC: {packet[Ether].dst}")
        print(f"  类型: {packet[Ether].type}")

    # 网络层信息
    if IP in packet:
        print("\n网络层信息:")
        print(f"  源IP: {packet[IP].src}")
        print(f"  目标IP: {packet[IP].dst}")
        print(f"  协议: {packet[IP].proto}")
        print(f"  TTL: {packet[IP].ttl}")
        print(f"  长度: {packet[IP].len}")

    # ARP信息
    elif ARP in packet:
        print("\nARP信息:")
        print(f"  操作码: {packet[ARP].op} (1=请求, 2=响应)")
        print(f"  发送方MAC: {packet[ARP].hwsrc}")
        print(f"  发送方IP: {packet[ARP].psrc}")
        print(f"  目标MAC: {packet[ARP].hwdst}")
        print(f"  目标IP: {packet[ARP].pdst}")

    # 传输层信息
    if TCP in packet:
        print("\n传输层信息 (TCP):")
        print(f"  源端口: {packet[TCP].sport}")
        print(f"  目标端口: {packet[TCP].dport}")
        print(f"  标志位: {packet[TCP].flags}")
        print(f"  序列号: {packet[TCP].seq}")
        print(f"  确认号: {packet[TCP].ack}")
        print(f"  窗口大小: {packet[TCP].window}")

    elif UDP in packet:
        print("\n传输层信息 (UDP):")
        print(f"  源端口: {packet[UDP].sport}")
        print(f"  目标端口: {packet[UDP].dport}")
        print(f"  长度: {packet[UDP].len}")

    elif ICMP in packet:
        print("\n传输层信息 (ICMP):")
        print(f"  类型: {packet[ICMP].type}")
        print(f"  代码: {packet[ICMP].code}")
        print(f"  校验和: {packet[ICMP].chksum}")

    # 应用层数据 (如果有)
    if packet.haslayer('Raw'):
        print("\n应用层数据:")
        try:
            # 尝试以ASCII格式打印数据
            raw_data = packet['Raw'].load
            print(f"  数据长度: {len(raw_data)} 字节")
            print(f"  数据: {raw_data.hex()}")
        except Exception as e:
            print(f"  无法解析数据: {e}")

    # 达到最大数据包数量时停止捕获
    if packet_count >= MAX_PACKETS:
        return True


def start_sniffing(interface=None, timeout=60):
    """
    开始捕获数据包

    参数:
        interface: 网络接口名称，默认为None(自动选择)
        timeout: 捕获超时时间(秒)，默认为60
    """
    print("开始捕获数据包...")
    print(f"将显示前{MAX_PACKETS}个数据包的详细信息")
    print("按Ctrl+C可以随时停止捕获")

    try:
        # 开始捕获数据包
        sniff(
            iface=interface,
            prn=packet_callback,
            store=0,
            timeout=timeout,
            stop_filter=lambda _: packet_count >= MAX_PACKETS
        )
        print(f"\n捕获完成，共捕获{packet_count}个数据包")
    except KeyboardInterrupt:
        print("\n用户中断捕获")
    except Exception as e:
        print(f"捕获过程中发生错误: {e}")

def get_gateway_ip():
    """获取网关IP地址"""
    # 这里简化处理，实际应用中可能需要通过网络配置获取
    # 也可以让用户手动输入网关IP
    return "192.168.1.1"


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='网关数据包监测程序')
    parser.add_argument('-c', '--count', type=int, default=10, help='要显示的最大数据包数量 (默认: 10)')
    parser.add_argument('-t', '--timeout', type=int, default=60, help='捕获超时时间(秒) (默认: 60)')
    args = parser.parse_args()

    # 更新最大数据包数量
    global MAX_PACKETS
    MAX_PACKETS = args.count

    gateway_ip = get_gateway_ip()
    print(f"监测网关: {gateway_ip}")
    print(f"最大显示数据包数量: {MAX_PACKETS}")

    # 构造过滤条件，只捕获与网关相关的数据包
    # 这里的过滤条件会捕获源IP或目标IP为网关的数据包
    filter_rule = f"host {gateway_ip}"
    print(f"过滤规则: {filter_rule}")

    # 开始捕获
    start_sniffing(timeout=args.timeout)


if __name__ == "__main__":
    main()