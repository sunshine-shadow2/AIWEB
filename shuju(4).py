import os
import time
import csv
import math
import threading
import numpy as np
from collections import defaultdict, deque
from scapy.all import sniff, IP, TCP, UDP, ICMP
from scapy.arch.windows import get_windows_if_list

class TrafficCapture:
    # 修改CSV头部，移除不需要的列
    CSV_HEADERS = [
        # 基础包级信息 (移除了src_ip, dst_ip, ip_id)
        'timestamp_us', 'ttl', 'ip_flags', 'ip_len',
        'protocol', 'src_port', 'dst_port', 'tcp_flags', 'tcp_seq', 'tcp_ack', 'tcp_window',
        # 流级统计特征
        'flow_duration', 'flow_pps', 'flow_bps', 'syn_ratio', 'three_way_handshake_ratio',
        'half_open_connections', 'src_ip_entropy',
        # 时间序列特征
        'window_1s_pps', 'window_1s_bps', 'window_1s_syn_ratio', 'traffic_burst_zscore', 'label'
    ]

    def __init__(self):
        # 初始化数据存储结构
        self.packet_buffer = []
        self.flow_table = defaultdict(lambda: {
            'start_time': None, 'end_time': None, 'packet_count': 0, 'byte_count': 0,
            'tcp_flags': defaultdict(int), 'syn_count': 0, 'syn_ack_count': 0, 'ack_count': 0
        })
        self.src_ips = defaultdict(int)
        self.window_stats = deque(maxlen=60)  # 存储60个1秒窗口的统计数据
        self.running = False
        self.capture_thread = None
        self.stat_thread = None
        self.selected_interface = None
        self.lock = threading.Lock()
        
        # 定义攻击标记条件
        self.attack_marker_id = 9999  # 特定ID值表示攻击流量
        self.attack_target_port = 919   # 目标端口91表示攻击流量

        # 创建输出目录
        os.makedirs('http', exist_ok=True)
        self.output_file = 'http/duotou.csv'

    def _calculate_entropy(self, data):
        """计算源IP分布的熵值"""
        total = sum(data.values())
        if total == 0: return 0
        entropy = 0
        for count in data.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy

    def _update_flow_stats(self, packet_info):
        """更新流统计信息"""
        flow_key = (packet_info['src_ip'], packet_info['src_port'],
                   packet_info['dst_ip'], packet_info['dst_port'], packet_info['protocol'])
        flow = self.flow_table[flow_key]
        current_time = packet_info['timestamp_us'] / 1e6  # 转换为秒

        # 更新流基本信息
        if flow['start_time'] is None:
            flow['start_time'] = current_time
        flow['end_time'] = current_time
        flow['packet_count'] += 1
        flow['byte_count'] += packet_info['ip_len']

        # TCP特定统计
        if packet_info['protocol'] == 'TCP':
            flags = packet_info['tcp_flags']
            flow['tcp_flags'][flags] += 1
            if 'S' in flags and 'A' not in flags:  # SYN标志
                flow['syn_count'] += 1
            elif 'S' in flags and 'A' in flags:  # SYN-ACK标志
                flow['syn_ack_count'] += 1
            elif 'A' in flags and 'S' not in flags:  # ACK标志
                flow['ack_count'] += 1

        # 更新源IP分布
        self.src_ips[packet_info['src_ip']] += 1

    def _calculate_flow_features(self, flow_key):
        """计算流级特征"""
        flow = self.flow_table[flow_key]
        if flow['packet_count'] == 0: return {}

        flow_duration = flow['end_time'] - flow['start_time']
        flow_duration = flow_duration if flow_duration > 0 else 0.001
        return {
            'flow_duration': flow_duration,
            'flow_pps': flow['packet_count'] / flow_duration,
            'flow_bps': flow['byte_count'] / flow_duration,
            'syn_ratio': flow['syn_count'] / flow['packet_count'] if flow['packet_count'] > 0 else 0,
            'three_way_handshake_ratio': flow['ack_count'] / flow['syn_count'] if flow['syn_count'] > 0 else 0,
            'half_open_connections': flow['syn_count'] - flow['ack_count'],
            'src_ip_entropy': self._calculate_entropy(self.src_ips)
        }

    def _calculate_time_series_features(self):
        """计算时间序列特征"""
        if len(self.window_stats) < 2:  # 需要至少2个窗口计算Z-score
            return {'window_1s_pps': 0, 'window_1s_bps': 0, 'window_1s_syn_ratio': 0, 'traffic_burst_zscore': 0}

        # 最近1秒窗口统计
        latest_window = self.window_stats[-1]
        # 计算Z-score (与历史窗口比较)
        pps_values = [w['pps'] for w in list(self.window_stats)[:-1]]  # 转换为列表后切片，排除当前窗口
        if len(pps_values) < 1: return {'window_1s_pps': 0, 'window_1s_bps': 0, 'window_1s_syn_ratio': 0, 'traffic_burst_zscore': 0}

        mean_pps = np.mean(pps_values)
        std_pps = np.std(pps_values) if len(pps_values) > 1 else 1
        zscore = (latest_window['pps'] - mean_pps) / std_pps if std_pps > 0 else 0

        return {
            'window_1s_pps': latest_window['pps'],
            'window_1s_bps': latest_window['bps'],
            'window_1s_syn_ratio': latest_window['syn_ratio'],
            'traffic_burst_zscore': zscore
        }

    def _packet_handler(self, packet):
        """Scapy数据包处理回调函数"""
        if not self.running or IP not in packet:
            return

        try:
            # 提取基础包信息
            ip_layer = packet[IP]
            timestamp_us = int(time.time_ns() / 1000)  # 微秒级时间戳
            protocol = ip_layer.proto
            proto_name = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(protocol, f"PROTO_{protocol}")

            # 初始化数据包信息（默认值）
            packet_info = {
                'timestamp_us': timestamp_us,
                'src_ip': ip_layer.src,
                'dst_ip': ip_layer.dst,
                'ttl': ip_layer.ttl,
                'ip_flags': str(ip_layer.flags),
                'ip_len': ip_layer.len,
                'ip_id': ip_layer.id,  # 添加IP ID字段
                'protocol': proto_name,
                'src_port': 'N/A',
                'dst_port': 'N/A',
                'tcp_flags': '',
                'tcp_seq': 'N/A',
                'tcp_ack': 'N/A',
                'tcp_window': 'N/A',
                'label': 0  # 默认正常流量
            }

            # 提取传输层信息
            try:
                if protocol == 6 and TCP in packet:  # TCP
                    tcp_layer = packet[TCP]
                    packet_info.update({
                        'src_port': tcp_layer.sport,
                        'dst_port': tcp_layer.dport,
                        'tcp_flags': tcp_layer.sprintf('%TCP.flags%'),
                        'tcp_seq': tcp_layer.seq,
                        'tcp_ack': tcp_layer.ack,
                        'tcp_window': tcp_layer.window
                    })
                elif protocol == 17 and UDP in packet:  # UDP
                    udp_layer = packet[UDP]
                    packet_info.update({
                        'src_port': udp_layer.sport,
                        'dst_port': udp_layer.dport
                    })
                # ICMP没有端口信息，保持默认值
            except Exception as e:
                print(f"提取传输层信息错误: {e}")

            # 检测攻击标记 - 通过IP ID字段或目标端口
            attack_detected = False
            attack_reason = ""
            
            # 检测IP ID攻击标记
            if ip_layer.id == self.attack_marker_id:
                packet_info['label'] = 1
                attack_detected = True
                attack_reason = f"IP ID={self.attack_marker_id}"
                
            # 检测目标端口攻击标记
            if (packet_info['protocol'] in ['TCP', 'UDP'] and 
                packet_info['dst_port'] == self.attack_target_port):
                packet_info['label'] = 1
                attack_detected = True
                attack_reason = f"目标端口={self.attack_target_port}"
            
            # 打印攻击检测信息
            if attack_detected:
                print(f"检测到攻击流量! 原因: {attack_reason}, 源IP: {ip_layer.src}, 目标IP: {ip_layer.dst}, 目标端口: {packet_info.get('dst_port', 'N/A')}")

            # 更新流统计
            with self.lock:
                self._update_flow_stats(packet_info)
                self.packet_buffer.append(packet_info)

        except Exception as e:
            print(f"处理数据包错误: {e}（数据包摘要: {packet.summary()}）")

    def _statistics_collector(self):
        """定期收集统计信息并写入CSV"""
        while self.running:
            start_time = time.time()
            with self.lock:
                if self.packet_buffer:
                    window_packets = len(self.packet_buffer)
                    window_bytes = sum(p['ip_len'] for p in self.packet_buffer)
                    syn_packets = sum(1 for p in self.packet_buffer if p['protocol'] == 'TCP' and 'S' in p['tcp_flags'])
                    syn_ratio = syn_packets / window_packets if window_packets > 0 else 0

                    self.window_stats.append({
                        'timestamp': start_time,
                        'pps': window_packets,
                        'bps': window_bytes,
                        'syn_ratio': syn_ratio
                    })

                    # 计算时间序列特征
                    time_series_features = self._calculate_time_series_features()

                    # 写入CSV
                    file_empty = not os.path.exists(self.output_file) or os.path.getsize(self.output_file) == 0
                    with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        if file_empty:
                            writer.writerow(self.CSV_HEADERS)
                        for packet in self.packet_buffer:
                            flow_key = (packet['src_ip'], packet['src_port'],
                                       packet['dst_ip'], packet['dst_port'], packet['protocol'])
                            flow_features = self._calculate_flow_features(flow_key)
                            # 构建行数据时跳过src_ip, dst_ip和ip_id字段
                            row = [
                                packet['timestamp_us'], 
                                packet['ttl'], 
                                packet['ip_flags'], 
                                packet['ip_len'],
                                packet['protocol'], 
                                packet['src_port'], 
                                packet['dst_port'],
                                packet['tcp_flags'], 
                                packet['tcp_seq'], 
                                packet['tcp_ack'],
                                packet['tcp_window'],
                                flow_features.get('flow_duration', 0),
                                flow_features.get('flow_pps', 0),
                                flow_features.get('flow_bps', 0),
                                flow_features.get('syn_ratio', 0),
                                flow_features.get('three_way_handshake_ratio', 0),
                                flow_features.get('half_open_connections', 0),
                                flow_features.get('src_ip_entropy', 0),
                                time_series_features['window_1s_pps'],
                                time_series_features['window_1s_bps'],
                                time_series_features['window_1s_syn_ratio'],
                                time_series_features['traffic_burst_zscore'],
                                packet['label']
                            ]
                            writer.writerow(row)
                    self.packet_buffer.clear()  # 清空缓冲区

            # 等待到1秒窗口结束
            elapsed = time.time() - start_time
            time.sleep(max(0, 1 - elapsed))

    def list_interfaces(self):
        """列出所有可用网络接口"""
        interfaces = get_windows_if_list()
        return [{
            'index': i,
            'name': iface['name'],
            'friendly_name': iface.get('friendly_name', iface['name']),
            'mac': iface.get('mac', 'Unknown'),
            'ips': iface.get('ips', [])
        } for i, iface in enumerate(interfaces) if iface.get('mac') != '00:00:00:00:00:00']

    def start_capture(self, interface_index):
        """开始流量捕获"""
        interfaces = self.list_interfaces()
        if interface_index < 0 or interface_index >= len(interfaces):
            raise ValueError("无效的接口索引")

        self.selected_interface = interfaces[interface_index]
        self.running = True

        # 启动统计收集线程
        self.stat_thread = threading.Thread(target=self._statistics_collector, daemon=True)
        self.stat_thread.start()

        # 启动数据包捕获
        print(f"开始在接口 {self.selected_interface['friendly_name']} 上捕获流量...")
        print(f"数据将保存到 {self.output_file}")
        print(f"攻击检测条件: IP ID={self.attack_marker_id} 或 目标端口={self.attack_target_port}")
        self.capture_thread = threading.Thread(
            target=sniff,
            kwargs={
                'iface': self.selected_interface['name'],
                'prn': self._packet_handler,
                'store': 0,
                'promisc': True
            },
            daemon=True
        )
        self.capture_thread.start()

    def stop_capture(self):
        """停止流量捕获"""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        if self.stat_thread and self.stat_thread.is_alive():
            self.stat_thread.join(timeout=2)
        print("流量捕获已停止")

if __name__ == '__main__':
    capture = TrafficCapture()
    interfaces = capture.list_interfaces()

    print("可用网络接口:")
    for i, iface in enumerate(interfaces):
        ips = ', '.join(iface['ips']) if iface['ips'] else '无IP'
        print(f"{i}. {iface['friendly_name']} (IP: {ips})")

    try:
        selected = int(input("请选择要监听的接口索引: "))
        capture.start_capture(selected)
        input("按Enter键停止捕获...\n")
    except KeyboardInterrupt:
        pass
    finally:
        capture.stop_capture()