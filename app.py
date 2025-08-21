import json
from flask import Flask, render_template, request, jsonify, current_app, Response
import os
import openai
import numpy as np
import pandas as pd
from duotoujiqi1 import detect_ddos_decision_tree
from interpreter import interpreter
from unittest.mock import patch
from interpreter.core.computer.computer import Computer
import ast
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP, ICMP
from scapy.layers.http import HTTP, HTTPRequest, HTTPResponse
from scapy.sendrecv import sniff
from scapy.arch.windows import get_windows_if_list
import sys
import ctypes
import logging
import signal
import threading
from datetime import datetime
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# 配置文件路径
CONFIG_FILE = 'config.json'
CONTEXT_FILE = 'context/'

# 全局变量
import threading
import json
import os
from collections import deque, defaultdict
from datetime import datetime
running = False
capture_thread = None
selected_interface = None
all_packets = deque(maxlen=100)  # 存储所有最新100条数据
captured_packets = deque(maxlen=100)  # 存储筛选后的数据包，最多100条
attack_packets = deque(maxlen=1000)  # 存储攻击数据包，最多100条
file1 = ""
filter_string = ""
packet_lock = threading.Lock()  # 数据包操作锁
filter_lock = threading.Lock()  # 筛选条件锁
attack_packet_lock = threading.Lock()  # 攻击数据包锁
current_attack_status = False  # 攻击状态标志
attack_status_lock = threading.Lock()  # 攻击状态锁

# 新增攻击检测器类
class AttackDetector:
    def __init__(self, required_consecutive=10):
        self.models = []
        self.consecutive_count = 0
        self.required_consecutive = required_consecutive
        self.current_status = False
        self.prediction_history = []
        self.history_lock = threading.Lock()
        self.max_history = 10  # 修改为最多存储10条记录

    def register_model(self, model_function):
        self.models.append(model_function)

    def detect(self, input_data):
        # 检查所有模型是否都预测为攻击
        all_positive = True
        predictions = []
        for model in self.models:
            result_df = model(input_data)
            prediction = result_df['ddos_prediction'].iloc[0]
            predictions.append({
                'model': model.__name__,  # 获取模型函数名
                'prediction': int(prediction),
                'features': input_data.iloc[0].to_dict(),
                'timestamp': datetime.now().isoformat()
            })
            if prediction != 1.0:
                all_positive = False

        # 保存预测记录
        with self.history_lock:
            self.prediction_history.extend(predictions)
            # 只保留最近10条记录
            if len(self.prediction_history) > self.max_history:
                self.prediction_history = self.prediction_history[-self.max_history:]

        if all_positive:
            self.consecutive_count += 1
            if self.consecutive_count >= self.required_consecutive:
                self.current_status = True
            else:
                self.current_status = False  # 尚未达到连续次数
        else:
            self.consecutive_count = 0
            self.current_status = False

        return self.current_status

# 初始化攻击检测器，要求连续10次检测为攻击
attack_detector = AttackDetector(required_consecutive=10)
# 注册决策树模型
attack_detector.register_model(detect_ddos_decision_tree)
  
  # 流跟踪（用于计算每个包的流指标）
flow_tracker = {}  # key: flow_key, value: (start_time, packet_count)
flow_pairs = set()  # 跟踪IP对数量
  
  # 读取配置文件
def read_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"api_base": "https://api.deepseek.com/v1", "model": "deepseek-chat"}

# 保存配置文件
def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# 从文件读取上下文
def load_context(m):
    try:
        with open(CONTEXT_FILE + m, 'r') as f:
            interpreter.messages = json.load(f)
        return True
    except FileNotFoundError:
        return False

# 保存上下文到文件
def save_context():
    if file1 == "":
        return
    m1 = file1
    m = f'{m1}.json'
    with open(CONTEXT_FILE + m[:10], 'w') as f:
        json.dump(interpreter.messages, f)

# 验证API
def validate_api():
    # 这里简单返回成功，实际中需要实现验证逻辑
    return jsonify({"success": True})

# 获取所有可用的网络接口
def get_network_interfaces():
    try:
        interfaces = get_windows_if_list()
        interface_list = []
        for i, iface in enumerate(interfaces):
            if iface.get('mac') != '00:00:00:00:00:00':
                interface_info = {
                    'index': i,
                    'name': iface['name'],
                    'friendly_name': iface.get('friendly_name', iface['name']),
                    'mac': iface.get('mac', 'Unknown'),
                    'ips': iface.get('ips', [])
                }
                interface_list.append(interface_info)
        return interface_list
    except Exception as e:
        logging.error(f"获取网络接口时发生错误: {e}")
        return []

import re

def matches_filter(packet_info, filter_str):
    # 过滤N/A IP的数据包
    if packet_info.get('src') == 'N/A' or packet_info.get('dst') == 'N/A':
        return False
    if not filter_str:
        return True
    
    # 支持逗号分隔的多条件筛选
    filters = [f.strip() for f in filter_str.split(',') if f.strip()]
    for f in filters:
        # 端口筛选特殊处理 (sport:xxx 或 dport:xxx)
        if f.startswith(('sport:', 'dport:')):
            parts = f.split(':', 1)
            if len(parts) == 2:
                port_type, port_str = parts
                port_field = 'sport' if port_type == 'sport' else 'dport'
                try:
                    port = int(port_str)
                    if packet_info.get(port_field) != port:
                        return False
                except ValueError:
                    return False
        else:
            # 检查是否是纯数字，可能是端口号
            if f.isdigit():
                 port = int(f)
                 # 仅匹配目的端口(dport)
                 dport = packet_info.get('dport')
                 try:
                     dport_int = int(dport) if dport != 'N/A' else None
                 except (ValueError, TypeError):
                     dport_int = None
                 # 检查dport是否等于目标端口
                 if not (dport_int is not None and dport_int == port):
                     return False
            else:
                try:
                    # 尝试将筛选条件编译为正则表达式
                    pattern = re.compile(f, re.IGNORECASE)
                    # 检查数据包的各个字段是否匹配筛选条件
                    match_found = False
                    for value in packet_info.values():
                        if isinstance(value, str) and pattern.search(value):
                            match_found = True
                            break
                        elif isinstance(value, (int, float)) and pattern.search(str(value)):
                            match_found = True
                            break
                    if not match_found:
                        return False
                except re.error:
                    # 如果正则表达式无效，则使用简单包含匹配
                    filter_lower = f.lower()
                    match_found = False
                    for value in packet_info.values():
                        if isinstance(value, str) and filter_lower in value.lower():
                            match_found = True
                            break
                        elif isinstance(value, (int, float)) and str(value) == filter_lower:
                            match_found = True
                            break
                    if not match_found:
                        return False
    return True

# 处理所有协议的数据包
def print_http_packet(packet):
    try:
        packet_info = {
                "timestamp": datetime.now().isoformat(),
                "size": len(packet),
                "type": "OTHER",
                "src": "N/A",
                "dst": "N/A",
                "sport": "N/A",
                "dport": "N/A",
                "method": "N/A",
                "path": "N/A",
                "status_code": "N/A",
                "attack_status": False
            }

        # 快速协议类型检测
        layers = [ICMP, TCP, UDP, IP, Ether]
        for layer in layers:
            if packet.haslayer(layer):
                packet_info["type"] = layer.__name__
                break

        # IP层信息提取
        ip_layer = packet.getlayer(IP)
        if ip_layer:
            packet_info["src"] = ip_layer.src
            packet_info["dst"] = ip_layer.dst

        # 传输层端口信息
        transport_layer = packet.getlayer(TCP) or packet.getlayer(UDP)
        if transport_layer:
            packet_info["sport"] = transport_layer.sport
            packet_info["dport"] = transport_layer.dport

        # HTTP信息提取（仅TCP）
        if packet_info["type"] == "TCP" and packet.haslayer(HTTP):
            if packet.haslayer(HTTPRequest):
                req = packet[HTTPRequest]
                packet_info["method"] = req.Method.decode() if req.Method else 'UNKNOWN'
                packet_info["path"] = req.Path.decode() if req.Path else '/'
            elif packet.haslayer(HTTPResponse):
                resp = packet[HTTPResponse]
                packet_info["status_code"] = resp.Status_Code.decode() if resp.Status_Code else 'UNKNOWN'

        # 批次化处理与筛选
        with packet_lock:
            all_packets.append(packet_info)
            
            # 合并相同时间戳和端口的数据包
            merged_packets = {}
            for pkt in list(all_packets):
                # 使用时间戳前10位(秒级)和端口作为合并键
                key = (pkt['timestamp'][:10], pkt['sport'], pkt['dport'])
                if key not in merged_packets:
                    merged_packets[key] = pkt.copy()
                    merged_packets[key]['merged_count'] = 1
                else:
                    merged_packets[key]['size'] += pkt['size']
                    merged_packets[key]['merged_count'] += 1
            
            # 只保留合并后的最新100条数据包
            merged_list = list(merged_packets.values())[-100:]
            all_packets.clear()
            all_packets.extend(merged_list)

        # 读取最新的筛选条件
        with filter_lock:
            current_filter = filter_string
        if matches_filter(packet_info, current_filter):
            with packet_lock:
                captured_packets.append(packet_info)

            # 计算每个合并包的指标并打印到控制台 (已精简未使用变量)
            current_time = datetime.fromisoformat(packet_info['timestamp'])
            bytecount = packet_info['size']

              # 仅保留参与csv_data生成的核心变量
              # 流相关指标计算 (用于后续特征提取)
            if packet_info['src'] != 'N/A' and packet_info['dst'] != 'N/A':
                  flow_key = (
                      packet_info['src'],
                      packet_info['dst'],
                      packet_info['type'],
                      packet_info['sport'],
                      packet_info['dport']
                  )
                  # 跟踪流的开始时间和包数
                  if flow_key not in flow_tracker:
                      flow_tracker[flow_key] = (current_time, 1)
                  else:
                      start_time, count = flow_tracker[flow_key]
                      count += 1
                      flow_tracker[flow_key] = (start_time, count)
                  
                  # 流对跟踪
                  flow_pair = (packet_info['src'], packet_info['dst'])
                  flow_pairs.add(flow_pair)
            # 新的批次特征提取逻辑
            global packet_buffer, batch_counter
            if 'packet_buffer' not in globals():
                packet_buffer = []
                batch_counter = 1

            # 添加当前数据包到缓冲区
            with packet_lock:
                packet_buffer.append(packet_info.copy())

            # 每100条数据包处理一次
            if len(packet_buffer) >= 100:
                # 特征提取函数（便于扩展）
                def extract_features(batch):
                    features = {}
                    # 1. 增强版时间戳密集度分析
                    timestamps = [datetime.fromisoformat(p['timestamp']) for p in batch]
                    
                    # 基于时间序列的统计计算
                    sorted_ts = sorted(timestamps)
                    
                    # 计算相邻时间差（秒）
                    # 时间差计算（秒级精度）
                    time_deltas = [
                        (sorted_ts[i+1] - sorted_ts[i]).total_seconds()
                        for i in range(len(sorted_ts)-1)
                    ] if len(sorted_ts) > 1 else []
                    
                    # 包速率计算（处理零除错）
                    pps_values = []
                    for t in time_deltas:
                        if t > 0.001:  # 过滤极小时间间隔
                            pps_values.append(1/t)
                        else:
                            pps_values.append(10000)  # 最大限速1000pps
                    time_span = (max(timestamps) - min(timestamps)).total_seconds() or 0.001

                    # 2. 流量速率 (总字节数/时间跨度)
                    total_bytes = sum(p['size'] for p in batch)
                    features['traffic_rate'] = total_bytes / time_span

                    # 3. 重复数据包度 (重复流占比)
                    flow_keys = [(p['src'], p['dst'], p['sport'], p['dport']) for p in batch]
                    unique_flows = len(set(flow_keys))
                    features['duplicate_ratio'] = 1 - (unique_flows / len(flow_keys))

                    # 源IP多样性指数
                    src_ips = [p['src'] for p in batch if p['src'] != 'N/A']
                    features['src_ip_diversity'] = len(set(src_ips)) / len(src_ips) if src_ips else 0

                    # 流对密度 (单位时间内唯一IP对数量)
                    flow_pairs = set((p['src'], p['dst']) for p in batch if p['src'] != 'N/A' and p['dst'] != 'N/A')
                    features['flow_pair_density'] = len(flow_pairs) / time_span if time_span > 0 else 0

                    # 时间间隔波动性 (时间差标准差)
                    features['time_delta_std'] = np.std(time_deltas) if time_deltas else 0

                    # 多维时间特征
                    # 核心特征指标
                    features.update({
                        'timestamp_density': len(batch)/time_span,
                        'traffic_rate': sum(p['size'] for p in batch)/time_span,
                        'duplicate_ratio': 1 - (len(set((p['src'],p['dst'],p['sport'],p['dport']) for p in batch))/len(batch)),
                        'src_ip_diversity': features['src_ip_diversity'],
                        'flow_pair_density': features['flow_pair_density'],
                        'time_delta_std': features['time_delta_std']
                    })

                    return features

                # 计算特征值
                batch_features = extract_features(packet_buffer)

                # 构建CSV数据
                csv_data = f"{batch_features['timestamp_density']:.2f},{batch_features['traffic_rate']:.2f},{batch_features['duplicate_ratio']:.4f},{batch_features['src_ip_diversity']:.4f},{batch_features['flow_pair_density']:.4f},{batch_features['time_delta_std']:.6f}"

                # 保存到http文件夹
                csv_path = os.path.join('http', 'test3.csv')
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                file_exists = os.path.exists(csv_path)

                with packet_lock:
                    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                        if not file_exists:
                            headers = ['timestamp_density','traffic_rate','duplicate_ratio','src_ip_diversity','flow_pair_density','time_delta_std']
                            f.write(','.join(headers) + '\n')
                        f.write(csv_data + '\n')

                print(f"批次 {batch_counter} 处理完成，已保存 {len(packet_buffer)} 条数据包特征")
                packet_buffer = []
                batch_counter += 1         
            # 调用detect_ddos函数进行预测
            # 将csv_data转换为DataFrame
            features = csv_data.split(',')
            feature_names = ['timestamp_density','traffic_rate','duplicate_ratio','src_ip_diversity','flow_pair_density','time_delta_std']
            input_data = pd.DataFrame([features], columns=feature_names).astype(float)
            
            # 使用攻击检测器进行检测（支持多模型）
            with attack_status_lock:
                current_attack_status = attack_detector.detect(input_data)
            
            # 记录当前数据包的攻击状态
            packet_info['attack_status'] = current_attack_status
            # 如果检测到攻击，保存当前批次的数据包到攻击列表
            if current_attack_status:
                with attack_packet_lock:
                    attack_packets.extend(packet_buffer.copy())
                    print(f"当前批次攻击数据包数: {len(attack_packets)}")
            with packet_lock:
                current_attack_status = is_attack

  
    except Exception as e:
        logging.debug(f"处理HTTP数据包出错: {e}")

# Scapy的数据包处理回调函数
def packet_handler(packet):
    try:
        print_http_packet(packet)  # 只处理HTTP包
    except Exception as e:
        logging.debug(f"处理数据包出错: {e}")

def apply_filter():
    """根据当前filter_string筛选数据包"""
    global captured_packets, all_packets, filter_string
    with packet_lock:
        captured_packets.clear()
        # 从所有数据包中筛选
        for p in all_packets:
            if matches_filter(p, filter_string):
                captured_packets.append(p)

def start_capture(port=80):
    global running, selected_interface, packet_batches
    global all_captured_packets, captured_packets
    running = True
    # 清空数据包列表和批次
    all_packets.clear()
    captured_packets.clear()
    if not selected_interface:
        logging.error("未选择网络接口！")
        return
    logging.info(f"已选择监听接口: {selected_interface['friendly_name']} (MAC: {selected_interface['mac']})"
)
    try:
        sniff(
            iface=selected_interface['name'],
            prn=packet_handler,
            store=0,
            promisc=True,
            stop_filter=lambda p: not running
        )
    except PermissionError:
        logging.error("权限错误：请确保以管理员身份运行此脚本")
    except Exception as e:
        logging.error(f"监听过程中发生严重错误: {e}")

# 保存配置的路由
@app.route('/save_config', methods=['POST'])
def save_config_route():
    api_base = request.json.get('api_base')
    model = request.json.get('model')
    if api_base and model:
        config = {"api_base": api_base, "model": model}
        save_config(config  )
        return jsonify({"success": True, "message": "配置已保存"})
    else:
        return jsonify({"success": False, "message": "缺少必要的配置信息"})

# 读取上下文的路由
@app.route('/load_context', methods=['POST'])
def load_context_route():
    data = request.get_json()
    file_name = data.get('file_name')
    api_base = data.get('api_base')
    model = data.get('model')

    if not file_name:
        return jsonify({"success": False, "message": "未提供文件名"})

    if load_context(file_name):
        return jsonify({"success": True, "message": "上下文已加载"})
    else:
        return jsonify({"success": False, "message": "未找到上下文文件"})

# 保存上下文的路由
@app.route('/save_context', methods=['POST'])
def save_context_route():
    save_context()
    return jsonify({"success": True, "message": "上下文已保存"})

# 发送用户请求到AI的路由
@app.route('/send_to_ai', methods=['POST'])
def send_to_ai():
    if not validate_api().json['success']:
        return jsonify({"success": False, "message": "API验证失败，请检查配置"})
    user_text = request.json.get('user_text', '').strip()
    if not user_text:
        return jsonify({"success": False, "message": "请输入请求内容！"})
    try:
        global file1
        file1 = user_text
        api_base = request.json.get('api_base', 'https://api.deepseek.com/v1')
        model = request.json.get('model', 'deepseek-chat')
        interpreter.api_key = os.getenv("OPENAI_API_KEY")
        interpreter.api_base = api_base
        interpreter.model = model
        interpreter.llm.model = model
        interpreter.llm.api_key = os.getenv("OPENAI_API_KEY")
        interpreter.llm.api_base = api_base
        interpreter.llm.temperature = 0.4

        def generate():
            response = interpreter.chat(f"{user_text}\n", stream=True)
            prompt = ""
            code = ""
            format = "代码内容"
            for chunk in response:
                if isinstance(chunk, dict) and chunk.get('role') == 'assistant' and chunk.get('type') == 'code':
                    code = chunk.get('content', '')
                    format = chunk.get('format', '')
                elif isinstance(chunk, dict) and chunk.get('role') == 'assistant' and chunk.get('type') == 'message':
                    prompt = chunk.get('content', '')
                data = json.dumps({"success": True, "format": format, "prompt": prompt, "code": code})
                yield f"{data}\n"

        return Response(generate(), mimetype='text/plain')
    except Exception as e:
        return jsonify({"success": False, "message": f"AI请求错误：{str(e)}"})

# 执行代码的路由
@app.route('/execute_code', methods=['POST'])
def execute_code():
    if not validate_api().json['success']:
        return jsonify({"success": False, "message": "API验证失败，请检查配置"})
    code_text = request.json.get('code_text', '').strip()
    if not code_text:
        return jsonify({"success": False, "message": "未输入任何代码！"})
    language1 = request.json.get('language', '')
    try:
        api_base = request.json.get('api_base', 'https://api.deepseek.com/v1')
        model = request.json.get('model', 'deepseek-chat')
        interpreter.api_key = os.getenv("OPENAI_API_KEY")
        interpreter.api_base = api_base
        interpreter.model = model
        interpreter.llm.model = model
        interpreter.llm.api_key = os.getenv("OPENAI_API_KEY")
        interpreter.llm.api_base = api_base
        interpreter.llm.temperature = 0.4
        computer = Computer(interpreter=interpreter)
        computer.import_computer_api = True

        def generate():
            if language1 == "python":
                command = code_text.strip()
            elif language1 == "powershell":
                command = f"chcp 65001 && {code_text.strip()}"
            else:
                command = code_text.strip()

            # 统一处理所有语言的输出
            for item in computer.run(language1, command, stream=True):
                # 过滤掉 active_line 类型的内容
                if item.get("format") == "active_line":
                    continue

                content = item.get("content", "")
                # 确保内容为字符串并编码
                if isinstance(content, bytes):
                    yield content.decode('utf-8') + "\n"
                else:
                    yield str(content) + "\n"
            print("执行完成\n")

        return Response(generate(), mimetype='text/plain')
    except Exception as e:
        return jsonify({"success": False, "message": f"代码执行失败：{str(e)}"})

# 发送执行结果到AI的路由
@app.route('/send_result_to_ai', methods=['POST'])
def send_result_to_ai():
    if not validate_api().json['success']:
        return jsonify({"success": False, "message": "API验证失败，请检查配置"})
    result_text = request.json.get('result_text', '').strip()
    if not result_text:
        return jsonify({"success": False, "message": "没有执行结果可以发送！"})
    try:
        api_base = request.json.get('api_base', 'https://api.deepseek.com/v1')
        model = request.json.get('model', 'deepseek-chat')
        interpreter.api_key = os.getenv("OPENAI_API_KEY")
        interpreter.api_base = api_base
        interpreter.model = model
        interpreter.llm.model = model
        interpreter.llm.api_key = os.getenv("OPENAI_API_KEY")
        interpreter.llm.api_base = api_base
        interpreter.llm.temperature = 0.4

        def generate():
            response = interpreter.chat(
                f"请分析以下执行结果有错误就改进，没有就进行下一步,如果任务已经完成就停止输出代码：\n{result_text}",
                stream=True)
            prompt = ""
            code = ""
            format = "代码内容"
            for chunk in response:
                if isinstance(chunk, dict) and chunk.get('role') == 'assistant' and chunk.get('type') == 'code':
                    code = chunk.get('content', '')
                    format = chunk.get('format', '')
                elif isinstance(chunk, dict) and chunk.get('role') == 'assistant' and chunk.get('type') == 'message':
                    prompt = chunk.get('content', '')
                data = json.dumps({"success": True, "format": format, "prompt": prompt, "code": code})
                yield f"{data}\n"

        return Response(generate(), mimetype='text/plain')
    except Exception as e:
        return jsonify({"success": False, "message": f"发送结果错误：{str(e)}"})

# 获取网络接口的路由
@app.route('/get_network_interfaces', methods=['GET'])
def get_network_interfaces_route():
    interfaces = get_network_interfaces()
    return jsonify({"success": True, "interfaces": interfaces})

# 开始捕获数据包的路由
@app.route('/start_capture', methods=['POST'])
def start_capture_route():
    global running, capture_thread, selected_interface, filter_string
    interface_index = request.json.get('interface_index')
    filter_string = request.json.get('filter_string', '')
    
    if interface_index is None:
        return jsonify({"success": False, "message": "未选择网络接口"})

    interfaces = get_network_interfaces()
    if interface_index < 0 or interface_index >= len(interfaces):
        return jsonify({"success": False, "message": "无效的接口索引"})

    selected_interface = interfaces[interface_index]


    if not running:
        capture_thread = threading.Thread(target=start_capture)
        capture_thread.start()
        return jsonify({"success": True, "message": f"已开始在接口 {selected_interface['friendly_name']} 上捕获所有流量，筛选条件: {filter_string}"})
    else:
        return jsonify({"success": False, "message": "数据包捕获已经在运行"})

# 停止捕获数据包的路由
@app.route('/stop_capture', methods=['GET'])
def stop_capture_route():
    global running
    if running:
        running = False
        return jsonify({"success": True, "message": "数据包捕获已停止"})
    else:
        return jsonify({"success": False, "message": "数据包捕获未运行"})

# 应用新的筛选条件
@app.route('/update_filter', methods=['POST'])
def update_filter_route():
    global filter_string
    new_filter = request.json.get('filter_string', '')
    with filter_lock:
        filter_string = new_filter
    # 更新筛选后的数据
    with packet_lock:
        captured_packets.clear()
        for p in all_packets:
            if matches_filter(p, filter_string):
                captured_packets.append(p)
    return jsonify({"success": True, "message": "筛选条件已更新"})

# 获取捕获的数据包的路由
@app.route('/get_total_pages', methods=['GET'])
def get_total_pages():
    total_pages = len(packet_batches)
    if all_captured_packets:
        total_pages += 1
    return jsonify({"success": True, "total_pages": total_pages})





@app.route('/get_captured_packets', methods=['GET'])
def get_captured_packets_route():
    global captured_packets, filter_string
    
    with packet_lock:
        # 按时间戳降序排序，确保最新数据在前，取前100条
        sorted_packets = sorted(captured_packets, key=lambda x: x['timestamp'], reverse=True)[:100]
        # 应用筛选条件
        filtered_packets = [p for p in sorted_packets if matches_filter(p, filter_string)]
    
    return jsonify({
        "success": True,
        "packets": filtered_packets
    })

@app.route('/get_attack_packets', methods=['GET'])
def get_attack_packets():
    with attack_packet_lock:
        # 返回攻击数据包的副本
        attack_data = list(attack_packets)
    return jsonify({"success": True, "packets": attack_data})


# 主页面路由7
@app.route('/')
def index():
    config = read_config()
    api_base = config.get('api_base', 'https://api.deepseek.com/v1')
    model = config.get('model', 'deepseek-chat')
    return render_template('index.html', api_base=api_base, model=model)

# 网络接口和数据包捕获页面路由
@app.route('/network_capture')
def network_capture():
    return render_template('network_capture.html')

# 获取当前攻击状态的路由
@app.route('/get_attack_status', methods=['GET'])
def get_attack_status():
    with attack_status_lock:
        return jsonify({"under_attack": current_attack_status})

@app.route('/get_model_predictions')
def get_model_predictions():
    with attack_detector.history_lock:
        return jsonify(attack_detector.prediction_history)

loaded_models = None
scaler = None

    




if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)