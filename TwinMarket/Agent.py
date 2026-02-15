"""
AI代理基础类模块

该模块定义了一个通用的AI代理基础类，用于与OpenAI API进行交互。
支持多种AI模型、重试机制、并发API调用等功能。
"""

# 标准库导入
import json
import os
import random
import sys
import time

# 第三方库导入
import yaml
from openai import OpenAI  # OpenAI官方客户端库
from tenacity import retry, wait_exponential, stop_after_attempt, wait_random  # 重试机制库

# ============================ 全局配置 ============================
# 默认系统提示词
sys_default_prompt = "You are a helpful assistant."


class BaseAgent:
    """
    AI代理基础类

    该类封装了与OpenAI API的交互逻辑，提供了统一的接口来调用各种语言模型。
    支持多个API密钥的随机选择、重试机制、参数可配置等特性。

    Attributes:
        config: 从配置文件加载的配置信息
        api_keys: API密钥列表
        model_name: 使用的模型名称
        base_url: API基础URL
        default_system_prompt: 默认系统提示词
        client: OpenAI客户端实例
    """

    def __init__(
        self, system_prompt=sys_default_prompt, config_path="./config_random/zyf.yaml"
    ):
        """
        初始化AI代理

        Args:
            system_prompt (str): 系统提示词，定义AI的角色和行为模式
            config_path (str): 配置文件路径，包含API密钥、模型名称等信息
        """
        # 获取配置文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_path)

        # 加载配置文件
        with open(config_path, "r") as config_file:
            self.config = yaml.load(config_file, Loader=yaml.FullLoader)

        # 提取配置信息
        self.api_keys = self.config["api_key"]  # API密钥列表
        self.model_name = self.config["model_name"]  # 模型名称
        self.base_url = self.config["base_url"]  # API基础URL
        self.default_system_prompt = system_prompt  # 默认系统提示词

        # 初始化OpenAI客户端，随机选择一个API密钥
        self.client = OpenAI(
            api_key=random.choice(self.api_keys),  # 从多个密钥中随机选择
            base_url=self.base_url,  # 设置API基础URL
            timeout=600.0,  # 设置超时时间为600秒（10分钟），避免请求超时
        )

    def __post_process(self, response):
        """
        处理OpenAI API的响应数据

        从原始API响应中提取有用信息，包括生成的文本内容和使用的token数量。

        Args:
            response: OpenAI API返回的原始响应对象

        Returns:
            dict: 包含响应内容和token使用情况的字典
        """
        return {
            "response": response.choices[0].message.content,  # 提取AI生成的文本内容
            "total_tokens": response.usage.total_tokens,  # 提取总的token使用数量
        }

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10) + wait_random(0, 2),
        stop=stop_after_attempt(10)
    )
    def __call_api(
        self,
        messages,
        temperature=0.9,
        max_tokens=8192,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        **kwargs,
    ):
        """
        调用OpenAI API并获取响应

        使用tenacity库实现重试机制，在API调用失败时会自动重试。
        重试间隔为1秒，最多重试10次。

        Args:
            messages (list): 对话消息列表
            temperature (float): 温度参数，控制输出的随机性
            max_tokens (int): 最大生成token数量
            top_p (float): 核采样参数
            frequency_penalty (float): 频率惩罚参数
            presence_penalty (float): 存在惩罚参数
            **kwargs: 其他可选参数

        Returns:
            response: OpenAI API的原始响应对象

        Raises:
            Exception: 当API调用失败时抛出异常
        """
        try:
            # 确保所有数值参数都是正确的类型（处理 kwargs 中可能存在的字符串类型参数）
            # 如果 kwargs 中有这些参数，优先使用 kwargs 中的值，并确保类型正确
            # 使用 Python 原生类型，确保 API 兼容性
            
            # 处理 temperature 参数
            if "temperature" in kwargs:
                temp_val = kwargs.pop("temperature")
                temperature = float(temp_val) if temp_val is not None else 0.9
            else:
                temperature = float(temperature) if temperature is not None else 0.9
            # 确保是 Python 原生 float 类型（处理 numpy 类型等）
            try:
                temperature = float(temperature)
            except (TypeError, ValueError):
                temperature = 0.9
            
            # 处理 max_tokens 参数
            if "max_tokens" in kwargs:
                tokens_val = kwargs.pop("max_tokens")
                max_tokens = int(tokens_val) if tokens_val is not None else 8192
            else:
                max_tokens = int(max_tokens) if max_tokens is not None else 8192
            # 确保是 Python 原生 int 类型
            try:
                max_tokens = int(max_tokens)
            except (TypeError, ValueError):
                max_tokens = 8192
            
            # 处理 top_p 参数
            if "top_p" in kwargs:
                top_p_val = kwargs.pop("top_p")
                top_p = float(top_p_val) if top_p_val is not None else 0.9
            else:
                top_p = float(top_p) if top_p is not None else 0.9
            # 确保是 Python 原生 float 类型
            try:
                top_p = float(top_p)
            except (TypeError, ValueError):
                top_p = 0.9
            
            # 处理 frequency_penalty 参数
            if "frequency_penalty" in kwargs:
                freq_val = kwargs.pop("frequency_penalty")
                frequency_penalty = float(freq_val) if freq_val is not None else 0.5
            else:
                frequency_penalty = float(frequency_penalty) if frequency_penalty is not None else 0.5
            # 确保是 Python 原生 float 类型
            try:
                frequency_penalty = float(frequency_penalty)
            except (TypeError, ValueError):
                frequency_penalty = 0.5
            
            # 处理 presence_penalty 参数
            if "presence_penalty" in kwargs:
                pres_val = kwargs.pop("presence_penalty")
                presence_penalty = float(pres_val) if pres_val is not None else 0.5
            else:
                presence_penalty = float(presence_penalty) if presence_penalty is not None else 0.5
            # 确保是 Python 原生 float 类型
            try:
                presence_penalty = float(presence_penalty)
            except (TypeError, ValueError):
                presence_penalty = 0.5
            
            # 最终验证：确保所有参数都是正确的 Python 原生类型
            # 如果 kwargs 中还有这些参数，也要处理（防止遗漏）
            for param_name, param_type, default_val in [
                ("temperature", float, 0.9),
                ("max_tokens", int, 8192),
                ("top_p", float, 0.9),
                ("frequency_penalty", float, 0.5),
                ("presence_penalty", float, 0.5),
            ]:
                if param_name in kwargs:
                    try:
                        if param_type == float:
                            kwargs[param_name] = float(kwargs[param_name])
                        elif param_type == int:
                            kwargs[param_name] = int(kwargs[param_name])
                    except (TypeError, ValueError):
                        kwargs[param_name] = default_val
            
            # 关键：确保 kwargs 中没有 temperature 等参数，避免覆盖
            # 这些参数应该作为位置参数传递，而不是在 kwargs 中
            for param_name in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"]:
                if param_name in kwargs:
                    # 如果 kwargs 中有这些参数，移除它们（使用位置参数的值）
                    kwargs.pop(param_name)
            
            # 最终类型验证和转换（强制转换为 Python 原生类型）
            # 使用 JSON 序列化/反序列化来确保类型是 JSON 兼容的 Python 原生类型
            try:
                # 先转换为 Python 原生类型
                temperature = float(temperature)
                max_tokens = int(max_tokens)
                top_p = float(top_p)
                frequency_penalty = float(frequency_penalty)
                presence_penalty = float(presence_penalty)
                
                # 使用 JSON 序列化/反序列化来强制转换为 JSON 原生类型
                # 这样可以确保即使是 numpy 类型也会被转换为 Python 原生类型
                test_params = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "frequency_penalty": frequency_penalty,
                    "presence_penalty": presence_penalty,
                }
                # 序列化后再反序列化，确保所有值都是 JSON 原生类型
                params_json = json.dumps(test_params)
                params_dict = json.loads(params_json)
                temperature = params_dict["temperature"]
                max_tokens = params_dict["max_tokens"]
                top_p = params_dict["top_p"]
                frequency_penalty = params_dict["frequency_penalty"]
                presence_penalty = params_dict["presence_penalty"]
            except (TypeError, ValueError, OverflowError) as e:
                # 如果转换失败，使用默认值
                print(f"[警告] 参数类型转换失败: {e}, 使用默认值", file=sys.stderr, flush=True)
                temperature = 0.9
                max_tokens = 8192
                top_p = 0.9
                frequency_penalty = 0.5
                presence_penalty = 0.5
            
            # 构建完整的 API 参数字典，确保所有参数都是 Python 原生类型
            # 这是最关键的步骤：使用字典构建参数，然后通过 JSON 序列化/反序列化确保类型正确
            api_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
            }
            
            # 添加 kwargs 中的其他参数（确保没有重复的参数）
            for key, value in kwargs.items():
                if key not in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "model", "messages"]:
                    api_params[key] = value
            
            # 最终类型验证：使用 JSON 序列化/反序列化来确保所有数值参数都是 Python 原生类型
            # 这是最可靠的方法，可以处理所有特殊情况（numpy、Decimal 等）
            try:
                # 只对数值参数进行 JSON 序列化/反序列化
                numeric_params = {
                    "temperature": api_params["temperature"],
                    "max_tokens": api_params["max_tokens"],
                    "top_p": api_params["top_p"],
                    "frequency_penalty": api_params["frequency_penalty"],
                    "presence_penalty": api_params["presence_penalty"],
                }
                numeric_json = json.dumps(numeric_params)
                numeric_dict = json.loads(numeric_json)
                # 更新参数字典中的数值参数
                api_params["temperature"] = float(numeric_dict["temperature"])
                api_params["max_tokens"] = int(numeric_dict["max_tokens"])
                api_params["top_p"] = float(numeric_dict["top_p"])
                api_params["frequency_penalty"] = float(numeric_dict["frequency_penalty"])
                api_params["presence_penalty"] = float(numeric_dict["presence_penalty"])
            except Exception:
                # 如果 JSON 转换失败，使用直接转换（应该不会发生，但作为后备）
                api_params["temperature"] = float(api_params["temperature"])
                api_params["max_tokens"] = int(api_params["max_tokens"])
                api_params["top_p"] = float(api_params["top_p"])
                api_params["frequency_penalty"] = float(api_params["frequency_penalty"])
                api_params["presence_penalty"] = float(api_params["presence_penalty"])
            
            # 最终验证：确保所有数值参数都是 Python 内置类型（不是 numpy 等）
            # 使用 type() 检查模块，确保是 builtins 模块的类型
            for param_name in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                param_value = api_params[param_name]
                param_type = type(param_value)
                # 如果不是 Python 内置的 float 类型，强制转换
                if param_type.__module__ != 'builtins' or not isinstance(param_value, float):
                    api_params[param_name] = float(param_value)
                    # 调试信息（只在出错时打印）
                    if param_type.__module__ != 'builtins':
                        print(f"[调试] {param_name} 类型转换: {param_type} -> float", file=sys.stderr, flush=True)
            
            # 确保 max_tokens 是 Python 内置的 int 类型
            if type(api_params["max_tokens"]).__module__ != 'builtins' or not isinstance(api_params["max_tokens"], int):
                api_params["max_tokens"] = int(api_params["max_tokens"])
            
            # 关键修复：使用显式参数传递，而不是 **api_params
            # 这样可以确保 OpenAI 客户端接收到的是正确的类型
            # 提取所有参数
            model = api_params.pop("model")
            messages = api_params.pop("messages")
            temperature = api_params.pop("temperature")
            max_tokens = api_params.pop("max_tokens")
            top_p = api_params.pop("top_p")
            frequency_penalty = api_params.pop("frequency_penalty")
            presence_penalty = api_params.pop("presence_penalty")
            
            # 最终类型强制转换（确保是 Python 内置类型）
            # 使用最严格的方式：先转换为 Python 原生类型，然后通过 JSON 验证
            try:
                # 构建临时字典进行 JSON 序列化/反序列化验证
                temp_check = {
                    "t": float(temperature),
                    "m": int(max_tokens),
                    "tp": float(top_p),
                    "fp": float(frequency_penalty),
                    "pp": float(presence_penalty),
                }
                temp_json = json.dumps(temp_check)
                temp_dict = json.loads(temp_json)
                temperature = float(temp_dict["t"])
                max_tokens = int(temp_dict["m"])
                top_p = float(temp_dict["tp"])
                frequency_penalty = float(temp_dict["fp"])
                presence_penalty = float(temp_dict["pp"])
            except Exception as e:
                # 如果 JSON 验证失败，使用直接转换
                print(f"[警告] JSON 验证失败: {e}, 使用直接转换", file=sys.stderr, flush=True)
                temperature = float(temperature) if temperature is not None else 0.9
                max_tokens = int(max_tokens) if max_tokens is not None else 8192
                top_p = float(top_p) if top_p is not None else 0.9
                frequency_penalty = float(frequency_penalty) if frequency_penalty is not None else 0.5
                presence_penalty = float(presence_penalty) if presence_penalty is not None else 0.5
            
            # 最终验证：确保所有参数都是 Python 内置类型
            # 检查类型模块，确保不是 numpy 等外部类型
            if type(temperature).__module__ != 'builtins':
                temperature = float(temperature)
            if type(max_tokens).__module__ != 'builtins':
                max_tokens = int(max_tokens)
            if type(top_p).__module__ != 'builtins':
                top_p = float(top_p)
            if type(frequency_penalty).__module__ != 'builtins':
                frequency_penalty = float(frequency_penalty)
            if type(presence_penalty).__module__ != 'builtins':
                presence_penalty = float(presence_penalty)
            
            # 关键修复：在调用 API 之前，最终强制转换为 Python 原生类型
            # 使用 type() 检查确保是 builtins 模块的类型，然后强制转换
            # 这是最可靠的方法，可以处理所有特殊情况
            temperature = float(temperature)
            max_tokens = int(max_tokens)
            top_p = float(top_p)
            frequency_penalty = float(frequency_penalty)
            presence_penalty = float(presence_penalty)
            
            # 验证：确保转换后的类型是 Python 内置类型
            # 如果不是，说明转换失败，使用默认值
            if not isinstance(temperature, float) or type(temperature).__module__ != 'builtins':
                print(f"[严重错误] temperature 类型错误: {type(temperature)}, 值: {temperature}, 使用默认值 0.9", file=sys.stderr, flush=True)
                temperature = 0.9
            if not isinstance(max_tokens, int) or type(max_tokens).__module__ != 'builtins':
                print(f"[严重错误] max_tokens 类型错误: {type(max_tokens)}, 值: {max_tokens}, 使用默认值 8192", file=sys.stderr, flush=True)
                max_tokens = 8192
            if not isinstance(top_p, float) or type(top_p).__module__ != 'builtins':
                print(f"[严重错误] top_p 类型错误: {type(top_p)}, 值: {top_p}, 使用默认值 0.9", file=sys.stderr, flush=True)
                top_p = 0.9
            if not isinstance(frequency_penalty, float) or type(frequency_penalty).__module__ != 'builtins':
                print(f"[严重错误] frequency_penalty 类型错误: {type(frequency_penalty)}, 值: {frequency_penalty}, 使用默认值 0.5", file=sys.stderr, flush=True)
                frequency_penalty = 0.5
            if not isinstance(presence_penalty, float) or type(presence_penalty).__module__ != 'builtins':
                print(f"[严重错误] presence_penalty 类型错误: {type(presence_penalty)}, 值: {presence_penalty}, 使用默认值 0.5", file=sys.stderr, flush=True)
                presence_penalty = 0.5
            
            # 最终检查：确保 api_params 中没有这些参数（避免重复）
            for param in ["temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty", "model", "messages"]:
                api_params.pop(param, None)
            
            # 关键修复：某些 API 服务器（如 qwen、deepseek、claude）要求 temperature 必须在 0-1 范围内
            # 如果 temperature 超出范围，限制在 0-1 之间
            # 对所有模型都进行范围检查，确保兼容性
            if temperature > 1.0:
                print(f"[警告] 模型 {self.model_name} 不支持 temperature > 1.0，当前值: {temperature}，限制为 1.0", file=sys.stderr, flush=True)
                temperature = 1.0
            elif temperature < 0.0:
                print(f"[警告] 模型 {self.model_name} 不支持 temperature < 0.0，当前值: {temperature}，限制为 0.0", file=sys.stderr, flush=True)
                temperature = 0.0
            
            # 调试信息已关闭（如需调试，可以取消注释）
            # if self.model_name in ["qwen-plus", "deepseek-v3.2"]:
            #     print(f"[调试] 模型: {self.model_name}, temperature 类型: {type(temperature)}, 值: {temperature}", file=sys.stderr, flush=True)
            
            # 关键修复：清理 messages 中最后一个 assistant 消息的尾随空白
            # 某些 API 服务器（如 Claude）要求 final assistant content 不能以尾随空白结尾
            if messages and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict) and last_message.get("role") == "assistant":
                    content = last_message.get("content")
                    if isinstance(content, str):
                        # 移除尾随空白字符（空格、制表符、换行符等）
                        last_message["content"] = content.rstrip()
            
            # 使用OpenAI客户端发送聊天完成请求
            # 使用显式参数传递，确保所有参数都是 Python 原生类型
            # 注意：不使用 **api_params，而是只传递必要的参数，避免任何可能的类型问题
            create_kwargs = {}
            for key, value in api_params.items():
                # 确保 kwargs 中的值也是正确的类型
                if key in ["temperature", "top_p", "frequency_penalty", "presence_penalty"]:
                    create_kwargs[key] = float(value) if value is not None else (0.9 if key == "temperature" or key == "top_p" else 0.5)
                elif key == "max_tokens":
                    create_kwargs[key] = int(value) if value is not None else 8192
                else:
                    create_kwargs[key] = value
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **create_kwargs,  # 其他参数（已经确保类型正确）
            )
            return response
        except Exception as e:
            # 记录API错误信息，包括更详细的错误详情
            error_msg = str(e)
            error_details = []
            
            # 尝试获取状态码
            status_code = None
            if hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                status_code = e.response.status_code
            elif hasattr(e, 'code'):
                status_code = e.code
            
            # 尝试获取错误详情（OpenAI 客户端可能使用不同的属性名）
            error_body = None
            # 尝试多种可能的属性名
            for attr_name in ['body', 'message', 'error', 'response']:
                if hasattr(e, attr_name):
                    attr_value = getattr(e, attr_name)
                    if attr_value:
                        try:
                            if isinstance(attr_value, dict):
                                error_body = attr_value
                                break
                            elif isinstance(attr_value, str):
                                error_body = json.loads(attr_value)
                                break
                        except:
                            if isinstance(attr_value, str) and len(attr_value) < 500:
                                error_body = attr_value
                                break
            
            # 如果还没有获取到，尝试从 response 对象获取
            if not error_body and hasattr(e, 'response'):
                try:
                    if hasattr(e.response, 'json'):
                        error_body = e.response.json()
                    elif hasattr(e.response, 'text'):
                        try:
                            error_body = json.loads(e.response.text)
                        except:
                            error_body = e.response.text
                except:
                    pass
            
            # 构建错误信息
            if status_code:
                error_details.append(f"Status: {status_code}")
            if error_body:
                if isinstance(error_body, dict):
                    error_details.append(f"Error: {error_body}")
                else:
                    error_details.append(f"Error: {error_body}")
            
            # 打印详细的错误信息（使用 stderr 确保立即输出）
            if error_details:
                error_output = f"[API错误] {' | '.join(error_details)}"
                print(error_output, file=sys.stderr, flush=True)
            else:
                print(f"[API错误] {error_msg}", file=sys.stderr, flush=True)
                # 打印异常类型和所有属性，帮助调试
                print(f"[API错误] 异常类型: {type(e).__name__}", file=sys.stderr, flush=True)
                if hasattr(e, '__dict__'):
                    print(f"[API错误] 异常属性: {e.__dict__}", file=sys.stderr, flush=True)
            
            raise  # 重新抛出异常，触发重试机制

    def get_response(
        self,
        user_input=None,
        system_prompt=None,
        temperature=0.9,
        max_tokens=4096,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5,
        debug=False,
        messages=None,  # 新增 messages 参数
        **kwargs,
    ):
        """
        获取AI的响应，支持多种输入模式

        该方法是主要的对外接口，支持传入单个用户输入或完整的对话消息列表。
        具有灵活的参数配置和错误处理机制。

        Args:
            user_input (str, optional): 用户输入的文本内容
            system_prompt (str, optional): 系统提示词，默认使用初始化时的提示词
            temperature (float): 温度参数，控制输出的随机性 (0-1)
            max_tokens (int): 最大生成token数量
            top_p (float): 核采样参数 (0-1)
            frequency_penalty (float): 频率惩罚参数 (-2.0-2.0)
            presence_penalty (float): 存在惩罚参数 (-2.0-2.0)
            debug (bool): 是否开启调试模式，会打印响应内容
            messages (list, optional): 完整的对话消息列表
            **kwargs: 其他传递给API的参数

        Returns:
            dict: 包含响应内容和token使用情况的字典，
                  或在出错时返回包含error字段的字典
        """
        try:
            # 使用默认系统提示词（如果未提供）
            if system_prompt is None:
                system_prompt = self.default_system_prompt

            # 初始化消息列表（如果未提供）
            if messages is None:
                messages = []

            # 检查并添加系统提示词（如果不存在）
            if not any(msg["role"] == "system" for msg in messages):
                messages.insert(0, {"role": "system", "content": system_prompt})

            # 添加用户输入到消息列表（如果提供了user_input）
            if user_input is not None:
                messages.append({"role": "user", "content": user_input})

            # 清理kwargs中的messages参数，避免参数冲突
            if "messages" in kwargs:
                kwargs.pop("messages")

            # 关键修复：在传递给 __call_api 之前，确保所有数值参数都是正确的类型
            # 这样可以避免类型问题在传递过程中累积
            try:
                temperature = float(temperature) if temperature is not None else 0.9
                max_tokens = int(max_tokens) if max_tokens is not None else 4096
                top_p = float(top_p) if top_p is not None else 0.9
                frequency_penalty = float(frequency_penalty) if frequency_penalty is not None else 0.5
                presence_penalty = float(presence_penalty) if presence_penalty is not None else 0.5
            except (TypeError, ValueError):
                # 如果转换失败，使用默认值
                temperature = 0.9
                max_tokens = 4096
                top_p = 0.9
                frequency_penalty = 0.5
                presence_penalty = 0.5
            
            # 确保 kwargs 中的这些参数也是正确的类型（如果存在）
            for param_name, param_type, default_val in [
                ("temperature", float, 0.9),
                ("max_tokens", int, 4096),
                ("top_p", float, 0.9),
                ("frequency_penalty", float, 0.5),
                ("presence_penalty", float, 0.5),
            ]:
                if param_name in kwargs:
                    try:
                        if param_type == float:
                            kwargs[param_name] = float(kwargs[param_name])
                        elif param_type == int:
                            kwargs[param_name] = int(kwargs[param_name])
                    except (TypeError, ValueError):
                        kwargs[param_name] = default_val

            # 调用底层API接口
            response = self.__call_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                **kwargs,
            )

            # 处理API响应，提取有用信息
            result = self.__post_process(response)

            # 调试模式：以绿色文字打印响应内容
            if debug:
                print("\033[92m" + f"[响应] {result['response']}" + "\033[0m")

            # 返回处理后的结果
            return result

        except Exception as e:
            # 错误处理：以红色文字打印错误信息
            print("\033[91m" + f"[错误] {str(e)}" + "\033[0m")
            return {"error": f"Error: {str(e)}"}


# ============================ 模块初始化和测试 ============================
# 记录模块加载的开始时间
start_time = time.time()
start_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
print("模块加载开始时间:", start_date)
