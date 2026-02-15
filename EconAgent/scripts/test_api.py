#!/usr/bin/env python3
"""测试API连接和认证"""

import json
import sys
from pathlib import Path

# 尝试导入openai
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("错误: openai库未安装")
    sys.exit(1)

def test_api(config_path):
    """测试API连接"""
    # 加载配置
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "https://api.openai.com/v1")
    model = config.get("model", "gpt-4o")
    timeout = config.get("timeout", 60)
    
    print("="*60)
    print("API连接测试")
    print("="*60)
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Timeout: {timeout}秒")
    print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
    print()
    
    # 检查openai版本
    try:
        from openai import OpenAI
        use_new_api = True
        print("检测到新版本openai库（>=1.0）")
    except (ImportError, AttributeError):
        use_new_api = False
        print("检测到旧版本openai库（<1.0）")
    
    # 设置客户端
    try:
        if use_new_api:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout
            )
        else:
            openai.api_key = api_key
            openai.api_base = base_url
            client = openai
    except Exception as e:
        print(f"错误: 无法创建客户端: {e}")
        return False
    
    # 测试1: 简单的聊天请求
    print("\n测试1: 发送简单聊天请求...")
    try:
        messages = [{"role": "user", "content": "Hello, please reply with 'OK'"}]
        
        if use_new_api:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                timeout=timeout
            )
            if response.choices:
                reply = response.choices[0].message.content
            else:
                reply = None
        else:
            # 旧版本
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    timeout=timeout
                )
            except TypeError:
                # 不支持timeout参数
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=0.7
                )
            
            if response.choices:
                reply = response.choices[0].message.content
            else:
                reply = None
        
        if reply:
            print(f"✓ 成功! 响应: {reply[:100]}")
            return True
        else:
            print("✗ 失败: API返回空响应")
            return False
            
    except Exception as e:
        print(f"✗ 失败: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test_api.py <config_path>")
        print("示例: python test_api.py mast_config.json")
        sys.exit(1)
    
    config_path = sys.argv[1]
    success = test_api(config_path)
    sys.exit(0 if success else 1)

