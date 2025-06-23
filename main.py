from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import time
import json
import re
import datetime
import math
from typing import List, Optional, Dict, Any, Callable

import inspect

# 定义请求参数模型
class TestRequest(BaseModel):
    message: str = "Hello from client"  # 可选参数，默认值为示例

# 定义响应模型
class TestResponse(BaseModel):
    status: str = "success"
    data: str
    message: str = "操作完成"

# 定义MCP标准模型结构
class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str
    permission: List = []

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# MCP标准请求体
class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = Field(100, gt=0)
    temperature: float = Field(0.7, ge=0, le=1.0)

# MCP标准响应体
class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: str

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage

# 工具请求与响应模型
class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any] = {}

class ToolResponse(BaseModel):
    status: str = "success"
    tool_name: str
    result: Any
    execution_time: float

class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    return_type: str

class ToolsList(BaseModel):
    tools: List[ToolInfo]

# 模型包装器
class ModelWrapper:
    def __init__(self):
        # 初始化模型
        self.model = self._load_model()
    
    def _load_model(self):
        # 实现模型加载逻辑
        return {"loaded": True}
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[str, Any]:
        # 实现模型推理逻辑
        # 这里返回符合MCP规范的响应
        return {
            "id": f"response-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "test-model",
            "choices": [
                {
                    "text": f"这是模型生成的内容，响应提示：{prompt}",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": 50,
                "total_tokens": len(prompt.split()) + 50
            }
        }

# 工具注册表
tools_registry = {}

# 工具装饰器
def tool(func: Callable) -> Callable:
    """
    将函数注册为AI工具
    """
    # 从函数签名中提取参数信息
    signature = inspect.signature(func)
    
    # 构建参数信息
    parameters = {}
    for name, param in signature.parameters.items():
        param_info = {
            "type": str(param.annotation.__name__) if param.annotation != inspect.Parameter.empty else "any",
            "required": param.default == inspect.Parameter.empty
        }
        if param.default != inspect.Parameter.empty and param.default is not None:
            param_info["default"] = param.default
        parameters[name] = param_info
    
    # 构建返回类型信息
    return_type = "any"
    if signature.return_annotation != inspect.Signature.empty:
        return_type = str(signature.return_annotation.__name__)
    
    # 注册工具
    tools_registry[func.__name__] = {
        "function": func,
        "description": func.__doc__ or "无描述",
        "parameters": parameters,
        "return_type": return_type
    }
    
    return func

# 定义工具函数
@tool
def test_function(message: str) -> str:
    """核心测试函数，返回处理后的字符串"""
    return f"Hello from test! 接收到的消息: {message}"

@tool
def calculate_date_difference(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    计算两个日期之间的差异
    
    参数:
    start_date: 开始日期，格式为 YYYY-MM-DD
    end_date: 结束日期，格式为 YYYY-MM-DD
    
    返回:
    包含天数差、工作日数、周数等信息的字典
    """
    try:
        start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        
        delta = end - start
        days = delta.days
        
        # 简单计算工作日(不考虑假期)
        weeks, remainder = divmod(days, 7)
        weekdays = min(remainder, 5) + min(days // 7 * 5, 5)
        
        return {
            "days": days,
            "weekdays": weekdays, 
            "weeks": weeks,
            "months": days / 30.44,  # 平均月长度
            "years": days / 365.25,  # 平均年长度(考虑闰年)
        }
    except Exception as e:
        raise ValueError(f"日期计算错误: {str(e)}")

@tool
def get_current_weather(city: str, country_code: str = "CN") -> Dict[str, Any]:
    """
    获取指定城市的当前天气信息
    
    参数:
    city: 城市名称
    country_code: 国家代码，默认为CN(中国)
    
    返回:
    包含天气信息的字典
    """
    # 模拟天气API调用
    weather_data = {
        "city": city,
        "country": country_code,
        "temperature": round(10 + 15 * math.sin(time.time() / 10000) + 5 * (hash(city) % 10) / 10, 1),
        "conditions": ["晴朗", "多云", "小雨", "大雨", "雷雨", "雾"][hash(city) % 6],
        "humidity": round(40 + 40 * math.sin(time.time() / 20000 + hash(city)), 1),
        "wind": {
            "speed": round(5 + 5 * math.sin(time.time() / 15000 + hash(city)), 1),
            "direction": ["北", "东北", "东", "东南", "南", "西南", "西", "西北"][hash(city) % 8]
        },
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    return weather_data

@tool
def perform_calculation(expression: str) -> Dict[str, Any]:
    """
    执行数学计算
    
    参数:
    expression: 要计算的表达式，如 "2 + 2 * 3"
    
    返回:
    计算结果
    """
    try:
        # 安全地评估表达式，只允许基本算术运算
        allowed_chars = set("0123456789+-*/() .")
        if not all(c in allowed_chars for c in expression):
            raise ValueError("表达式包含不允许的字符")
            
        # 使用eval计算表达式(在实际产品中应该使用更安全的方法)
        result = eval(expression)
        
        return {
            "expression": expression,
            "result": result,
            "type": type(result).__name__
        }
    except Exception as e:
        raise ValueError(f"计算错误: {str(e)}")

@tool
def summarize_text(text: str, max_length: int = 100) -> str:
    """
    对文本进行简单摘要
    
    参数:
    text: 要摘要的文本
    max_length: 摘要的最大长度，默认100字符
    
    返回:
    文本摘要
    """
    if len(text) <= max_length:
        return text
        
    # 简单实现：截取开头部分加省略号
    summary = text[:max_length - 3] + "..."
    return summary

@tool
def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    从文本中提取实体信息(简单实现)
    
    参数:
    text: 要分析的文本
    
    返回:
    包含不同类型实体的字典
    """
    entities = {
        "dates": [],
        "emails": [],
        "urls": [],
        "numbers": []
    }
    
    # 提取日期(简单模式YYYY-MM-DD)
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    entities["dates"] = re.findall(date_pattern, text)
    
    # 提取邮箱
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    entities["emails"] = re.findall(email_pattern, text)
    
    # 提取URL
    url_pattern = r'https?://[^\s]+'
    entities["urls"] = re.findall(url_pattern, text)
    
    # 提取数字
    number_pattern = r'\b\d+\.?\d*\b'
    entities["numbers"] = re.findall(number_pattern, text)
    
    return entities

# 初始化FastAPI应用
app = FastAPI(
    title="MCP测试工具API",
    description="为MCP项目提供测试功能的API服务，符合MCP标准",
    version="1.0.0"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加自定义中间件处理UTF-8编码
@app.middleware("http")
async def add_utf8_charset(request: Request, call_next):
    """确保所有JSON响应使用UTF-8编码"""
    response = await call_next(request)
    if response.headers.get("content-type") == "application/json":
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

# 自定义响应处理
class UTF8JSONResponse(JSONResponse):
    """确保JSON响应使用UTF-8编码"""
    media_type = "application/json; charset=utf-8"
    
    def render(self, content):
        return json.dumps(
            content,
            ensure_ascii=False,  # 确保不将非ASCII字符转义为\uXXXX
            allow_nan=False,
            indent=None,
            separators=(",", ":")
        ).encode("utf-8")

# 实例化模型
model = ModelWrapper()

# 依赖项：获取模型实例
def get_model():
    return model

# API路由：GET请求
@app.get("/test", response_model=TestResponse, tags=["测试功能"], response_class=UTF8JSONResponse)
def run_test_get(message: str = "默认消息"):
    """
    通过GET方式调用测试函数  
    - **message**: 自定义消息（可选）
    """
    try:
        result = test_function(message)
        return TestResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部错误: {str(e)}")

# API路由：POST请求（支持更复杂的参数）
@app.post("/test", response_model=TestResponse, tags=["测试功能"], response_class=UTF8JSONResponse)
def run_test_post(request: TestRequest):
    """
    通过POST方式调用测试函数  
    - **request.message**: 客户端发送的消息
    """
    try:
        result = test_function(request.message)
        return TestResponse(data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")

# 健康检查接口
@app.get("/health", tags=["系统状态"], response_class=UTF8JSONResponse)
def health_check():
    """系统健康检查接口"""
    return {"status": "ok", "message": "服务运行正常"}

# MCP标准接口实现
@app.get("/v1/models", response_model=ModelList, tags=["MCP标准接口"], response_class=UTF8JSONResponse)
def list_models():
    """获取可用模型列表"""
    return {
        "object": "list",
        "data": [
            {
                "id": "test-model",
                "object": "model",
                "owned_by": "test-organization",
                "permission": []
            }
        ]
    }

@app.post("/v1/completions", response_model=CompletionResponse, tags=["MCP标准接口"], response_class=UTF8JSONResponse)
def create_completion(request: CompletionRequest, model_wrapper: ModelWrapper = Depends(get_model)):
    """
    创建文本完成请求
    - **model**: 模型名称
    - **prompt**: 提示文本
    - **max_tokens**: 最大生成令牌数
    - **temperature**: 温度参数，控制随机性
    """
    try:
        response = model_wrapper.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")

# 工具系统API接口
@app.get("/tools", response_model=ToolsList, tags=["工具系统"], response_class=UTF8JSONResponse)
def list_tools():
    """获取所有可用工具列表"""
    tools_list = []
    for name, info in tools_registry.items():
        tools_list.append({
            "name": name,
            "description": info["description"],
            "parameters": info["parameters"],
            "return_type": info["return_type"]
        })
    return {"tools": tools_list}

@app.post("/tools/{tool_name}", response_model=ToolResponse, tags=["工具系统"], response_class=UTF8JSONResponse)
def execute_tool(tool_name: str, parameters: Dict[str, Any] = {}):
    """
    执行指定的工具
    - **tool_name**: 工具名称
    - **parameters**: 工具参数
    """
    if tool_name not in tools_registry:
        raise HTTPException(status_code=404, detail=f"工具 '{tool_name}' 不存在")
    
    try:
        start_time = time.time()
        tool_func = tools_registry[tool_name]["function"]
        result = tool_func(**parameters)
        execution_time = time.time() - start_time
        
        return {
            "status": "success",
            "tool_name": tool_name,
            "result": result,
            "execution_time": round(execution_time, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工具执行失败: {str(e)}")

@app.post("/tools", response_model=ToolResponse, tags=["工具系统"], response_class=UTF8JSONResponse)
def execute_tool_by_name(request: ToolRequest):
    """
    通过名称和参数执行工具
    - **request.tool_name**: 工具名称
    - **request.parameters**: 工具参数
    """
    return execute_tool(request.tool_name, request.parameters)

# 主函数：启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")