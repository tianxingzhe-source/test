from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import time
from typing import List, Optional, Dict, Any

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

# 原始函数逻辑
def test_function(message: str) -> str:
    """核心测试函数，返回处理后的字符串"""
    return f"Hello from test! 接收到的消息: {message}"

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

# 实例化模型
model = ModelWrapper()

# 依赖项：获取模型实例
def get_model():
    return model

# API路由：GET请求
@app.get("/test", response_model=TestResponse, tags=["测试功能"])
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
@app.post("/test", response_model=TestResponse, tags=["测试功能"])
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
@app.get("/health", tags=["系统状态"])
def health_check():
    """系统健康检查接口"""
    return {"status": "ok", "message": "服务运行正常"}

# MCP标准接口实现
@app.get("/v1/models", response_model=ModelList, tags=["MCP标准接口"])
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

@app.post("/v1/completions", response_model=CompletionResponse, tags=["MCP标准接口"])
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

# 主函数：启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")