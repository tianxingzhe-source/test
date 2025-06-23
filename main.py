from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 定义请求参数模型
class TestRequest(BaseModel):
    message: str = "Hello from client"  # 可选参数，默认值为示例

# 定义响应模型
class TestResponse(BaseModel):
    status: str = "success"
    data: str
    message: str = "操作完成"

# 初始化FastAPI应用
app = FastAPI(
    title="MCP测试工具API",
    description="为MCP项目提供测试功能的API服务",
    version="1.0.0"
)

# 原始函数逻辑
def test_function(message: str) -> str:
    """核心测试函数，返回处理后的字符串"""
    return f"Hello from test! 接收到的消息: {message}"

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

# 主函数：启动服务
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
