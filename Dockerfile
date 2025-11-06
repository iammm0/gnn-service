# 使用官方 Python 镜像作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制并安装项目依赖
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码到容器内
COPY . /app/

# 复制预下载的模型到容器
COPY models/bert-base-chinese /app/models/bert-base-chinese

# 开放 FastAPI 默认的端口
EXPOSE 8000

# 运行 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]