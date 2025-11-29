FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY *.py ./
COPY *.csv ./
COPY *.json ./

# 安装Python依赖
RUN uv sync

# 创建必要的目录
RUN mkdir -p data reports cache_market

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai

# 默认命令
CMD ["uv", "run", "python", "fupan.py", "--date", "$(date +%Y%m%d)"]
