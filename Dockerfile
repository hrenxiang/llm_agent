# 使用官方的 Python 3.10 版本作为基础镜像
FROM python:3.10

# 设置工作目录
WORKDIR /app

# 复制项目文件到容器中
COPY . /app

# 安装系统依赖和 pip 加速
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    build-essential \
    wget \
    curl \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# 复制 pyproject.toml 和 poetry.lock 文件
COPY pyproject.toml poetry.lock /app/

# 安装 pip 以及 Poetry（使用国内源加速）
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple poetry \
    && pip install -r requirements.txt --no-cache-dir

# 直接在docker容器内全局安装项目依赖
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 运行 FastAPI 应用，使用 uvicorn 作为服务器
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]