FROM python:3.11-slim

# 压测本身是 IO 密集，镜像尽量小；nvidia-smi 由宿主机驱动提供，
# 容器内若需要显存监控要挂 nvidia runtime，见 README
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖，利用镜像层缓存：代码改动不会导致重装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 结果目录挂出来，容器重启不丢压测数据
RUN mkdir -p /app/data/results
VOLUME ["/app/data"]

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/health')"

CMD ["python", "scripts/serve.py", "--host", "0.0.0.0", "--port", "8080"]
