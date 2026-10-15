# 为什么用 slim：官方精简镜像，体积约为完整版的 1/4，攻击面也更小
FROM python:3.12-slim

WORKDIR /app

# 为什么先拷 requirements：依赖不变时这一层直接命中缓存，
# 改业务代码不用重装依赖，构建从分钟级降到秒级
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 为什么设置 PYTHONUNBUFFERED：让日志实时刷到 stdout，
# 否则 docker logs 会看到"延迟几分钟"的假象（Python 默认缓冲）
ENV PYTHONUNBUFFERED=1

# 为什么用非 root 用户运行：最小权限原则，
# 万一容器被攻破，攻击者拿到的也只是普通用户权限
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 为什么 0.0.0.0：容器里只听 127.0.0.1 的话，端口映射出去也访问不到
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
