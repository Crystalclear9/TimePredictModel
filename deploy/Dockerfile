# CPU 预测服务镜像；SGLang GPU 服务需要单独部署。
FROM python:3.11-slim
# libgomp1 支持 XGBoost；安装 chrony 工具不代表容器内已自动完成校时。
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 chrony \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# 复制包配置和源码后安装，未复制数据集、模型包或语义编码器缓存。
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir .
USER 65532:65532
# 运行参数由容器启动时提供；认证文件、Redis 地址和模型目录需另行配置。
ENTRYPOINT ["timepredict"]
