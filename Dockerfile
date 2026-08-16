FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_PROJECT_ENVIRONMENT=/usr/local \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.aliyun.com/debian|g; s|http://deb.debian.org/debian-security|https://mirrors.aliyun.com/debian-security|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv==0.8.5

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    sed -i '/^\[\[tool\.uv\.index\]\]/,/^default = true$/d' pyproject.toml \
    && sed -i 's|https://pypi.tuna.tsinghua.edu.cn|https://mirrors.aliyun.com/pypi|g' uv.lock \
    && uv sync --frozen --no-dev --no-install-project --default-index https://mirrors.aliyun.com/pypi/simple/ \
    && uv pip install --system --default-index https://mirrors.aliyun.com/pypi/simple/ \
        beautifulsoup4==4.14.3 \
        openpyxl==3.1.5 \
        python-docx==1.2.0 \
        python-pptx==1.0.2

COPY . .

CMD ["sh", "-lc", "${START_COMMAND:-sleep infinity}"]
