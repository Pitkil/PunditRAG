FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
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
RUN sed -i '/^\[\[tool\.uv\.index\]\]/,/^default = true$/d' pyproject.toml \
    && rm -f uv.lock \
    && uv sync --no-dev --no-install-project --default-index https://mirrors.aliyun.com/pypi/simple/

COPY . .

CMD ["sh", "-lc", "${START_COMMAND:-sleep infinity}"]
