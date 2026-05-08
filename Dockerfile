FROM rust:1.90-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV VIRTUAL_ENV=/home/reviewer/.venv
ENV PATH="/home/reviewer/.venv/bin:${PATH}"

RUN rustup toolchain install nightly
RUN rustup default nightly

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    protobuf-compiler \
    ca-certificates \
    libgmp-dev \
    libmpfr-dev \
    libmpc-dev \
    ncat \
    time \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/reviewer/app

# Install circuit Python dependencies from code/circuit/pyproject.toml.
RUN python3 -m venv "${VIRTUAL_ENV}"
RUN pip install --upgrade pip setuptools wheel
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0
RUN pip install \
    numpy==2.3.3 \
    onnx==1.18.0 \
    "onnxruntime>=1.24.4" \
    optimum==1.27.0 \
    python-dotenv==1.1.1 \
    transformers==4.56.1

CMD ["/bin/bash"]
