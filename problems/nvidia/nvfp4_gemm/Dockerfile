FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Install SSH server and dependencies
RUN apt-get update && \
    apt-get install -y openssh-server git curl ca-certificates nano && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure SSH (exactly like docs)
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    ssh-keygen -A

# Create SSH directory for root
RUN mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

# Install NVIDIA Nsight Compute (ncu) for profiling
RUN apt-get update && \
    apt-get install -y nvidia-cuda-toolkit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/root/.local/bin:${PATH}

# CUTLASS + reference kernels
RUN cd /usr/local && git clone --depth 1 https://github.com/NVIDIA/cutlass || true
RUN cd /root && git clone https://github.com/vyomakesh0728/reference-kernels.git || true

# Any extra Python deps
RUN uv pip install --system numpy ninja packaging nvidia-cutlass-dsl

# Create the start script inside the image
RUN <<'EOF'
cat > /start.sh <<'EOS'
#!/bin/bash
set -e

# Set up SSH key from environment variable
if [ ! -z "$PUBLIC_KEY" ]; then
  echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# Set SSH port from environment variable
if [ ! -z "$SSH_PORT" ]; then
  sed -i '/^#*Port /d' /etc/ssh/sshd_config || true
  echo "Port $SSH_PORT" >> /etc/ssh/sshd_config
fi

echo "Starting SSH server on port ${SSH_PORT:-22}"
exec /usr/sbin/sshd -D
EOS
chmod +x /start.sh
EOF

EXPOSE 22
ENTRYPOINT ["/start.sh"]

