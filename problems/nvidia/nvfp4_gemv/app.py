import modal

app = modal.App("nvfp4-gemv-b200")


image = modal.Image.from_registry(
    "nvidia/cuda:13.0.2-base-ubuntu24.04",
    add_python="3.11",
).run_commands(
        [
            # Update and install essentials
            "apt-get update",
            "DEBIAN_FRONTEND=noninteractive apt-get install -y "
            "python3-pip python3-venv git wget curl gh build-essential pkg-config libssl-dev cargo",
            # Install CUDA toolkit + cuda-gdb
            "DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-toolkit-13-0 cuda-gdb-13-0",
            # Install uv and claude cli
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
            "curl -fsSL https://claude.ai/install.sh | bash",
        # Install rustup and set toolchain to stable
        "bash -lc 'if ! command -v rustup >/dev/null 2>&1; then "
        'curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y; fi\'',
        "bash -lc 'source \"$HOME/.cargo/env\" && rustup default stable'",
        # Clone CUTLASS and repos
        "cd /usr/local && git clone https://github.com/NVIDIA/cutlass",
        'cd "$HOME" && git clone https://github.com/vyomakesh0728/reference-kernels.git',
        'cd "$HOME" && git clone https://github.com/gpu-mode/popcorn-cli.git',
        # Python packages directly required (use uv from ~/.local/bin)
        "bash -lc 'export PATH=\"$HOME/.local/bin:$PATH\"; ~/.local/bin/uv venv'",
        'bash -lc \'export PATH="$HOME/.local/bin:$PATH"; '
        "source .venv/bin/activate && ~/.local/bin/uv pip install torch numpy ninja'",
        # Install NVIDIA Nsight Compute CLI (optional profiling tool; ignore if missing)
        "wget https://developer.download.nvidia.com/devtools/nsight-compute/NsightCompute-2024.2.0-linux-installer.run || true",
        "sh NsightCompute-2024.2.0-linux-installer.run --accept-eula --silent || true",
        "python3 --version",
        "bash -lc 'export PATH=\"$HOME/.local/bin:$PATH\"; ~/.local/bin/uv --version'",
    ]
)


@app.function(
    image=image,
    gpu="B200",  # fallback to "H200" or another Modal-supported string if needed
    cpu=30,
    memory=184,
    timeout=60 * 60,
)
def run():
    """
    Placeholder entrypoint. Replace with your workload invocation.
    """
    print("B200 deployment image is ready. Replace `run` with your workload.")
