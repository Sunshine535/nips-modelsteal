## Remote Server

- SSH: `ssh frp-server` (key-based auth, no password)
- GPU: 8× A100-SXM4-80GB
- Python env: uv venv at `.venv` (Python 3.12 + PyTorch 2.10 + CUDA 12.8)
- Activate: `source .venv/bin/activate`
- Use `screen` for background jobs: `screen -dmS exp bash -c '...'`
- Pip mirror: https://mirrors.aliyun.com/pypi/simple/
- HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`
