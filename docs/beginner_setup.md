# Beginner setup (Windows 10/11)

1. Install Python 3.11 (64-bit), Git for Windows and VS Code. During Python installation tick **Add Python to PATH**.
2. Open Command Prompt and run `nvidia-smi`. It should display your NVIDIA GPU and driver. If not, install/update the laptop GPU driver from Acer/NVIDIA.
3. Extract this project ZIP. Open the extracted folder in VS Code, then Terminal → New Terminal.
4. Create and activate an environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.
5. Install PyTorch using the exact command from https://pytorch.org/get-started/locally/ (Windows, Pip, Python, CUDA option supported by your driver). Then verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If CUDA is False, stop and fix the driver/PyTorch installation.
6. Install app packages: `python -m pip install --upgrade pip` then `pip install -r requirements-app.txt`.
7. Run `python app.py`. Open `http://127.0.0.1:7860` in Chrome. First launch downloads the model (several GB).
8. First test: prompt `a red bicycle leaning against a brick wall, soft morning light`, 512×512, 15–20 steps, guidance 7, seed -1.
9. Stop with Ctrl+C. On later days activate `.venv` and run `python app.py`.

If CUDA out-of-memory: close games and GPU-heavy apps, restart the app, keep 512×512 and lower steps to 10–15. Avoid 768/1024 until the baseline works.
