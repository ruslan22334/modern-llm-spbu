# ДЗ 3 — Triton

Написал девять ядер на Triton, от сложения с константой до скалярного FlashAttention. Код в `kernels.py`, проверка через PyTorch в `check.py`. На RTX 3070 Ti прошли все девять проверок.

Запуск из корня репозитория на Python 3.12 с NVIDIA GPU.

```powershell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r hw3/requirements.txt
python -m hw3.check
```
