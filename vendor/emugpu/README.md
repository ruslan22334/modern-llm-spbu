# emugpu — библиотека для распределённых практикумов

Реальные PyTorch-процессы на одном компьютере, явное назначение устройств и mesh с именованными осями. Работает полностью на CPU. Можно назначить нескольким процессам одну общую CUDA GPU либо разные GPU.

В комплекте: коммуникационные примитивы, generic first-order autograd, ручной DP, простые примеры, отладка и Torch/Triton kernels. Готовых TP+SP, CP, EP и FSDP, преподавательских решений и проверок этих алгоритмов нет.

## Установка

Python ≥3.10, PyTorch ≥2.4, NumPy ≥1.24. Из распакованного каталога:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install .
```

Для редактирования исходников: `python -m pip install -e .`. Готовый wheel можно установить через `python -m pip install dist/emugpu-0.3.2-py3-none-any.whl`. Установка автоматически загрузит зависимости, если их ещё нет. Для CPU-only окружения можно сначала установить Torch с CPU index:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

CUDA требует CUDA-enabled PyTorch. Для отладки: `python -m pip install '.[debug]'`; для Triton на Linux: `python -m pip install '.[triton]'`. Основной CPU runtime не импортирует Triton и не требует драйвер NVIDIA.

## Первые запуски

```bash
python examples/all_reduce.py
python examples/mesh.py
python examples/ring.py
python examples/hybrid_mesh.py --dp 4 --tpsp 2
python examples/gradient_exchange.py --world-size 4
python examples/train_dp.py --world-size 4 --steps 5
python examples/train_dp.py --world-size 16 --steps 5 --trace trace-dp.json
```

DP пример делает autograd + SGD, использует разный размер batch на ranks и сравнивает градиенты и обновлённые параметры с dense Torch. Два аргумента `world-size` и `steps` означают число OS-процессов и число общих training iterations.

## Каталог конфигураций

```bash
python examples/run_config.py --list
python examples/run_config.py cpu-dp4-tpsp2
python examples/run_config.py gpu8-dp4-tpsp2 --cpu
python examples/run_config.py gpu8-dp4-two-devices --dry-run
```

[Все конфигурации, команды, CPU/GPU placement и параллельная отладка](docs/CONFIGURATIONS_RU.md). Mesh-примеры проверяют коммуникации; реализации TP+SP/CP/EP/FSDP остаются заданием студентов.

## Своя программа

```python
import torch
from emugpu import World

def worker(ctx):
    mesh = ctx.mesh((2, 2), ("rows", "cols"))
    group = mesh["cols"].get_group()
    value = torch.tensor([ctx.global_rank + 1.], device=ctx.device)
    return {"rank": ctx.global_rank, "group": group.ranks,
            "sum": float(group.all_reduce(value))}

if __name__ == "__main__":
    print(World(4, device="cpu").run(worker))
```

Нужны top-level worker и main guard. Модель/optimizer создавайте внутри worker. Возвращаемый результат должен сериализоваться Torch. `World.run` возвращает rank-ordered список, CUDA tensors результатов загружаются на CPU.

## Устройства и число процессов

```python
World(16, device="cpu")
World(4, devices=["cuda:0"] * 4)                       # одна GPU
World(4, devices=["cuda:0", "cuda:0", "cuda:1", "cuda:1"])
World(2, devices=[["cuda:0", "cuda:1"], ["cuda:2", "cuda:3"]])
```

`ctx.device` — primary device, `ctx.devices` — все назначенные. Размещать части модели на дополнительных устройствах нужно самим. Число ranks — любое положительное целое; фактический предел задают ресурсы машины. Индексы CUDA относятся к GPU, видимым через `CUDA_VISIBLE_DEVICES`; global rank не является номером GPU.

**Транспорт всегда CPU/Gloo**, в том числе при CUDA compute. Процессы на одной GPU делят реальную VRAM. Это среда проверки семантики, а не симулятор скорости NCCL или памяти нескольких физических GPU.

## Что читать дальше

- [API процессов, mesh, коммуникаций и отладки](docs/API_RU.md).
- [DP×TP+SP топология и обмен для своего Muon](docs/HYBRID_AND_GRADIENTS_RU.md).
- [Как программировать DP и нормировать loss](docs/DP_RU.md).
- `src/emugpu/runtime.py`: процессы, raw primitives, CPU staging, tracing.
- `src/emugpu/mesh.py`: именованные оси и группы.
- `src/emugpu/collectives.py`: generic differentiable operations.
- `src/emugpu/dp.py`: единственный готовый training parallelism.
- `examples/`: короткие рабочие программы.

Проверка исходников:

```bash
# Linux/macOS, из этого каталога:
PYTHONPATH="$PWD/src" python -m unittest discover -s tests -v
# Если библиотека уже установлена, PYTHONPATH не нужен:
python -m unittest discover -s tests -v
```

CUDA/Triton execution не проверяется CPU тестами. Для optional CPU tile-модели: `python -m emugpu.kernels.cpu_tiles`; настоящие Triton kernels: `python -m emugpu.kernels.run --device cuda`.
