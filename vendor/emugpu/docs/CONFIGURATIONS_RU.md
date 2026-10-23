# Готовые конфигурации и запуск

Все команды выполняются из распакованного каталога после `python -m pip install .`.
`run_config.py` запускает настоящие отдельные PyTorch-процессы. Число процессов не ограничено восемью.

```bash
python examples/run_config.py --list
python examples/run_config.py cpu-dp4-tpsp2 --dry-run
python examples/run_config.py cpu-dp4-tpsp2
```

`--dry-run` показывает каждый rank, устройства, координату mesh и полные списки групп, не запускает процессы и не требует CUDA. `--cpu` заменяет все устройства на CPU, сохраняя число ranks, mesh и количество device slots. `--steps 10` задаёт десять обновлений SGD для DP; для mesh-примеров этот параметр не используется.

## Один компьютер, CPU

| Имя конфигурации | Процессы | Mesh | Что выполняется |
|---|---:|---|---|
| `cpu-1`, `cpu-2`, `cpu-4`, `cpu-8`, `cpu-16`, `cpu-32` | число в имени | `workers=N` | AllReduce внутри группы |
| `cpu-dp2`, `cpu-dp4`, `cpu-dp8`, `cpu-dp16` | число после dp | `dp=N` | MLP, backward, ручной DP, SGD и сравнение с dense Torch |
| `cpu-dp1-tpsp4` | 4 | `(1,4)` | Топология одной реплики, 4 model ranks |
| `cpu-dp4-tpsp2` | 8 | `(4,2)` | Топология четырёх реплик по 2 model ranks |
| `cpu-dp2-tpsp4` | 8 | `(2,4)` | Топология двух реплик по 4 model ranks |
| `cpu-dp8-tpsp2` | 16 | `(8,2)` | Топология восьми реплик по 2 model ranks |
| `cpu-mesh-2x2x2x2` | 16 | `(dp,tp,cp,ep)=(2,2,2,2)` | Именованные группы для упражнений |
| `cpu-fsdp-axis4` | 4 | `fsdp=4` | Группа для упражнения |
| `cpu-dp4-two-devices` | 4 | `dp=4` | Две CPU device slots, явное размещение слоёв и DP |

```bash
python examples/run_config.py cpu-4
python examples/run_config.py cpu-dp4 --steps 10
python examples/run_config.py cpu-dp4-tpsp2 --trace mesh.json
python examples/run_config.py cpu-dp2-tpsp4
python examples/run_config.py cpu-dp8-tpsp2
python examples/run_config.py cpu-mesh-2x2x2x2
python examples/run_config.py cpu-fsdp-axis4
```

**Mesh-примеры проверяют группы и коммуникации. Готового обучения с TP+SP, CP, EP или FSDP здесь нет:** эти алгоритмы реализуют студенты. На CPU доступны Torch autograd, матричные операции и коммуникационные примитивы. Triton kernels требуют GPU; на CPU можно использовать обычный Torch и учебный tile simulator.

## Одна GPU на нескольких процессах

| Конфигурация | Процессы | GPU | Поведение |
|---|---:|---:|---|
| `gpu1-single` | 1 | 1 | Одиночный rank |
| `gpu1-dp4` | 4 | 1 | Четыре DP реплики на `cuda:0` |
| `gpu1-dp1-tpsp4` | 4 | 1 | Mesh `(1,4)` |
| `gpu1-dp4-tpsp2` | 8 | 1 | Mesh `(4,2)` |
| `gpu1-dp2-tpsp4` | 8 | 1 | Mesh `(2,4)` |
| `gpu1-dp8-tpsp2` | 16 | 1 | Mesh `(8,2)` |

```bash
python examples/run_config.py gpu1-dp4 --steps 10
python examples/run_config.py gpu1-dp4-tpsp2
# Та же топология полностью на CPU:
python examples/run_config.py gpu1-dp4-tpsp2 --cpu
```

Все процессы делят память одной физической GPU. Транспорт остаётся CPU/Gloo, CUDA tensors перед обменом staging-копируются на CPU. Эмуляция проверяет корректность алгоритма; восемь ranks на одной GPU не дают память/скорость восьми физических GPU.

## Несколько GPU

| Конфигурация | Процессы | Физические GPU | Назначение |
|---|---:|---:|---|
| `gpu2-dp2`, `gpu4-dp4`, `gpu8-dp8` | 2 / 4 / 8 | 2 / 4 / 8 | Один rank на GPU, DP training |
| `gpu2-dp4-shared` | 4 | 2 | Ranks 0,1 на GPU0; ranks 2,3 на GPU1, DP training |
| `gpu4-dp1-tpsp4` | 4 | 4 | Mesh `(1,4)`, один rank на GPU |
| `gpu8-dp4-tpsp2` | 8 | 8 | Mesh `(4,2)`, один rank на GPU |
| `gpu8-dp2-tpsp4` | 8 | 8 | Mesh `(2,4)`, один rank на GPU |
| `gpu16-dp8-tpsp2` | 16 | 16 | Mesh `(8,2)`, один rank на GPU |
| `gpu8-dp4-two-devices` | 4 | 8 | По две GPU на процесс; два слоя явно размещены на разных GPU, DP training |

```bash
python examples/run_config.py gpu2-dp4-shared --steps 10
python examples/run_config.py gpu8-dp4-tpsp2
python examples/run_config.py gpu8-dp4-two-devices --steps 10
# Оба сценария можно проверить без CUDA:
python examples/run_config.py gpu8-dp4-tpsp2 --cpu
python examples/run_config.py gpu8-dp4-two-devices --cpu --steps 10
```

`cuda:N` — индекс внутри `CUDA_VISIBLE_DEVICES`. Например, `CUDA_VISIBLE_DEVICES=3,5 python examples/run_config.py gpu2-dp2` использует физические GPU3 и GPU5 как `cuda:0` и `cuda:1`.

### Восемь CPU ranks для DP×TP+SP

```python
from emugpu import World

def worker(ctx):
    mesh = ctx.mesh((4, 2), ("dp", "tpsp"))
    dp = mesh["dp"].get_group()
    tpsp = mesh["tpsp"].get_group()
    # Здесь реализуйте свою модель и TP/SP операции.
    return {"rank": ctx.global_rank, "dp": dp.ranks, "tpsp": tpsp.ranks}

if __name__ == "__main__":
    print(World(8, device="cpu").run(worker))
```

TP/SP группы: `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`.
DP группы: `[0,2,4,6]`, `[1,3,5,7]`.

`World(4, devices=[["cuda:0","cuda:1"], ...])` создаёт **4**, а не 8 ranks. CPU аналог — `World(4, devices=[["cpu","cpu"] for _ in range(4)])`; две CPU slots указывают на один CPU backend. Дополнительные устройства не создают дополнительные коммуникационные ranks. `dp-placement` демонстрирует перенос activation между двумя слоями внутри процесса и DP между четырьмя процессами; это не реализация TP/SP.

## Параллельная отладка

```bash
python -m pip install '.[debug]'
python -m emugpu debug-config --world-size 8 --output launch-8.json
python examples/run_config.py cpu-dp4-tpsp2 --debug --wait-for-debugger all
```

Перенесите configurations и compounds из `launch-8.json` в `.vscode/launch.json`, выберите compound для всех ranks. Rank `r` слушает порт `5678+r`. Сначала подключите все восемь отладчиков: каждый процесс ждёт своего attach, затем могут начаться коллективные операции. Если остановить один rank на breakpoint, остальные могут ждать его в коллективе; продолжайте их согласованно. `--debug-base-port` меняет базовый порт; у генератора `debug-config` используйте соответствующий параметр из `--help`. `--wait-for-debugger rank0` останавливает перед стартом только rank0; `none` позволяет запускаться без attach.

## Собственная конфигурация

Измените `configurations()` в `examples/run_config.py`: shape, названия осей, по одному назначению устройств на rank и mode (`mesh`, `dp` или `dp-placement`). Произведение размеров mesh должно совпадать с числом назначений. Для произвольной программы используйте `World(N, devices=...).run(worker)`; не нужно добавлять preset или менять библиотеку. Все ranks должны создавать mesh и группы в одинаковом порядке.
