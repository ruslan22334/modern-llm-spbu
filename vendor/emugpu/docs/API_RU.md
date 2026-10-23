# emugpu 0.3.1: процессы, устройства, mesh и коммуникации

Библиотека запускает реальные PyTorch-процессы на одном хосте. Число ranks — любое положительное целое, включая 1, 16 и 32; верхнего программного лимита нет. Реальный предел задают RAM, VRAM, число дескрипторов и стоимость процессов. Коммуникации используют CPU/Gloo, в том числе при CUDA-вычислениях. Это среда проверки распределённых алгоритмов, а не симулятор скорости сети или объёма памяти нескольких GPU.

## Минимальная программа

```python
import torch
from emugpu import World

def train(ctx):
    mesh = ctx.mesh((2, 2), ("rows", "cols"))
    cols = mesh["cols"].get_group()
    rows = mesh["rows"].get_group()
    x = torch.tensor([ctx.global_rank + 1.], device=ctx.device)
    return {
        "rank": ctx.global_rank,
        "coordinate": mesh.get_coordinate(),
        "col_ranks": cols.ranks,
        "col_sum": cols.all_reduce(x).item(),
        "row_sum": rows.all_reduce(x).item(),
    }

if __name__ == "__main__":
    print(World(4, device="cpu").run(train))
```

Ожидаемые суммы `cols` по global ranks: `[3,3,7,7]`, `rows`: `[4,6,4,6]`. Результаты возвращаются в порядке **global rank**, даже если процессы завершаются в другом порядке.

```
mesh shape=(2,2), names=(rows,cols)
                 cols coordinate
                  0     1
 rows coordinate 0  r0 ── r1     `cols` group [0,1]
               1  r2 ── r3     `cols` group [2,3]
                  │     │
                [0,2] [1,3]    `rows` groups
```

`World.run` требует importable функцию верхнего уровня и `if __name__ == '__main__'`. Создавайте модель и optimizer внутри worker. Python globals между процессами не общие. Результаты должны сериализоваться `torch.save`; возвращаемые CUDA tensors загружаются в родителе на CPU.

## Назначение CPU и GPU

```python
# 16 CPU процессов:
world = World(16, device="cpu")

# 4 процесса делят одну GPU:
world = World(4, devices=["cuda:0"] * 4)

# 4 процесса на четырёх GPU:
world = World(4, devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

# 2 процесса на каждой из двух GPU:
world = World(4, devices=["cuda:0", "cuda:0", "cuda:1", "cuda:1"])

# Явное назначение нескольких устройств одному процессу:
world = World(2, devices=[["cuda:0", "cuda:1"], ["cuda:2", "cuda:3"]])
```

`devices` содержит ровно одну запись на rank. Запись — устройство либо непустой список устройств. `ctx.device` — первое устройство записи, `ctx.devices` — весь tuple. Библиотека выбирает primary CUDA device; размещать разные части модели на остальных `ctx.devices` нужно самим через `.to(device)`. Назначение не запрещает процессу доступ к другим видимым GPU и не распределяет модель автоматически.

Индексы CUDA относятся к устройствам, **видимым этому интерпретатору**. При внешнем `CUDA_VISIBLE_DEVICES=5,7` физические GPU 5 и 7 становятся `cuda:0` и `cuda:1`. CPU-only PyTorch не поддерживает CUDA. На общей GPU все процессы используют одну физическую VRAM и отдельные CUDA allocators; четырёхкратная вместимость не появляется.

```bash
emugpu run my_app:train --world-size 4 --device cpu
emugpu run my_app:train --world-size 4 --devices '["cuda:0","cuda:0","cuda:1","cuda:1"]'
```

Python API `launch("my_app:train", world_size=4, devices=[...], project_dir=".")` поддерживает те же назначения. `run(worker, world_size=4, devices=[...])` — краткая форма `World(...).run(...)`.

## Построение mesh

```python
from emugpu import init_device_mesh

# Внутри worker:
mesh = ctx.mesh((2, 2), ("rows", "cols"))  # внутри World(4)
# Альтернативная Torch-like форма:
mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("rows", "cols"), ctx=ctx)

mesh.get_coordinate()      # координата текущего rank
mesh.mesh                  # CPU int64 Tensor с global ranks
mesh.shape                 # tuple размерностей
mesh.mesh_dim_names        # tuple имён
mesh.size()                # общее число ranks
mesh.size("cols")            # размер локальной `cols` группы
mesh.get_local_rank("cols")  # индекс текущего rank внутри `cols` группы
mesh["cols"].get_group()     # Context для `cols` оси
mesh.get_group("cols")       # то же
mesh["rows", "cols"].get_group()  # flatten выбранных осей в указанном порядке
```

Произведение размеров mesh равно world size. Каждая ось задаёт **семейство групп**: координаты остальных осей зафиксированы, выбранная координата меняется. На `(2,2)` группа `cols` имеет 2 ranks. Совмещённая `("rows","cols")` группа — 4 ranks. Для World(16) можно построить `(2,2,2,2)` с именами `("a","b","c","d")`: каждая осевая группа будет содержать 2 ranks.

Mesh строят **все world ranks в одном порядке с одинаковыми аргументами**. Одиночные оси создаются сразу. Комбинированные оси создаются при первом запросе: все world ranks должны запросить их в одном порядке, даже если часть ranks позже не будет вычислять на этой группе. Повторный запрос использует кэш.

```python
# Пользовательская перестановка, logical rank follows mesh order:
mesh = ctx.mesh((2,2), ("rows","cols"), ranks=[[3,1],[2,0]])
# `cols` группы [3,1], [2,0], `rows` группы [3,2], [1,0].
```

Context переводит логические ranks в backend ranks. AllGather и AllToAll возвращают данные в порядке mesh, а ReduceScatter выдаёт slice по логическому rank. Нельзя подменять group-local rank глобальным `ctx.global_rank` при шардировании.

Это собственный Torch-like фасад. `mesh['cols'].get_group()` возвращает **emugpu.Context**, а не PyTorch ProcessGroup. Для прямого вызова `torch.distributed` есть `comm.group`, но он имеет backend порядок отсортированных global ranks: при переставленном mesh используйте API emugpu. В DTensor передавать этот mesh нельзя. При необходимости native `torch.distributed.device_mesh.DeviceMesh` можно построить после инициализации Gloo отдельно; данный курс использует emugpu API.

## Context и адресация

| Свойство | Значение |
|---|---|
| `ctx.global_rank` | rank в WORLD |
| `ctx.rank` | логический rank в текущей группе |
| `ctx.size` | число участников текущей группы |
| `ctx.ranks` | global ranks в логическом порядке группы |
| `ctx.device` / `ctx.devices` | primary / все назначенные устройства |
| `ctx.group` | backend Gloo ProcessGroup |

Все `src`/`dst` в примитивах — **локальные ranks данной группы**. Например, у группы `[2,3]` `src=0` означает global rank 2. Переставленная `[3,1]` с `src=0` означает global rank 3.

Для нестандартной группы все world ranks вызывают `comm = ctx.new_group([0,2,5])`. Участники получают Context, остальные — `None`. Последующие операции вызывает только группа. Группы создавайте из WORLD context.

При завершении worker launcher освобождает созданные подгруппы и WORLD. Закрытие borrowed Context не уничтожает чужую WORLD группу. Подгруппы принадлежат создавшему их world context; держите его открытым до окончания работы.

## Примитивы

Raw операции возвращают новые **detached** tensors. Исходные tensors не меняются. CPU staging и Gloo происходят внутри; выход возвращается на исходное устройство, для `recv` и `broadcast(None)` — на `comm.device`.

| Вызов | Семантика / требования |
|---|---|
| `all_reduce(x, op='sum')` | SUM / MEAN / MAX; одинаковые shape/dtype, результат у всех |
| `all_gather(x, axis=0)` | конкатенация равных shards в logical rank order |
| `reduce_scatter(x, op='sum', axis=0)` | reduce полных inputs, equal slices каждому rank |
| `all_gather_v(x)` | список по source; variable rows, equal trailing shape/dtype |
| `broadcast(x=None, src=0)` | у src Tensor, у остальных None или такой же shape/dtype |
| `reduce(x, dst=0, op='sum')` | Tensor только у dst, у остальных None |
| `gather(x, dst=0, axis=0)` | конкатенация только у dst, у остальных None |
| `reduce_scatter_v(x, split_sizes, op='sum', axis=0)` | reduce полных inputs, variable slices по одинаковому split manifest |
| `gather_v(x, dst=0)` | variable-row tensors, список только у dst |
| `scatter_v(blocks=None, src=0)` | src даёт список variable-row tensors, по одному каждому rank |
| `scatter(x=None, src=0, axis=0)` | src даёт полный Tensor, каждый получает equal slice |
| `all_to_all(blocks)` | p tensors: `blocks[dst]`; выход `[from_source]`; variable rows |
| `all_to_all_single(x, input_split_sizes=None, output_split_sizes=None)` | flat axis0 exchange; splits — nonnegative row counts |
| `send(x, dst, tag=0)` / `recv(src, tag=0, shape=None, dtype=None)` | blocking P2P с заголовком dtype/shape |
| `send_recv(x, dst, src, tag=0)` | обе стороны posted до wait; безопасно для кольца |
| `isend(x, dst, tag=0)` | Work; обязательно `wait()` |
| `irecv(src, tag=0, shape=None, dtype=None)` | Work; `wait()` возвращает Tensor |
| `barrier()` | синхронизация группы |

MEAN разрешён только для floating tensors. Dtype: fp64/fp32/fp16/bf16, int64/int32/int16/int8/uint8/bool; dense strided tensors. Sparse/complex не поддерживаются. Int16 typed collectives передают int32 и восстанавливают int16. Для AG/RS/scatter нужны корректные axes, для RS/scatter длина оси делится на размер группы. A2A поддерживает пустые chunks; rank с нулём tokens всё равно участвует в обмене. P2P tags — целые 0…500000000, header поддерживает не более 16 dimensions.

RS использует AR + slice, reduce/gather/scatter — AR/AG/broadcast fallbacks. `all_gather_v` использует A2A, посылая копию input каждому участнику. Семантика правильная, транспортная стоимость отличается от оптимизированных GPU реализаций. Эти API не предоставляют `async_op` для collectives; асинхронный интерфейс есть у P2P.

```python
# Blocking send в кольце deadlock-опасен; используйте paired exchange:
prev = (comm.rank-1) % comm.size
next_rank = (comm.rank+1) % comm.size
incoming = comm.send_recv(x, dst=next_rank, src=prev, tag=7)

# Или явные handles: сначала post обе стороны, потом wait.
receive = comm.irecv(prev, tag=8)
send = comm.isend(x, next_rank, tag=8)
incoming = receive.wait()
send.wait()
```

`irecv` асинхронно posts header, payload получает при `wait()`. `isend` держит header/payload buffers до `wait()`. Завершите все Work до выхода из worker/закрытия группы. Self P2P не нужен: используйте clone; `send_recv` при обоих self peers возвращает clone.

## Generic autograd

Raw операции detached. Используйте `emugpu.collectives` для first-order gradients:

```python
from emugpu import collectives
full = collectives.all_gather(local, comm, axis=0)
local = collectives.reduce_scatter(full, comm, axis=0)
summed = collectives.all_reduce(partial, comm)
received = collectives.all_to_all(send_blocks, comm)
rotated = collectives.ring_shift(x, comm, shift=1, tag=100)
```

| Операция | Forward | Backward |
|---|---|---|
| all_gather | AG | RS SUM |
| reduce_scatter | RS SUM | AG |
| all_reduce | AR SUM | AR SUM |
| all_to_all | variable A2A | inverse A2A |
| ring_shift | to +shift | gradient to −shift |

Эти adjoints соответствуют разным local losses у ranks. Реплицированные losses могут требовать других adjoints и нормировок — это часть вашего алгоритма. Все ranks проходят согласованный forward/backward. Higher-order gradients не предоставляются.

## Ошибки и трассировка

Перед collective каждый rank обменивается metadata внутри **своей группы**. Проверяются порядок вызовов, arguments, shape/dtype. Рассогласование вызывает `EmulationError` у участников; отсутствие rank заканчивается `CommunicationTimeout`. Проверка не исправляет произвольный diverging control flow. P2P не имеет collective preflight; совпадение peers/tags обеспечиваете сами.

`World(4, timeout=30, run_timeout=300)` отдельно ограничивает Gloo waits и полный runtime программы. Без явного watchdog — `max(30, timeout*10)`. В debug mode watchdog отключён, если run_timeout не задан. Gloo timeout во время pause остаётся активен.

```python
world.save_trace("trace.json")  # parent after run
# Inside ranks, all group members call:
comm.save_trace("tp-group.json")  # пишется у logical group rank0
```

Для нескольких независимых групп задавайте разные filenames, иначе каждый group rank0 перезапишет один файл. `gather_trace()` на группе возвращает только события этой группы; на WORLD — все локально записанные group events. Trace содержит global `rank`, local `group_rank`, global member list `group` и логические номера операций. Это не latency и не единый clock. Viewer/trace schema доступны через `export_viewer`.

## Отладка

```bash
emugpu debug-config --world-size 16 --base-port 5678 --output .vscode/launch-16.json
emugpu run my_app:train --world-size 16 --device cpu \
  --debug --wait-for-debugger all --timeout 600
```

У ranks 0…15 порты 5678…5693. VS Code compound запускает отдельный attach к каждому процессу. В `World` доступны `debug=True`, `debug_base_port=5678`, `wait_for_debugger='all'`, `timeout=600`. Один pause может задержать всех участников следующего collective — это ожидаемая семантика.

