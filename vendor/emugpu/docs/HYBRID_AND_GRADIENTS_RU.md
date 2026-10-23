# DP×TP+SP mesh и обмен градиентами

## 8 GPU: dp=4, tpsp=2

Стандартная топология — **8 процессов, одна GPU на rank**, mesh `(4,2)`:

```
            tpsp=0   tpsp=1
  dp=0        r0       r1       одна модельная реплика
  dp=1        r2       r3
  dp=2        r4       r5
  dp=3        r6       r7
```

Группы TP+SP: `[0,1]`, `[2,3]`, `[4,5]`, `[6,7]`.
Группы DP: `[0,2,4,6]`, `[1,3,5,7]` — replicas одного и того же model shard.

```python
# Launch под main guard:
World(8, devices=[f"cuda:{i}" for i in range(8)]).run(worker)

# Внутри worker:
mesh = ctx.mesh((4,2), ("dp", "tpsp"))
dp_group = mesh["dp"].get_group()
model_group = mesh["tpsp"].get_group()
```

Это реализовано как **топология, placement и отдельные communication groups**. TP+SP-модель студенты реализуют сами на model_group. Для её одинаковых parameter shards градиенты усредняют по dp_group. SP действует в той же model_group, отдельное дополнительное число ranks для SP не нужно.

Рабочая демонстрация групп, без реализации model parallelism:

```bash
python examples/hybrid_mesh.py --dp 4 --tpsp 2 --device cpu
# 16 процессов:
python examples/hybrid_mesh.py --dp 8 --tpsp 2 --device cpu
# 8 физических GPU:
python examples/hybrid_mesh.py --dp 4 --tpsp 2 \
  --devices '["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6","cuda:7"]'
# Те же 8 ranks на одной GPU:
python examples/hybrid_mesh.py --dp 4 --tpsp 2 --devices '["cuda:0","cuda:0","cuda:0","cuda:0","cuda:0","cuda:0","cuda:0","cuda:0"]'
```

## Две GPU на процесс

Такое назначение тоже поддерживается:

```python
World(4, devices=[["cuda:0","cuda:1"], ["cuda:2","cuda:3"],
                  ["cuda:4","cuda:5"], ["cuda:6","cuda:7"]]).run(worker)
```

У worker есть `ctx.devices`, primary — `ctx.device`. Размещение двух частей модели и обмен между ними внутри процесса пишется вручную. DP helper поддерживает parameters на любом из назначенных `ctx.devices`; после gradient synchronization каждый gradient остаётся на своём устройстве. У этой схемы **4 process ranks**, поэтому mesh `(4,2)` построить нельзя: его произведение равно 8. Второе устройство не превращается во второй communication rank. Для dp4×tpsp2 с обычными rank-based collectives используйте 8 процессов.

Максимум в 8 процессов отсутствует. Любое положительное число ranks разрешено; практический предел — ресурсы машины. Все compute/communication функции остаются single-host.

## Коммуникации для собственного Muon

Доступны:

| Операция | Применение |
|---|---|
| all_reduce SUM/MEAN | синхронизировать DP gradients |
| all_to_all | variable-size gradient/update buffers, список по destinations |
| all_to_all_single | flat buffer + unequal input/output split sizes |
| all_gather / all_gather_v | собрать shards; ragged только по leading axis |
| reduce_scatter / reduce_scatter_v | sum/mean и выдача равных / variable shards |
| gather / gather_v | собрать у одного owner |
| scatter / scatter_v | распределить равные/неравные chunks от owner |
| broadcast | распространить рассчитанное обновление |
| send/recv, isend/irecv, send_recv | собственный обмен и pipelining |

Пример маршрутизации **полных матриц разных shapes** к owner и обратно:

```bash
python examples/gradient_exchange.py --world-size 4
python examples/gradient_exchange.py --world-size 16
```

В примере matrices flatten в 1D: trailing shapes тогда одинаковы, а lengths могут отличаться. Shared shape manifest известен из model parameters; если manifest неизвестен, metadata нужно передать отдельно. Empty destinations передаются как tensors длины 0, не пропускаются.

Внутри owner оставлено место для вашего optimizer transformation. Пример возвращает копию матрицы, проверяя только транспорт и восстановление порядка. Реализации Muon в библиотеке нет.

Для matrix optimizer различайте optimizer ownership и parameter sharding. Если правило обновления определено над **целой матрицей**, применение этого правила отдельно к её TP shards в общем случае даст другое обновление. Сначала соберите необходимую матрицу/градиент, вычислите transformation и раздайте соответствующие update shards. Выбор группы зависит от layout: DP aggregation идёт по dp_group, сбор частей матрицы — по группе владельцев этих частей.

Raw communication detached: это подходит для optimizer.step внутри no_grad. Для operations в обучаемом графе есть `emugpu.collectives.all_to_all` с inverse exchange в backward; generic adjoints и replicated-loss conventions нужно учитывать отдельно.

`gather_v` использует all_gather_v fallback, поэтому временно получает данные на всех ranks. `scatter_v` использует variable AllToAll с пустыми sends у non-source ranks. Это проверка семантики; shared GPU и CPU/Gloo staging не моделируют NCCL throughput.

Для packed gradients с разным числом элементов у owners доступен `comm.reduce_scatter_v(flat_grad, split_sizes, op="mean")`. Splits одинаковы на всех ranks и суммируются в длину flat buffer; результат принадлежит текущему logical rank. После owner transformation можно собрать updates через all_gather_v. ReduceScatterV реализован через полный AllReduce и slice, поэтому не обеспечивает память/traffic native sharded optimizer.
