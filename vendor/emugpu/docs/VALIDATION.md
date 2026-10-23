# Проверка поставки 0.3.2

Дата: 1 октября 2026.

- **29 tests — PASS**: 26 source-tree проверки и 3 installed-wheel проверки. Runtime/dtype/errors, mesh, generic autograd, CPU kernels, DP, variable-size exchanges и упаковка без готовых распределённых моделей.
- Каталог presets проверен на соответствие числа устройств размеру mesh. DP с двумя CPU device slots на процесс проверен на четырёх отдельных процессах против dense Torch (градиенты и три обновления SGD).
- **dp4×tpsp2**: 8 реальных CPU-процессов, проверены membership и reductions отдельных групп.
- **dp8×tpsp2**: 16 реальных CPU-процессов, проверены membership и reductions. Программного предела в 8 ranks нет; constructor также принимает World(32).
- GatherV, ScatterV, ReduceScatterV и AllToAllSingle с разными split sizes проверены на singleton и переставленных группах, включая zero-size chunks.
- DP проверен с неодинаковыми/пустыми batch, unused parameters, SGD/Adam. Назначение двух устройств на процесс проверено как конфигурация placement.
- Пример `gradient_exchange.py` проверяет отправку полных матриц разных shapes владельцам, обратный exchange и variable gather/scatter. Это transport example, без optimizer transformation.
- Wheel/source distribution/student archive не содержат преподавательских решений или готовых TP+SP/CP/EP/FSDP моделей.

Это CPU validation. CUDA/Triton и выполнение модели на двух физических GPU внутри одного процесса не исполнялись в этой проверке. CPU tests не подтверждают GPU execution или NCCL performance. ReduceScatterV использует AllReduce+slice, GatherV — AllGatherV fallback.
