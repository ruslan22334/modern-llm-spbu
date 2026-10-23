# Ручной data parallel

Каждый rank хранит полную модель и optimizer. Ranks считают forward/backward на разных batch shards. Перед optimizer.step gradients синхронизируются по DP группе.

```python
import torch
from emugpu import DataParallel

# внутри worker; local_model уже на ctx.device:
mesh = ctx.mesh((ctx.size,), ("dp",))
model = DataParallel(local_model, mesh["dp"].get_group())
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer.zero_grad(set_to_none=True)
output = model(local_x)
# MSE: mean по всем output elements, поэтому count = local_target.numel():
loss = (output - local_target).square().mean()
loss.backward()
model.sync_gradients(local_count=local_target.numel(), loss_reduction="mean")
optimizer.step()
```

`DataParallel` — nn.Module wrapper. При создании он копирует параметры и buffers с group-local src=0. Forward вызывает local module. **Синхронизация не автоматическая**: каждый rank вызывает sync_gradients после backward. Не создавайте optimizer до переноса модуля на нужное устройство.

## Нормировка

Пусть n_r — число loss elements или samples, по которым получен local mean; N=Σ n_r.

- Local mean: global gradient = Σ(n_r × local_grad_r) / N.
- Local sum: global gradient = Σ(local_grad_r) / N.
- Равные n_r: достаточно mean gradients, но helper поддерживает и разные n_r.

`local_count` должно соответствовать знаменателю вашего mean. Если loss усреднён по samples, но sum по output features — count равен batch size. Если loss — mean по всем elements — count равен numel. Именно первый контракт использует `examples/train_dp.py`.

Raw gradient AllReduce / world size при неодинаковых batch sizes даст неверное weighting local means.

## Функции без wrapper

```python
from emugpu import broadcast_module, sync_gradients
broadcast_module(local_model, dp_group, src=0)
# ... local forward and backward ...
sync_gradients(local_model, dp_group, local_count=n, loss_reduction="sum")
optimizer.step()
```

`src` — rank внутри группы. Модель должна иметь одинаковую структуру/flags на всех участниках. Все они вызывают helper в одном порядке. Frozen parameters не синхронизируются. Parameter, unused на части ranks, получает zero contribution от них; если unused везде, сохраняется grad=None.

## Пустой batch

Не делайте mean по пустому Tensor — он даст NaN. Используйте differentiable zero loss и local_count=0:

```python
output = model(local_x)     # возможно, shape [0,O]
loss = (output - local_y).square().sum()
loss.backward()
model.sync_gradients(local_count=local_y.numel(), loss_reduction="sum")
```

Глобальный count должен быть >0. Empty rank участвует во всех exchanges. Его параметрические градиенты после synchronization такие же, как у других replicas, поэтому optimizer update одинаковый.

Это ручная учебная реализация: без gradient hooks, overlap, bucketing, data loader sharding и автоматической синхронизации batch-dependent buffers на каждом forward. Dropout/RNG и BatchNorm статистики требуют отдельного согласования. Практика использует Linear/ReLU.

Модуль может содержать parameters/buffers на нескольких устройствах из `comm.devices`. Их размещение и transfers в forward выполняются вашим кодом; DP helper синхронизирует каждый parameter gradient и сохраняет его исходное устройство. На всех ranks структура/shape/dtype параметров должна совпадать, а physical CUDA indices могут отличаться.
