"""Library regressions: collective errors, empty shards and group ownership."""
import importlib.util
import math
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

from emugpu.runtime import CommunicationTimeout, Context, EmulationError, World, initialize


def regression_worker(comm):
    messages = {}
    invalid = {
        "object": object(),
        "complex": torch.ones(2, dtype=torch.complex64),
        "sparse": torch.eye(2).to_sparse(),
        "meta": torch.ones(2, device="meta"),
    }
    for name, bad in invalid.items():
        for operation in ("all_reduce", "all_gather", "reduce_scatter", "broadcast", "all_to_all"):
            x = bad if comm.rank == 0 else torch.ones(comm.size)
            try:
                if operation == "all_to_all":
                    comm.all_to_all([x] * comm.size)
                elif operation == "broadcast":
                    comm.broadcast(x, src=0)
                else:
                    getattr(comm, operation)(x)
            except EmulationError as exc:
                messages[f"{name}/{operation}"] = str(exc)
            else:
                raise AssertionError(f"undetected invalid input: {name}/{operation}")
            comm.barrier()  # preflight failures must leave communication usable

    try:
        comm.all_gather(torch.ones(2), axis=4 if comm.rank == 0 else 0)
    except EmulationError as exc:
        messages["axis"] = str(exc)
    else:
        raise AssertionError("invalid axis was not detected")
    comm.barrier()
    try:
        comm.broadcast(torch.ones(2), src=comm.size if comm.rank == 0 else 0)
    except EmulationError as exc:
        messages["source"] = str(exc)
    else:
        raise AssertionError("invalid source was not detected")
    comm.barrier()

    # A zero-length shard is valid, including ranks for which chunk() returned
    # no chunk in the old implementation.
    zero = comm.reduce_scatter(torch.empty(0, 3))
    assert zero.shape == (0, 3)
    assert comm.all_gather(zero).shape == (0, 3)
    assert comm.reduce_scatter(torch.empty(2, 0), axis=1).shape == (2, 0)

    # Borrowed Context/initialize must not destroy a group owned by World or an
    # external torch.distributed user, including context-manager and double close.
    with Context() as borrowed:
        borrowed.barrier()
    assert dist.is_initialized()
    borrowed.close()
    with initialize() as borrowed_initialized:
        borrowed_initialized.barrier()
    assert dist.is_initialized()
    comm.barrier()

    # Optional expected shape mismatch drains payload first, so P2P remains
    # usable on both sides instead of blocking the sender.
    if comm.rank == 0:
        comm.send(torch.arange(4), 1, tag=412)
    elif comm.rank == 1:
        try:
            comm.recv(0, tag=412, shape=(9,))
        except EmulationError as exc:
            assert "received (4,)" in str(exc)
        else:
            raise AssertionError("invalid P2P expected shape was not detected")
    comm.barrier()
    return messages


def allreduce_worker(comm, add=0, *, multiplier=1):
    result = comm.all_reduce(torch.tensor([comm.rank + add]))
    assert result.item() == sum(range(comm.size)) + comm.size * add
    return {"pid": os.getpid(), "result": result.item() * multiplier}


def timeout_worker(comm):
    if comm.rank == 0:
        comm.all_reduce(torch.ones(2))
    else:
        time.sleep(4)


def dtype_worker(comm):
    dtypes = (torch.float32, torch.float64, torch.float16, torch.bfloat16,
              torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
    for index, dtype in enumerate(dtypes):
        if dtype == torch.int16:
            values = [torch.tensor([32767 - rank, -32768 + rank], dtype=dtype) for rank in range(comm.size)]
        elif dtype == torch.int8:
            values = [torch.tensor([100 + rank, -100], dtype=dtype) for rank in range(comm.size)]
        elif dtype == torch.uint8:
            values = [torch.tensor([250 - rank, 128], dtype=dtype) for rank in range(comm.size)]
        elif dtype == torch.bool:
            values = [torch.tensor([rank % 2 == 0, rank % 2 != 0], dtype=dtype) for rank in range(comm.size)]
        else:
            values = [torch.tensor([rank + 1, rank + 2], dtype=dtype) for rank in range(comm.size)]
        x = values[comm.rank]
        expected_sum = torch.stack(values).sum(dim=0, dtype=dtype)
        expected_max = torch.stack(values).max(dim=0).values
        summed = comm.all_reduce(x)
        maximum = comm.all_reduce(x, op="max")
        gathered = comm.all_gather(x)
        scattered = comm.reduce_scatter(x.repeat(comm.size))
        broadcast = comm.broadcast(x, src=comm.size - 1)
        for actual, expected in ((summed, expected_sum), (maximum, expected_max),
                                 (gathered, torch.cat(values)), (scattered, expected_sum),
                                 (broadcast, values[-1])):
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for src, actual in enumerate(comm.all_to_all([x] * comm.size)):
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, values[src], rtol=0, atol=0)
        previous, following = (comm.rank - 1) % comm.size, (comm.rank + 1) % comm.size
        incoming = comm.send_recv(x, following, previous, tag=700 + index)
        assert incoming.dtype == dtype
        torch.testing.assert_close(incoming, values[previous], rtol=0, atol=0)
        if comm.rank == 0:
            comm.send(x, 1, tag=800 + index)
        elif comm.rank == 1:
            incoming = comm.recv(0, tag=800 + index)
            assert incoming.dtype == dtype
            torch.testing.assert_close(incoming, values[0], rtol=0, atol=0)
        comm.barrier()
    widened_event = next(event for event in comm.trace if event["op"] == "all_reduce" and event["dtype"] == "torch.int16")
    assert widened_event["details"]["transport_dtype"] == "torch.int32"
    assert widened_event["details"]["transport_tensor_bytes"] == 8
    assert widened_event["details"]["tensor_payload_bytes"] == 4
    return True


class RuntimeRegressionTests(unittest.TestCase):
    def test_invalid_tensor_empty_shards_and_borrowed_context(self):
        results = World(2, timeout=10).run(regression_worker)
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(results[0]), 22)
        self.assertTrue(all("rank 0" in message for message in results[0].values()))

    def test_spawn_ranks_and_forward_arguments_2_4_8(self):
        for size in (2, 4, 8):
            with self.subTest(size=size):
                results = World(size).run(allreduce_worker, args=(5,), kwargs={"multiplier": 3})
                self.assertEqual(len({entry["pid"] for entry in results}), size)
                self.assertNotIn(os.getpid(), {entry["pid"] for entry in results})
                self.assertTrue(all(entry["result"] == 3 * (sum(range(size)) + size * 5) for entry in results))

    def test_backend_timeout_is_public_error(self):
        with self.assertRaises(CommunicationTimeout):
            World(2, timeout=1).run(timeout_worker)

    def test_advertised_dtypes_all_primitives_and_integer_overflow(self):
        self.assertEqual(World(2).run(dtype_worker), [True, True])

    def test_import_has_no_environment_side_effect(self):
        import emugpu.runtime as runtime
        specification = importlib.util.spec_from_file_location("_emugpu_import_check", runtime.__file__)
        module = importlib.util.module_from_spec(specification)
        with patch.dict(os.environ, {}, clear=True), patch("sys.platform", "darwin"):
            specification.loader.exec_module(module)
            self.assertNotIn("GLOO_SOCKET_IFNAME", os.environ)
            self.assertNotIn("USE_LIBUV", os.environ)

    def test_existing_group_contract_and_numeric_validation(self):
        with patch.object(dist, "is_initialized", return_value=True), patch.object(dist, "get_backend", return_value="nccl"):
            with self.assertRaisesRegex(EmulationError, "Gloo"):
                Context()
        for invalid in (0, -1, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                World(2, timeout=invalid)
            with self.assertRaises(ValueError):
                World(2, run_timeout=invalid)
        with self.assertRaises(ValueError):
            World(4, debug=True, debug_base_port=65534)
        with self.assertRaises(ValueError):
            World(2, wait_for_debugger="unknown")


if __name__ == "__main__":
    unittest.main()
