# Lightning Bagua

This is starter project template which shall simplify initial steps for each new PL project...

[![CI testing](https://github.com/Lightning-AI/lightning-Bagua/actions/workflows/ci-testing.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-Bagua/actions/workflows/ci-testing.yml)
[![General checks](https://github.com/Lightning-AI/lightning-Bagua/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-Bagua/actions/workflows/ci-checks.yml)
[![Documentation Status](https://readthedocs.org/projects/lightning-Bagua/badge/?version=latest)](https://lightning-Bagua.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-Bagua/main.svg?badge_token=mqheL1-cTn-280Vx4cJUdg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-Bagua/main?badge_token=mqheL1-cTn-280Vx4cJUdg)

\* the Read-The-Docs is failing as this one leads to the public domain which requires the repo to be public too

[Bagua](https://github.com/BaguaSys/bagua) is a deep learning training acceleration framework which supports multiple advanced distributed
training algorithms including:

- [Gradient AllReduce](https://tutorials.baguasys.com/algorithms/gradient-allreduce) for centralized synchronous communication, where gradients are averaged among all workers.
- [Decentralized SGD](https://tutorials.baguasys.com/algorithms/decentralized) for decentralized synchronous communication, where each worker exchanges data with one or a few specific workers.
- [ByteGrad](https://tutorials.baguasys.com/algorithms/bytegrad) and [QAdam](https://tutorials.baguasys.com/algorithms/q-adam) for low precision communication, where data is compressed into low precision  before communication.
- [Asynchronous Model Average](https://tutorials.baguasys.com/algorithms/async-model-average) for asynchronous communication, where workers are not required to be  synchronized in the same iteration in a lock-step style.

By default, Bagua uses *Gradient AllReduce* algorithm, which is also the algorithm implemented in DDP, but Bagua can usually produce a higher training throughput due to its backend written in Rust.

```python
# train on 4 GPUs (using Bagua mode)
trainer = Trainer(strategy="bagua", accelerator="gpu", devices=4)
```

By specifying the `algorithm` in the `BaguaStrategy`, you can select more advanced training algorithms featured by Bagua:

```python
# train on 4 GPUs, using Bagua Gradient AllReduce algorithm
trainer = Trainer(
    strategy=BaguaStrategy(algorithm="gradient_allreduce"),
    accelerator="gpu",
    devices=4,
)

# train on 4 GPUs, using Bagua ByteGrad algorithm
trainer = Trainer(
    strategy=BaguaStrategy(algorithm="bytegrad"),
    accelerator="gpu",
    devices=4,
)

# train on 4 GPUs, using Bagua Decentralized SGD
trainer = Trainer(
    strategy=BaguaStrategy(algorithm="decentralized"),
    accelerator="gpu",
    devices=4,
)

# train on 4 GPUs, using Bagua Low Precision Decentralized SGD
trainer = Trainer(
    strategy=BaguaStrategy(algorithm="low_precision_decentralized"),
    accelerator="gpu",
    devices=4,
)

# train on 4 GPUs, using Asynchronous Model Average algorithm, with a synchronization interval of 100ms
trainer = Trainer(
    strategy=BaguaStrategy(algorithm="async", sync_interval_ms=100),
    accelerator="gpu",
    devices=4,
)
```

To use *QAdam*, we need to initialize [QAdamOptimizer](https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/q_adam/index.html#bagua.torch_api.algorithms.q_adam.QAdamOptimizer) first:

```python
from pytorch_lightning.strategies import BaguaStrategy
from bagua.torch_api.algorithms.q_adam import QAdamOptimizer


class MyModel(pl.LightningModule):
    ...

    def configure_optimizers(self):
        # initialize QAdam Optimizer
        return QAdamOptimizer(self.parameters(), lr=0.05, warmup_steps=100)


model = MyModel()
trainer = Trainer(
    accelerator="gpu",
    devices=4,
    strategy=BaguaStrategy(algorithm="qadam"),
)
trainer.fit(model)
```

Bagua relies on its own [launcher](https://tutorials.baguasys.com/getting-started/#launch-job) to schedule jobs. Below, find examples using `bagua.distributed.launch` which follows `torch.distributed.launch` API:

```bash
# start training with 8 GPUs on a single node
python -m bagua.distributed.launch --nproc_per_node=8 train.py
```

If the ssh service is available with passwordless login on each node, you can launch the distributed job on a single node with `baguarun` which has a similar syntax as `mpirun`. When staring the job, `baguarun` will automatically spawn new processes on each of your training node provided by `--host_list` option and each node in it is described as an ip address followed by a ssh port.

```bash
# Run on node1 (or node2) to start training on two nodes (node1 and node2), 8 GPUs per node
baguarun --host_list hostname1:ssh_port1,hostname2:ssh_port2 --nproc_per_node=8 --master_port=port1 train.py
```

Note

You can also start training in the same way as Distributed Data Parallel. However, system optimizations like [Bagua-Net](https://tutorials.baguasys.com/more-optimizations/bagua-net) and [Performance autotuning](https://tutorials.baguasys.com/performance-autotuning/) can only be enabled through bagua launcher. It is worth noting that with `Bagua-Net`, Distributed Data Parallel can also achieve better performance without modifying the training script.

See [Bagua Tutorials](https://tutorials.baguasys.com/) for more details on installation and advanced features.
