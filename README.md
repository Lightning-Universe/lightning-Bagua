# Lightning ⚡ Bagua

**Deep Learning Training Acceleration with [Bagua](https://tutorials.baguasys.com/) and [Lightning AI](https://lightning.ai)**

[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://lightning.ai/)
[![PyPI Status](https://badge.fury.io/py/lightning-bagua.svg)](https://badge.fury.io/py/lightning-bagua)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lightning-bagua)](https://pypi.org/project/lightning-bagua/)
[![PyPI Status](https://pepy.tech/badge/lightning-bagua)](https://pepy.tech/project/lightning-bagua)
[![Deploy Docs](https://github.com/Lightning-AI/lightning-Bagua/actions/workflows/docs-deploy.yml/badge.svg)](https://lightning-ai.github.io/lightning-Bagua/)

[![General checks](https://github.com/Lightning-AI/lightning-bagua/actions/workflows/ci-checks.yml/badge.svg?event=push)](https://github.com/Lightning-AI/lightning-bagua/actions/workflows/ci-checks.yml)
[![Build Status](https://dev.azure.com/Lightning-AI/compatibility/_apis/build/status/Lightning-AI.lightning-Bagua?branchName=main)](https://dev.azure.com/Lightning-AI/compatibility/_build/latest?definitionId=47&branchName=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Lightning-AI/lightning-Bagua/main.svg)](https://results.pre-commit.ci/latest/github/Lightning-AI/lightning-Bagua/main)

[Bagua](https://github.com/BaguaSys/bagua) is a deep learning training acceleration framework which supports multiple advanced distributed
training algorithms including:

- [Gradient AllReduce](https://tutorials.baguasys.com/algorithms/gradient-allreduce) for centralized synchronous communication, where gradients are averaged among all workers.
- [Decentralized SGD](https://tutorials.baguasys.com/algorithms/decentralized) for decentralized synchronous communication, where each worker exchanges data with one or a few specific workers.
- [ByteGrad](https://tutorials.baguasys.com/algorithms/bytegrad) and [QAdam](https://tutorials.baguasys.com/algorithms/q-adam) for low precision communication, where data is compressed into low precision  before communication.
- [Asynchronous Model Average](https://tutorials.baguasys.com/algorithms/async-model-average) for asynchronous communication, where workers are not required to be  synchronized in the same iteration in a lock-step style.

By default, Bagua uses *Gradient AllReduce* algorithm, which is also the algorithm implemented in DDP, but Bagua can usually produce a higher training throughput due to its backend written in Rust.

## Installation

```bash
pip install -U lightning-bagua
```

## Usage

Simply set the strategy argument in the Trainer:

```python
from lightning import Trainer

# train on 4 GPUs (using Bagua mode)
trainer = Trainer(strategy="bagua", accelerator="gpu", devices=4)
```

See [Bagua Tutorials](https://tutorials.baguasys.com/) for more details on installation and advanced features.
