# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest import mock

import pytest
import torch
from lightning_utilities import module_available

if module_available("lightning"):
    from lightning.pytorch import Trainer
    from lightning.pytorch.demos.boring_classes import BoringModel, ManualOptimBoringModel
    from lightning.pytorch.trainer.states import TrainerFn
    from lightning.pytorch.utilities.exceptions import MisconfigurationException
elif module_available("pytorch_lightning"):
    from pytorch_lightning import Trainer
    from pytorch_lightning.demos.boring_classes import BoringModel, ManualOptimBoringModel
    from pytorch_lightning.trainer.states import TrainerFn
    from pytorch_lightning.utilities.exceptions import MisconfigurationException

from lightning_bagua import BaguaStrategy
from tests import RunIf


class BoringModel4QAdam(BoringModel):
    def configure_optimizers(self):
        from bagua.torch_api.algorithms.q_adam import QAdamOptimizer

        optimizer = QAdamOptimizer(self.layer.parameters(), lr=0.05, warmup_steps=20)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


@pytest.mark.xfail(raises=AssertionError, reason="Internal error in Bagua")  # Unexpected rsp=<Response [500]'
@RunIf(min_gpus=1)
def test_manual_optimization(tmpdir):
    model = ManualOptimBoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=1,
        strategy=BaguaStrategy(),
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(model)


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    reason="Async does not support this CUDA architecture",
)
@RunIf(min_gpus=2, standalone=True)
def test_async_algorithm(tmpdir):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy=BaguaStrategy(algorithm="async"),
        accelerator="gpu",
        devices=2,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model)

    for param in model.parameters():
        assert torch.norm(param) < 3


@RunIf(min_gpus=1)
@pytest.mark.parametrize(
    "algorithm", ["gradient_allreduce", "bytegrad", "qadam", "decentralized", "low_precision_decentralized"]
)
def test_configuration(algorithm, tmpdir):
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy=BaguaStrategy(algorithm=algorithm),
        accelerator="gpu",
        devices=1,
    )
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer

    with mock.patch(
        "bagua.torch_api.data_parallel.bagua_distributed.BaguaDistributedDataParallel.__init__", return_value=None
    ), mock.patch("bagua.torch_api.communication.is_initialized", return_value=True):
        if algorithm == "qadam":
            with pytest.raises(MisconfigurationException, match="Bagua QAdam can only accept one QAdamOptimizer"):
                trainer.strategy._configure_bagua_model(trainer)
        else:
            trainer.strategy._configure_bagua_model(trainer)


@RunIf(min_gpus=1)
def test_qadam_configuration(tmpdir):
    model = BoringModel4QAdam()
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=1,
        strategy=BaguaStrategy(algorithm="qadam"),
        accelerator="gpu",
        devices=1,
    )
    trainer.state.fn = TrainerFn.FITTING
    trainer.strategy.connect(model)
    trainer.lightning_module.trainer = trainer
    trainer.strategy.setup_optimizers(trainer)

    with mock.patch(
        "bagua.torch_api.data_parallel.bagua_distributed.BaguaDistributedDataParallel.__init__", return_value=None
    ), mock.patch("bagua.torch_api.communication.is_initialized", return_value=True):
        trainer.strategy._configure_bagua_model(trainer)
