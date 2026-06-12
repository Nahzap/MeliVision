"""Tests for DataLoader worker release and eval-loader policy."""

from unittest.mock import MagicMock, patch

import pytest

from src.data.data_engine import GrainDataEngine
from src.data.dataloader_utils import (
    close_dataset_resources,
    dispose_data_engine,
    dispose_dataloader_workers,
    free_training_memory,
)


class _FakeIterator:
    def __init__(self):
        self.shutdown_calls = 0

    def _shutdown_workers(self):
        self.shutdown_calls += 1


class _FakeLoader:
    def __init__(self, *, num_workers=4, persistent_workers=True, iterator=None):
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self._iterator = iterator
        self.iter_calls = 0

    def __iter__(self):
        self.iter_calls += 1
        return iter(())


class _FakeDataset:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_dispose_dataloader_workers_shuts_down_active_iterator():
    iterator = _FakeIterator()
    loader = _FakeLoader(iterator=iterator)

    dispose_dataloader_workers(loader)

    assert iterator.shutdown_calls == 1
    assert loader._iterator is None


def test_dispose_dataloader_workers_noop_without_active_iterator():
    loader = _FakeLoader(iterator=None, persistent_workers=True)

    dispose_dataloader_workers(loader)

    assert loader.iter_calls == 0
    assert loader._iterator is None


def test_release_train_workers_ignores_exclusive_flag():
    train_loader = MagicMock()
    train_loader.num_workers = 4
    engine = GrainDataEngine(
        config={"data": {"loader_policy": {"exclusive_train_workers_during_val": False}}},
        train_dataset=None,
        val_dataset=None,
        train_loader=train_loader,
        val_loader=MagicMock(),
        train_workers=4,
        val_workers=0,
    )

    with patch("src.data.data_engine.dispose_dataloader_workers") as mock_dispose:
        engine.release_train_workers()
        mock_dispose.assert_called_once_with(train_loader)


def test_release_all_workers_keeps_datasets_open_during_training():
    train_ds = _FakeDataset()
    val_ds = _FakeDataset()
    engine = GrainDataEngine(
        config={},
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_loader=_FakeLoader(num_workers=2),
        val_loader=_FakeLoader(num_workers=2),
        train_workers=2,
        val_workers=2,
    )

    with patch("src.data.dataloader_utils.free_training_memory"):
        engine.release_all_workers()

    assert not train_ds.closed
    assert not val_ds.closed


def test_release_all_workers_closes_datasets_on_dispose():
    train_ds = _FakeDataset()
    engine = GrainDataEngine(
        config={},
        train_dataset=train_ds,
        val_dataset=None,
        train_loader=_FakeLoader(num_workers=2),
        val_loader=_FakeLoader(num_workers=0),
        train_workers=2,
        val_workers=0,
    )

    with patch("src.data.dataloader_utils.free_training_memory"):
        engine.release_all_workers(close_datasets=True)

    assert train_ds.closed


def test_dispose_data_engine_delegates_to_release_all_workers():
    engine = MagicMock()
    dispose_data_engine(engine)
    engine.release_all_workers.assert_called_once()


def test_close_dataset_resources_noop_without_close():
    close_dataset_resources(object())


def test_free_training_memory_closes_matplotlib_figures():
    import matplotlib.pyplot as plt

    plt.figure()
    assert plt.get_fignums()
    free_training_memory("test")
    assert not plt.get_fignums()


def test_create_eval_dataloaders_forces_zero_workers():
    from scripts.post_training import create_eval_dataloaders

    config = {
        "data": {
            "train_dir": "data/processed/train",
            "val_dir": "data/processed/val",
            "batch_size": 4,
            "num_workers": 8,
            "val_num_workers": 4,
            "image_size": 252,
            "samples_per_class": 2,
            "loader_policy": {"persistent_workers": True},
        },
        "grain_detection": {"enabled": True, "use_mask": True},
        "augmentation": {"performance_mode": True},
    }

    with patch("src.data.data_engine.GrainDataEngine.from_config") as mock_from_config:
        mock_engine = MagicMock()
        mock_engine.train_loader = MagicMock()
        mock_engine.val_loader = MagicMock()
        mock_engine.test_loader = None
        mock_from_config.return_value = mock_engine

        train_loader, val_loader, test_loader, engine = create_eval_dataloaders(config)

        passed_config = mock_from_config.call_args[0][0]
        assert passed_config["data"]["num_workers"] == 0
        assert passed_config["data"]["val_num_workers"] == 0
        assert passed_config["data"]["loader_policy"]["persistent_workers"] is False
        assert train_loader is mock_engine.train_loader
        assert val_loader is mock_engine.val_loader
        assert test_loader is None
        assert engine is mock_engine
