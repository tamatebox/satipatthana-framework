import pytest
from dataclasses import dataclass
from satipatthana.configs.base import BaseConfig


@dataclass
class SimpleConfig(BaseConfig):
    name: str = "default"
    count: int = 0

    def validate(self):
        if self.count < 0:
            raise ValueError("Count must be non-negative")


def test_base_config_from_dict_valid():
    data = {"name": "test", "count": 10}
    config = SimpleConfig.from_dict(data)
    assert config.name == "test"
    assert config.count == 10


def test_base_config_from_dict_extra_keys_ignored():
    data = {"name": "test", "extra": "ignored", "count": 5}
    config = SimpleConfig.from_dict(data)
    assert config.name == "test"
    assert config.count == 5
    # Extra keys should not cause error, just be ignored (and logged)


def test_base_config_validation_error():
    data = {"name": "invalid", "count": -1}
    with pytest.raises(ValueError, match="Count must be non-negative"):
        SimpleConfig.from_dict(data)


def test_base_config_from_object():
    # Should handle being passed an object instead of dict gracefully
    obj = SimpleConfig(name="obj", count=1)
    config = SimpleConfig.from_dict(obj)
    assert config == obj
