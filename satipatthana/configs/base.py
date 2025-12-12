from dataclasses import dataclass
from typing import Dict, Any, Type, TypeVar, List
import inspect
from satipatthana.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound="BaseConfig")


@dataclass
class BaseConfig:
    """Base class for all configuration objects."""

    def __post_init__(self):
        self.validate()

    def validate(self):
        """
        Override this method to perform validation and type conversion.
        """
        pass

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Creates an instance from a dictionary, ignoring extra keys but logging warnings.
        """
        if not isinstance(data, dict):
            if isinstance(data, cls):
                return data
            raise ValueError(f"Expected dict, got {type(data)} for {cls.__name__}")

        sig = inspect.signature(cls)
        # Extract only relevant parameters
        filtered_data = {k: v for k, v in data.items() if k in sig.parameters}

        # Log unknown keys for debugging (excluding private keys starting with _)
        unknown_keys = [k for k in data.keys() if k not in sig.parameters and not k.startswith("_")]
        if unknown_keys:
            # Using debug level to avoid spamming logs, or warning if strictness is desired
            logger.debug(f"Unknown keys for {cls.__name__}: {', '.join(unknown_keys)}. These will be ignored.")

        return cls(**filtered_data)
