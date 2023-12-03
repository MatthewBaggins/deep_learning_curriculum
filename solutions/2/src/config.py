from dataclasses import dataclass, field
import math

from typing_extensions import Self


REFERENCE_MODEL_SIZE = 6

@dataclass(frozen=True, slots=True)
class Config:
    """Experiment configuration"""
    model_size: int
    """Number of (hidden) channels in each of the two convolutional layers.
    """
    dataset_size: int
    """Fraction of training data used for training
    in terms of the powers of (1/2).
    """
    # KW
    fc_dim: int = field(default=64, kw_only=True)
    """Fully connected layer dimension"""
    kernel_size: int = field(default=3, kw_only=True)
    """Convolution kernel size"""
    n_epochs: int = field(default=1, kw_only=True)
    
    def __post_init__(self) -> None:
        assert isinstance(self.model_size, int)
        assert isinstance(self.dataset_size, int)
        assert 0 < self.model_size, f"model_size={self.model_size}"
        assert 0 <= self.dataset_size, f"dataset_size={self.dataset_size}"
        assert 0 < self.n_epochs, f"n_epochs={self.n_epochs}"
    
    @classmethod
    def default(cls) -> Self:
        return cls(
            model_size=1,
            dataset_size=0
        )

    @property
    def n_channels(self) -> int:
        return REFERENCE_MODEL_SIZE * int(math.sqrt(2) ** self.model_size)
    
    @property
    def dataset_fraction(self) -> float:
        return 0.5 ** self.dataset_size
