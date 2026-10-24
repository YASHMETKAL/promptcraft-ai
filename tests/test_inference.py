import pytest

from src.inference import generate_optimized_prompt


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        generate_optimized_prompt(object(), object(), "  ")
