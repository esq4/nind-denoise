"""
Test suite for the OperationsFactory class in the nind_denoise.pipeline.base module.
"""

import pytest

from nind_denoise.pipeline.base import OperationsFactory


@pytest.mark.asyncio
async def test_operations_factory_iteration():
    stages = ["stage1", "stage2"]
    factory = OperationsFactory(stages)

    expected_stages = [stage async for stage in factory]
    assert expected_stages == stages, f"Expected {stages}, but got {expected_stages}"


@pytest.mark.asyncio
async def test_operations_factory_empty_iteration():
    factory = OperationsFactory([])

    expected_stages = [stage async for stage in factory]
    assert expected_stages == [], f"Expected empty list, but got {expected_stages}"


@pytest.mark.asyncio
async def test_operations_factory_single_stage_iteration():
    factory = OperationsFactory(["single_stage"])

    expected_stages = [stage async for stage in factory]
    assert expected_stages == [
        "single_stage"
    ], f"Expected ['single_stage'], but got {expected_stages}"
