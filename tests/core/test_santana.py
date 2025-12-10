"""
Tests for SantanaLog class.
"""

import pytest
import torch
from satipatthana.core.santana import SantanaLog


class TestSantanaLog:
    """Tests for SantanaLog functionality."""

    def test_initialization(self):
        """Test default initialization."""
        log = SantanaLog()
        assert len(log.states) == 0
        assert len(log.meta_history) == 0
        assert len(log.energies) == 0
        assert len(log) == 0

    def test_add_state(self):
        """Test adding states to the log."""
        log = SantanaLog()
        state = torch.randn(4, 32)  # (Batch, Dim)

        log.add(state)

        assert len(log) == 1
        assert log.states[0].shape == (4, 32)
        assert len(log.meta_history) == 1
        assert log.meta_history[0] == {}

    def test_add_state_with_meta_and_energy(self):
        """Test adding states with metadata and energy."""
        log = SantanaLog()
        state = torch.randn(4, 32)
        meta = {"step": 0, "gate_open": True}
        energy = 0.5

        log.add(state, meta=meta, energy=energy)

        assert len(log) == 1
        assert log.meta_history[0] == meta
        assert log.energies[0] == energy

    def test_to_tensor(self):
        """Test converting states to tensor."""
        log = SantanaLog()
        batch_size, dim = 4, 32

        # Add 3 states
        for _ in range(3):
            log.add(torch.randn(batch_size, dim))

        tensor = log.to_tensor()

        assert tensor.shape == (3, batch_size, dim)

    def test_to_tensor_empty(self):
        """Test to_tensor on empty log."""
        log = SantanaLog()
        tensor = log.to_tensor()
        assert tensor.numel() == 0

    def test_get_final_state(self):
        """Test getting final state."""
        log = SantanaLog()

        # Empty log
        assert log.get_final_state() is None

        # Add states
        state1 = torch.randn(4, 32)
        state2 = torch.randn(4, 32)
        log.add(state1)
        log.add(state2)

        final = log.get_final_state()
        assert torch.allclose(final, state2)

    def test_get_initial_state(self):
        """Test getting initial state."""
        log = SantanaLog()

        # Empty log
        assert log.get_initial_state() is None

        # Add states
        state1 = torch.randn(4, 32)
        state2 = torch.randn(4, 32)
        log.add(state1)
        log.add(state2)

        initial = log.get_initial_state()
        assert torch.allclose(initial, state1)

    def test_get_total_energy(self):
        """Test total energy calculation."""
        log = SantanaLog()
        log.add(torch.randn(4, 32), energy=0.5)
        log.add(torch.randn(4, 32), energy=0.3)
        log.add(torch.randn(4, 32), energy=0.1)

        assert log.get_total_energy() == pytest.approx(0.9)

    def test_get_final_energy(self):
        """Test getting final energy."""
        log = SantanaLog()

        # Empty log
        assert log.get_final_energy() is None

        log.add(torch.randn(4, 32), energy=0.5)
        log.add(torch.randn(4, 32), energy=0.1)

        assert log.get_final_energy() == 0.1

    def test_clear(self):
        """Test clearing the log."""
        log = SantanaLog()
        log.add(torch.randn(4, 32), meta={"step": 0}, energy=0.5)
        log.add(torch.randn(4, 32), meta={"step": 1}, energy=0.3)

        log.clear()

        assert len(log) == 0
        assert len(log.meta_history) == 0
        assert len(log.energies) == 0

    def test_state_detached(self):
        """Test that stored states are detached from computation graph."""
        log = SantanaLog()
        state = torch.randn(4, 32, requires_grad=True)

        log.add(state)

        # Stored state should not require grad
        assert not log.states[0].requires_grad

    def test_to_batch_list(self):
        """Test splitting batched log into individual logs."""
        log = SantanaLog()
        batch_size = 4
        dim = 32

        # Add 3 steps
        for _ in range(3):
            log.add(torch.randn(batch_size, dim), energy=0.5)

        batch_logs = log.to_batch_list(batch_size)

        assert len(batch_logs) == batch_size
        for b, individual_log in enumerate(batch_logs):
            assert len(individual_log) == 3
            for state in individual_log.states:
                assert state.shape == (1, dim)
