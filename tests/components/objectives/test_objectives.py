import pytest
import torch
import torch.nn as nn
from samadhi.components.objectives.supervised_regression import SupervisedRegressionObjective
from samadhi.components.objectives.unsupervised import UnsupervisedObjective
from samadhi.components.objectives.anomaly import AnomalyObjective
from samadhi.components.objectives.supervised_classification import SupervisedClassificationObjective
from samadhi.components.objectives.robust_regression import RobustRegressionObjective
from samadhi.components.objectives.cosine_similarity import CosineSimilarityObjective


@pytest.fixture
def anomaly_mock_config():
    # AnomalyObjective needs objective.anomaly_margin and objective.anomaly_weight
    # Mock SamadhiConfig structure
    objective_config = MockConfig(
        {
            "stability_coeff": 0.1,
            "entropy_coeff": 0.1,
            "balance_coeff": 0.1,
            "anomaly_margin": 1.0,  # Specific margin for testing
            "anomaly_weight": 2.0,  # Specific weight for testing
            "recon_coeff": 1.0,
            "huber_delta": 1.0,
        }
    )
    return MockConfig(
        {"n_probes": 5, "refine_steps": 2, "objective": objective_config, "input_dim": 10}
    )  # Nested objective config + input_dim


@pytest.fixture
def dummy_anomaly_data():
    batch_size = 4
    dim = 10

    # x: input data
    # y: labels (0 for normal, 1 for anomaly)
    # s0, s_final: intermediate states
    # decoded: reconstructed output
    # metadata: includes probs and s_history for stability, entropy, balance losses

    # Example: 2 normal, 2 anomaly
    x = torch.randn(batch_size, dim)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Normal, Normal, Anomaly, Anomaly

    s0 = torch.randn(batch_size, dim)
    s_final = torch.randn(batch_size, dim)

    # Create decoded outputs that will trigger different margin loss scenarios
    decoded = torch.zeros(batch_size, dim)
    # Normal samples (y=0): should have low recon error, so decoded should be close to x
    decoded[0] = x[0] * 1.01  # Very low error
    decoded[1] = x[1] * 0.99  # Very low error
    # Anomaly samples (y=1):
    # - decoded[2] makes recon error < margin (should contribute to loss)
    # - decoded[3] makes recon error > margin (should not contribute much/any loss)

    # To control recon_errors, we can make `decoded` specifically.
    # We want recon_errors for anomalies to be controllable.
    # Let's assume x is mostly zeros for simplicity to control MSE easily.
    x_test = torch.zeros(batch_size, dim)
    # Normal data: easy to reconstruct
    x_test[0] = torch.ones(dim) * 0.1
    decoded[0] = torch.ones(dim) * 0.101  # Low error
    x_test[1] = torch.ones(dim) * 0.2
    decoded[1] = torch.ones(dim) * 0.202  # Low error

    # Anomaly data:
    # Set target recon error for anomaly_idx=2 to be less than margin (e.g., 0.5 < 1.0)
    # MSE = mean((x-decoded)^2)
    # If x is 0, MSE = mean(decoded^2)
    # Let recon_error[2] = 0.5 (so decoded[2] should be ~sqrt(0.5) if dim=1)
    # For dim=10, sum((decoded)^2) / 10 = 0.5 => sum((decoded)^2) = 5
    # So, each element decoded[2,j] could be sqrt(0.5) / sqrt(10)
    recon_error_low = 0.5  # This will be < margin (1.0)
    decoded[2] = torch.full((dim,), (recon_error_low * dim) ** 0.5 / dim**0.5)  # x_test[2] is 0
    # Set target recon error for anomaly_idx=3 to be more than margin (e.g., 1.5 > 1.0)
    recon_error_high = 1.5  # This will be > margin (1.0)
    decoded[3] = torch.full((dim,), (recon_error_high * dim) ** 0.5 / dim**0.5)  # x_test[3] is 0

    x = x_test

    # Mock metadata
    probs = torch.softmax(torch.randn(batch_size, 5), dim=1)
    s_history = [torch.randn(batch_size, dim) for _ in range(3)]
    metadata = {"probs": probs, "s_history": s_history}

    return x, y, s0, s_final, decoded, metadata


def test_anomaly_objective(anomaly_mock_config, dummy_anomaly_data):
    objective = AnomalyObjective(anomaly_mock_config, device="cpu")
    x, y, s0, s_final, decoded, metadata = dummy_anomaly_data

    total_loss, components = objective.compute_loss(
        x=x,
        y=y,
        s0=s0,
        s_final=s_final,
        decoded_s_final=decoded,
        metadata=metadata,
        num_refine_steps=anomaly_mock_config.refine_steps,
    )

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0

    # Verify component losses are present
    assert "recon_loss_normal" in components
    assert "recon_loss_anomaly" in components
    assert "stability_loss" in components
    assert "entropy_loss" in components
    assert "balance_loss" in components

    # --- Verify Margin Loss Logic ---
    margin = anomaly_mock_config.objective.anomaly_margin  # 1.0
    anomaly_weight = anomaly_mock_config.objective.anomaly_weight  # 2.0

    # Calculate expected recon_errors for normal samples (y=0)
    # x_test[0] = 0.1, decoded[0] = 0.101 => error approx (0.001)^2 = 1e-6
    # x_test[1] = 0.2, decoded[1] = 0.202 => error approx (0.002)^2 = 4e-6
    # Mean of these is roughly 2.5e-6, should be very small
    expected_normal_recon_errors = torch.mean((decoded[y == 0] - x[y == 0]) ** 2, dim=1)
    assert components["recon_loss_normal"] == pytest.approx(expected_normal_recon_errors.mean().item(), rel=1e-3)

    # Calculate expected recon_errors for anomaly samples (y=1)
    # x_test[2]=0, decoded[2] has recon_error_low = 0.5
    # x_test[3]=0, decoded[3] has recon_error_high = 1.5
    expected_anomaly_recon_errors = torch.mean((decoded[y == 1] - x[y == 1]) ** 2, dim=1)

    # Anomaly #2: recon_error = 0.5 ( < margin 1.0) => contributes to loss: (1.0 - 0.5) = 0.5
    # Anomaly #3: recon_error = 1.5 ( > margin 1.0) => contributes to loss: max(0, 1.0 - 1.5) = 0
    # Mean anomaly loss should be (0.5 + 0) / 2 = 0.25
    expected_loss_anomaly_component = torch.relu(margin - expected_anomaly_recon_errors).mean().item()
    assert components["recon_loss_anomaly"] == pytest.approx(expected_loss_anomaly_component, rel=1e-3)

    # Verify overall total loss makes sense (cannot be exact due to random stability/entropy/balance)
    # It should be greater than recon_loss_normal + (anomaly_weight * recon_loss_anomaly)
    base_recon_part = components["recon_loss_normal"] + (anomaly_weight * components["recon_loss_anomaly"])
    assert total_loss.item() > base_recon_part - 1e-6  # Allow for small positive other losses


class MockConfig(dict):
    def __getattr__(self, name):
        return self.get(name)


@pytest.fixture
def mock_config():
    return MockConfig(
        {
            "n_probes": 5,
            "stability_coeff": 0.1,
            "entropy_coeff": 0.1,
            "balance_coeff": 0.1,
            "refine_steps": 2,
            "input_dim": 10,
        }
    )


@pytest.fixture
def dummy_data():
    batch_size = 4
    dim = 10
    x = torch.randn(batch_size, dim)
    y = torch.randn(batch_size, dim)
    s0 = torch.randn(batch_size, dim)
    s_final = torch.randn(batch_size, dim)
    decoded = torch.randn(batch_size, dim)

    # Mock metadata
    probs = torch.softmax(torch.randn(batch_size, 5), dim=1)
    s_history = [torch.randn(batch_size, dim) for _ in range(3)]

    metadata = {"probs": probs, "s_history": s_history}

    return x, y, s0, s_final, decoded, metadata


def test_supervised_regression_objective(mock_config, dummy_data):
    objective = SupervisedRegressionObjective(mock_config, device="cpu")
    x, y, s0, s_final, decoded, metadata = dummy_data

    total_loss, components = objective.compute_loss(
        x=x,
        y=y,
        s0=s0,
        s_final=s_final,
        decoded_s_final=decoded,
        metadata=metadata,
        num_refine_steps=mock_config.refine_steps,
    )

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0
    assert "recon_loss" in components
    assert "stability_loss" in components
    assert "entropy_loss" in components
    assert "balance_loss" in components


def test_supervised_classification_objective(mock_config, dummy_data):
    x, _, s0, s_final, _, metadata = dummy_data
    # Re-mock y and decoded_s_final for classification
    # decoded_s_final: (Batch, NumClasses) -> Logits. Let's say 2 classes.
    decoded_s_final_cls = torch.randn(x.size(0), 2)
    # y: (Batch,) -> Class Indices (0 or 1)
    y_cls = torch.randint(0, 2, (x.size(0),))

    objective = SupervisedClassificationObjective(mock_config)

    loss, components = objective.compute_loss(x, y_cls, s0, s_final, decoded_s_final_cls, metadata, num_refine_steps=2)

    assert isinstance(loss, torch.Tensor)
    assert "total_loss" in components
    assert "classification_loss" in components
    assert "stability_loss" in components


def test_cosine_similarity_objective(mock_config, dummy_data):
    objective = CosineSimilarityObjective(mock_config, device="cpu")
    x, y, s0, s_final, decoded, metadata = dummy_data

    total_loss, components = objective.compute_loss(
        x=x,
        y=y,
        s0=s0,
        s_final=s_final,
        decoded_s_final=decoded,
        metadata=metadata,
        num_refine_steps=mock_config.refine_steps,
    )

    assert isinstance(total_loss, torch.Tensor)
    assert "cosine_loss" in components
    # CosineEmbeddingLoss output is usually small positive float if vectors are not identical
    assert components["cosine_loss"] >= 0


def test_robust_regression_objective(mock_config, dummy_data):
    # Mock huber_delta in config
    mock_config.huber_delta = 1.0

    objective = RobustRegressionObjective(mock_config, device="cpu")
    x, y, s0, s_final, decoded, metadata = dummy_data

    total_loss, components = objective.compute_loss(
        x=x,
        y=y,
        s0=s0,
        s_final=s_final,
        decoded_s_final=decoded,
        metadata=metadata,
        num_refine_steps=mock_config.refine_steps,
    )

    assert isinstance(total_loss, torch.Tensor)
    assert "robust_loss" in components
    assert components["robust_loss"] >= 0


def test_supervised_regression_objective_no_target(mock_config, dummy_data):
    objective = SupervisedRegressionObjective(mock_config, device="cpu")
    x, _, s0, s_final, decoded, metadata = dummy_data

    with pytest.raises(ValueError):
        objective.compute_loss(
            x=x,
            y=None,
            s0=s0,
            s_final=s_final,
            decoded_s_final=decoded,
            metadata=metadata,
            num_refine_steps=mock_config.refine_steps,
        )


def test_unsupervised_objective(mock_config, dummy_data):
    objective = UnsupervisedObjective(mock_config, device="cpu")
    x, y, s0, s_final, decoded, metadata = dummy_data

    # In unsupervised, y should be ignored, but we pass it to verify it doesn't crash
    total_loss, components = objective.compute_loss(
        x=x,
        y=y,
        s0=s0,
        s_final=s_final,
        decoded_s_final=decoded,
        metadata=metadata,
        num_refine_steps=mock_config.refine_steps,
    )

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.item() > 0
    # Unsupervised uses x as target for recon_loss
    expected_recon = nn.MSELoss()(decoded, x).item()
    assert abs(components["recon_loss"] - expected_recon) < 1e-6
