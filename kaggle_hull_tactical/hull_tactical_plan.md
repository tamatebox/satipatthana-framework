# Hull Tactical Market Prediction: Samadhi Model Adaptation Plan

## 1. Overview
This document outlines the strategy for applying the **Samadhi Model**, specifically the `MlpSamadhiModel` variant, to the [Kaggle Hull Tactical: Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction) competition.

The core challenge is to predict `market_forward_excess_returns` for the S&P 500 while adhering to volatility constraints. We will adapt Samadhi's "convergence to attractor" philosophy to perform **forecasting via state completion (inpainting)**. The `MlpSamadhiModel` is chosen for its suitability with tabular data, utilizing MLP-based adapters and decoders for robust representation learning.

## 2. Conceptual Approach: "Masked Target Convergence"

Instead of a traditional `Input -> Output` regression model, we treat the prediction task as a **Missing Value Imputation** problem within a unified state space.

### The Unified State Vector
We define a single state vector $S$ that concatenates market features, historical context, and the target variable.

$$ S = [ \underbrace{\mathbf{x}_{market}}_{\text{Features (98)}}, \quad \underbrace{\mathbf{x}_{lagged}}_{\text{Context (1)}}, \quad \underbrace{y_{target}}_{\text{Target (1)}} ] $$

*   **$\\mathbf{x}_{market}$**: The 98 features provided (M*, E*, P*, etc.).
*   **$\\mathbf{x}_{lagged}$**: `lagged_market_forward_excess_returns` (available at test time).
*   **$y_{target}$**: `market_forward_excess_returns` (The value we want to predict).

### Inference Mechanism (Vicāra Loop)
1.  **Initialization**: At test time, we know $\\mathbf{x}_{market}$ and $\\mathbf{x}_{lagged}$, but $y_{target}$ is unknown. We initialize the state with a mask (e.g., 0) for the target.
    $$ S_0 = [ \mathbf{x}_{market}, \mathbf{x}_{lagged}, 0 ] $$
2.  **Refinement**: We feed $S_0$ into the Samadhi Model. The Vicāra loop iteratively refines this state.
    $$ S_{t+1} = S_t + \alpha \cdot \Phi(S_t) $$
    Because the model has learned the joint probability distribution of features and targets (the "attractors" of the market), it will naturally pull the `0` in the target slot towards the value that is most consistent with the provided market context.
3.  **Convergence**: The final state $S_{final}$ contains the predicted return in its last dimension.

## 3. Implementation Plan

### Step 1: Environment & Data Setup
*   Ensure Kaggle API is configured.
*   Download `hull-tactical-market-prediction` dataset.
*   Explore data distribution and handle missing values (crucial for early historical data).

### Step 2: Data Processing (The "Masking" Loader)
*   **File**: `samadhi/data/hull_tactical_dataset.py`
*   **`HullTacticalDataset` Class**:
    *   **Feature Columns**: All columns except `date_id`, `forward_returns`, `risk_free_rate`, `market_forward_excess_returns`.
    *   **Target Column**: `market_forward_excess_returns`.
    *   **Lagged Target Column**: `lagged_market_forward_excess_returns` (for use in test time and during training if available).
    *   **Missing Value Handling**: Initially, `fillna(0)` is applied to all feature and target columns. This can be refined later.
    *   **Normalization**: All features and the target are normalized using mean and standard deviation from the training data.
    *   **`__getitem__` Method**:
        *   **`original_state`**: The fully observed, normalized state (features, lagged target, true target). Used as the reconstruction target for loss calculation.
        *   **`input_state`**: If `mask_target` is `True`, the `target_col` (i.e., `market_forward_excess_returns`) is set to `0.0` within the `input_state`. This is the masked input fed to the Samadhi model.

### Step 3: Model Configuration
*   **Model Type**: `MlpSamadhiModel` will be used for tabular data.
*   **`input_dim`**: This represents the total dimension of the raw input to the MLP Adapter, which is the number of features plus the target column. Based on `train.csv` (excluding `date_id`, `forward_returns`, `risk_free_rate`), this is $94 (\text{features}) + 1 (\text{target}) = 95$.
*   **`dim`**: This is the dimension of the internal latent space where Vitakka and Vicara operate. It can be set independently (e.g., 64, 128) and should be tuned.
*   **Vicāra Type**: Use `ProbeVicara` or `WeightedVicara` to allow non-linear interactions between features and the target slot.
*   **Attention**: `hard` attention might be useful to select a specific "market regime" (cluster) that best matches the current situation.

### Step 4: Training Strategy
*   **Pre-training**: Utilize `SupervisedSamadhiTrainer.pretrain_autoencoder` to pre-train the Vitakka's adapter (encoder) and the model's decoder. This helps in learning a robust latent representation and stabilizes the initial state of the model before full Samadhi training.
*   **Objective**: Minimize Reconstruction Loss (MSE) between the *refined state* and the *true state*.
    *   Focus weight heavily on the last dimension (Target) to prioritize prediction accuracy over feature reconstruction.
    *   Add `Stability Loss` to ensure the prediction settles on a stable value.

### Step 5: Submission Pipeline
*   Integrate with the Kaggle Evaluation API (`import kaggle_evaluation.hull_tactical_market_prediction_evaluation_gateway`).
*   Implement the loop:
    ```python
    import kaggle_evaluation.hull_tactical_market_prediction_evaluation_gateway as evaluation_gateway

    def make_policy(model):
        def predict(test_df, revealed_targets):
            # 1. Preprocess test_df -> S_0 [Features, Lagged, 0]
            # 2. Run Samadhi(S_0) -> S_final
            # 3. Extract target from S_final
            # 4. Construct portfolio weights (betting strategy based on energy/confidence)
            return submission_df
        return predict

    inference_server = evaluation_gateway.run()
    inference_server.serve(make_policy(model))
    ```

## 4. Hypothesis & Edge
*   **Denoising**: Markets are noisy. Samadhi's primary strength is purification. By treating the target as just another "noisy" (missing) feature to be restored, we leverage the model's native capability.
*   **Regime-Based Prediction**: Vitakka (Search) will map the current market state to a known "concept" (market regime). The Vicāra (Refinement) for that specific regime will then predict the return. This is effectively a dynamic, regime-switching model.
*   **Confidence-Weighted Betting**: We can use the model's final "Energy" or "Stability" score to size our bets. If the model struggles to converge (high energy), it means the market condition is unfamiliar -> Bet small (Low Volatility). If it converges cleanly -> Bet big.
