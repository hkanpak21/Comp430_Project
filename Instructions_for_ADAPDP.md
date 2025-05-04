1. Introduction

This document specifies the requirements for extending the Secure Split Federated Learning (SFL) Framework (defined in v1.0 PRD) to incorporate adaptive differential privacy mechanisms based on the "Adap DP-FL" paper by Fu et al. (2022). This extension modifies the existing Gaussian noise mechanism for client gradients (Mechanism 2) and the associated privacy accounting.

The core constraints remain: all noise mechanisms and the privacy accountant must be implemented manually without external libraries (like Opacus or dp-accounting). This version focuses specifically on adapting the Gaussian noise components.

Reference: This PRD builds upon and modifies the requirements specified in Secure_Split_Federated_Learning_Framework_v1.0_PRD.md. All functionalities from v1.0 are expected unless explicitly modified here.

2. Goals

G1 (Updated): Modify the Gaussian noise mechanism (Mechanism 2) to use adaptive gradient clipping thresholds (Ck_t) per client per round.

G2 (Updated): Modify the Gaussian noise mechanism (Mechanism 2) to use an adaptive noise scale (sigma_t) that decreases based on central model convergence.

G3 (Updated): Update the manual Moments Accountant to correctly track the cumulative privacy budget (ε, δ) considering the round-dependent adaptive noise scale (sigma_t).

G4: Ensure the adaptive mechanisms are configurable and integrated smoothly into the existing SFLV1 simulation workflow.

3. Core Functionality Modifications (Extending v1.0)

3.1. SFLV1 Framework Implementation
* No fundamental change to the Client, Main Server, Fed Server structure or basic SFLV1 communication flow, but requires additions for new parameter broadcasts and validation.

3.2. Noise Mechanism 1: Laplacian Noise on Smashed Data
* No change. This mechanism remains as specified in v1.0 (fixed sensitivity, configurable epsilon_prime).

3.3. Noise Mechanism 2: Adaptive Gaussian Noise on Client Gradients (Replaces v1.0 Section 4.3)
* Requirement: Implement adaptive clipping and adaptive noise scale manually.
* Location: SFLClient, during gradient computation before sending to Fed Server.
* Adaptive Gradient Clipping:
* Threshold Calculation: Instead of a fixed config.clip_norm, each client k calculates its clipping threshold Ck_t for round t based on its own results from round t-1. Use the method described in Fu et al. (2022, Eq 6, 7, 8).
* Ck_t = config.adaptive_clipping_factor * noisy_mean_norm_(t-1)
* noisy_mean_norm_(t-1) is the mean L2 norm of the noise-added clipped per-sample gradients from the previous round (t-1), calculated by the client itself. This avoids extra server communication for broadcasting norms. Requires clients to store necessary info from the previous round.
* Initialize C_k^0 based on an initial noisy step or a predefined value.
* Per-Sample Clipping: The micro-batching loop for obtaining per-sample gradients remains. However, each per-sample gradient ∇WCk,t_sample is now clipped using the dynamically calculated Ck_t.
* Aggregation: Sum the clipped per-sample gradients.
* Adaptive Noise Scale:
* Noise Scale Parameter: Introduce sigma_t, the noise scale for round t. Initialize with config.initial_sigma. This sigma_t is global and determined by the server.
* Server-Side Logic (e.g., in Fed Server or main loop):
* Requires a central validation dataset.
* After each aggregation round t, evaluate the global model's loss (J(wt)) on the validation set.
* Track the validation loss trend. If J(w_{t-k}) > ... > J(w_{t-1}) > J(wt) for k = config.noise_decay_patience (e.g., 3 rounds), then update the next round's noise scale: sigma_{t+1} = config.adaptive_noise_decay_factor * sigma_t. Otherwise, sigma_{t+1} = sigma_t.
* Communication: The server must broadcast the current sigma_t to all clients at the beginning of each round (or whenever it changes).
* Noise Addition:
* Add Gaussian noise to the summed clipped per-sample gradients.
* The standard deviation of the added noise for the summed gradients is sigma_t * Ck_t (where sigma_t is the current global noise scale received from the server and Ck_t is the client's calculated adaptive clipping threshold for that round).
* Formula Ref: Fu et al. add N(0, (sigma_t * Ck_t)^2) to the average gradient. Adjust variance accordingly if adding noise to the sum. Noise added to sum has std dev sigma_t * Ck_t.

3.4. Privacy Accounting: Manual Moments Accountant (Updated)
* Requirement: Update the manual Moments Accountant to handle the adaptive noise scale (sigma_t).
* Method: The accountant still tracks moments based on the Gaussian mechanism.
* step(noise_multiplier, sampling_rate, num_steps) - Modification: The fundamental calculation of RDP per step now depends on the noise scale used in that specific step. Instead of a fixed noise_multiplier, the step function (or the internal _compute_log_moment) needs the effective noise scale sigma_t that was actually used for the gradients generated in those num_steps.
* The RDP privacy loss (epsilon(alpha)) for a given round t is calculated using the specific sigma_t applied during that round (Ref: Fu et al., Theorem 1, which uses sigma_t).
* The accountant must sum these per-round RDP losses, where each round might have used a different sigma_t.
* get_privacy_spent(delta): Logic remains the same (calculating ε from total accumulated RDP ε(α) and δ), but the accumulated RDP ε(α) is now the sum of potentially different per-round RDP contributions.

3.5. Configuration Management (Additions)
* Requirement: Add new configuration parameters.
* New Parameters:
* adaptive_clipping_factor (alpha in Fu et al., e.g., 1.0)
* initial_sigma (Starting global noise scale sigma_0)
* adaptive_noise_decay_factor (beta in Fu et al., e.g., 0.999)
* noise_decay_patience (Number of consecutive loss decreases required, e.g., 3)
* validation_set_ratio (Portion of data held out centrally for validation loss tracking, e.g., 0.1)
* initial_clipping_threshold (Value for C_k^0)

3.6. Evaluation (Additions)
* Requirement: Track and potentially log the central validation loss used for noise adaptation.
* Output: Final report should still include test accuracy and the final (ε, δ) calculated by the updated manual accountant. Optionally log the evolution of sigma_t.

4. Design & Architecture Modifications

SFLClient needs logic to calculate Ck_t and store previous round info if needed. Must accept sigma_t from server.

FedServer (or the main experiment script) needs logic for:

Managing a central validation dataset/loader.

Evaluating global model validation loss each round.

Tracking the loss trend.

Updating sigma_t based on the trend and beta.

Broadcasting the current sigma_t to clients.

ManualPrivacyAccountant needs modification in its step or internal logic to accept and use the round-specific sigma_t for calculating per-step/per-round RDP contributions before summation.

5. README Updates

Update the Configuration section to include the new adaptive parameters (adaptive_clipping_factor, initial_sigma, etc.) and explain their roles.

Update the Features section to mention adaptive clipping and adaptive noise scale reduction.

Briefly explain the server's role in tracking validation loss for noise adaptation.

6. Constraints

NO external DP libraries (Opacus, dp-accounting, etc.). All DP mechanisms and accounting must be manual.

Focus only on adapting the Gaussian noise mechanism (Mechanism 2). Laplacian noise (Mechanism 1) remains unchanged from v1.0.

Implementation based strictly on the details provided in these PRD instructions and the referenced equations/concepts from Fu et al. (2022) as interpreted here.

This follow-up PRD provides the necessary instructions to extend the framework with adaptive noise capabilities while adhering to the manual implementation constraint.