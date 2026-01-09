# RNN Implementation and Memory Ablation

This repository explores Recurrent Neural Networks (RNNs) from first principles, with a focus on **understanding memory as a mechanism**, not as a black-box abstraction.

The goal is to empirically verify when and why a latent state becomes *memory*, and when it does not.

---

## Motivation

Recurrent Neural Networks (RNNs) are among the simplest neural architectures designed to work with time series data. Conceptually, they form the foundation for more advanced sequence models such as LSTMs, GRUs, and historically, ideas that later influenced Transformers.

The core idea is straightforward: we feed a time series into a neural network **step by step**, where each step corresponds to one point in time.

At time step `t`, the model receives an input vector $x_t$ and produces a prediction $\hat{x}_{t+1}$ for the next time step. The prediction is compared with the true next value using Mean Squared Error:

$$
\text{MSE}(\hat{x}_{t+1}, x_{t+1})
$$

Training minimizes this loss via gradient descent over the entire sequence.

Importantly, a time series can be seen as a function whose analytical form is unknown. The neural network becomes a flexible parametric approximation of that function. In this sense, training an RNN is equivalent to fitting a highly non-linear function directly from data.

This repository focuses on building and understanding this mechanism from scratch before moving to more complex sequence models.

---

## Core Question

**When does a latent state actually become memory?**

More specifically:

- Does adding a recurrent latent state help when the data has no temporal structure?
- Does recurrence matter when the data contains real-world dependencies?
- What happens to the hidden state dynamics when recurrence is removed?

---

## Experimental Setup

At each time step, the model receives:

$$
z_t = \text{cat}(x_t, y_{t-1})
$$

and produces a new latent state:

$$
y_t = \sigma(W z_t)
$$

The latent state is then mapped back to the input space using a readout layer:

$$
\hat{x}_{t+1} = W_{\text{out}} y_t + b_{\text{out}}
$$

The latent vector `y` is not a prediction by itself — it acts as internal state.

---

## Sanity Check: Random Data (No Memory)

If the data is randomly generated and has no temporal dependence:

$$
p(x_t \mid x_{t-1}) = p(x_t)
$$

then the past carries no information about the future.

In this case, the following two setups should behave similarly:

- **With memory:**

$$
z_t = \text{cat}(x_t, y_{t-1})
$$

- **Without memory:**

$$
z_t = x_t
$$

because there is nothing meaningful to store in the latent state.

### Outcome

For random data:

- Adding a latent state does not improve behavior
- The recurrent pathway receives no meaningful temporal signal
- The model naturally exhibits similar dynamics with or without memory

This is not a model failure — it is a direct consequence of the data having no temporal structure.

![Latent state evolution on random data (with memory)](plots/FIGURE%201%20-%20Latent%20state%20evolution%20on%20random%20data%20(with%20memory).png)

<!-- FIGURE 2: Latent state evolution on random data (no memory) -->

---

## PyTorch Implementation

A minimal PyTorch RNN is implemented using:

- Explicit latent state
- Explicit recurrence
- Mean Squared Error loss
- Autoregressive rollout after training

The model is trained by unrolling the sequence and performing backpropagation through time.

---

## Rollout Behavior

After training, the model is run in free-running mode:

- The last real data point is used as the initial input
- Each prediction is fed back as the next input
- The latent state is propagated forward

This allows us to observe whether the model can extrapolate temporal dynamics beyond observed data.

<!-- FIGURE 3: Observed history vs rollout on random data -->

---

## Ablation on Financial Time Series

We repeat the experiment on real-world data assumed to have temporal structure.

### Data

Five economically related time series are used:

- S&P 500
- Nasdaq
- Dow Jones
- EUR/USD
- USD/JPY

The assumption is not strict mathematical dependency, but shared market dynamics.

Each feature is normalized independently.

<!-- FIGURE 4: Normalized financial time series -->

---

## Memory vs No-Memory Models

Two models are trained:

1. **Memory-based model**

$$
z_t = \text{cat}(x_t, y_{t-1})
$$

2. **Memory-less model**

$$
z_t = x_t
$$

All other components (loss, optimizer, hidden size) are kept identical.

---

## Results

### Training Loss

- The memory-based model exhibits different convergence behavior
- The no-memory model converges toward a degenerate solution

<!-- FIGURE 5: Training loss (with memory vs no memory) and Hidden state evolution (with memory vs no memory) -->

### Hidden State Dynamics

- With memory, the hidden state remains active over time
- Different latent dimensions activate at different moments
- The latent state carries information forward

Without memory:

- The hidden state collapses
- Many units saturate near constant values
- Temporal dynamics fade away

---

## Rollout Comparison

### With Memory

- Predictions remain dynamic
- Features interact and evolve
- Temporal structure is propagated forward

### Without Memory

- Predictions rapidly collapse
- The model converges to a fixed point
- The rollout becomes flat and unresponsive

<!-- FIGURE 6: Rollout with memory and Rollout without memory -->

---

## Conclusion

This experiment demonstrates that:

- A latent vector becomes memory **only when it is used recurrently**
- Without feedback from the previous hidden state, temporal dynamics disappear
- Recurrence is the minimal mechanism enabling temporal reasoning
- Without memory, the model can only regress to the mean

This behavior is a direct consequence of the model architecture, not an implementation artifact.

---

## Scope

This repository is intentionally minimal.

The focus is not on performance, benchmarks, or hyperparameter tuning, but on isolating and understanding the
