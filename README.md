
# Deep Reinforcement Learning for Enhanced Index Tracking

This project implements a deep reinforcement learning solution for Index Tracking and Enhanced Index Tracking problems using Proximal Policy Optimization (PPO) with Temporal Convolution Networks (TCNs). The system learns optimal portfolio allocation strategies to either replicate or outperform a target index (S&P 500).

## 📘 Overview

The study combines financial data processing with deep reinforcement learning to approache both Index Tracking and Enhanced Index Tracking problems.

The project consists of the following components:

1. **`dataset`.py** – Retrieves and preprocesses S&P 500 constituent data from Yahoo Finance.
2. **`single_agent.py`** – Implements the complete RL pipeline including:
    - Custom financial environment with transaction cost modeling.
    - TCN-based policy and value networks.
    - PPO training and evaluation framework.

⚠️ Important Note: The `single_agent.py` implementation is designed to run exclusively on **NVIDIA GPUs** with CUDA support. CPU execution is not supported due to heavy computational requirements.

## 🧠 Methodology

- **Data**: S&P 500 constituents (2013-2024) with complete history.
- **Sources**:
    - **Yahoo Finance API** – for price and volume data S&P 500 Constituents Database – for survivor bias filtering.
- **Model**: 
    - **PPO Algorithm** -  with actor-critic architecture.
    - **Temporal Convolution Networks** - for sequential data processing.
    - **Custom reward function** - for balancing tracking error and excess returns.
- **Software**: Python with libraries like `pytorch`, `pandas`, `numpy`.

## 📊 Performance Metrics

The system optimizes for the minimizing tracking error (index replication) and Maximizing excess returns (enhanced tracking), while controlling transaction costs. Training progress is logged to TensorBoard with metrics including:
- Policy and value losses
- Portfolio performance statistics
- Computational efficiency

## 📁 File Structure

- `dataset.py` – Data retrieval and preprocessing pipeline
- `single_agent.py` – GPU-only RL implementation (environment, networks, training, evaluation).
- `README.md` – Project documentation
