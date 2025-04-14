import os
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"  # Enable full autotuning

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import time
import datetime

print("============================================================================================")
# Set device and enable TF32 for Ampere GPUs
T.backends.cuda.matmul.allow_tf32 = True
T.backends.cudnn.allow_tf32 = True
T.set_float32_matmul_precision('high')

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
T.cuda.set_device(device)
T.cuda.empty_cache()

# Enable cudnn benchmarking
T.backends.cudnn.benchmark = True

print(f"Device set to : {T.cuda.get_device_name(device)}")
print("============================================================================================")

#############################
### Financial Environment ###
#############################

class FinancialEnvironment:
    def __init__(self, data, n_stocks, n_s, M, num_envs, EIT: False):
        """
        Args:
            data: Matrix with all the features data
            n_stocks: Number of stocks
            n_s: Number of look-back periods
            M: Trading days per step
            num_envs: Number of parallel environments
            EIT: EIT problem, default IT problem (different reward)
        """
        self.stock_prices = data[2]  # Shape [T, n_stocks]
        self.index_prices = data[4]  # Shape [T]

        self.state_space = T.cat(
            (
                data[0],  # Volume: [T-1, n_stocks]
                data[1],  # VIX: [T-1, 1]
                data[3],  # Stock returns: [T-1, n_stocks]
                data[5]   # Index returns: [T-1, 1]
            ),
            dim=1  # Concatenate along feature dimension
        )

        self.n_s = n_s  # Look-back periods
        self.M = M  # Trading days per step
        self.time_steps = n_s * M # Trading days before strategic rebalancement
        self.num_features = self.state_space.shape[1]  
        self.state_dim = self.time_steps * self.num_features
        self.action_dim = n_stocks
        self.num_envs = num_envs
        self.eit = EIT
        print(data[0].shape)  # Expected: [num_timesteps, n_stocks]
        print(data[1].unsqueeze(1).shape)  # Expected: [num_timesteps, 1]
        print(self.state_space.shape)  # Should be [num_timesteps, (n_stocks + 1 + n_stocks + 1)]

    def reset(self):
        low_start = self.time_steps  
        max_start = self.stock_prices.shape[0] - self.time_steps - self.M

        # Generate initial steps within valid range
        self.initial_step = T.randint(
            low=low_start,
            high=max_start,
            size=(self.num_envs,),
            device=device
        )
        self.current_step = self.initial_step.clone()
        self.portfolio_value = 2.0e+10
        self.shares = T.zeros((self.num_envs, self.M, self.action_dim), device=device)
        return self.get_state()

    def test_reset(self):
        self.initial_step = T.randint(
            low=self.stock_prices.shape[0] - self.time_steps,
            high=self.stock_prices.shape[0] - self.time_steps + 1,
            size=(self.num_envs,),
            device=device
            )
        self.current_step = self.initial_step.clone()
        self.portfolio_value = 2.0e+10
        self.shares = T.zeros((self.num_envs, self.M, self.action_dim), device=device)
        print(self.initial_step)
        states = T.stack([
            self.state_space[s - self.time_steps : s]  # [window_size, num_features]
            for s in self.current_step
        ])
        return states.contiguous()

    def get_state(self):
        states = T.stack([
            self.state_space[s - self.time_steps : s]  
            for s in self.current_step
        ]) # [self.num_envs, M, n_stocks]
        return states.contiguous()
    
    # Vectorized computation of transaction cost calculation
    def batched_transaction_cost(self, v_new, tp_value_i_, weights, stock_prices, fee_1=0.005, fee_2=0.005, fee_3=1):
        fee_3_tensor = T.tensor(fee_3, device=stock_prices.device, dtype=stock_prices.dtype)

        delta = T.abs(tp_value_i_ - weights * v_new.unsqueeze(-1))
        cost = T.minimum(
            T.maximum((fee_1 / stock_prices) * delta, fee_3_tensor),
            delta * fee_2
        )
        return T.nansum(cost, dim=1)  # Sum over stocks [batch_size]
    
    # Vectorized fixed-point iteration
    def batched_banach_iteration(self, tp_value_, tp_value_i_, weights, stock_prices, epsilon=1e-5, max_iters=10):
        v_old = T.zeros_like(tp_value_).double()
        tp_value_ = tp_value_.double()
        costs = self.batched_transaction_cost(v_old, tp_value_i_, weights, stock_prices)
        v_new = tp_value_ - costs
        i = 0
        while (T.abs(v_new - v_old) >= epsilon).any():
            v_old = v_new.detach().clone()
            costs = self.batched_transaction_cost(v_old, tp_value_i_, weights, stock_prices)
            v_new = tp_value_ - costs
            i += 1
            if i == max_iters:
                print("batched_banach_iteration takes too long")
                break
        return v_new.float(), costs.float()

    def step(self, weights):
        weights = T.as_tensor(weights, dtype=T.float32, device=device)
        window_size = self.time_steps  # n_s * M

        # Vectorized Index Handling 
        start_indices = self.current_step
        end_indices = start_indices + self.M

        # Batched Data Extraction
        stock_prices = T.stack([
            self.stock_prices[s.item():e.item()]
            for s, e in zip(start_indices, end_indices)
        ])  # [self.num_envs, M, n_stocks]

        index_values = T.stack([
            self.index_prices[s.item():e.item()]
            for s, e in zip(start_indices, end_indices)
        ])  # [self.num_envs, M]


        # Batched Portfolio Simulation 
        tp_value = T.zeros((self.num_envs, self.M), device=device)
        tp_value_ = T.zeros((self.num_envs, self.M), device=device)
        shares = self.shares
        step_costs = T.zeros((self.num_envs, self.M), device=device)
        for m in range(self.M):
            # Vectorized transaction cost calculation
            tp_value_i_ = shares[:, m-1] * stock_prices[:, m]
            tp_value_[:, m] = tp_value_i_.sum(-1)
            tp_value_[:, 0] = self.portfolio_value
            tp_value[:, m], costs = self.batched_banach_iteration(
                tp_value_[:, m],
                tp_value_i_,
                weights,
                stock_prices[:, m]
            )
            step_costs[:, m] = costs
            # Vectorized share update
            shares[:, m] = (weights * tp_value[:, m].unsqueeze(-1)) / stock_prices[:, m]
        
        # Batched Returns & Rewards 
        tp_returns = tp_value_[:, 1:] / tp_value_[:, :-1] - 1  # [self.num_envs, M-1]
        index_returns = index_values[:, 1:] / index_values[:, :-1] - 1 # [self.num_envs, M-1]
        track_error = T.sqrt(T.mean((tp_returns - index_returns) ** 2, dim=1))  # [self.num_envs]
        ex_returns = T.mean((tp_returns - index_returns), dim=1) # [self.num_envs]
        step_costs = T.sum(step_costs, dim=1)
        rewards = - 100 * track_error + 1000 * ex_returns * self.eit

        # Update States with Fixed Window
        self.current_step += self.M
        next_states = T.stack([
            self.state_space[s.item() - window_size : s.item()] 
            for s in self.current_step
        ])
        self.shares = shares
        self.portfolio_value = tp_value_[:, -1]
        dones = self.current_step >= (self.initial_step + self.time_steps)

        return (
            next_states.contiguous(),
            rewards,
            dones,
            {"Tracking error": track_error,
             "Transaction costs": step_costs,
             "Excess returns": ex_returns,
             }
        )
#################################
### Actor and Critic Networks ###
#################################

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dilation: int):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1,
                               padding=padding, dilation=dilation)
        self.lrelu = nn.LeakyReLU()

        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1, stride = stride) if in_channels != out_channels or stride > 1 else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        residual = x
        # First causal convolution
        out = self.lrelu(self.conv1(x))
        # Second causal convolution
        out = self.lrelu(self.conv2(out))
        # Trim the output to match the input sequence length
        if out.shape[-1] != residual.shape[-1] // self.conv1.stride[0]:
            out = out[:, :, :residual.shape[-1] // self.conv1.stride[0]]
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
            residual = residual[:, :, :out.shape[-1]]  # Trim residual to match output length
        return self.lrelu(out + residual)

class PolicyNetwork(nn.Module):
    def __init__(self, num_features: int , action_dim: int, time_steps: int,
                 num_channels=[128, 128, 128, 128], kernel_size: int = 5, stride: int = 1):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 3 ** i 
            in_channels = num_features if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride, dilation=dilation ))

        self.network = nn.Sequential(*layers)
        self.pfc1 = nn.Linear(num_channels[-1] * (time_steps // stride ** len(num_channels)) , 512)
        self.tanh = nn.Tanh()
        self.mean_layer = nn.Linear(512, action_dim) # mean output layer
        self.std_layer = nn.Linear(512, action_dim) # std output layer

        # initialization
        nn.init.xavier_normal_(self.pfc1.weight)
        nn.init.constant_(self.pfc1.bias, 0.0)
        nn.init.orthogonal_(self.mean_layer.weight, 0.1)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        nn.init.orthogonal_(self.std_layer.weight, 0.1)  
        nn.init.constant_(self.std_layer.bias, 0.0)
        
    def forward(self, state):
        state = state.permute(0,2,1)

        x = self.network(state)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layers

        x = self.pfc1(x)  # (self.num_envs, 512)
        x = self.tanh(x)
        
        mean = self.mean_layer(x)
        std = nn.functional.softplus(self.std_layer(x)) + 1e-7

        return mean, std # Shape: (self.num_envs, action_dim)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, nodes: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(state_dim, nodes),
            nn.Tanh(),
            nn.Linear(nodes, nodes),
            nn.Tanh(),
            nn.Linear(nodes, nodes),
            nn.Tanh(),
            nn.Linear(nodes, 1)
        )

        self.layers.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state):
        values = self.layers(state).squeeze(-1)

        return values  # Shape: (self.num_envs,)

#####################
### Rollout Class ###
#####################

class RolloutBuffer:
    def __init__(self, batch_size: int, mini_batch_size: int, state_shape: tuple, action_dim: int):
        """
        Args:
            batch_size: Total experiences per training epoch
            mini_batch_size: Size of individual training batches
            state_shape: Shape of state space
            action_dim: Dimension of action space (number of stocks)
        """
        # Initializing with explicit shape parameters
        self.states = T.empty((batch_size, *state_shape), dtype=T.float32, device=device)
        self.next_states = T.empty((batch_size, *state_shape), dtype=T.float32, device=device)
        self.weights = T.empty((batch_size, action_dim), dtype=T.float32, device=device)
        self.actions = T.empty((batch_size, action_dim), dtype=T.float32, device=device)
        self.log_probs = T.empty(batch_size, dtype=T.float32, device=device)
        self.values = T.empty(batch_size, dtype=T.float32, device=device)
        self.rewards = T.empty(batch_size, dtype=T.float32, device=device)
        self.dones = T.empty(batch_size, dtype=T.bool, device=device)

        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self._ptr = 0

    def store_memory(self,
                    states: T.Tensor,
                    next_states: T.Tensor,
                    weights: T.Tensor,
                    actions: T.Tensor,
                    log_probs: T.Tensor,
                    values: T.Tensor,
                    rewards: T.Tensor,
                    dones: T.Tensor) -> None:
        
        n_elements = states.size(0) # number of parallel environments

        # Copy data to pre-allocated buffers
        self.states[self._ptr:self._ptr+n_elements] = states.detach()
        self.next_states[self._ptr:self._ptr+n_elements] = next_states.detach()
        self.weights[self._ptr:self._ptr+n_elements] = weights.detach()
        self.actions[self._ptr:self._ptr+n_elements] = actions.detach()
        self.log_probs[self._ptr:self._ptr+n_elements] = log_probs.detach()
        self.values[self._ptr:self._ptr+n_elements] = values.detach()
        self.rewards[self._ptr:self._ptr+n_elements] = rewards.detach()
        self.dones[self._ptr:self._ptr+n_elements] = dones.detach()

        self._ptr += n_elements

    def get_memory(self) -> tuple:
        # Shuffled indices
        indices = T.randperm(self.batch_size, device=device)

        return (
            self.states,
            self.next_states,
            self.weights,
            self.actions,
            self.log_probs,
            self.values,
            self.rewards,
            self.dones,
            [indices[i:i+self.mini_batch_size] for i in range(0, self.batch_size, self.mini_batch_size)]
        )

    def clear_memory(self) -> None:
        # Reset buffer without reallocating memory
        self._ptr = 0
        self.states = self.states.detach()
        self.next_states = self.next_states.detach()
        self.weights = self.weights.detach()
        self.actions = self.actions.detach()
        self.log_probs = self.log_probs.detach()
        self.values = self.values.detach()
        self.rewards = self.rewards.detach()
        self.dones = self.dones.detach()

############################
### Agent Training Class ###
############################

class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, num_features: int, time_steps: int,
                 num_envs: int, num_steps: int, batch_size: int, mini_batch_size: int,
                 lr: float = 3e-4, gamma: float = 0.99, lambda_: float = 0.95,
                 epsilon: float = 0.2, e_1: float = 0.5, e_2: float = 0.0):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (number of stocks)
            num_features: Number of financial features per timestep
            time_steps: Number of historical timesteps in state
            num_envs: Number of parallel environments
            num_steps: Number of steps in one episode
            batch_size: Total experiences per training epoch
            mini_batch_size: Size of individual training batches
        """
        self.policy_network = T.compile(
            PolicyNetwork(num_features, action_dim, time_steps).to(device),
            mode='max-autotune',
            dynamic=False,
            fullgraph=True
        )

        self.value_network = T.compile(
            ValueNetwork(state_dim).to(device),
            mode='max-autotune',
            dynamic=False,
            fullgraph=True
        )

        self.policy_optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=lr,
            betas=(0.999, 0.999),
            fused=True
        )
        self.value_optimizer = optim.Adam(
            self.value_network.parameters(),
            lr=lr,
            betas=(0.999, 0.999),
            fused=True
        )

        # Training parameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.e_1 = e_1  # Value loss coefficient
        self.e_2 = e_2  # Entropy coefficient
        self.num_features = num_features
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.time_steps = time_steps
        self.mini_batch_size = mini_batch_size

        # Mixed precision training
        self.scaler = T.cuda.amp.GradScaler()

        # Memory buffer
        state_shape = (time_steps, num_features)
        self.buffer = RolloutBuffer(
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            state_shape=state_shape,
            action_dim=action_dim
        )

    def select_action(self, states: T.Tensor):
        with T.no_grad():
            # Get actions and log probabilities
            means, stds = self.policy_network(states)
            normal_dist = Normal(means, stds)

            actions = normal_dist.sample()
            log_probs = normal_dist.log_prob(actions).sum(-1)
            weights = nn.functional.softmax(actions, dim=-1)

            values = self.value_network(states)

        return weights, actions, log_probs, values

    def remember(self, states: T.Tensor, next_states: T.Tensor, weights: T.Tensor, actions: T.Tensor,
                 log_probs: T.Tensor, values: T.Tensor, rewards: T.Tensor, dones: T.Tensor):
        self.buffer.store_memory(states, next_states, weights, actions, log_probs, values, rewards, dones)

    def update(self):
        # Get trajectories from buffer
        states, next_states, weights, actions, old_log_probs, values, rewards, dones, batches = self.buffer.get_memory()
        
        # Advantage and Cumulative Discounted Rewards calculation
        rewards = rewards.view(self.num_steps, self.num_envs)
        values = values.view(self.num_steps, self.num_envs)
        dones = dones.view(self.num_steps, self.num_envs)
        next_states = next_states.view(self.num_steps, self.num_envs, self.time_steps, self.num_features)

        advantages = T.zeros_like(rewards, device=device, requires_grad=False)
        cd_rewards = T.zeros_like(rewards, device=device, requires_grad=False)
        last_gae = T.zeros(self.num_envs, device=device)
        last_cd_reward = T.zeros(self.num_envs, device=device)
        terminal_value = self.value_network(next_states[-1])

        with T.no_grad():
          for t in reversed(range(len(rewards))):
            next_values = values[t+1] if dones[t].all() == False else terminal_value

            delta = rewards[t] + self.gamma * next_values  - values[t]
            delta_ = rewards[t] + self.gamma * terminal_value * (dones[t])

            last_gae = delta + self.gamma * self.lambda_ * last_gae
            advantages[t] = last_gae

            last_cd_reward = delta_ + self.gamma * last_cd_reward
            cd_rewards[t] = last_cd_reward

        #cd_rewards = advantages + values
        advantages = advantages.view(self.num_envs * self.num_steps)
        cd_rewards = cd_rewards.view(self.num_envs * self.num_steps)

        # Mini-batch training
        for batch_indices in batches:
            b_advantages = advantages[batch_indices]

            # Policy loss computation
            means, stds = self.policy_network(states[batch_indices])
            dist = Normal(means, stds)
            new_log_probs = dist.log_prob(actions[batch_indices]).sum(-1) 


            log_ratio = (new_log_probs - old_log_probs[batch_indices].detach())
            ratio = log_ratio.exp()

            clipped_ratio = T.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            policy_loss = -T.min(
                ratio * b_advantages,
                clipped_ratio * b_advantages
            ).mean()

            # Value loss computation
            value_pred = self.value_network(states[batch_indices])
            value_loss = ((value_pred - cd_rewards[batch_indices]) ** 2).mean()

            # Entropy loss computation
            entropy = - dist.entropy().sum(-1).mean()

            # Total loss
            loss = policy_loss + self.e_1 * value_loss + self.e_2 * entropy

            # Backprop and optimize
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.policy_optimizer)
            self.scaler.unscale_(self.value_optimizer)
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
            self.scaler.step(self.policy_optimizer)
            self.scaler.step(self.value_optimizer)
            self.scaler.update()

        self.clear()
        return (
            loss.item(),
            policy_loss.item(),
            value_loss.item(),
            entropy.item()
        )

    def clear(self) -> None:
        self.buffer.clear_memory()

###################################
### Data Import & Preprocessing ###
###################################

class DataLoader:
    def __init__(self, file_path="train_df.csv"):
        self.df = pd.read_csv(file_path).set_index("Date")
        self.n_stocks = (self.df.shape[1] - 2) // 2
    def gpu_preprocess(self) -> list[T.Tensor]:
        # Convert raw data to tensors (full length T)
        volume = T.tensor(
            self.df.iloc[:, :self.n_stocks].values,
            dtype=T.float32,
            device=device
        )  # [T, n_stocks]
        vix_price = T.tensor(
            self.df.iloc[:, -1].values,
            dtype=T.float32,
            device=device
        )  # [T]
        stock_prices = T.tensor(
            self.df.iloc[:, self.n_stocks:-2].values,
            dtype=T.float32,
            device=device
        )  # [T, n_stocks]
        index_prices = T.tensor(
            self.df.iloc[:, -2].values,
            dtype=T.float32,
            device=device
        )  # [T]

        # Calculate returns (T-1 elements)
        stock_returns = stock_prices[1:] - stock_prices[:-1]  # [T-1, n_stocks]
        index_returns = index_prices[1:] - index_prices[:-1]  # [T-1]

        return [
            volume[1:],                # [T-1, n_stocks]
            vix_price[1:].unsqueeze(1),  # [T-1, 1]
            stock_prices[1:],          # [T-1, n_stocks]
            stock_returns,             # [T-1, n_stocks]
            index_prices[1:],          # [T-1]
            index_returns.unsqueeze(1)  # [T-1, 1]
        ]

#####################
### Training Loop ###
#####################

def configure_and_run_training():
    T._dynamo.reset()  # Clear compiled graph
    
    # Seeding
    seed = 10
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)

    # TensorBoard setup
    run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S')}"
    writer = SummaryWriter(f'runs/{run_name}')

    # 
    H = 20000  # Total policy rollouts
    K = 200    # Parallel environments
    M = 63     # Monthly rebalancing
    n = 4      # Number of steps

    # Environment setup
    data_loader = DataLoader()
    data = data_loader.gpu_preprocess()  

    env = FinancialEnvironment(
        data=data,
        n_stocks=data_loader.n_stocks,
        n_s=int(252//M),
        M=M,
        num_envs=K,
        EIT = False
    )
    env.reset()

    # Agent setup
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_features=env.num_features,
        time_steps=env.time_steps,
        num_envs=K,
        num_steps=n,
        batch_size=int(K * n),
        mini_batch_size=int((K*n)//5),
        lr=1e-5
    )

    # Warmup
    print("Warming up CUDA graphs...")
    warmup_states = env.reset()
    warmup_states_tensor = T.as_tensor(warmup_states, dtype=T.float32, device=device)

    # Warmup with Normal policy
    for _ in range(3):  
        means, _  = agent.policy_network(warmup_states_tensor)
        dummy_loss = means.pow(2).mean() 

        agent.policy_optimizer.zero_grad()
        dummy_loss.backward()
        agent.policy_optimizer.step()
    T.cuda.empty_cache()

    # Main Training Loop
    start_time = time.time()
    print(f"Training started on {T.cuda.get_device_name(device)}")

    for rollout in range(H):
        epoch_start = time.time()

        # Environment reset
        states = env.reset()
        states_tensor = T.as_tensor(states, dtype=T.float32, device=device)

        # Pre-allocate metrics
        total_rewards = T.zeros(int(252//M), device=device)
        total_errors = T.zeros(int(252//M), device=device)
        total_costs = T.zeros(int(252//M), device=device)
        total_ex_ret = T.zeros(int(252//M), device=device)
        # Episode collection
        for step in range(int(252//M)):
            weights, actions, log_probs, values = agent.select_action(states_tensor) 
            next_states, rewards, dones, info = env.step(weights)

            agent.remember(
                states_tensor,  # Already GPU tensor
                next_states,    # GPU tensor
                weights,
                actions,        # GPU tensor
                log_probs,      # GPU tensor
                values,         # GPU tensor
                rewards,        # GPU tensor
                dones           # GPU tensor
            )

            total_rewards[step] = rewards.mean()
            total_errors[step] = info["Tracking error"].mean()
            total_ex_ret[step] = info["Excess returns"].mean()
            total_costs[step] = info["Transaction costs"].mean()
            states_tensor = next_states

        # Policy update
        total_loss, policy_loss, value_loss, entropy_loss = agent.update()
        epoch_time = time.time() - epoch_start
        
        # Logging result on tensorboard
        ep_reward = total_rewards.sum().cpu().item()
        avg_error = total_errors.mean().cpu().item()
        avg_ex_ret = total_ex_ret.mean().cpu().item()
        ep_costs = total_costs.sum().cpu().item()

        writer.add_scalar("Perf/Avg Reward", ep_reward, rollout)
        writer.add_scalar("Perf/Tracking Error", avg_error, rollout)
        writer.add_scalar("Perf/Excess Returns", avg_ex_ret, rollout)
        writer.add_scalar("Perf/Transaction Costs", ep_costs, rollout)
        writer.add_scalar("Train/Total Loss", total_loss, rollout)
        writer.add_scalar("Train/Policy Loss", policy_loss, rollout)
        writer.add_scalar("Train/Value Loss", value_loss, rollout)
        writer.add_scalar("Train/Entropy", entropy_loss, rollout)
        writer.add_scalar("Time/Epoch Time", epoch_time, rollout)

        # Progress reporting
        if rollout % 10 == 0:
            print(f"Epoch {rollout+1}/{H} | Time: {epoch_time:.2f}s | "
                  f"Loss: {total_loss:.3f} (P: {policy_loss:.4f}/V: {value_loss:.4f}) | "
                  f"Reward: {ep_reward:.4f} | Error: {avg_error:.4f} | Transaction costs: {ep_costs:.1f} | "
                  f" Ex Returns: {avg_ex_ret:.5f}")

    # Finalization
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    secs = total_time % 60

    print(f"\nTraining Completed in {hours}h {mins}m {secs:.1f}s")
    print(f"Average Steps/Second: {(H * K * n) / total_time:.1f}")

    # Save final model
    final_checkpoint = {
        'policy': agent.policy_network.state_dict(),
        'value': agent.value_network.state_dict(),
        'config': {'H': H, 'K': K}
    }
    T.save(final_checkpoint, 'final_agent.pth')
    writer.close()

# Execute the training
configure_and_run_training()

####################
### Testing Loop ###
####################

def evaluate_agent():
    T._dynamo.reset()  # Clear compiled graph
    
    # Seeding
    seed = 10
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed_all(seed)

    #
    H = 1  # Total policy rollouts
    K = 2    # Parallel environments
    M = 63     # Monthly rebalancing
    n = 4      # Number of steps

    # Environment setup
    test_data_loader = DataLoader(file_path="test_df.csv")
    test_data = test_data_loader.gpu_preprocess()

    test_env = FinancialEnvironment(
        data=test_data,
        n_stocks=test_data_loader.n_stocks,
        n_s=int(252 // M),
        M=M,
        num_envs=K,
        EIT = False
    )

    # Agent setup
    checkpoint = T.load('final_agent.pth', map_location=device)
    agent = PPOAgent(
        state_dim=test_env.state_dim,
        action_dim=test_env.action_dim,
        num_features=test_env.num_features,
        time_steps=test_env.time_steps,
        num_envs=K,
        num_steps=n,               
        batch_size=K * n, # not used during evaluation, but required for init
        mini_batch_size=(K * n) // 8, #
        lr=1e-5 #
    )
    agent.policy_network.load_state_dict(checkpoint['policy'])
    agent.value_network.load_state_dict(checkpoint['value'])
    agent.policy_network.eval()
    agent.value_network.eval()


    start_time = time.time()
    print(f"Testing started on {T.cuda.get_device_name(device)}")
    with T.no_grad():
        for rollout in range(H):
            epoch_start = time.time()
            states = test_env.test_reset()
            states_tensor = T.as_tensor(states, dtype=T.float32, device=device)

            total_rewards = T.zeros(int(252//M), device=device)
            total_errors = T.zeros(int(252//M), device=device)
            total_ex_ret = T.zeros(int(252//M), device=device)
            total_costs = T.zeros(int(252//M), device=device)
            print(states_tensor.size())
            
            # Episode collection
            for step in range(int(252//M)):
                print(step)
                means, _ = agent.policy_network(states_tensor)
                actions = means
                weights = nn.functional.softmax(actions, dim=-1)
                
                next_states, rewards, dones, info = test_env.step(weights)
                
                total_rewards[step] = rewards.mean()  # GPU->GPU
                total_errors[step] = info["Tracking error"].mean()  # GPU->GPU
                total_ex_ret[step] = info["Excessive returns"].mean() # GPU->GPU
                total_costs[step] = info["Transaction costs"].sum() # GPU->GPU
                states_tensor = next_states

            # ======== LOGGING ========
            ep_reward = total_rewards.sum().cpu().item()
            avg_error = total_errors.mean().cpu().item()
            avg_ex_ret = total_ex_ret.mean().cpu().item()
            ep_costs = total_costs.sum().cpu().item()
            epoch_time = time.time() - epoch_start
    
            # Progress reporting
            print(f"Epoch {rollout+1}/{H} | Time: {epoch_time:.2f}s | "
                  f"Reward: {ep_reward:.4f} | Error: {avg_error:.5f} |"
                  f"Excessive Returns: {avg_ex_ret:.6f} | Transaction costs: {ep_costs:.0f}")

    # ======== FINALIZATION ========
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    mins = int((total_time % 3600) // 60)
    secs = total_time % 60

    print(f"\nTesting Completed in {hours}h {mins}m {secs:.1f}s")
    print(f"Average Steps/Second: {(H * K * 4) / total_time:.1f}")


# Run the evaluation
evaluate_agent()