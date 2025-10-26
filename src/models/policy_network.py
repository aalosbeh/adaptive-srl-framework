"""
Policy and Value Networks for Federated Deep Reinforcement Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class PolicyNetwork(nn.Module):
    """Actor network for PPO algorithm"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Action mean and log_std
        self.action_mean = nn.Linear(input_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution
        
        Args:
            state: State tensor of shape (batch, state_dim)
        
        Returns:
            Tuple of (action_mean, action_std)
        """
        x = self.shared_layers(state)
        action_mean = self.action_mean(x)
        action_std = torch.exp(self.action_log_std).expand_as(action_mean)
        
        return action_mean, action_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            deterministic: If True, return mean action
        
        Returns:
            Tuple of (action, log_prob)
        """
        action_mean, action_std = self.forward(state)
        
        if deterministic:
            action = action_mean
            log_prob = None
        else:
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of actions
        
        Args:
            state: State tensor
            action: Action tensor
        
        Returns:
            Tuple of (log_prob, entropy)
        """
        action_mean, action_std = self.forward(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Critic network for PPO algorithm"""
    
    def __init__(self, state_dim: int, hidden_dims: list = [256, 256]):
        super().__init__()
        
        self.state_dim = state_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get state value
        
        Args:
            state: State tensor of shape (batch, state_dim)
        
        Returns:
            Value tensor of shape (batch, 1)
        """
        return self.network(state)


class PPOAgent:
    """Proximal Policy Optimization Agent"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize networks
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value = ValueNetwork(state_dim).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr_value)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.get_action(state_tensor, deterministic)
        
        action = action.cpu().numpy()[0]
        log_prob = log_prob.cpu().item() if log_prob is not None else None
        
        return action, log_prob
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation
        
        Args:
            rewards: Reward tensor (T,)
            values: Value tensor (T,)
            dones: Done flags (T,)
            next_value: Next state value (1,)
        
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Append next value
        values_with_next = torch.cat([values, next_value])
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_with_next[t + 1] * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        num_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update policy and value networks using PPO
        
        Args:
            states: State tensor (N, state_dim)
            actions: Action tensor (N, action_dim)
            old_log_probs: Old log probabilities (N,)
            advantages: Advantage tensor (N,)
            returns: Return tensor (N,)
            num_epochs: Number of update epochs
            batch_size: Batch size for updates
        
        Returns:
            Dictionary of training metrics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0
        }
        
        dataset_size = states.shape[0]
        num_batches = (dataset_size + batch_size - 1) // batch_size
        
        for epoch in range(num_epochs):
            # Shuffle data
            indices = torch.randperm(dataset_size)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, dataset_size)
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, entropy = self.policy.evaluate_actions(batch_states, batch_actions)
                values = self.value(batch_states).squeeze(-1)
                
                # Policy loss (PPO clip objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
                
                # Update policy
                self.policy_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Accumulate metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.mean().item()
                metrics['total_loss'] += total_loss.item()
        
        # Average metrics
        total_updates = num_epochs * num_batches
        for key in metrics:
            metrics[key] /= total_updates
        
        return metrics
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters for federated learning"""
        return {
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict()
        }
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters from federated aggregation"""
        self.policy.load_state_dict(parameters['policy'])
        self.value.load_state_dict(parameters['value'])
    
    def save(self, path: str):
        """Save agent checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

