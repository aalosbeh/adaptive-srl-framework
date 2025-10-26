"""
Federated Deep Reinforcement Learning Trainer
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import copy
from tqdm import tqdm


class FederatedTrainer:
    """Trainer for federated deep reinforcement learning"""
    
    def __init__(
        self,
        num_institutions: int,
        state_dim: int,
        action_dim: int,
        num_rounds: int = 50,
        local_epochs: int = 5,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        noise_multiplier: float = 1.1,
        clip_norm: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize federated trainer
        
        Args:
            num_institutions: Number of participating institutions
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            num_rounds: Number of federated learning rounds
            local_epochs: Number of local training epochs per round
            epsilon: Privacy budget (differential privacy)
            delta: Privacy parameter (differential privacy)
            noise_multiplier: Noise multiplier for differential privacy
            clip_norm: Gradient clipping norm
            device: Device for computation
        """
        self.num_institutions = num_institutions
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        self.device = device
        
        # Initialize local agents for each institution
        from src.models.policy_network import PPOAgent
        self.local_agents = [
            PPOAgent(state_dim, action_dim, device=device)
            for _ in range(num_institutions)
        ]
        
        # Global model parameters
        self.global_parameters = self.local_agents[0].get_parameters()
        
        # Training history
        self.history = defaultdict(list)
        
    def add_differential_privacy_noise(
        self,
        parameters: Dict[str, torch.Tensor],
        sensitivity: float
    ) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to parameters
        
        Args:
            parameters: Model parameters
            sensitivity: Sensitivity of the function
        
        Returns:
            Noisy parameters
        """
        noisy_parameters = {}
        
        # Calculate noise variance based on privacy budget
        # σ² = 2Δ²log(1.25/δ)/ε²
        noise_variance = (
            2 * sensitivity ** 2 * np.log(1.25 / self.delta) / (self.epsilon ** 2)
        )
        noise_std = np.sqrt(noise_variance)
        
        for key, param in parameters.items():
            if isinstance(param, dict):
                noisy_parameters[key] = self.add_differential_privacy_noise(param, sensitivity)
            else:
                noise = torch.randn_like(param) * noise_std * self.noise_multiplier
                noisy_parameters[key] = param + noise
        
        return noisy_parameters
    
    def clip_parameters(
        self,
        parameters: Dict[str, torch.Tensor],
        clip_norm: float
    ) -> Dict[str, torch.Tensor]:
        """
        Clip parameter updates to bound sensitivity
        
        Args:
            parameters: Model parameters
            clip_norm: Clipping norm
        
        Returns:
            Clipped parameters
        """
        clipped_parameters = {}
        
        for key, param in parameters.items():
            if isinstance(param, dict):
                clipped_parameters[key] = self.clip_parameters(param, clip_norm)
            else:
                param_norm = torch.norm(param)
                if param_norm > clip_norm:
                    clipped_parameters[key] = param * (clip_norm / param_norm)
                else:
                    clipped_parameters[key] = param
        
        return clipped_parameters
    
    def aggregate_parameters(
        self,
        local_parameters_list: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate local parameters using weighted averaging
        
        Args:
            local_parameters_list: List of local parameters from institutions
            weights: Weights for each institution (default: uniform)
        
        Returns:
            Aggregated global parameters
        """
        if weights is None:
            weights = [1.0 / len(local_parameters_list)] * len(local_parameters_list)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Initialize aggregated parameters
        aggregated = {}
        
        # Get structure from first parameter set
        first_params = local_parameters_list[0]
        
        def aggregate_dict(param_dicts: List[Dict], weights: np.ndarray) -> Dict:
            """Recursively aggregate nested dictionaries"""
            result = {}
            for key in param_dicts[0].keys():
                if isinstance(param_dicts[0][key], dict):
                    result[key] = aggregate_dict([p[key] for p in param_dicts], weights)
                else:
                    # Weighted average of tensors
                    stacked = torch.stack([p[key] for p in param_dicts])
                    result[key] = torch.sum(
                        stacked * torch.tensor(weights, device=stacked.device).view(-1, *([1] * (stacked.ndim - 1))),
                        dim=0
                    )
            return result
        
        aggregated = aggregate_dict(local_parameters_list, weights)
        
        return aggregated
    
    def train_local(
        self,
        institution_id: int,
        local_data: Dict,
        num_epochs: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Train local agent on institution's data
        
        Args:
            institution_id: ID of the institution
            local_data: Local training data
            num_epochs: Number of training epochs
        
        Returns:
            Tuple of (local_parameters, metrics)
        """
        agent = self.local_agents[institution_id]
        
        # Extract data
        states = torch.FloatTensor(local_data['states']).to(self.device)
        actions = torch.FloatTensor(local_data['actions']).to(self.device)
        rewards = torch.FloatTensor(local_data['rewards']).to(self.device)
        dones = torch.FloatTensor(local_data['dones']).to(self.device)
        old_log_probs = torch.FloatTensor(local_data['log_probs']).to(self.device)
        
        # Compute values
        with torch.no_grad():
            values = agent.value(states).squeeze(-1)
            next_states = torch.FloatTensor(local_data['next_states']).to(self.device)
            next_value = agent.value(next_states[-1:]).squeeze(-1)
        
        # Compute advantages and returns
        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        
        # Update agent
        metrics = agent.update(
            states, actions, old_log_probs, advantages, returns,
            num_epochs=num_epochs
        )
        
        # Get updated parameters
        local_parameters = agent.get_parameters()
        
        return local_parameters, metrics
    
    def train(
        self,
        institutional_data: List[Dict],
        institution_weights: Optional[List[float]] = None
    ) -> Dict[str, List]:
        """
        Train federated model across all institutions
        
        Args:
            institutional_data: List of data dictionaries for each institution
            institution_weights: Weights for each institution (based on data size)
        
        Returns:
            Training history
        """
        print(f"Starting federated training for {self.num_rounds} rounds...")
        
        for round_idx in tqdm(range(self.num_rounds), desc="Federated Rounds"):
            print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")
            
            # Distribute global model to all institutions
            for agent in self.local_agents:
                agent.set_parameters(self.global_parameters)
            
            # Local training at each institution
            local_parameters_list = []
            round_metrics = defaultdict(list)
            
            for inst_id in range(self.num_institutions):
                print(f"Training institution {inst_id + 1}/{self.num_institutions}...")
                
                local_params, metrics = self.train_local(
                    inst_id,
                    institutional_data[inst_id],
                    self.local_epochs
                )
                
                # Clip parameters for privacy
                local_params = self.clip_parameters(local_params, self.clip_norm)
                
                # Add differential privacy noise
                local_params = self.add_differential_privacy_noise(
                    local_params,
                    sensitivity=self.clip_norm
                )
                
                local_parameters_list.append(local_params)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    round_metrics[key].append(value)
            
            # Aggregate parameters
            self.global_parameters = self.aggregate_parameters(
                local_parameters_list,
                weights=institution_weights
            )
            
            # Record metrics
            for key, values in round_metrics.items():
                avg_value = np.mean(values)
                self.history[key].append(avg_value)
                print(f"{key}: {avg_value:.4f}")
            
            self.history['round'].append(round_idx + 1)
        
        print("\nFederated training completed!")
        return dict(self.history)
    
    def evaluate(
        self,
        test_data: Dict,
        institution_id: int = 0
    ) -> Dict[str, float]:
        """
        Evaluate trained model
        
        Args:
            test_data: Test data dictionary
            institution_id: Institution ID to use for evaluation
        
        Returns:
            Evaluation metrics
        """
        agent = self.local_agents[institution_id]
        agent.set_parameters(self.global_parameters)
        agent.policy.eval()
        agent.value.eval()
        
        states = torch.FloatTensor(test_data['states']).to(self.device)
        actions = torch.FloatTensor(test_data['actions']).to(self.device)
        rewards = torch.FloatTensor(test_data['rewards']).to(self.device)
        
        with torch.no_grad():
            # Evaluate actions
            log_probs, _ = agent.policy.evaluate_actions(states, actions)
            values = agent.value(states).squeeze(-1)
        
        metrics = {
            'mean_reward': rewards.mean().item(),
            'mean_value': values.mean().item(),
            'mean_log_prob': log_probs.mean().item()
        }
        
        return metrics
    
    def save(self, path: str):
        """Save trainer state"""
        torch.save({
            'global_parameters': self.global_parameters,
            'history': dict(self.history),
            'config': {
                'num_institutions': self.num_institutions,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'num_rounds': self.num_rounds,
                'local_epochs': self.local_epochs,
                'epsilon': self.epsilon,
                'delta': self.delta
            }
        }, path)
    
    def load(self, path: str):
        """Load trainer state"""
        checkpoint = torch.load(path)
        self.global_parameters = checkpoint['global_parameters']
        self.history = defaultdict(list, checkpoint['history'])
        
        # Distribute to local agents
        for agent in self.local_agents:
            agent.set_parameters(self.global_parameters)

