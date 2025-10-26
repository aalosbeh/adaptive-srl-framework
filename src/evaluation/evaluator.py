"""
Comprehensive Evaluation Module for SRL Framework
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from scipy import stats
import json
import os


class Evaluator:
    """Evaluator for the Adaptive Multi-Modal SRL Framework"""
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def evaluate_metacognitive_estimation(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate metacognitive state estimation accuracy
        
        Args:
            predictions: Dictionary of predicted states
            ground_truth: Dictionary of ground truth states
        
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        components = ['awareness', 'monitoring', 'control']
        
        for component in components:
            if component in predictions and component in ground_truth:
                pred = predictions[component]
                true = ground_truth[component]
                
                # Mean Absolute Error
                mae = np.mean(np.abs(pred - true))
                
                # Root Mean Squared Error
                rmse = np.sqrt(np.mean((pred - true) ** 2))
                
                # Correlation
                correlation, _ = stats.pearsonr(pred.flatten(), true.flatten())
                
                # Accuracy (within threshold)
                threshold = 0.1
                accuracy = np.mean(np.abs(pred - true) < threshold)
                
                metrics[f'{component}_mae'] = mae
                metrics[f'{component}_rmse'] = rmse
                metrics[f'{component}_correlation'] = correlation
                metrics[f'{component}_accuracy'] = accuracy
        
        # Overall metrics
        all_mae = [metrics[f'{c}_mae'] for c in components if f'{c}_mae' in metrics]
        all_accuracy = [metrics[f'{c}_accuracy'] for c in components if f'{c}_accuracy' in metrics]
        
        metrics['overall_mae'] = np.mean(all_mae) if all_mae else 0.0
        metrics['overall_accuracy'] = np.mean(all_accuracy) if all_accuracy else 0.0
        
        return metrics
    
    def evaluate_intervention_effectiveness(
        self,
        interventions: List[Dict],
        outcomes: List[Dict]
    ) -> Dict[str, float]:
        """
        Evaluate effectiveness of interventions
        
        Args:
            interventions: List of intervention records
            outcomes: List of outcome records
        
        Returns:
            Dictionary of effectiveness metrics
        """
        metrics = {}
        
        # Group by intervention type
        intervention_types = ['content', 'strategy', 'feedback', 'social']
        
        for int_type in intervention_types:
            type_interventions = [i for i in interventions if i['type'] == int_type]
            type_outcomes = [outcomes[i] for i, int_dict in enumerate(interventions) if int_dict['type'] == int_type]
            
            if type_outcomes:
                # Compute effectiveness metrics
                learning_gains = [o.get('learning_gain', 0) for o in type_outcomes]
                engagement_improvements = [o.get('engagement_improvement', 0) for o in type_outcomes]
                
                metrics[f'{int_type}_avg_learning_gain'] = np.mean(learning_gains)
                metrics[f'{int_type}_avg_engagement'] = np.mean(engagement_improvements)
                metrics[f'{int_type}_effectiveness'] = (
                    np.mean(learning_gains) * 0.6 + np.mean(engagement_improvements) * 0.4
                )
        
        # Overall effectiveness
        all_effectiveness = [
            metrics[f'{t}_effectiveness']
            for t in intervention_types
            if f'{t}_effectiveness' in metrics
        ]
        metrics['overall_effectiveness'] = np.mean(all_effectiveness) if all_effectiveness else 0.0
        
        return metrics
    
    def evaluate_privacy_preservation(
        self,
        epsilon: float,
        delta: float,
        noise_added: np.ndarray,
        original_params: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate privacy preservation metrics
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter
            noise_added: Noise added to parameters
            original_params: Original parameters
        
        Returns:
            Dictionary of privacy metrics
        """
        metrics = {}
        
        # Privacy budget
        metrics['epsilon'] = epsilon
        metrics['delta'] = delta
        
        # Noise statistics
        metrics['noise_mean'] = np.mean(noise_added)
        metrics['noise_std'] = np.std(noise_added)
        metrics['noise_to_signal_ratio'] = np.std(noise_added) / (np.std(original_params) + 1e-8)
        
        # Privacy score (higher is better, normalized)
        # Based on epsilon (lower is better for privacy)
        privacy_score = 1.0 / (1.0 + epsilon)
        metrics['privacy_score'] = privacy_score
        
        return metrics
    
    def evaluate_federated_performance(
        self,
        training_history: Dict[str, List],
        convergence_threshold: float = 0.01
    ) -> Dict[str, float]:
        """
        Evaluate federated learning performance
        
        Args:
            training_history: Training history dictionary
            convergence_threshold: Threshold for convergence detection
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Extract loss history
        if 'total_loss' in training_history:
            losses = training_history['total_loss']
            
            # Final loss
            metrics['final_loss'] = losses[-1] if losses else float('inf')
            
            # Convergence round (when loss change < threshold)
            convergence_round = None
            for i in range(1, len(losses)):
                if abs(losses[i] - losses[i-1]) < convergence_threshold:
                    convergence_round = i
                    break
            
            metrics['convergence_round'] = convergence_round if convergence_round else len(losses)
            
            # Loss reduction
            if len(losses) > 1:
                metrics['loss_reduction'] = (losses[0] - losses[-1]) / losses[0]
        
        # Communication efficiency
        if 'round' in training_history:
            metrics['total_rounds'] = len(training_history['round'])
        
        return metrics
    
    def compute_statistical_significance(
        self,
        method1_results: np.ndarray,
        method2_results: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute statistical significance between two methods
        
        Args:
            method1_results: Results from method 1
            method2_results: Results from method 2
        
        Returns:
            Dictionary of statistical test results
        """
        # T-test
        t_stat, p_value = stats.ttest_ind(method1_results, method2_results)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(method1_results) - np.mean(method2_results)
        pooled_std = np.sqrt(
            (np.std(method1_results) ** 2 + np.std(method2_results) ** 2) / 2
        )
        cohens_d = mean_diff / (pooled_std + 1e-8)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'significant': p_value < 0.05
        }
    
    def plot_convergence(
        self,
        training_history: Dict[str, List],
        save_path: str = None
    ):
        """Plot training convergence"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Policy loss
        if 'policy_loss' in training_history:
            axes[0, 0].plot(training_history['policy_loss'])
            axes[0, 0].set_title('Policy Loss')
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Value loss
        if 'value_loss' in training_history:
            axes[0, 1].plot(training_history['value_loss'], color='orange')
            axes[0, 1].set_title('Value Loss')
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True)
        
        # Entropy
        if 'entropy' in training_history:
            axes[1, 0].plot(training_history['entropy'], color='green')
            axes[1, 0].set_title('Policy Entropy')
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Entropy')
            axes[1, 0].grid(True)
        
        # Total loss
        if 'total_loss' in training_history:
            axes[1, 1].plot(training_history['total_loss'], color='red')
            axes[1, 1].set_title('Total Loss')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metacognitive_estimation(
        self,
        predictions: Dict[str, np.ndarray],
        ground_truth: Dict[str, np.ndarray],
        save_path: str = None
    ):
        """Plot metacognitive state estimation results"""
        components = ['awareness', 'monitoring', 'control']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, component in enumerate(components):
            if component in predictions and component in ground_truth:
                pred = predictions[component].flatten()[:1000]  # Sample for visualization
                true = ground_truth[component].flatten()[:1000]
                
                axes[idx].scatter(true, pred, alpha=0.5, s=10)
                axes[idx].plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
                axes[idx].set_xlabel('Ground Truth')
                axes[idx].set_ylabel('Prediction')
                axes[idx].set_title(f'{component.capitalize()} Estimation')
                axes[idx].legend()
                axes[idx].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(
        self,
        all_metrics: Dict[str, Dict],
        save_dir: str = None
    ):
        """
        Generate comprehensive evaluation report
        
        Args:
            all_metrics: Dictionary of all evaluation metrics
            save_dir: Directory to save report
        """
        if save_dir is None:
            save_dir = self.output_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metrics as JSON
        metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        
        # Generate markdown report
        report_path = os.path.join(save_dir, 'evaluation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Evaluation Report\n\n")
            f.write("## Adaptive Multi-Modal AI Framework for Self-Regulated Learning\n\n")
            
            for category, metrics in all_metrics.items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            f.write(f"- **{key}**: {value:.4f}\n")
                        else:
                            f.write(f"- **{key}**: {value}\n")
                    f.write("\n")
        
        print(f"Evaluation report saved to {report_path}")
        print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    # Example usage
    evaluator = Evaluator(output_dir='results/evaluation')
    
    # Simulate some results
    predictions = {
        'awareness': np.random.rand(1000, 1),
        'monitoring': np.random.rand(1000, 1),
        'control': np.random.rand(1000, 1)
    }
    
    ground_truth = {
        'awareness': predictions['awareness'] + np.random.normal(0, 0.1, (1000, 1)),
        'monitoring': predictions['monitoring'] + np.random.normal(0, 0.1, (1000, 1)),
        'control': predictions['control'] + np.random.normal(0, 0.1, (1000, 1))
    }
    
    # Evaluate
    metrics = evaluator.evaluate_metacognitive_estimation(predictions, ground_truth)
    print("Metacognitive Estimation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

