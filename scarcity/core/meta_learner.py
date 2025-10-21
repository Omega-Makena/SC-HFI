"""
Meta-Learner Module

The MetaLearner class implements the meta-learning component that learns
to adapt and optimize the overall ensemble system.
"""

import logging
import numpy as np


class MetaLearner:
    """
    Meta-learning component for the hierarchical federated ensemble.
    
    The meta-learner learns across tasks and experts to improve routing decisions,
    adaptation strategies, and overall system performance.
    
    In Stage 3, it aggregates structured insights from clients instead of raw weights.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the MetaLearner.
        
        Args:
            **kwargs: Configuration parameters for meta-learning
        """
        self.logger = logging.getLogger(f"{__name__}.MetaLearner")
        self.insight_history = []  # Store historical insights for meta-learning
        
        # Reptile-style meta-learning parameters
        self.global_params = {
            "mean": 0.0,
            "std": 1.0,
            "learning_rate": 0.01,
            "num_updates": 0
        }
        self.running_statistics = {
            "all_means": [],
            "all_stds": [],
            "all_losses": []
        }
        
        self.logger.info("Initializing MetaLearner with Reptile-style meta-learning")
        
    def aggregate(self, insights: list):
        """
        Aggregate structured insights from clients.
        
        This is the Scarcity Framework's key innovation: instead of averaging
        raw model weights, we process high-level insights about learning dynamics.
        
        Args:
            insights: List of insight dictionaries from clients
            
        Returns:
            Dictionary of aggregated knowledge
        """
        self.logger.info(f"MetaLearner: Aggregating {len(insights)} insights")
        
        if not insights:
            self.logger.warning("MetaLearner: No insights to aggregate")
            return {}
        
        # Store insights for future meta-learning
        self.insight_history.extend(insights)
        
        # Check if insights are expert-based or traditional
        is_expert_mode = 'selected_expert' in insights[0]
        
        if is_expert_mode:
            # Expert-based aggregation
            loss_improvements = [insight["loss_improvement"] for insight in insights]
            final_losses = [insight["final_loss"] for insight in insights]
            
            # Count expert usage
            expert_counts = {}
            for insight in insights:
                expert_name = insight["selected_expert"]
                expert_counts[expert_name] = expert_counts.get(expert_name, 0) + 1
            
            aggregated_knowledge = {
                "avg_loss_improvement": float(np.mean(loss_improvements)),
                "avg_final_loss": float(np.mean(final_losses)),
                "expert_usage": expert_counts,
                "num_insights": len(insights),
                "total_insights_seen": len(self.insight_history),
                "mode": "expert_routing"
            }
            
            self.logger.info(
                f"MetaLearner: Expert-based aggregation - "
                f"Avg loss improvement: {aggregated_knowledge['avg_loss_improvement']:.4f}, "
                f"Avg final loss: {aggregated_knowledge['avg_final_loss']:.4f}"
            )
            self.logger.info(
                f"MetaLearner: Expert usage - {expert_counts}"
            )
            
        else:
            # Traditional aggregation with uncertainty
            uncertainties = [insight["uncertainty"] for insight in insights]
            mean_grads = [insight["mean_grad"] for insight in insights]
            loss_improvements = [insight["loss_improvement"] for insight in insights]
            final_losses = [insight["final_loss"] for insight in insights]
            
            # Aggregate across clients
            avg_uncertainty = np.mean(uncertainties)
            std_uncertainty = np.std(uncertainties)
            avg_mean_grad = np.mean(mean_grads)
            avg_loss_improvement = np.mean(loss_improvements)
            avg_final_loss = np.mean(final_losses)
            
            # Identify high and low uncertainty clients
            high_uncertainty_clients = [
                i["client_id"] for i in insights 
                if i["uncertainty"] > avg_uncertainty
            ]
            low_uncertainty_clients = [
                i["client_id"] for i in insights 
                if i["uncertainty"] <= avg_uncertainty
            ]
            
            aggregated_knowledge = {
                "avg_uncertainty": float(avg_uncertainty),
                "std_uncertainty": float(std_uncertainty),
                "avg_mean_grad": float(avg_mean_grad),
                "avg_loss_improvement": float(avg_loss_improvement),
                "avg_final_loss": float(avg_final_loss),
                "high_uncertainty_clients": high_uncertainty_clients,
                "low_uncertainty_clients": low_uncertainty_clients,
                "num_insights": len(insights),
                "total_insights_seen": len(self.insight_history),
                "mode": "uncertainty_based"
            }
            
            self.logger.info(
                f"MetaLearner: Aggregated knowledge - "
                f"Avg uncertainty: {avg_uncertainty:.4f} (±{std_uncertainty:.4f}), "
                f"Avg grad: {avg_mean_grad:.4f}, "
                f"Avg loss improvement: {avg_loss_improvement:.4f}"
            )
            
            self.logger.info(
                f"MetaLearner: High uncertainty clients: {high_uncertainty_clients}"
            )
            self.logger.info(
                f"MetaLearner: Low uncertainty clients: {low_uncertainty_clients}"
            )
        
        return aggregated_knowledge
    
    def meta_train(self, tasks, experts):
        """
        Perform meta-training across multiple tasks and experts.
        
        Args:
            tasks: Collection of tasks for meta-training
            experts: Available experts in the ensemble
        """
        self.logger.info("MetaLearner: meta_train() called")
        # TODO: Implement meta-training logic
        pass
    
    def adapt(self, task, num_steps: int = 5):
        """
        Adapt to a new task using meta-learned knowledge.
        
        Args:
            task: New task to adapt to
            num_steps: Number of adaptation steps
            
        Returns:
            Adapted model or parameters
        """
        self.logger.info("MetaLearner: adapt() called")
        # TODO: Implement adaptation logic
        pass
    
    def optimize_ensemble(self, performance_data):
        """
        Optimize ensemble configuration based on performance data.
        
        Args:
            performance_data: Historical performance metrics
        """
        self.logger.info("MetaLearner: optimize_ensemble() called")
        # TODO: Implement ensemble optimization
        pass
    
    def suggest_experts(self, task_description):
        """
        Suggest which experts to use for a given task.
        
        Args:
            task_description: Description or features of the task
            
        Returns:
            Recommended expert configuration
        """
        self.logger.info("MetaLearner: suggest_experts() called")
        # TODO: Implement expert suggestion logic
        pass
    
    def update_global_params(self, insights: list):
        """
        Update global meta-parameters based on insights from all clients.
        
        This implements a simple Reptile-style approach where we track
        statistics across rounds and use them to provide better initialization.
        
        Args:
            insights: List of insight dictionaries from clients
            
        Returns:
            Updated global parameters
        """
        self.logger.info(f"MetaLearner: Updating global params from {len(insights)} clients")
        
        if not insights:
            self.logger.warning("MetaLearner: No insights to update from")
            return self.global_params
        
        # Extract statistics from insights
        current_means = []
        current_stds = []
        current_losses = []
        
        for insight in insights:
            # Handle expert-based insights
            if 'expert_summaries' in insight:
                for summary in insight['expert_summaries']:
                    if summary.get('expert_type') == 'StructureExpert':
                        current_means.append(summary.get('mean', 0.0))
                        current_stds.append(summary.get('std', 1.0))
                
                if 'final_loss' in insight:
                    current_losses.append(insight['final_loss'])
            
            # Handle uncertainty-based insights
            elif 'mean_grad' in insight:
                # Use gradient info as proxy
                current_losses.append(insight.get('final_loss', 0.0))
        
        # Update running statistics
        if current_means:
            self.running_statistics['all_means'].extend(current_means)
            self.running_statistics['all_stds'].extend(current_stds)
        if current_losses:
            self.running_statistics['all_losses'].extend(current_losses)
        
        # Compute global parameters using running average
        if self.running_statistics['all_means']:
            # Use exponential moving average for stability
            alpha = 0.1  # Learning rate for meta-learning
            new_mean = np.mean(current_means) if current_means else 0.0
            new_std = np.mean(current_stds) if current_stds else 1.0
            
            self.global_params['mean'] = (1 - alpha) * self.global_params['mean'] + alpha * new_mean
            self.global_params['std'] = (1 - alpha) * self.global_params['std'] + alpha * new_std
        
        if current_losses:
            avg_loss = np.mean(current_losses)
            # Adapt learning rate based on loss (simple heuristic)
            if avg_loss > 10.0:
                self.global_params['learning_rate'] = 0.05  # Higher LR for high loss
            elif avg_loss > 5.0:
                self.global_params['learning_rate'] = 0.02
            else:
                self.global_params['learning_rate'] = 0.01  # Standard LR
        
        self.global_params['num_updates'] += 1
        
        self.logger.info(
            f"MetaLearner: Updated global params - "
            f"mean={self.global_params['mean']:.4f}, "
            f"std={self.global_params['std']:.4f}, "
            f"lr={self.global_params['learning_rate']:.4f}, "
            f"updates={self.global_params['num_updates']}"
        )
        
        return self.global_params.copy()
    
    def broadcast_params(self):
        """
        Broadcast current global meta-parameters to clients.
        
        Returns:
            Dictionary of global initialization parameters
        """
        self.logger.info(
            f"MetaLearner: Broadcasting params - "
            f"mean={self.global_params['mean']:.4f}, "
            f"std={self.global_params['std']:.4f}, "
            f"lr={self.global_params['learning_rate']:.4f}"
        )
        
        return {
            "meta_mean": self.global_params['mean'],
            "meta_std": self.global_params['std'],
            "meta_lr": self.global_params['learning_rate'],
            "meta_updates": self.global_params['num_updates']
        }
    
    def summarize(self):
        """
        Generate a summary of meta-learning progress and insights.
        
        Returns:
            Summary of meta-learner state and performance
        """
        summary = {
            "global_params": self.global_params.copy(),
            "total_insights": len(self.insight_history),
            "running_stats": {
                "num_means": len(self.running_statistics['all_means']),
                "num_stds": len(self.running_statistics['all_stds']),
                "num_losses": len(self.running_statistics['all_losses'])
            }
        }
        
        if self.running_statistics['all_losses']:
            summary["avg_historical_loss"] = float(np.mean(self.running_statistics['all_losses']))
        
        self.logger.info(f"MetaLearner: Summary - {summary}")
        return summary

