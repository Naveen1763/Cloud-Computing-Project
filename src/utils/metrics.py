import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class ResourceEfficiencyIndex:
    """Compute Resource Efficiency Index (REI) for autoscaler evaluation."""
    
    def __init__(self, 
                 slo_weight: float = 0.4,
                 efficiency_weight: float = 0.3,
                 stability_weight: float = 0.3):
        """Initialize REI calculator with weights.
        
        Args:
            slo_weight: Weight for SLO satisfaction (default: 0.4)
            efficiency_weight: Weight for resource efficiency (default: 0.3)
            stability_weight: Weight for scaling stability (default: 0.3)
        """
        assert abs(slo_weight + efficiency_weight + stability_weight - 1.0) < 1e-6
        
        self.weights = {
            'slo': slo_weight,
            'efficiency': efficiency_weight,
            'stability': stability_weight
        }
        
    def compute_rei(self, metrics: Dict[str, float]) -> float:
        """Compute overall REI score.
        
        Args:
            metrics: Dictionary containing:
                - slo_satisfaction_rate: (0-1) rate of meeting SLOs
                - resource_efficiency: (0-1) average resource utilization
                - scaling_stability: (0-1) stability score
                
        Returns:
            REI score (0-1, higher is better)
        """
        slo_score = metrics.get('slo_satisfaction_rate', 0)
        efficiency_score = metrics.get('resource_efficiency', 0)
        stability_score = metrics.get('scaling_stability', 0)
        
        rei = (self.weights['slo'] * slo_score +
               self.weights['efficiency'] * efficiency_score +
               self.weights['stability'] * stability_score)
        
        return rei
    
    def compute_archetype_specific_rei(self, 
                                     metrics: Dict[str, float],
                                     archetype: str) -> float:
        """Compute REI with archetype-specific weights.
        
        Different workload types have different priorities:
        - SPIKE: Prioritize SLO (avoid cold starts)
        - PERIODIC: Prioritize efficiency (predictable load)
        - RAMP: Balance all three
        - STATIONARY_NOISY: Prioritize stability (avoid oscillation)
        """
        archetype_weights = {
            'SPIKE': {'slo': 0.6, 'efficiency': 0.2, 'stability': 0.2},
            'PERIODIC': {'slo': 0.3, 'efficiency': 0.5, 'stability': 0.2},
            'RAMP': {'slo': 0.35, 'efficiency': 0.35, 'stability': 0.3},
            'STATIONARY_NOISY': {'slo': 0.3, 'efficiency': 0.3, 'stability': 0.4}
        }
        
        weights = archetype_weights.get(archetype, self.weights)
        
        slo_score = metrics.get('slo_satisfaction_rate', 0)
        efficiency_score = metrics.get('resource_efficiency', 0)
        stability_score = metrics.get('scaling_stability', 0)
        
        rei = (weights['slo'] * slo_score +
               weights['efficiency'] * efficiency_score +
               weights['stability'] * stability_score)
        
        return rei


class AutoscalerMetrics:
    """Calculate detailed metrics for autoscaler evaluation."""
    
    def __init__(self, slo_threshold_ms: float = 500):
        """Initialize metrics calculator.
        
        Args:
            slo_threshold_ms: Response time threshold for SLO (default: 500ms)
        """
        self.slo_threshold_ms = slo_threshold_ms
        
    def calculate_slo_metrics(self, response_times: List[float]) -> Dict[str, float]:
        """Calculate SLO-related metrics.
        
        Args:
            response_times: List of response times in milliseconds
            
        Returns:
            Dictionary with SLO metrics
        """
        if not response_times:
            return {
                'slo_satisfaction_rate': 0.0,
                'avg_response_time': 0.0,
                'p95_response_time': 0.0,
                'p99_response_time': 0.0,
                'cold_start_rate': 0.0
            }
            
        response_times = np.array(response_times)
        violations = response_times > self.slo_threshold_ms
        
        # Detect cold starts (response time > 5x normal)
        median_rt = np.median(response_times)
        cold_starts = response_times > (5 * median_rt)
        
        return {
            'slo_satisfaction_rate': 1.0 - np.mean(violations),
            'avg_response_time': np.mean(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'p99_response_time': np.percentile(response_times, 99),
            'cold_start_rate': np.mean(cold_starts)
        }
    
    def calculate_efficiency_metrics(self, 
                                   utilization_series: List[float],
                                   replica_counts: List[int]) -> Dict[str, float]:
        """Calculate resource efficiency metrics.
        
        Args:
            utilization_series: CPU/memory utilization over time (0-1)
            replica_counts: Number of replicas over time
            
        Returns:
            Dictionary with efficiency metrics
        """
        if not utilization_series or not replica_counts:
            return {
                'resource_efficiency': 0.0,
                'avg_utilization': 0.0,
                'over_provisioning_rate': 0.0,
                'under_provisioning_rate': 0.0
            }
            
        utilization = np.array(utilization_series)
        replicas = np.array(replica_counts)
        
        # Calculate average utilization
        avg_util = np.mean(utilization)
        
        # Resource efficiency: how close to target utilization (e.g., 0.7)
        target_util = 0.7
        efficiency = 1.0 - np.mean(np.abs(utilization - target_util))
        
        # Over/under provisioning
        over_provisioned = utilization < 0.5  # Less than 50% utilized
        under_provisioned = utilization > 0.9  # More than 90% utilized
        
        return {
            'resource_efficiency': efficiency,
            'avg_utilization': avg_util,
            'over_provisioning_rate': np.mean(over_provisioned),
            'under_provisioning_rate': np.mean(under_provisioned),
            'total_replica_minutes': np.sum(replicas)
        }
    
    def calculate_stability_metrics(self, 
                                  replica_counts: List[int],
                                  scaling_actions: List[Tuple[int, str]]) -> Dict[str, float]:
        """Calculate scaling stability metrics.
        
        Args:
            replica_counts: Number of replicas over time
            scaling_actions: List of (timestamp, action) tuples
            
        Returns:
            Dictionary with stability metrics
        """
        if not replica_counts:
            return {
                'scaling_stability': 0.0,
                'oscillation_rate': 0.0,
                'avg_scaling_interval': 0.0,
                'scaling_actions_per_hour': 0.0
            }
            
        replicas = np.array(replica_counts)
        
        # Calculate oscillation (rapid up/down scaling)
        replica_changes = np.diff(replicas)
        oscillations = 0
        for i in range(1, len(replica_changes)):
            if replica_changes[i] * replica_changes[i-1] < 0:  # Sign change
                oscillations += 1
                
        oscillation_rate = oscillations / len(replica_changes) if len(replica_changes) > 0 else 0
        
        # Scaling frequency
        total_time_minutes = len(replica_counts)
        scaling_actions_per_hour = len(scaling_actions) / (total_time_minutes / 60) if total_time_minutes > 0 else 0
        
        # Average interval between scaling actions
        if len(scaling_actions) > 1:
            intervals = [scaling_actions[i][0] - scaling_actions[i-1][0] 
                        for i in range(1, len(scaling_actions))]
            avg_interval = np.mean(intervals)
        else:
            avg_interval = total_time_minutes
            
        # Stability score (inverse of oscillation and scaling frequency)
        stability = 1.0 / (1.0 + oscillation_rate + scaling_actions_per_hour / 10)
        
        return {
            'scaling_stability': stability,
            'oscillation_rate': oscillation_rate,
            'avg_scaling_interval': avg_interval,
            'scaling_actions_per_hour': scaling_actions_per_hour
        }
    
    def calculate_all_metrics(self,
                            response_times: List[float],
                            utilization_series: List[float],
                            replica_counts: List[int],
                            scaling_actions: List[Tuple[int, str]]) -> Dict[str, float]:
        """Calculate all metrics for REI computation.
        
        Returns:
            Dictionary with all metrics including component scores
        """
        slo_metrics = self.calculate_slo_metrics(response_times)
        efficiency_metrics = self.calculate_efficiency_metrics(utilization_series, replica_counts)
        stability_metrics = self.calculate_stability_metrics(replica_counts, scaling_actions)
        
        # Combine all metrics
        all_metrics = {}
        all_metrics.update(slo_metrics)
        all_metrics.update(efficiency_metrics)
        all_metrics.update(stability_metrics)
        
        return all_metrics


if __name__ == "__main__":
    # Test metrics calculation
    print("Testing Autoscaler Metrics and REI...")
    
    # Simulate some data
    np.random.seed(42)
    
    # Scenario 1: Good autoscaler
    print("\n=== Scenario 1: Well-tuned Autoscaler ===")
    response_times = np.random.normal(200, 50, 1000)  # Good response times
    utilization = np.random.normal(0.7, 0.1, 100)  # Around target
    replicas = [10] * 50 + [15] * 50  # Stable scaling
    scaling_actions = [(30, 'scale_up')]  # Minimal actions
    
    metrics_calc = AutoscalerMetrics()
    metrics = metrics_calc.calculate_all_metrics(
        response_times.tolist(),
        utilization.tolist(),
        replicas,
        scaling_actions
    )
    
    rei_calc = ResourceEfficiencyIndex()
    rei = rei_calc.compute_rei(metrics)
    
    print(f"SLO Satisfaction: {metrics['slo_satisfaction_rate']:.3f}")
    print(f"Resource Efficiency: {metrics['resource_efficiency']:.3f}")
    print(f"Scaling Stability: {metrics['scaling_stability']:.3f}")
    print(f"Overall REI: {rei:.3f}")
    
    # Scenario 2: Poor autoscaler
    print("\n=== Scenario 2: Poorly-tuned Autoscaler ===")
    response_times = np.random.normal(600, 200, 1000)  # Many violations
    utilization = np.random.uniform(0.2, 0.95, 100)  # Erratic
    replicas = [5, 20, 5, 25, 10, 30, 5] * 14  # Oscillating
    scaling_actions = [(i*5, 'scale') for i in range(20)]  # Frequent
    
    metrics = metrics_calc.calculate_all_metrics(
        response_times.tolist(),
        utilization.tolist(),
        replicas[:100],
        scaling_actions
    )
    
    rei = rei_calc.compute_rei(metrics)
    
    print(f"SLO Satisfaction: {metrics['slo_satisfaction_rate']:.3f}")
    print(f"Resource Efficiency: {metrics['resource_efficiency']:.3f}")
    print(f"Scaling Stability: {metrics['scaling_stability']:.3f}")
    print(f"Overall REI: {rei:.3f}")
    
    # Test archetype-specific REI
    print("\n=== Archetype-specific REI ===")
    for archetype in ['SPIKE', 'PERIODIC', 'RAMP', 'STATIONARY_NOISY']:
        rei_archetype = rei_calc.compute_archetype_specific_rei(metrics, archetype)
        print(f"{archetype}: {rei_archetype:.3f}")