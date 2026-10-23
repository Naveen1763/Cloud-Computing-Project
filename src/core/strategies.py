"""
Archetype-specific scaling strategies for AAPA.

Each strategy implements different scaling logic tailored to specific
workload patterns (SPIKE, PERIODIC, RAMP, STATIONARY_NOISY).
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ScalingStrategy(ABC):
    """Abstract base class for scaling strategies."""
    
    def __init__(self, config: dict):
        """Initialize strategy with configuration."""
        self.config = config
        self.target_cpu = config.get('target_cpu', 0.7)
        self.cooldown = config.get('cooldown', 300)  # seconds
        self.min_replicas = config.get('min_replicas', 1)
        self.max_replicas = config.get('max_replicas', 100)
        self.last_scaling_time = 0
        self.scaling_reason = ""
        
    @abstractmethod
    def calculate_target_replicas(
        self,
        current_replicas: int,
        current_cpu: float,
        current_requests: int,
        confidence: float
    ) -> int:
        """Calculate target number of replicas."""
        pass
        
    def get_scaling_reason(self) -> str:
        """Get explanation for last scaling decision."""
        return self.scaling_reason
        
    def _apply_confidence_adjustment(self, target: int, confidence: float) -> int:
        """Apply uncertainty-based adjustment to target replicas."""
        if confidence < 0.7:
            # Be more conservative when uncertain
            margin = 1 + (1 - confidence) * 0.5
            target = int(np.ceil(target * margin))
        return target


class SpikeStrategy(ScalingStrategy):
    """
    Strategy for SPIKE workloads.
    
    Features:
    - Maintains warm pool of replicas
    - Aggressive scale-up
    - Slow scale-down
    - Pre-emptive scaling based on spike detection
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.warm_pool_size = config.get('warm_pool_size', 2)
        self.spike_multiplier = config.get('spike_multiplier', 1.5)
        self.target_cpu = 0.3  # Lower target for headroom
        self.cooldown = 1200  # 20 minutes
        
    def calculate_target_replicas(
        self,
        current_replicas: int,
        current_cpu: float,
        current_requests: int,
        confidence: float
    ) -> int:
        """
        Calculate target replicas for spike workloads.
        
        Maintains warm pools and scales aggressively on load increase.
        """
        # Always maintain minimum warm pool
        min_replicas = max(self.min_replicas, self.warm_pool_size)
        
        # Calculate CPU-based target
        if current_cpu > 0:
            cpu_target = int(np.ceil(current_replicas * current_cpu / self.target_cpu))
        else:
            cpu_target = min_replicas
            
        # Apply spike multiplier for aggressive scaling
        if current_cpu > self.target_cpu:
            cpu_target = int(np.ceil(cpu_target * self.spike_multiplier))
            self.scaling_reason = f"Spike detected, scaling up aggressively (CPU: {current_cpu:.2f})"
        else:
            self.scaling_reason = f"Maintaining warm pool (CPU: {current_cpu:.2f})"
            
        # Apply confidence adjustment
        target = self._apply_confidence_adjustment(cpu_target, confidence)
        
        # Ensure within bounds
        target = max(min_replicas, min(target, self.max_replicas))
        
        return target


class PeriodicStrategy(ScalingStrategy):
    """
    Strategy for PERIODIC workloads.
    
    Features:
    - Predictive scaling based on historical patterns
    - Higher CPU utilization targets
    - Faster scale-down
    - Time-based pre-scaling
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.target_cpu = 0.75  # Higher target for efficiency
        self.cooldown = 180  # 3 minutes
        self.prediction_horizon = config.get('prediction_horizon', 900)  # 15 minutes
        self.historical_buffer = []
        
    def calculate_target_replicas(
        self,
        current_replicas: int,
        current_cpu: float,
        current_requests: int,
        confidence: float
    ) -> int:
        """
        Calculate target replicas for periodic workloads.
        
        Uses predictive scaling based on patterns.
        """
        # Store historical data
        self.historical_buffer.append((current_cpu, current_requests))
        if len(self.historical_buffer) > 60:  # Keep last hour
            self.historical_buffer.pop(0)
            
        # Calculate base target from CPU
        if current_cpu > 0:
            base_target = int(np.ceil(current_replicas * current_cpu / self.target_cpu))
        else:
            base_target = self.min_replicas
            
        # Apply prediction if enough history
        if len(self.historical_buffer) >= 10:
            predicted_load = self._predict_future_load()
            if predicted_load > current_cpu:
                base_target = int(np.ceil(current_replicas * predicted_load / self.target_cpu))
                self.scaling_reason = f"Pre-scaling for predicted load: {predicted_load:.2f}"
            else:
                self.scaling_reason = f"Normal scaling (CPU: {current_cpu:.2f})"
        else:
            self.scaling_reason = f"Insufficient history for prediction"
            
        # Apply confidence adjustment
        target = self._apply_confidence_adjustment(base_target, confidence)
        
        # Ensure within bounds
        target = max(self.min_replicas, min(target, self.max_replicas))
        
        return target
        
    def _predict_future_load(self) -> float:
        """Simple prediction based on recent trend."""
        recent_cpu = [x[0] for x in self.historical_buffer[-10:]]
        trend = np.polyfit(range(len(recent_cpu)), recent_cpu, 1)[0]
        predicted = recent_cpu[-1] + trend * 5  # 5 minutes ahead
        return max(0, min(1, predicted))


class RampStrategy(ScalingStrategy):
    """
    Strategy for RAMP workloads.
    
    Features:
    - Trend-following scaling
    - Moderate CPU targets
    - Smooth scaling transitions
    - Linear extrapolation
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.target_cpu = 0.6
        self.cooldown = 420  # 7 minutes
        self.trend_window = config.get('trend_window', 10)
        self.replica_history = []
        
    def calculate_target_replicas(
        self,
        current_replicas: int,
        current_cpu: float,
        current_requests: int,
        confidence: float
    ) -> int:
        """
        Calculate target replicas for ramp workloads.
        
        Follows trends smoothly.
        """
        # Update history
        self.replica_history.append(current_replicas)
        if len(self.replica_history) > self.trend_window:
            self.replica_history.pop(0)
            
        # Calculate base target
        if current_cpu > 0:
            base_target = int(np.ceil(current_replicas * current_cpu / self.target_cpu))
        else:
            base_target = self.min_replicas
            
        # Apply trend following if enough history
        if len(self.replica_history) >= 3:
            trend = np.polyfit(range(len(self.replica_history)), 
                              self.replica_history, 1)[0]
            if abs(trend) > 0.1:  # Significant trend
                trend_target = int(current_replicas + trend * 2)
                base_target = max(base_target, trend_target)
                self.scaling_reason = f"Following trend: {trend:.2f} replicas/min"
            else:
                self.scaling_reason = f"Stable load (CPU: {current_cpu:.2f})"
        else:
            self.scaling_reason = f"Building history"
            
        # Apply confidence adjustment
        target = self._apply_confidence_adjustment(base_target, confidence)
        
        # Ensure within bounds
        target = max(self.min_replicas, min(target, self.max_replicas))
        
        return target


class StationaryNoisyStrategy(ScalingStrategy):
    """
    Strategy for STATIONARY_NOISY workloads.
    
    Features:
    - Conservative scaling
    - Longer cooldowns
    - Noise filtering
    - Stability emphasis
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.target_cpu = 0.55
        self.cooldown = 720  # 12 minutes
        self.stability_window = config.get('stability_window', 5)
        self.cpu_history = []
        
    def calculate_target_replicas(
        self,
        current_replicas: int,
        current_cpu: float,
        current_requests: int,
        confidence: float
    ) -> int:
        """
        Calculate target replicas for stationary noisy workloads.
        
        Emphasizes stability over responsiveness.
        """
        # Update history
        self.cpu_history.append(current_cpu)
        if len(self.cpu_history) > self.stability_window:
            self.cpu_history.pop(0)
            
        # Use smoothed CPU instead of instantaneous
        if len(self.cpu_history) >= 3:
            smoothed_cpu = np.median(self.cpu_history)
        else:
            smoothed_cpu = current_cpu
            
        # Calculate target based on smoothed value
        if smoothed_cpu > 0:
            target = int(np.ceil(current_replicas * smoothed_cpu / self.target_cpu))
        else:
            target = self.min_replicas
            
        # Only scale if significant difference
        if abs(target - current_replicas) <= 1:
            target = current_replicas
            self.scaling_reason = f"Maintaining stability (smoothed CPU: {smoothed_cpu:.2f})"
        else:
            self.scaling_reason = f"Significant change detected (smoothed CPU: {smoothed_cpu:.2f})"
            
        # Apply confidence adjustment
        target = self._apply_confidence_adjustment(target, confidence)
        
        # Ensure within bounds
        target = max(self.min_replicas, min(target, self.max_replicas))
        
        return target


def create_strategies(config: dict) -> dict:
    """
    Create strategy instances from configuration.
    
    Args:
        config: Configuration dict with strategy parameters
        
    Returns:
        Dict mapping archetype names to strategy instances
    """
    strategies = {
        'SPIKE': SpikeStrategy(config.get('spike', {})),
        'PERIODIC': PeriodicStrategy(config.get('periodic', {})),
        'RAMP': RampStrategy(config.get('ramp', {})),
        'STATIONARY_NOISY': StationaryNoisyStrategy(config.get('stationary_noisy', {})),
        'default': StationaryNoisyStrategy(config.get('default', {}))
    }
    
    return strategies