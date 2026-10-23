#!/usr/bin/env python3
"""
Phase 3 v2: Improved SimPy-based serverless autoscaling simulator.
Fixed issues with warm pools, scaling intervals, and spike prediction.
"""

import simpy
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import pickle
import yaml
from collections import deque

from src.models.uncertainty import ConformalScalingPolicy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Request:
    """Represents a serverless function invocation request."""
    arrival_time: float
    function_id: str
    execution_time: float
    memory_mb: int = 128
    
    
@dataclass
class Pod:
    """Represents a Kubernetes pod running function instances."""
    id: str
    cpu_cores: float = 0.1  # 100m
    memory_mb: int = 128
    startup_time: float = 2.0  # Cold start time in seconds
    is_warm: bool = False
    last_used: float = 0.0
    warmup_time: float = 0.0  # Time when pod was warmed up
    

@dataclass
class Metrics:
    """Tracks simulation metrics."""
    total_requests: int = 0
    completed_requests: int = 0
    slo_violations: int = 0
    cold_starts: int = 0
    total_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)
    pod_minutes: float = 0.0
    scaling_actions: int = 0
    queued_time: float = 0.0  # Time spent in queue
    
    def get_slo_violation_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.slo_violations / self.total_requests
    
    def get_mean_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return np.mean(self.response_times)
    
    def get_p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return np.percentile(self.response_times, 99)


class Autoscaler(ABC):
    """Abstract base class for autoscaling strategies."""
    
    @abstractmethod
    def get_desired_replicas(self, current_load: float, current_replicas: int, 
                           history: List[float], predictions: Dict[str, Any]) -> int:
        """Calculate desired number of replicas."""
        pass
        

class HPAAutoscaler(Autoscaler):
    """Standard Kubernetes Horizontal Pod Autoscaler (reactive)."""
    
    def __init__(self, target_cpu_utilization: float = 0.7, 
                 scale_down_cooldown: float = 300):
        self.target_cpu_utilization = target_cpu_utilization
        self.scale_down_cooldown = scale_down_cooldown
        self.last_scale_down = -float('inf')
        
    def get_desired_replicas(self, current_load: float, current_replicas: int,
                           history: List[float], predictions: Dict[str, Any]) -> int:
        # Simple CPU-based scaling
        if current_replicas == 0:
            return 1
            
        # Estimate CPU utilization based on load
        cpu_per_request = 0.01  # 10m CPU per request
        current_cpu = current_load * cpu_per_request
        avg_cpu_per_replica = current_cpu / current_replicas
        
        # Calculate desired replicas
        desired = int(np.ceil(current_replicas * avg_cpu_per_replica / self.target_cpu_utilization))
        
        # Apply scale-down cooldown
        if desired < current_replicas:
            if predictions.get('time', 0) - self.last_scale_down < self.scale_down_cooldown:
                return current_replicas
            self.last_scale_down = predictions.get('time', 0)
            
        return max(1, desired)


class GenericPredictiveAutoscaler(Autoscaler):
    """Generic predictive autoscaler using simple forecasting."""
    
    def __init__(self, prediction_window: int = 5, target_utilization: float = 0.7):
        self.prediction_window = prediction_window
        self.target_utilization = target_utilization
        
    def get_desired_replicas(self, current_load: float, current_replicas: int,
                           history: List[float], predictions: Dict[str, Any]) -> int:
        # Simple linear regression prediction
        if len(history) < 3:
            # Fall back to reactive if not enough history
            cpu_per_request = 0.01
            current_cpu = current_load * cpu_per_request
            if current_replicas == 0:
                return 1
            avg_cpu_per_replica = current_cpu / current_replicas
            return max(1, int(np.ceil(current_replicas * avg_cpu_per_replica / self.target_utilization)))
            
        # Predict future load using linear trend
        x = np.arange(len(history))
        y = np.array(history)
        
        # Simple linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # Predict load for next window
        predicted_load = m * (len(history) + self.prediction_window) + c
        predicted_load = max(0, predicted_load)
        
        # Calculate required replicas
        cpu_per_request = 0.01
        predicted_cpu = predicted_load * cpu_per_request
        replica_capacity = 0.1 * self.target_utilization  # 100m CPU * target util
        
        return max(1, int(np.ceil(predicted_cpu / replica_capacity)))


class AAPAAutoscaler(Autoscaler):
    """Improved Archetype-Aware Predictive Autoscaler with better spike handling."""
    
    def __init__(self, model_path: str = "../models/lightgbm_model.txt",
                 encoder_path: str = "../models/label_encoder.pkl"):
        # Load classification model
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=model_path)
        
        # Load label encoder
        with open(encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
            
        # Improved archetype-specific strategies
        self.strategies = {
            'SPIKE': {
                'target_util': 0.3,  # Lower target for headroom
                'scale_down_cooldown': 600,  # 10 min - shorter
                'warm_pool_size': 3,  # More warm pods
                'burst_multiplier': 2.0,  # More aggressive
                'spike_threshold': 1.5,  # Lower threshold
                'predictive_window': 3  # Look ahead 3 intervals
            },
            'PERIODIC': {
                'target_util': 0.75,
                'scale_down_cooldown': 300,
                'warm_pool_size': 1,
                'burst_multiplier': 1.2,
                'spike_threshold': 3.0,
                'predictive_window': 5
            },
            'RAMP': {
                'target_util': 0.6,
                'scale_down_cooldown': 600,
                'warm_pool_size': 1,
                'burst_multiplier': 1.3,
                'spike_threshold': 2.5,
                'predictive_window': 4
            },
            'STATIONARY_NOISY': {
                # 0.55 per the paper's Table III and src/core/strategies.py.
                # This file previously used 0.65, so the simulator ran a
                # different STATIONARY policy than the one being cited.
                'target_util': 0.55,
                'scale_down_cooldown': 900,  # Longer for stability
                'warm_pool_size': 1,
                'burst_multiplier': 1.1,
                'spike_threshold': 4.0,
                'predictive_window': 2
            }
        }
        
        self.last_scale_down = -float('inf')
        self.current_archetype = None
        self.confidence = 0.0
        self.spike_history = deque(maxlen=10)  # Track recent spike events
        
    def classify_workload(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify workload and return archetype with confidence."""
        # Get prediction probabilities
        probs = self.model.predict(features.reshape(1, -1))[0]
        
        # Get predicted class and confidence
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        archetype = self.label_encoder.inverse_transform([pred_idx])[0]
        
        return archetype, confidence
        
    def _select_strategy(self) -> Dict:
        """Scaling parameters for the current prediction.

        Subclasses may widen this - see ConformalAAPAAutoscaler, which returns
        an envelope over every archetype the classifier could not rule out.
        """
        return self.strategies[self.current_archetype]

    def predict_spike(self, history: List[float], strategy: Dict) -> bool:
        """Improved spike prediction using pattern recognition."""
        if len(history) < 5:
            return False
            
        recent = history[-5:]
        
        # Check for rapid increase pattern
        if len(recent) >= 2:
            growth_rate = (recent[-1] - recent[-2]) / (recent[-2] + 1e-6)
            if growth_rate > 0.5:  # 50% growth
                return True
                
        # Check if we're in a low period after recent spikes
        if self.spike_history and len(self.spike_history) >= 2:
            # If we had spikes recently and load is low, prepare for next spike
            avg_recent = np.mean(recent)
            if avg_recent < np.median(history) * 0.5:
                return True
                
        return False
        
    def get_desired_replicas(self, current_load: float, current_replicas: int,
                           history: List[float], predictions: Dict[str, Any]) -> int:
        # Extract features if provided
        if 'features' in predictions:
            self.current_archetype, self.confidence = self.classify_workload(predictions['features'])
        
        if not self.current_archetype:
            # Fall back to generic if no classification
            return GenericPredictiveAutoscaler().get_desired_replicas(
                current_load, current_replicas, history, predictions
            )
            
        # Get strategy for current archetype
        strategy = self._select_strategy()
        
        # For SPIKE workloads with low confidence, be MORE aggressive, not less
        if self.current_archetype == 'SPIKE' and self.confidence < 0.8:
            adjusted_target_util = strategy['target_util'] * 0.8  # Even lower target
        else:
            adjusted_target_util = strategy['target_util']
        
        # Calculate base replicas needed
        cpu_per_request = 0.01
        current_cpu = current_load * cpu_per_request
        replica_capacity = 0.1 * adjusted_target_util
        
        base_replicas = int(np.ceil(current_cpu / replica_capacity))
        
        # Enhanced archetype-specific adjustments
        if self.current_archetype == 'SPIKE':
            # Check for spike indicators
            spike_detected = False
            
            # Current spike detection
            if len(history) >= 2:
                if current_load > strategy['spike_threshold'] * np.median(history[-10:]):
                    spike_detected = True
                    self.spike_history.append(predictions.get('time', 0))
                    
            # Predictive spike detection
            if self.predict_spike(history, strategy):
                spike_detected = True
                
            if spike_detected:
                # Aggressive scaling for spikes
                base_replicas = int(base_replicas * strategy['burst_multiplier'])
                # Ensure minimum replicas during spike
                base_replicas = max(base_replicas, 5)
                
        elif self.current_archetype == 'PERIODIC':
            # Enhanced periodic prediction
            if len(history) >= 60:
                # Use last hour for pattern detection
                y = np.array(history[-60:])
                
                # Simple peak detection
                peaks = []
                for i in range(1, len(y)-1):
                    if y[i] > y[i-1] and y[i] > y[i+1] and y[i] > np.median(y) * 1.5:
                        peaks.append(i)
                        
                # If we're approaching a typical peak time
                if peaks and len(history) % 20 < 5:  # Near 20-min cycle
                    base_replicas = int(base_replicas * 1.2)
                    
        elif self.current_archetype == 'RAMP':
            # Trend following with momentum
            if len(history) >= 5:
                recent_trend = np.polyfit(range(5), history[-5:], 1)[0]
                if recent_trend > 0.1:  # Positive trend
                    # Scale ahead of the trend
                    future_load = current_load + recent_trend * strategy['predictive_window']
                    future_cpu = future_load * cpu_per_request
                    base_replicas = int(np.ceil(future_cpu / replica_capacity))
        
        # Always maintain warm pool
        desired = base_replicas + strategy['warm_pool_size']
        
        # Apply scale-down cooldown
        if desired < current_replicas:
            if predictions.get('time', 0) - self.last_scale_down < strategy['scale_down_cooldown']:
                return current_replicas
            self.last_scale_down = predictions.get('time', 0)
            
        return max(1, desired)


class ConformalAAPAAutoscaler(AAPAAutoscaler):
    """AAPA driven by conformal prediction sets instead of a softmax threshold.

    The parent picks one archetype by argmax and softens its strategy when the
    softmax score looks low. That score is uncalibrated - the booster reports
    ~0.98 confidence while being 96% accurate, and worse, it under-covers SPIKE
    specifically - so the softening rarely fires when it should.

    This version asks the conformal predictor which archetypes cannot be ruled
    out at level alpha, then provisions for the worst case among them. If the
    set covers the truth with probability >= 1 - alpha, the resulting capacity
    is adequate with probability >= 1 - alpha. When the set is a singleton the
    behaviour is identical to the parent, so any difference in the results
    comes entirely from the windows the model is genuinely unsure about.
    """

    def __init__(self, model_path: str, encoder_path: str,
                 conformal_path: str, alpha: Optional[float] = None):
        super().__init__(model_path=model_path, encoder_path=encoder_path)

        with open(conformal_path, 'rb') as f:
            self.conformal = pickle.load(f)

        if alpha is not None:
            self.conformal.alpha = alpha

        self.policy = ConformalScalingPolicy(
            self.strategies, alpha=self.conformal.alpha
        )
        self.prediction_set: List[str] = []
        # Diagnostics, surfaced by run_simulation.py
        self.set_sizes: List[int] = []

    def classify_workload(self, features: np.ndarray) -> Tuple[str, float]:
        """Classify, and record the conformal set alongside the top-1 label."""
        probs = self.model.predict(features.reshape(1, -1))

        info = self.conformal.predict_with_uncertainty(probs)[0]
        self.prediction_set = [str(a) for a in info['prediction_set']]
        self.set_sizes.append(info['set_size'])

        return str(info['archetype']), float(info['softmax_confidence'])

    def _select_strategy(self) -> Dict:
        """Conservative envelope over every archetype still in play."""
        if not self.prediction_set:
            return super()._select_strategy()
        return self.policy.resolve(self.prediction_set)


class ConformalTargetOnlyAutoscaler(ConformalAAPAAutoscaler):
    """Set-targeted hedging on the CPU target alone.

    This is the minimal form of the idea: given the conformal set, compute the
    replicas each member archetype would require and take the maximum, i.e.

        replicas = max over a in C(x) of ceil(load / target_cpu[a])

    Because capacity is inversely proportional to the CPU target, the maximum
    is attained by whichever plausible archetype has the lowest target, so this
    is exactly "provision for the most demanding archetype still in play".

    ConformalAAPAAutoscaler additionally envelopes warm pool, cooldown, burst
    multiplier and spike threshold. Running both isolates how much of the
    benefit comes from the CPU target alone versus the full envelope.
    """

    def get_desired_replicas(self, current_load: float, current_replicas: int,
                             history: List[float], predictions: Dict[str, Any]) -> int:
        if 'features' in predictions:
            self.current_archetype, self.confidence = self.classify_workload(
                predictions['features']
            )

        if not self.prediction_set:
            return super().get_desired_replicas(
                current_load, current_replicas, history, predictions
            )

        cpu_per_request = 0.01
        current_cpu = current_load * cpu_per_request

        candidates = []
        for archetype in self.prediction_set:
            strategy = self.strategies.get(archetype)
            if strategy is None:
                continue
            replica_capacity = 0.1 * strategy['target_util']
            candidates.append(int(np.ceil(current_cpu / replica_capacity)))

        if not candidates:
            return super().get_desired_replicas(
                current_load, current_replicas, history, predictions
            )

        desired = max(candidates)

        # Cooldown from the top-1 archetype, matching the reference controller,
        # which applies no per-archetype cooldown widening.
        strategy = self.strategies[self.current_archetype]
        if desired < current_replicas:
            if predictions.get('time', 0) - self.last_scale_down < strategy['scale_down_cooldown']:
                return current_replicas
            self.last_scale_down = predictions.get('time', 0)

        return max(1, desired)


class Top1TargetOnlyAutoscaler(ConformalTargetOnlyAutoscaler):
    """Ablation control: archetype-targeted scaling WITHOUT the conformal set.

    Identical to ConformalTargetOnlyAutoscaler except that it sizes for the
    single argmax archetype rather than hedging over the prediction set. It
    therefore also drops the warm pools and burst logic that the full
    AAPAAutoscaler carries.

    This separates two effects that are otherwise confounded:
      * removing warm pools / burst multipliers, and
      * hedging over a conformal set.

    If this control matches ConformalTargetOnlyAutoscaler, the gain came from
    simplifying the strategy, not from conformal prediction, and the
    contribution must be stated accordingly.
    """

    def _plausible_archetypes(self) -> List[str]:
        return [self.current_archetype] if self.current_archetype else []

    def get_desired_replicas(self, current_load: float, current_replicas: int,
                             history: List[float], predictions: Dict[str, Any]) -> int:
        if 'features' in predictions:
            self.current_archetype, self.confidence = self.classify_workload(
                predictions['features']
            )
        # Collapse the set to the top-1 label, then reuse the parent's sizing.
        self.prediction_set = self._plausible_archetypes()
        return super().get_desired_replicas(
            current_load, current_replicas, history, predictions
        )


class ServerlessSimulator:
    """Improved simulation environment with better request handling."""
    
    def __init__(self, env: simpy.Environment, autoscaler: Autoscaler,
                 slo_threshold: float = 0.5):  # 500ms SLO
        self.env = env
        self.autoscaler = autoscaler
        self.slo_threshold = slo_threshold
        
        # Resources
        self.pods: List[Pod] = []
        self.request_queue = simpy.Store(env)
        
        # Metrics
        self.metrics = Metrics()
        self.load_history: List[float] = []
        self.request_rate_history: List[float] = []  # Track actual request rates
        
        # Improved configuration
        self.pod_startup_time = 2.0  # seconds
        self.scale_interval = 5.0  # Reduced from 15s for faster response
        self.pod_warmup_time = 0.1  # Time to warm up a pod
        self.request_buffer_size = 100  # Buffer requests instead of dropping
        
        # Track request arrivals for better load calculation
        self.recent_arrivals = deque(maxlen=100)
        
    def request_generator(self, workload: List[float], function_id: str,
                         execution_time_dist: Callable):
        """Generate requests based on workload trace."""
        for minute, rate in enumerate(workload):
            # Generate requests for this minute
            if rate > 0:
                # Poisson process within the minute
                inter_arrival = 60.0 / rate if rate > 0 else float('inf')
                
                start_time = self.env.now
                while self.env.now < start_time + 60:
                    # Create request
                    req = Request(
                        arrival_time=self.env.now,
                        function_id=function_id,
                        execution_time=execution_time_dist()
                    )
                    
                    self.metrics.total_requests += 1
                    self.recent_arrivals.append(self.env.now)
                    
                    # Try to put in queue with timeout
                    try:
                        yield self.request_queue.put(req)
                    except:
                        # Queue full - this is an SLO violation
                        self.metrics.slo_violations += 1
                    
                    # Wait for next request
                    yield self.env.timeout(np.random.exponential(inter_arrival))
                    
            # Wait for next minute
            yield self.env.timeout(max(0, (minute + 1) * 60 - self.env.now))
            
    def pod_warmer(self):
        """Process that keeps pods warm by running dummy requests."""
        while True:
            # Check all pods and warm up cold ones
            for pod in self.pods:
                if not pod.is_warm and self.env.now - pod.warmup_time > self.pod_warmup_time:
                    # Warm up the pod
                    pod.is_warm = True
                    pod.last_used = self.env.now
                    
            # Run every second
            yield self.env.timeout(1.0)
            
    def request_handler(self):
        """Improved request processing with better pod selection."""
        while True:
            # Get request from queue
            req = yield self.request_queue.get()
            
            # Calculate queue time
            queue_time = self.env.now - req.arrival_time
            self.metrics.queued_time += queue_time
            
            # Find best available pod (prefer warm pods)
            selected_pod = None
            warm_pods = [p for p in self.pods if p.is_warm]
            cold_pods = [p for p in self.pods if not p.is_warm]
            
            if warm_pods:
                # Use least recently used warm pod for load balancing
                selected_pod = min(warm_pods, key=lambda p: p.last_used)
                startup_delay = 0
            elif cold_pods:
                # Use a cold pod
                selected_pod = cold_pods[0]
                startup_delay = self.pod_startup_time
                self.metrics.cold_starts += 1
            else:
                # No pods available - request fails
                self.metrics.slo_violations += 1
                continue
                
            # Process request
            if startup_delay > 0:
                yield self.env.timeout(startup_delay)
                selected_pod.is_warm = True
                
            # Execute function
            yield self.env.timeout(req.execution_time)
            
            # Calculate total response time
            response_time = self.env.now - req.arrival_time
            self.metrics.response_times.append(response_time)
            self.metrics.total_response_time += response_time
            self.metrics.completed_requests += 1
            
            # Check SLO
            if response_time > self.slo_threshold:
                self.metrics.slo_violations += 1
                
            # Update pod state
            selected_pod.last_used = self.env.now
            
    def calculate_current_load(self) -> float:
        """Calculate current load based on recent request arrivals."""
        if not self.recent_arrivals:
            return 0.0
            
        # Count requests in last scale_interval seconds
        current_time = self.env.now
        recent_count = sum(1 for t in self.recent_arrivals 
                          if current_time - t <= self.scale_interval)
        
        # Requests per second
        return recent_count / self.scale_interval
            
    def autoscaling_controller(self, features_list: List[np.ndarray] = None):
        """Improved autoscaling with better load tracking."""
        minute_idx = 0
        
        while True:
            # Calculate actual current load
            current_load = self.calculate_current_load()
            self.load_history.append(current_load)
            
            # Track request rate for analysis
            if len(self.recent_arrivals) > 0:
                rate = len([t for t in self.recent_arrivals 
                           if self.env.now - t <= 60]) / 60.0
                self.request_rate_history.append(rate)
            
            # Prepare predictions dict
            predictions = {'time': self.env.now}
            
            # Align features with current time
            if features_list:
                # Calculate which minute we're in
                current_minute = int(self.env.now / 60)
                if current_minute < len(features_list):
                    predictions['features'] = features_list[current_minute]
                
            # Get desired replicas
            current_replicas = len(self.pods)
            desired_replicas = self.autoscaler.get_desired_replicas(
                current_load, current_replicas, self.load_history, predictions
            )
            
            # Scale up/down
            if desired_replicas > current_replicas:
                # Scale up
                for i in range(desired_replicas - current_replicas):
                    pod = Pod(
                        id=f"pod-{len(self.pods)}",
                        warmup_time=self.env.now
                    )
                    self.pods.append(pod)
                self.metrics.scaling_actions += 1
                    
            elif desired_replicas < current_replicas:
                # Scale down - remove least recently used pods
                self.pods.sort(key=lambda p: p.last_used)
                to_remove = current_replicas - desired_replicas
                self.pods = self.pods[to_remove:]
                self.metrics.scaling_actions += 1
                
            # Track pod minutes
            self.metrics.pod_minutes += len(self.pods) * (self.scale_interval / 60)
            
            # Wait for next scaling interval
            yield self.env.timeout(self.scale_interval)
            minute_idx += 1
            
    def run(self, workload: List[float], function_id: str,
            execution_time_dist: Callable, features_list: List[np.ndarray] = None,
            duration: float = None):
        """Run the simulation."""
        # Start processes
        self.env.process(self.request_generator(workload, function_id, execution_time_dist))
        self.env.process(self.request_handler())
        self.env.process(self.pod_warmer())  # New pod warming process
        self.env.process(self.autoscaling_controller(features_list))
        
        # Run simulation
        if duration is None:
            duration = len(workload) * 60  # Convert minutes to seconds
            
        self.env.run(until=duration)
        
        return self.metrics


def run_experiment(workload_df: pd.DataFrame, autoscaler: Autoscaler,
                  execution_time_dist: Callable) -> Dict[str, Any]:
    """Run a single experiment with given workload and autoscaler."""
    # Extract workload trace (invocations per minute)
    workload = workload_df['invocations'].values
    
    # Extract features if using AAPA
    features_list = None
    if isinstance(autoscaler, AAPAAutoscaler):
        # Get features for each minute
        features_list = []
        feature_cols = [col for col in workload_df.columns if col.startswith('feature_')]
        for i in range(len(workload)):
            if i < len(feature_cols):
                features_list.append(workload_df.iloc[i][feature_cols].values)
                
    # Create simulation environment
    env = simpy.Environment()
    sim = ServerlessSimulator(env, autoscaler)
    
    # Run simulation
    metrics = sim.run(
        workload=workload,
        function_id="test-function",
        execution_time_dist=execution_time_dist,
        features_list=features_list
    )
    
    # Calculate REI score
    rei = calculate_rei(metrics)
    
    return {
        'metrics': metrics,
        'rei': rei,
        'autoscaler': autoscaler.__class__.__name__
    }


def calculate_rei(metrics: Metrics) -> float:
    """Calculate Resource Efficiency Index."""
    # Component scores (normalized to 0-1)
    s_slo = 1 - metrics.get_slo_violation_rate()
    
    # Normalize resource usage (assuming max 100 pod-minutes)
    s_eff = 1 - min(metrics.pod_minutes / 100, 1.0)
    
    # Normalize stability (assuming max 50 scaling actions)
    s_stab = 1 - min(metrics.scaling_actions / 50, 1.0)
    
    # Weighted combination
    alpha, beta, gamma = 0.4, 0.4, 0.2
    rei = alpha * s_slo + beta * s_eff + gamma * s_stab
    
    return rei


def main():
    """Test the improved simulator."""
    logger.info("Testing improved serverless autoscaling simulator...")
    
    # Create sample spike workload
    workload = np.concatenate([
        np.ones(10) * 10,  # Baseline
        np.ones(3) * 150,  # Spike
        np.ones(10) * 10,  # Return to baseline
        np.ones(3) * 200,  # Bigger spike
        np.ones(10) * 10,  # Baseline
        np.ones(2) * 180,  # Another spike
        np.ones(12) * 10   # Baseline
    ])
    
    # Execution time distribution (exponential with mean 100ms)
    execution_time_dist = lambda: np.random.exponential(0.1)
    
    # Test different autoscalers
    autoscalers = {
        'HPA': HPAAutoscaler(),
        'Generic Predictive': GenericPredictiveAutoscaler(),
        # AAPA would need real features
    }
    
    results = {}
    for name, autoscaler in autoscalers.items():
        logger.info(f"Running simulation with {name}...")
        
        # Create environment and simulator
        env = simpy.Environment()
        sim = ServerlessSimulator(env, autoscaler)
        
        # Run simulation
        metrics = sim.run(workload, "test-func", execution_time_dist)
        rei = calculate_rei(metrics)
        
        results[name] = {
            'slo_violation_rate': metrics.get_slo_violation_rate(),
            'mean_response_time': metrics.get_mean_response_time(),
            'p99_response_time': metrics.get_p99_response_time(),
            'pod_minutes': metrics.pod_minutes,
            'cold_starts': metrics.cold_starts,
            'scaling_actions': metrics.scaling_actions,
            'rei': rei
        }
        
    # Print results
    print("\n" + "="*60)
    print("Improved Simulation Results")
    print("="*60)
    
    for name, result in results.items():
        print(f"\n{name}:")
        for metric, value in result.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()