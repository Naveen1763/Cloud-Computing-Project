"""
AAPA: Archetype-Aware Predictive Autoscaler

This module implements the main AAPA autoscaler that combines workload
classification with uncertainty-aware scaling strategies.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScalingDecision:
    """Represents a scaling decision."""
    target_replicas: int
    confidence: float
    archetype: str
    reason: str


class AAPAAutoscaler:
    """
    Archetype-Aware Predictive Autoscaler for Kubernetes.
    
    This autoscaler classifies workloads into archetypes and applies
    differentiated scaling strategies with uncertainty quantification.
    """
    
    def __init__(self, classifier, strategies: Dict, uncertainty_model=None):
        """
        Initialize AAPA autoscaler.
        
        Args:
            classifier: Trained workload classifier
            strategies: Dict mapping archetype names to strategy instances
            uncertainty_model: Optional uncertainty quantification model
        """
        self.classifier = classifier
        self.strategies = strategies
        self.uncertainty_model = uncertainty_model
        self.current_archetype = None
        self.confidence_threshold = 0.7
        
    def make_scaling_decision(
        self, 
        features: np.ndarray,
        current_replicas: int,
        current_cpu: float,
        current_requests: int
    ) -> ScalingDecision:
        """
        Make a scaling decision based on current state and predictions.
        
        Args:
            features: Feature vector for current time window
            current_replicas: Current number of replicas
            current_cpu: Current average CPU utilization (0-1)
            current_requests: Current request rate
            
        Returns:
            ScalingDecision object with target replicas and metadata
        """
        # Classify workload archetype
        archetype, confidence = self._classify_workload(features)
        
        # Apply uncertainty adjustment if confidence is low
        if confidence < self.confidence_threshold:
            archetype, confidence = self._apply_uncertainty_adjustment(
                archetype, confidence, current_replicas
            )
        
        # Get strategy for archetype
        strategy = self.strategies.get(archetype)
        if strategy is None:
            logger.warning(f"No strategy found for archetype {archetype}, using default")
            strategy = self.strategies.get('default')
            
        # Calculate target replicas using strategy
        target_replicas = strategy.calculate_target_replicas(
            current_replicas=current_replicas,
            current_cpu=current_cpu,
            current_requests=current_requests,
            confidence=confidence
        )
        
        # Create scaling decision
        decision = ScalingDecision(
            target_replicas=target_replicas,
            confidence=confidence,
            archetype=archetype,
            reason=strategy.get_scaling_reason()
        )
        
        logger.info(f"Scaling decision: {decision}")
        return decision
        
    def _classify_workload(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Classify workload archetype with confidence score.
        
        Args:
            features: Feature vector
            
        Returns:
            Tuple of (archetype_name, confidence_score)
        """
        # Get prediction and probabilities
        prediction = self.classifier.predict(features.reshape(1, -1))[0]
        probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]
        
        # Map prediction to archetype name
        archetype_map = {
            0: 'PERIODIC',
            1: 'SPIKE', 
            2: 'STATIONARY_NOISY',
            3: 'RAMP'
        }
        archetype = archetype_map.get(prediction, 'UNKNOWN')
        confidence = float(np.max(probabilities))
        
        return archetype, confidence
        
    def _apply_uncertainty_adjustment(
        self, 
        archetype: str, 
        confidence: float,
        current_replicas: int
    ) -> Tuple[str, float]:
        """
        Apply uncertainty-aware adjustments when confidence is low.
        
        Args:
            archetype: Predicted archetype
            confidence: Confidence score
            current_replicas: Current replica count
            
        Returns:
            Adjusted (archetype, confidence) tuple
        """
        if self.uncertainty_model is not None:
            # Use uncertainty model for adjustment
            adjusted = self.uncertainty_model.adjust_prediction(
                archetype, confidence, current_replicas
            )
            return adjusted
        else:
            # Simple fallback: use conservative strategy when uncertain
            if confidence < 0.5:
                return 'STATIONARY_NOISY', confidence
            return archetype, confidence
            
    def update_state(self, actual_cpu: float, actual_requests: int):
        """
        Update autoscaler state based on observed metrics.
        
        This can be used for online learning or adaptation.
        
        Args:
            actual_cpu: Observed CPU utilization
            actual_requests: Observed request rate
        """
        # Placeholder for online adaptation logic
        pass