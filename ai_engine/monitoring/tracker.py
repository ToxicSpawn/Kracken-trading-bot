"""
Monitoring and tracking for LLM training and inference.
Supports Weights & Biases, MLflow, and Prometheus.
"""
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# W&B integration
_wandb_available = False
_wandb_initialized = False

try:
    import wandb
    _wandb_available = True
except ImportError:
    pass

# MLflow integration
_mlflow_available = False

try:
    import mlflow
    import mlflow.pytorch
    _mlflow_available = True
except ImportError:
    pass

# Prometheus integration
_prometheus_available = False

try:
    from prometheus_client import Counter, Histogram, Gauge
    _prometheus_available = True
except ImportError:
    pass


class MetricsTracker:
    """Unified metrics tracking interface."""

    def __init__(
        self,
        project_name: str = "llm-inference",
        use_wandb: bool = True,
        use_mlflow: bool = False,
        use_prometheus: bool = True,
    ):
        """
        Initialize metrics tracker.

        Args:
            project_name: Project name for tracking
            use_wandb: Enable W&B tracking
            use_mlflow: Enable MLflow tracking
            use_prometheus: Enable Prometheus metrics
        """
        self.project_name = project_name
        self.use_wandb = use_wandb and _wandb_available
        self.use_mlflow = use_mlflow and _mlflow_available
        self.use_prometheus = use_prometheus and _prometheus_available

        # Initialize W&B
        if self.use_wandb:
            try:
                wandb.init(project=project_name, resume="allow")
                global _wandb_initialized
                _wandb_initialized = True
                logger.info("W&B initialized")
            except Exception as e:
                logger.warning(f"W&B initialization failed: {e}")
                self.use_wandb = False

        # Initialize MLflow
        if self.use_mlflow:
            try:
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
                mlflow.set_experiment(project_name)
                logger.info("MLflow initialized")
            except Exception as e:
                logger.warning(f"MLflow initialization failed: {e}")
                self.use_mlflow = False

        # Initialize Prometheus metrics
        if self.use_prometheus:
            try:
                self.inference_counter = Counter(
                    "llm_inference_total",
                    "Total number of inference requests",
                    ["model", "status"]
                )
                self.inference_latency = Histogram(
                    "llm_inference_latency_seconds",
                    "Inference latency in seconds",
                    ["model"]
                )
                self.inference_tokens = Histogram(
                    "llm_inference_tokens",
                    "Number of tokens generated",
                    ["model"]
                )
                self.model_loaded = Gauge(
                    "llm_model_loaded",
                    "Whether model is loaded (1) or not (0)",
                    ["model"]
                )
                logger.info("Prometheus metrics initialized")
            except Exception as e:
                logger.warning(f"Prometheus initialization failed: {e}")
                self.use_prometheus = False

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to all enabled trackers."""
        if self.use_wandb and _wandb_initialized:
            try:
                wandb.log(metrics, step=step)
            except Exception as e:
                logger.warning(f"W&B logging failed: {e}")

        if self.use_mlflow:
            try:
                mlflow.log_metrics(metrics, step=step or 0)
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

    def log_inference(
        self,
        model_name: str,
        latency: float,
        tokens: int,
        status: str = "success",
    ):
        """Log inference metrics."""
        metrics = {
            "inference/latency": latency,
            "inference/tokens": tokens,
            "inference/model": model_name,
        }
        self.log_metrics(metrics)

        if self.use_prometheus:
            try:
                self.inference_counter.labels(model=model_name, status=status).inc()
                self.inference_latency.labels(model=model_name).observe(latency)
                self.inference_tokens.labels(model=model_name).observe(tokens)
            except Exception as e:
                logger.warning(f"Prometheus logging failed: {e}")

    def log_training_metrics(
        self,
        epoch: int,
        loss: float,
        learning_rate: float,
        **kwargs
    ):
        """Log training metrics."""
        metrics = {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/epoch": epoch,
        }
        metrics.update({f"train/{k}": v for k, v in kwargs.items()})
        self.log_metrics(metrics, step=epoch)

    def log_model_loaded(self, model_name: str, loaded: bool = True):
        """Log model loading status."""
        if self.use_prometheus:
            try:
                self.model_loaded.labels(model=model_name).set(1 if loaded else 0)
            except Exception as e:
                logger.warning(f"Prometheus logging failed: {e}")

    def finish(self):
        """Finish tracking session."""
        if self.use_wandb and _wandb_initialized:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"W&B finish failed: {e}")


# Global tracker instance
_global_tracker: Optional[MetricsTracker] = None


def get_tracker() -> MetricsTracker:
    """Get global metrics tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MetricsTracker(
            project_name=os.getenv("METRICS_PROJECT_NAME", "llm-inference"),
            use_wandb=os.getenv("USE_WANDB", "true").lower() == "true",
            use_mlflow=os.getenv("USE_MLFLOW", "false").lower() == "true",
            use_prometheus=os.getenv("USE_PROMETHEUS", "true").lower() == "true",
        )
    return _global_tracker

