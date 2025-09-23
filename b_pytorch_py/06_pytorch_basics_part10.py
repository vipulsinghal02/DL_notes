# %% [markdown]
# # PyTorch Basics Part 10: MLOps and Advanced Production
#
# Comprehensive guide to deploying PyTorch models in production with mathematical foundations for monitoring, optimization, and quality assurance
#
# ## Mathematical Framework for Production ML Systems
#
# Production machine learning requires rigorous mathematical foundations for monitoring, optimization, and reliability:
#
# ### Core Mathematical Concepts for MLOps
#
# **1. Statistical Process Control:**
# - **Control Charts**: Monitor metric $X_t$ with control limits $\mu \pm k\sigma$
# - **CUSUM**: Cumulative sum $S_t = \max(0, S_{t-1} + (X_t - \mu_0) - k)$
# - **EWMA**: Exponentially weighted moving average $Z_t = \lambda X_t + (1-\lambda)Z_{t-1}$
#
# **2. Drift Detection:**
# - **Population Stability Index**: $\text{PSI} = \sum_{i=1}^{10} (\text{Expected}_i - \text{Actual}_i) \times \ln\left(\frac{\text{Expected}_i}{\text{Actual}_i}\right)$
# - **KL Divergence**: $D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$
# - **Kolmogorov-Smirnov Test**: $D_{n,m} = \sup_x |F_{1,n}(x) - F_{2,m}(x)|$
#
# **3. Performance Metrics:**
# - **Service Level Indicators**: $\text{SLI} = \frac{\text{Good Events}}{\text{Total Events}}$
# - **Availability**: $A = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}$ where MTBF = Mean Time Between Failures
# - **Throughput**: $\lambda = \frac{N}{T}$ (requests per unit time)
# - **Latency Percentiles**: $P_{99} = \inf\{x : F(x) \geq 0.99\}$
#
# **4. A/B Testing Statistics:**
# - **Two-Sample t-test**: $t = \frac{\bar{X}_1 - \bar{X}_2}{s_p\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}$
# - **Effect Size (Cohen's d)**: $d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}}$
# - **Statistical Power**: $\text{Power} = P(\text{reject } H_0 | H_1 \text{ is true})$
# - **Minimum Detectable Effect**: $\text{MDE} = t_{\alpha/2} + t_{\beta} \times \frac{\sigma\sqrt{2}}{sqrt{n}}$
#
# **5. Model Optimization:**
# - **Quantization**: $q = \text{round}\left(\frac{x}{s}\right) + z$ where $s$ is scale, $z$ is zero-point
# - **Pruning**: Remove weights where $|w_{ij}| < \theta$ (threshold-based)
# - **Knowledge Distillation**: $\mathcal{L} = \alpha \mathcal{L}_{CE}(y, \sigma(z_s)) + (1-\alpha)\mathcal{L}_{KL}(\sigma(z_t/T), \sigma(z_s/T))$
#
# **6. Resource Allocation:**
# - **Little's Law**: $L = \lambda W$ (average number in system = arrival rate × average time in system)
# - **Queueing Theory**: $\rho = \frac{\lambda}{\mu}$ (utilization = arrival rate / service rate)
# - **Auto-scaling**: $\text{instances} = \max\left(\lceil\frac{\text{load}}{\text{capacity per instance}}\rceil, \text{min instances}\right)$

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import datetime
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Experiment Tracking and Management
#
# **Mathematical Foundation for Experiment Design:**
#
# Systematic experiment tracking requires statistical rigor to ensure reproducible and meaningful results:
#
# **1. Experimental Design Principles:**
# - **Randomization**: Ensures $E[\epsilon_i] = 0$ and reduces bias
# - **Replication**: Increases statistical power through larger sample size $n$
# - **Factorial Design**: Tests $k$ factors with $2^k$ treatments
#
# **2. Significance Testing:**
# - **Null Hypothesis**: $H_0: \mu_1 = \mu_2$ (no difference between conditions)
# - **Type I Error**: $\alpha = P(\text{reject } H_0 | H_0 \text{ true})$
# - **Type II Error**: $\beta = P(\text{accept } H_0 | H_1 \text{ true})$
# - **Multiple Comparisons**: Bonferroni correction $\alpha' = \frac{\alpha}{m}$ for $m$ tests
#
# **3. Effect Size and Practical Significance:**
# - **Cohen's d**: $d = \frac{\bar{X}_1 - \bar{X}_2}{s_{pooled}}$ where $s_{pooled} = \sqrt{\frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}}$
# - **Confidence Intervals**: $\bar{X} \pm t_{\alpha/2} \frac{s}{\sqrt{n}}$
#
# Systematic experiment tracking is crucial for reproducible machine learning and enables teams to compare approaches systematically.

# %%
# Experiment tracking system
class ExperimentTracker:
    def __init__(self, experiment_name: str, base_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.base_dir / experiment_name / self.run_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking structures
        self.metrics = {}
        self.hyperparameters = {}
        self.artifacts = {}
        self.logs = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.experiment_dir / 'experiment.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{experiment_name}_{self.run_id}")

        self.logger.info(f"Started experiment: {experiment_name}")
        self.logger.info(f"Run ID: {self.run_id}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")

    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters for this experiment run"""
        self.hyperparameters.update(hyperparams)
        self.logger.info(f"Hyperparameters: {hyperparams}")

        # Save to file
        with open(self.experiment_dir / 'hyperparameters.json', 'w') as f:
            json.dump(self.hyperparameters, f, indent=2)

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []

        metric_entry = {
            'value': value,
            'step': step,
            'timestamp': time.time()
        }

        self.metrics[name].append(metric_entry)
        self.logger.info(f"Metric - {name}: {value} (step: {step})")

    def log_artifact(self, name: str, artifact: Any, artifact_type: str = 'pickle'):
        """Log an artifact (model, plot, etc.)"""
        artifact_path = self.experiment_dir / f"{name}.{artifact_type}"

        if artifact_type == 'pickle':
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifact, f)
        elif artifact_type == 'torch':
            torch.save(artifact, artifact_path)
        elif artifact_type == 'json':
            with open(artifact_path, 'w') as f:
                json.dump(artifact, f, indent=2)

        self.artifacts[name] = str(artifact_path)
        self.logger.info(f"Artifact saved: {name} -> {artifact_path}")

    def save_model(self, model: nn.Module, name: str = 'model'):
        """Save PyTorch model"""
        model_path = self.experiment_dir / f"{name}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'hyperparameters': self.hyperparameters,
            'run_id': self.run_id
        }, model_path)

        self.artifacts[f"{name}_model"] = str(model_path)
        self.logger.info(f"Model saved: {model_path}")

    def finish_experiment(self):
        """Finalize experiment and save summary"""
        summary = {
            'experiment_name': self.experiment_name,
            'run_id': self.run_id,
            'hyperparameters': self.hyperparameters,
            'final_metrics': {name: values[-1] for name, values in self.metrics.items()},
            'artifacts': self.artifacts,
            'duration': time.time() - self.metrics[list(self.metrics.keys())[0]][0]['timestamp'] if self.metrics else 0
        }

        with open(self.experiment_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed metrics
        with open(self.experiment_dir / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

        self.logger.info("Experiment finished")
        return summary

# Example usage of experiment tracking
# Create a simple model for demonstration
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# Demonstrate experiment tracking
tracker = ExperimentTracker("classification_experiment")

# Log hyperparameters
hyperparams = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'hidden_size': 64,
    'num_epochs': 10,
    'dropout_rate': 0.2
}
tracker.log_hyperparameters(hyperparams)

# Create synthetic data
X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=True)

# Create model
model = SimpleClassifier(20, hyperparams['hidden_size'], 3)
optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
criterion = nn.CrossEntropyLoss()

# Training loop with tracking
print("Training with experiment tracking...")
for epoch in range(hyperparams['num_epochs']):
    epoch_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    # Log metrics
    avg_loss = epoch_loss / len(dataloader)
    accuracy = 100 * correct / total

    tracker.log_metric('train_loss', avg_loss, epoch)
    tracker.log_metric('train_accuracy', accuracy, epoch)

    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

# Save model and finish experiment
tracker.save_model(model)
tracker.log_artifact('training_data_stats', {'mean': X.mean().item(), 'std': X.std().item()})

summary = tracker.finish_experiment()
print(f"\nExperiment completed. Run ID: {tracker.run_id}")
print(f"Final accuracy: {summary['final_metrics']['train_accuracy']['value']:.2f}%")

# %% [markdown]
# ## Model Registry and Versioning
#
# A model registry manages different versions of trained models, tracks their performance, and facilitates model deployment and rollback. This is essential for maintaining model quality and enabling safe deployments.

# %%
# Model Registry System
class ModelRegistry:
    def __init__(self, registry_dir: str = "./model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)
        self.metadata_file = self.registry_dir / 'registry_metadata.json'

        # Load existing registry or create new
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'models': {}}

    def register_model(self,
                      model: nn.Module,
                      model_name: str,
                      version: str,
                      metrics: Dict[str, float],
                      metadata: Dict[str, Any] = None,
                      stage: str = 'staging'):
        """Register a new model version"""

        if model_name not in self.metadata['models']:
            self.metadata['models'][model_name] = {'versions': {}}

        # Create version directory
        version_dir = self.registry_dir / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = version_dir / 'model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_architecture': model.__class__.__name__,
            'model_config': self._extract_model_config(model)
        }, model_path)

        # Save metadata
        version_metadata = {
            'version': version,
            'stage': stage,
            'metrics': metrics,
            'metadata': metadata or {},
            'created_at': datetime.datetime.now().isoformat(),
            'model_path': str(model_path),
            'model_size_mb': model_path.stat().st_size / (1024*1024) if model_path.exists() else 0
        }

        self.metadata['models'][model_name]['versions'][version] = version_metadata
        self._save_metadata()

        print(f"Registered {model_name} version {version} in {stage} stage")
        return version_metadata

    def promote_model(self, model_name: str, version: str, target_stage: str):
        """Promote model to a different stage (staging -> production)"""
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        if version not in self.metadata['models'][model_name]['versions']:
            raise ValueError(f"Version {version} not found for model {model_name}")

        # Update stage
        self.metadata['models'][model_name]['versions'][version]['stage'] = target_stage
        self.metadata['models'][model_name]['versions'][version]['promoted_at'] = datetime.datetime.now().isoformat()

        self._save_metadata()
        print(f"Promoted {model_name} version {version} to {target_stage}")

    def get_model_by_stage(self, model_name: str, stage: str = 'production'):
        """Get the latest model version for a given stage"""
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        versions = self.metadata['models'][model_name]['versions']
        stage_versions = [(v, data) for v, data in versions.items() if data['stage'] == stage]

        if not stage_versions:
            raise ValueError(f"No models found in {stage} stage for {model_name}")

        # Return the latest version (by creation time)
        latest_version = max(stage_versions, key=lambda x: x[1]['created_at'])
        return latest_version[0], latest_version[1]

    def load_model(self, model_name: str, version: str, model_class):
        """Load a specific model version"""
        if model_name not in self.metadata['models']:
            raise ValueError(f"Model {model_name} not found")

        version_data = self.metadata['models'][model_name]['versions'].get(version)
        if not version_data:
            raise ValueError(f"Version {version} not found for model {model_name}")

        # Load model
        checkpoint = torch.load(version_data['model_path'])

        # Instantiate model (would need more sophisticated config handling in practice)
        model = model_class(**version_data['metadata'].get('model_config', {}))
        model.load_state_dict(checkpoint['model_state_dict'])

        return model, version_data

    def list_models(self, stage: Optional[str] = None):
        """List all models, optionally filtered by stage"""
        models_info = []

        for model_name, model_data in self.metadata['models'].items():
            for version, version_data in model_data['versions'].items():
                if stage is None or version_data['stage'] == stage:
                    models_info.append({
                        'model_name': model_name,
                        'version': version,
                        'stage': version_data['stage'],
                        'metrics': version_data['metrics'],
                        'created_at': version_data['created_at'],
                        'size_mb': version_data['model_size_mb']
                    })

        return sorted(models_info, key=lambda x: x['created_at'], reverse=True)

    def compare_models(self, model_name: str, versions: List[str], metric: str):
        """Compare performance of different model versions"""
        comparison = []

        for version in versions:
            version_data = self.metadata['models'][model_name]['versions'].get(version)
            if version_data:
                comparison.append({
                    'version': version,
                    'stage': version_data['stage'],
                    'metric_value': version_data['metrics'].get(metric, 'N/A'),
                    'created_at': version_data['created_at']
                })

        return sorted(comparison, key=lambda x: x['metric_value'] if x['metric_value'] != 'N/A' else -1, reverse=True)

    def _extract_model_config(self, model: nn.Module):
        """Extract model configuration (simplified)"""
        # This would be more sophisticated in practice
        if hasattr(model, 'fc1') and hasattr(model, 'fc2'):
            return {
                'input_size': model.fc1.in_features,
                'hidden_size': model.fc1.out_features,
                'num_classes': model.fc2.out_features
            }
        return {}

    def _save_metadata(self):
        """Save registry metadata to disk"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

# Demonstrate model registry
registry = ModelRegistry()

# Register different model versions
models_to_register = [
    {'version': 'v1.0', 'metrics': {'accuracy': 0.85, 'precision': 0.82}, 'stage': 'staging'},
    {'version': 'v1.1', 'metrics': {'accuracy': 0.87, 'precision': 0.84}, 'stage': 'staging'},
    {'version': 'v1.2', 'metrics': {'accuracy': 0.89, 'precision': 0.86}, 'stage': 'staging'}
]

print("Registering model versions...")
for model_info in models_to_register:
    # Create a slightly different model for each version
    test_model = SimpleClassifier(20, 64 + int(model_info['version'][-1]) * 10, 3)

    registry.register_model(
        model=test_model,
        model_name="text_classifier",
        version=model_info['version'],
        metrics=model_info['metrics'],
        metadata={'description': f'Model version {model_info["version"]} with improved accuracy'},
        stage=model_info['stage']
    )

# Promote best model to production
registry.promote_model('text_classifier', 'v1.2', 'production')

# List all models
print("\nAll registered models:")
all_models = registry.list_models()
for model_info in all_models:
    print(f"  {model_info['model_name']} {model_info['version']} ({model_info['stage']}) - "
          f"Accuracy: {model_info['metrics']['accuracy']:.3f}")

# Get production model
prod_version, prod_data = registry.get_model_by_stage('text_classifier', 'production')
print(f"\nProduction model: {prod_version} with accuracy {prod_data['metrics']['accuracy']:.3f}")

# Compare models
print("\nModel comparison by accuracy:")
comparison = registry.compare_models('text_classifier', ['v1.0', 'v1.1', 'v1.2'], 'accuracy')
for comp in comparison:
    print(f"  Version {comp['version']}: {comp['metric_value']:.3f} ({comp['stage']})")

# %% [markdown]
# ## Model Monitoring and Drift Detection
#
# Production models need continuous monitoring to detect performance degradation, data drift, and other issues. This system tracks model health and alerts when intervention is needed.

# %%
# Model Monitoring System
class ModelMonitor:
    def __init__(self, model_name: str, monitoring_dir: str = "./monitoring"):
        self.model_name = model_name
        self.monitoring_dir = Path(monitoring_dir) / model_name
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

        # Initialize monitoring data
        self.performance_log = []
        self.feature_stats = {'mean': {}, 'std': {}, 'min': {}, 'max': {}}
        self.prediction_stats = {'distribution': [], 'confidence_scores': []}
        self.alerts = []

        # Drift detection parameters
        self.drift_threshold = 0.05  # Statistical significance level
        self.performance_threshold = 0.1  # Acceptable performance drop

    def log_prediction(self, features: torch.Tensor, predictions: torch.Tensor,
                      confidences: torch.Tensor, true_labels: Optional[torch.Tensor] = None):
        """Log model predictions and compute statistics"""
        timestamp = time.time()

        # Feature statistics
        batch_stats = self._compute_feature_stats(features)

        # Prediction statistics
        pred_dist = torch.bincount(predictions, minlength=3).float() / len(predictions)
        avg_confidence = confidences.mean().item()

        # Performance metrics (if ground truth available)
        performance = None
        if true_labels is not None:
            accuracy = (predictions == true_labels).float().mean().item()
            performance = {'accuracy': accuracy, 'count': len(predictions)}

        # Log entry
        log_entry = {
            'timestamp': timestamp,
            'batch_size': len(predictions),
            'feature_stats': batch_stats,
            'prediction_distribution': pred_dist.tolist(),
            'average_confidence': avg_confidence,
            'performance': performance
        }

        self.performance_log.append(log_entry)

        # Check for drift and anomalies
        self._check_drift(batch_stats, pred_dist)
        if performance:
            self._check_performance_degradation(performance['accuracy'])

    def _compute_feature_stats(self, features: torch.Tensor):
        """Compute statistics for input features"""
        return {
            'mean': features.mean(dim=0).tolist(),
            'std': features.std(dim=0).tolist(),
            'min': features.min(dim=0)[0].tolist(),
            'max': features.max(dim=0)[0].tolist()
        }

    def set_baseline(self, baseline_features: torch.Tensor, baseline_predictions: torch.Tensor):
        """Set baseline statistics for drift detection"""
        self.baseline_stats = self._compute_feature_stats(baseline_features)
        self.baseline_pred_dist = torch.bincount(baseline_predictions, minlength=3).float() / len(baseline_predictions)
        print(f"Baseline set with {len(baseline_features)} samples")

    def _check_drift(self, current_stats: Dict, current_pred_dist: torch.Tensor):
        """Check for data and prediction drift"""
        if not hasattr(self, 'baseline_stats'):
            return

        # Feature drift detection (simplified KL divergence approximation)
        feature_drift_scores = []
        for i, (baseline_mean, current_mean) in enumerate(zip(self.baseline_stats['mean'], current_stats['mean'])):
            baseline_std = self.baseline_stats['std'][i]
            current_std = current_stats['std'][i]

            # Simple drift score based on mean and std deviation
            if baseline_std > 0:
                drift_score = abs(baseline_mean - current_mean) / baseline_std
                feature_drift_scores.append(drift_score)

        max_feature_drift = max(feature_drift_scores) if feature_drift_scores else 0

        # Prediction drift (KL divergence)
        pred_drift = self._kl_divergence(self.baseline_pred_dist, current_pred_dist)

        # Check thresholds
        if max_feature_drift > 2.0:  # 2 standard deviations
            self._create_alert('feature_drift', f'Feature drift detected: {max_feature_drift:.3f}')

        if pred_drift > self.drift_threshold:
            self._create_alert('prediction_drift', f'Prediction drift detected: {pred_drift:.3f}')

    def _kl_divergence(self, p: torch.Tensor, q: torch.Tensor):
        """Compute KL divergence between two distributions"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        p = p + epsilon
        q = q + epsilon

        return torch.sum(p * torch.log(p / q)).item()

    def _check_performance_degradation(self, current_accuracy: float):
        """Check for performance degradation"""
        # Get recent performance
        recent_performance = [entry['performance']['accuracy']
                            for entry in self.performance_log[-10:]
                            if entry['performance']]

        if len(recent_performance) >= 5:
            baseline_accuracy = np.mean(recent_performance[:5])
            if current_accuracy < baseline_accuracy - self.performance_threshold:
                self._create_alert('performance_degradation',
                                 f'Accuracy dropped from {baseline_accuracy:.3f} to {current_accuracy:.3f}')

    def _create_alert(self, alert_type: str, message: str):
        """Create and log an alert"""
        alert = {
            'timestamp': time.time(),
            'type': alert_type,
            'message': message,
            'severity': self._get_alert_severity(alert_type)
        }

        self.alerts.append(alert)
        print(f"ALERT [{alert['severity']}]: {alert_type} - {message}")

    def _get_alert_severity(self, alert_type: str):
        """Determine alert severity"""
        severity_map = {
            'feature_drift': 'MEDIUM',
            'prediction_drift': 'MEDIUM',
            'performance_degradation': 'HIGH'
        }
        return severity_map.get(alert_type, 'LOW')

    def get_monitoring_report(self):
        """Generate monitoring report"""
        if not self.performance_log:
            return "No monitoring data available"

        # Recent performance
        recent_entries = self.performance_log[-10:]
        recent_accuracy = [e['performance']['accuracy'] for e in recent_entries if e['performance']]

        # Alert summary
        alert_counts = {}
        for alert in self.alerts:
            alert_counts[alert['type']] = alert_counts.get(alert['type'], 0) + 1

        report = {
            'model_name': self.model_name,
            'monitoring_period': f"{len(self.performance_log)} batches",
            'recent_accuracy': {
                'mean': np.mean(recent_accuracy) if recent_accuracy else 'N/A',
                'std': np.std(recent_accuracy) if recent_accuracy else 'N/A',
                'samples': len(recent_accuracy)
            },
            'alerts': {
                'total_alerts': len(self.alerts),
                'by_type': alert_counts,
                'recent_alerts': self.alerts[-5:] if self.alerts else []
            },
            'health_status': self._get_health_status()
        }

        return report

    def _get_health_status(self):
        """Determine overall model health"""
        recent_alerts = [a for a in self.alerts if time.time() - a['timestamp'] < 3600]  # Last hour
        high_severity_alerts = [a for a in recent_alerts if a['severity'] == 'HIGH']

        if high_severity_alerts:
            return 'CRITICAL'
        elif len(recent_alerts) > 5:
            return 'WARNING'
        else:
            return 'HEALTHY'

    def save_monitoring_data(self):
        """Save monitoring data to disk"""
        monitoring_data = {
            'performance_log': self.performance_log,
            'alerts': self.alerts,
            'baseline_stats': getattr(self, 'baseline_stats', None)
        }

        with open(self.monitoring_dir / 'monitoring_data.json', 'w') as f:
            json.dump(monitoring_data, f, indent=2)

        print(f"Monitoring data saved to {self.monitoring_dir}")

# Demonstrate model monitoring
monitor = ModelMonitor("text_classifier")

# Set baseline with normal data
baseline_features = torch.randn(1000, 20)
baseline_predictions = torch.randint(0, 3, (1000,))
monitor.set_baseline(baseline_features, baseline_predictions)

# Simulate normal operation
print("\nSimulating normal model operation...")
for i in range(5):
    # Normal data similar to baseline
    features = torch.randn(100, 20) * 1.1  # Slight variation
    predictions = torch.randint(0, 3, (100,))
    confidences = torch.rand(100) * 0.3 + 0.7  # High confidence
    true_labels = torch.randint(0, 3, (100,))

    monitor.log_prediction(features, predictions, confidences, true_labels)

# Simulate data drift
print("\nSimulating data drift...")
for i in range(3):
    # Shifted data distribution
    features = torch.randn(100, 20) + 2.0  # Mean shift
    predictions = torch.randint(0, 3, (100,))
    confidences = torch.rand(100) * 0.4 + 0.5  # Lower confidence
    true_labels = torch.randint(0, 3, (100,))

    monitor.log_prediction(features, predictions, confidences, true_labels)

# Simulate performance degradation
print("\nSimulating performance degradation...")
for i in range(3):
    features = torch.randn(100, 20)
    predictions = torch.randint(0, 3, (100,))
    confidences = torch.rand(100) * 0.3 + 0.4  # Lower confidence
    # Generate labels with lower accuracy
    true_labels = torch.randint(0, 3, (100,))
    # Artificially create poor performance
    incorrect_mask = torch.rand(100) < 0.4  # 40% incorrect
    true_labels[incorrect_mask] = (predictions[incorrect_mask] + 1) % 3

    monitor.log_prediction(features, predictions, confidences, true_labels)

# Generate monitoring report
report = monitor.get_monitoring_report()
print("\n" + "="*50)
print("MONITORING REPORT")
print("="*50)
print(f"Model: {report['model_name']}")
print(f"Health Status: {report['health_status']}")
print(f"Monitoring Period: {report['monitoring_period']}")

if report['recent_accuracy']['samples'] > 0:
    print(f"Recent Accuracy: {report['recent_accuracy']['mean']:.3f} ± {report['recent_accuracy']['std']:.3f}")

print(f"Total Alerts: {report['alerts']['total_alerts']}")
for alert_type, count in report['alerts']['by_type'].items():
    print(f"  - {alert_type}: {count}")

if report['alerts']['recent_alerts']:
    print("\nRecent Alerts:")
    for alert in report['alerts']['recent_alerts']:
        alert_time = datetime.datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S')
        print(f"  [{alert['severity']}] {alert_time}: {alert['message']}")

# Save monitoring data
monitor.save_monitoring_data()

# %% [markdown]
# ## A/B Testing Framework for Models
#
# A/B testing allows safe deployment of new models by comparing their performance against existing models in production. This framework manages traffic splitting, statistical analysis, and rollback capabilities.

# %%
# A/B Testing Framework
class ABTestingFramework:
    def __init__(self, test_name: str, results_dir: str = "./ab_tests"):
        self.test_name = test_name
        self.results_dir = Path(results_dir) / test_name
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Test configuration
        self.variants = {}  # variant_name -> model info
        self.traffic_allocation = {}  # variant_name -> percentage
        self.metrics = {}  # variant_name -> list of metrics
        self.test_active = False
        self.start_time = None

        # Statistical parameters
        self.significance_level = 0.05
        self.minimum_sample_size = 100

    def add_variant(self, variant_name: str, model: nn.Module, traffic_percentage: float,
                   is_control: bool = False):
        """Add a variant to the A/B test"""
        self.variants[variant_name] = {
            'model': model,
            'is_control': is_control,
            'model_path': self.results_dir / f"{variant_name}_model.pth"
        }

        self.traffic_allocation[variant_name] = traffic_percentage
        self.metrics[variant_name] = []

        # Save model
        torch.save(model.state_dict(), self.variants[variant_name]['model_path'])

        print(f"Added variant '{variant_name}' with {traffic_percentage}% traffic allocation")
        print(f"  Control variant: {is_control}")
        print(f"  Model saved to: {self.variants[variant_name]['model_path']}")

    def start_test(self, duration_hours: float = 24):
        """Start the A/B test"""
        # Validate configuration
        total_traffic = sum(self.traffic_allocation.values())
        if abs(total_traffic - 100) > 0.01:
            raise ValueError(f"Traffic allocation must sum to 100%, got {total_traffic}%")

        control_variants = [name for name, info in self.variants.items() if info['is_control']]
        if len(control_variants) != 1:
            raise ValueError("Exactly one control variant must be specified")

        self.test_active = True
        self.start_time = time.time()
        self.duration = duration_hours * 3600  # Convert to seconds

        print(f"A/B test '{self.test_name}' started")
        print(f"Duration: {duration_hours} hours")
        print(f"Variants: {list(self.variants.keys())}")
        print(f"Traffic allocation: {self.traffic_allocation}")

    def assign_variant(self, user_id: str) -> str:
        """Assign a user to a variant based on traffic allocation"""
        if not self.test_active:
            raise ValueError("A/B test is not active")

        # Simple hash-based assignment for consistency
        hash_value = hash(user_id + self.test_name) % 100

        cumulative_percentage = 0
        for variant_name, percentage in self.traffic_allocation.items():
            cumulative_percentage += percentage
            if hash_value < cumulative_percentage:
                return variant_name

        # Fallback to first variant
        return list(self.variants.keys())[0]

    def get_model_for_variant(self, variant_name: str) -> nn.Module:
        """Get the model for a specific variant"""
        if variant_name not in self.variants:
            raise ValueError(f"Variant '{variant_name}' not found")

        return self.variants[variant_name]['model']

    def log_result(self, user_id: str, variant_name: str, metrics: Dict[str, float]):
        """Log a result for the A/B test"""
        if not self.test_active:
            return

        result = {
            'timestamp': time.time(),
            'user_id': user_id,
            'variant': variant_name,
            'metrics': metrics
        }

        self.metrics[variant_name].append(result)

    def get_test_status(self):
        """Get current test status and preliminary results"""
        if not self.test_active:
            return {'status': 'inactive'}

        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.duration - elapsed_time)

        # Sample sizes
        sample_sizes = {variant: len(results) for variant, results in self.metrics.items()}

        # Preliminary results
        variant_stats = {}
        for variant_name, results in self.metrics.items():
            if results:
                # Calculate statistics for each metric
                metrics_summary = {}
                for metric_name in results[0]['metrics'].keys():
                    values = [r['metrics'][metric_name] for r in results]
                    metrics_summary[metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'count': len(values)
                    }

                variant_stats[variant_name] = metrics_summary

        return {
            'status': 'active',
            'elapsed_hours': elapsed_time / 3600,
            'remaining_hours': remaining_time / 3600,
            'sample_sizes': sample_sizes,
            'variant_stats': variant_stats,
            'ready_for_analysis': all(size >= self.minimum_sample_size for size in sample_sizes.values())
        }

    def run_statistical_analysis(self, metric_name: str = 'accuracy'):
        """Run statistical analysis comparing variants"""
        from scipy import stats

        if not self.test_active:
            raise ValueError("Test must be active to run analysis")

        # Get control variant
        control_variant = next(name for name, info in self.variants.items() if info['is_control'])

        # Extract metric values
        results = {}
        for variant_name, metrics_list in self.metrics.items():
            if metrics_list and metric_name in metrics_list[0]['metrics']:
                values = [m['metrics'][metric_name] for m in metrics_list]
                results[variant_name] = values

        if control_variant not in results:
            raise ValueError(f"No data for control variant '{control_variant}'")

        # Statistical tests
        analysis_results = {
            'metric': metric_name,
            'control_variant': control_variant,
            'comparisons': {}
        }

        control_values = results[control_variant]

        for variant_name, variant_values in results.items():
            if variant_name == control_variant:
                continue

            # Two-sample t-test
            t_stat, p_value = stats.ttest_ind(variant_values, control_values)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(variant_values) - 1) * np.var(variant_values) +
                                 (len(control_values) - 1) * np.var(control_values)) /
                                (len(variant_values) + len(control_values) - 2))

            effect_size = (np.mean(variant_values) - np.mean(control_values)) / pooled_std if pooled_std > 0 else 0

            # Confidence interval for difference
            diff_mean = np.mean(variant_values) - np.mean(control_values)
            diff_se = np.sqrt(np.var(variant_values)/len(variant_values) + np.var(control_values)/len(control_values))
            ci_lower = diff_mean - 1.96 * diff_se
            ci_upper = diff_mean + 1.96 * diff_se

            analysis_results['comparisons'][variant_name] = {
                'sample_size': len(variant_values),
                'mean': np.mean(variant_values),
                'std': np.std(variant_values),
                'vs_control': {
                    'difference': diff_mean,
                    'p_value': p_value,
                    'significant': p_value < self.significance_level,
                    'effect_size': effect_size,
                    'confidence_interval': [ci_lower, ci_upper]
                }
            }

        # Control variant stats
        analysis_results['control_stats'] = {
            'sample_size': len(control_values),
            'mean': np.mean(control_values),
            'std': np.std(control_values)
        }

        return analysis_results

    def make_decision(self, analysis_results: Dict, min_effect_size: float = 0.1):
        """Make a decision based on statistical analysis"""
        recommendations = []

        best_variant = None
        best_improvement = -float('inf')

        for variant_name, comparison in analysis_results['comparisons'].items():
            vs_control = comparison['vs_control']

            if vs_control['significant'] and vs_control['effect_size'] > min_effect_size:
                if vs_control['difference'] > best_improvement:
                    best_improvement = vs_control['difference']
                    best_variant = variant_name

                recommendations.append({
                    'variant': variant_name,
                    'action': 'PROMOTE',
                    'reason': f'Statistically significant improvement of {vs_control["difference"]:.4f}',
                    'confidence': f'p-value: {vs_control["p_value"]:.4f}'
                })

            elif vs_control['significant'] and vs_control['difference'] < -min_effect_size:
                recommendations.append({
                    'variant': variant_name,
                    'action': 'REJECT',
                    'reason': f'Statistically significant degradation of {vs_control["difference"]:.4f}',
                    'confidence': f'p-value: {vs_control["p_value"]:.4f}'
                })

            else:
                recommendations.append({
                    'variant': variant_name,
                    'action': 'INCONCLUSIVE',
                    'reason': 'No significant difference detected or effect size too small',
                    'confidence': f'p-value: {vs_control["p_value"]:.4f}'
                })

        decision = {
            'best_variant': best_variant or analysis_results['control_variant'],
            'recommendations': recommendations,
            'overall_recommendation': 'PROMOTE' if best_variant else 'KEEP_CONTROL'
        }

        return decision

    def stop_test(self):
        """Stop the A/B test"""
        self.test_active = False

        # Save final results
        final_results = {
            'test_name': self.test_name,
            'variants': {name: {'is_control': info['is_control']} for name, info in self.variants.items()},
            'traffic_allocation': self.traffic_allocation,
            'start_time': self.start_time,
            'end_time': time.time(),
            'duration_hours': (time.time() - self.start_time) / 3600,
            'final_metrics': self.metrics
        }

        with open(self.results_dir / 'final_results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            # Deep convert the nested structure
            def deep_convert(obj):
                if isinstance(obj, dict):
                    return {k: deep_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [deep_convert(v) for v in obj]
                else:
                    return convert_numpy(obj)

            json.dump(deep_convert(final_results), f, indent=2)

        print(f"A/B test '{self.test_name}' stopped and results saved")

# Demonstrate A/B testing
try:
    from scipy import stats

    # Create A/B test
    ab_test = ABTestingFramework("model_improvement_test")

    # Create two model variants
    control_model = SimpleClassifier(20, 64, 3)
    treatment_model = SimpleClassifier(20, 96, 3)  # Larger model

    # Add variants to test
    ab_test.add_variant('control', control_model, traffic_percentage=50, is_control=True)
    ab_test.add_variant('treatment', treatment_model, traffic_percentage=50, is_control=False)

    # Start test
    ab_test.start_test(duration_hours=0.1)  # Short test for demo

    # Simulate user interactions
    print("\nSimulating user interactions...")

    # Generate realistic performance differences
    np.random.seed(42)

    for i in range(200):  # Simulate 200 users
        user_id = f"user_{i}"
        variant = ab_test.assign_variant(user_id)

        # Simulate different performance for variants
        if variant == 'control':
            # Control has baseline performance
            accuracy = np.random.normal(0.85, 0.05)
            latency = np.random.normal(50, 10)  # milliseconds
        else:
            # Treatment has slightly better accuracy but higher latency
            accuracy = np.random.normal(0.88, 0.05)  # 3% improvement
            latency = np.random.normal(65, 12)  # 15ms higher latency

        # Ensure valid ranges
        accuracy = np.clip(accuracy, 0, 1)
        latency = max(latency, 10)

        ab_test.log_result(user_id, variant, {
            'accuracy': accuracy,
            'latency_ms': latency
        })

    # Check test status
    status = ab_test.get_test_status()
    print(f"\nTest Status: {status['status']}")
    print(f"Sample sizes: {status['sample_sizes']}")
    print(f"Ready for analysis: {status['ready_for_analysis']}")

    if status['ready_for_analysis']:
        # Run statistical analysis
        print("\nRunning statistical analysis...")
        analysis = ab_test.run_statistical_analysis('accuracy')

        print(f"\nAccuracy Analysis Results:")
        print(f"Control ({analysis['control_variant']}): {analysis['control_stats']['mean']:.4f} ± {analysis['control_stats']['std']:.4f}")

        for variant_name, comparison in analysis['comparisons'].items():
            vs_control = comparison['vs_control']
            print(f"{variant_name}: {comparison['mean']:.4f} ± {comparison['std']:.4f}")
            print(f"  Difference: {vs_control['difference']:.4f} (p-value: {vs_control['p_value']:.4f})")
            print(f"  Significant: {vs_control['significant']} (effect size: {vs_control['effect_size']:.3f})")
            print(f"  95% CI: [{vs_control['confidence_interval'][0]:.4f}, {vs_control['confidence_interval'][1]:.4f}]")

        # Make decision
        decision = ab_test.make_decision(analysis, min_effect_size=0.1)

        print(f"\nDecision: {decision['overall_recommendation']}")
        print(f"Best variant: {decision['best_variant']}")

        for rec in decision['recommendations']:
            print(f"\n{rec['variant']}: {rec['action']}")
            print(f"  Reason: {rec['reason']}")
            print(f"  Confidence: {rec['confidence']}")

        # Analyze latency as well
        latency_analysis = ab_test.run_statistical_analysis('latency_ms')
        print(f"\nLatency Analysis Results:")
        print(f"Control: {latency_analysis['control_stats']['mean']:.1f}ms ± {latency_analysis['control_stats']['std']:.1f}ms")

        for variant_name, comparison in latency_analysis['comparisons'].items():
            vs_control = comparison['vs_control']
            print(f"{variant_name}: {comparison['mean']:.1f}ms (difference: {vs_control['difference']:.1f}ms, p-value: {vs_control['p_value']:.4f})")

    # Stop test
    ab_test.stop_test()

except ImportError:
    print("scipy not available. Install with: pip install scipy")
    print("Showing A/B testing framework structure instead:")

    # Show framework without statistical analysis
    ab_test = ABTestingFramework("demo_test")
    control_model = SimpleClassifier(20, 64, 3)
    ab_test.add_variant('control', control_model, traffic_percentage=100, is_control=True)

    print("\nA/B Testing Framework Features:")
    print("  • Traffic allocation and variant assignment")
    print("  • Statistical significance testing")
    print("  • Effect size calculation")
    print("  • Automated decision making")
    print("  • Experiment tracking and logging")

# %% [markdown]
# ## Advanced Deployment Strategies
#
# Modern ML deployment requires sophisticated strategies for scaling, reliability, and performance. This includes edge deployment, model serving optimization, and distributed inference.

# %%
# Model Serving and Deployment Strategies
class ModelServer:
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.model = None
        self.model_config = {}
        self.deployment_config = {}
        self.performance_stats = {
            'requests_served': 0,
            'total_inference_time': 0,
            'errors': 0,
            'start_time': time.time()
        }

    def load_model(self, model: nn.Module, config: Dict[str, Any] = None):
        """Load model for serving"""
        self.model = model
        self.model.eval()
        self.model_config = config or {}

        # Warm up model
        self._warmup_model()
        print(f"Model {self.model_name} v{self.model_version} loaded and warmed up")

    def _warmup_model(self, warmup_batches: int = 10):
        """Warm up model with dummy requests"""
        if self.model is None:
            return

        # Determine input shape from model config or use default
        input_shape = self.model_config.get('input_shape', [1, 20])

        with torch.no_grad():
            for _ in range(warmup_batches):
                dummy_input = torch.randn(*input_shape)
                _ = self.model(dummy_input)

    def predict(self, input_data: torch.Tensor, return_probabilities: bool = True):
        """Make prediction with performance tracking"""
        start_time = time.time()

        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")

            with torch.no_grad():
                # Ensure input is in correct format
                if input_data.dim() == 1:
                    input_data = input_data.unsqueeze(0)

                # Forward pass
                logits = self.model(input_data)

                # Get predictions
                predictions = torch.argmax(logits, dim=1)

                result = {
                    'predictions': predictions.tolist(),
                    'input_shape': list(input_data.shape),
                    'batch_size': input_data.size(0)
                }

                if return_probabilities:
                    probabilities = torch.softmax(logits, dim=1)
                    result['probabilities'] = probabilities.tolist()
                    result['confidence_scores'] = torch.max(probabilities, dim=1)[0].tolist()

                # Update performance stats
                inference_time = time.time() - start_time
                self.performance_stats['requests_served'] += 1
                self.performance_stats['total_inference_time'] += inference_time

                result['inference_time_ms'] = inference_time * 1000

                return result

        except Exception as e:
            self.performance_stats['errors'] += 1
            return {
                'error': str(e),
                'inference_time_ms': (time.time() - start_time) * 1000
            }

    def batch_predict(self, input_batch: List[torch.Tensor], max_batch_size: int = 32):
        """Handle batch predictions with automatic batching"""
        results = []

        # Process in chunks
        for i in range(0, len(input_batch), max_batch_size):
            batch_chunk = input_batch[i:i + max_batch_size]

            # Stack tensors
            stacked_input = torch.stack(batch_chunk)

            # Predict
            batch_result = self.predict(stacked_input)

            if 'error' not in batch_result:
                # Split results back
                for j in range(len(batch_chunk)):
                    individual_result = {
                        'prediction': batch_result['predictions'][j],
                        'inference_time_ms': batch_result['inference_time_ms'] / len(batch_chunk)
                    }

                    if 'probabilities' in batch_result:
                        individual_result['probabilities'] = batch_result['probabilities'][j]
                        individual_result['confidence_score'] = batch_result['confidence_scores'][j]

                    results.append(individual_result)
            else:
                # Handle error for entire chunk
                for _ in batch_chunk:
                    results.append(batch_result)

        return results

    def get_performance_stats(self):
        """Get server performance statistics"""
        uptime = time.time() - self.performance_stats['start_time']
        requests_served = self.performance_stats['requests_served']

        stats = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'uptime_seconds': uptime,
            'requests_served': requests_served,
            'errors': self.performance_stats['errors'],
            'error_rate': self.performance_stats['errors'] / max(requests_served, 1),
            'requests_per_second': requests_served / max(uptime, 1)
        }

        if requests_served > 0:
            stats['average_inference_time_ms'] = (self.performance_stats['total_inference_time'] / requests_served) * 1000
        else:
            stats['average_inference_time_ms'] = 0

        return stats

    def health_check(self):
        """Health check for load balancer"""
        try:
            # Quick inference test
            test_input = torch.randn(1, self.model_config.get('input_features', 20))
            result = self.predict(test_input)

            if 'error' in result:
                return {'status': 'unhealthy', 'reason': result['error']}

            stats = self.get_performance_stats()

            # Check various health indicators
            if stats['error_rate'] > 0.1:  # More than 10% error rate
                return {'status': 'unhealthy', 'reason': 'High error rate'}

            if result['inference_time_ms'] > 1000:  # Slower than 1 second
                return {'status': 'degraded', 'reason': 'Slow inference'}

            return {'status': 'healthy', 'stats': stats}

        except Exception as e:
            return {'status': 'unhealthy', 'reason': f'Health check failed: {str(e)}'}

# Load Balancer for multiple model servers
class ModelLoadBalancer:
    def __init__(self, balancing_strategy: str = 'round_robin'):
        self.servers = []
        self.balancing_strategy = balancing_strategy
        self.current_index = 0
        self.server_weights = {}

    def add_server(self, server: ModelServer, weight: float = 1.0):
        """Add a server to the load balancer"""
        self.servers.append(server)
        self.server_weights[id(server)] = weight
        print(f"Added server {server.model_name} v{server.model_version} with weight {weight}")

    def remove_server(self, server: ModelServer):
        """Remove a server from the load balancer"""
        if server in self.servers:
            self.servers.remove(server)
            del self.server_weights[id(server)]
            print(f"Removed server {server.model_name} v{server.model_version}")

    def get_next_server(self):
        """Get next server based on balancing strategy"""
        healthy_servers = [s for s in self.servers if self._is_server_healthy(s)]

        if not healthy_servers:
            raise RuntimeError("No healthy servers available")

        if self.balancing_strategy == 'round_robin':
            server = healthy_servers[self.current_index % len(healthy_servers)]
            self.current_index += 1
            return server

        elif self.balancing_strategy == 'least_loaded':
            # Choose server with lowest request rate
            server_loads = [(s.get_performance_stats()['requests_per_second'], s) for s in healthy_servers]
            return min(server_loads, key=lambda x: x[0])[1]

        elif self.balancing_strategy == 'weighted':
            # Weighted random selection
            weights = [self.server_weights[id(s)] for s in healthy_servers]
            total_weight = sum(weights)
            r = np.random.random() * total_weight

            cumsum = 0
            for server, weight in zip(healthy_servers, weights):
                cumsum += weight
                if r <= cumsum:
                    return server

            return healthy_servers[-1]  # Fallback

        else:
            return healthy_servers[0]  # Default fallback

    def _is_server_healthy(self, server: ModelServer):
        """Check if server is healthy"""
        health = server.health_check()
        return health['status'] in ['healthy', 'degraded']

    def predict(self, input_data: torch.Tensor):
        """Route prediction request to appropriate server"""
        try:
            server = self.get_next_server()
            result = server.predict(input_data)
            result['served_by'] = f"{server.model_name}_v{server.model_version}"
            return result
        except RuntimeError as e:
            return {'error': str(e)}

    def get_cluster_status(self):
        """Get status of all servers in the cluster"""
        cluster_status = {
            'total_servers': len(self.servers),
            'balancing_strategy': self.balancing_strategy,
            'servers': []
        }

        healthy_count = 0
        total_requests = 0
        total_errors = 0

        for server in self.servers:
            health = server.health_check()
            stats = server.get_performance_stats()

            server_info = {
                'model_name': server.model_name,
                'model_version': server.model_version,
                'health_status': health['status'],
                'weight': self.server_weights[id(server)],
                'stats': stats
            }

            cluster_status['servers'].append(server_info)

            if health['status'] in ['healthy', 'degraded']:
                healthy_count += 1

            total_requests += stats['requests_served']
            total_errors += stats['errors']

        cluster_status['healthy_servers'] = healthy_count
        cluster_status['total_requests'] = total_requests
        cluster_status['total_errors'] = total_errors
        cluster_status['cluster_error_rate'] = total_errors / max(total_requests, 1)

        return cluster_status

# Demonstrate model serving and load balancing
print("Setting up model serving cluster...")

# Create multiple model servers
server1 = ModelServer("classifier", "v1.0")
server2 = ModelServer("classifier", "v1.1")
server3 = ModelServer("classifier", "v1.2")

# Load models (using different sizes to simulate different performance)
model1 = SimpleClassifier(20, 64, 3)
model2 = SimpleClassifier(20, 96, 3)
model3 = SimpleClassifier(20, 128, 3)

server1.load_model(model1, {'input_features': 20, 'input_shape': [1, 20]})
server2.load_model(model2, {'input_features': 20, 'input_shape': [1, 20]})
server3.load_model(model3, {'input_features': 20, 'input_shape': [1, 20]})

# Create load balancer
load_balancer = ModelLoadBalancer(balancing_strategy='weighted')
load_balancer.add_server(server1, weight=1.0)
load_balancer.add_server(server2, weight=2.0)  # Higher weight
load_balancer.add_server(server3, weight=1.5)

# Simulate load
print("\nSimulating production load...")
for i in range(50):
    # Generate random input
    input_data = torch.randn(20)

    # Route through load balancer
    result = load_balancer.predict(input_data)

    if 'error' not in result and i % 10 == 0:
        print(f"Request {i}: Served by {result['served_by']}, "
              f"Prediction: {result['predictions'][0]}, "
              f"Confidence: {result['confidence_scores'][0]:.3f}, "
              f"Latency: {result['inference_time_ms']:.2f}ms")

# Get cluster status
cluster_status = load_balancer.get_cluster_status()

print("\n" + "="*60)
print("CLUSTER STATUS REPORT")
print("="*60)
print(f"Total Servers: {cluster_status['total_servers']}")
print(f"Healthy Servers: {cluster_status['healthy_servers']}")
print(f"Balancing Strategy: {cluster_status['balancing_strategy']}")
print(f"Total Requests: {cluster_status['total_requests']}")
print(f"Cluster Error Rate: {cluster_status['cluster_error_rate']:.4f}")

print("\nServer Details:")
for server_info in cluster_status['servers']:
    stats = server_info['stats']
    print(f"\n{server_info['model_name']} v{server_info['model_version']}:")
    print(f"  Health: {server_info['health_status']}")
    print(f"  Weight: {server_info['weight']}")
    print(f"  Requests: {stats['requests_served']}")
    print(f"  RPS: {stats['requests_per_second']:.2f}")
    print(f"  Avg Latency: {stats['average_inference_time_ms']:.2f}ms")
    print(f"  Error Rate: {stats['error_rate']:.4f}")

# Demonstrate batch processing
print("\nDemonstrating batch processing...")
batch_inputs = [torch.randn(20) for _ in range(25)]  # 25 individual inputs
batch_results = server2.batch_predict(batch_inputs, max_batch_size=8)

print(f"Processed batch of {len(batch_inputs)} inputs")
print(f"Average latency per item: {np.mean([r['inference_time_ms'] for r in batch_results]):.2f}ms")
print(f"Total batch processing time: {sum(r['inference_time_ms'] for r in batch_results):.2f}ms")

print("\n" + "="*60)
print("DEPLOYMENT BEST PRACTICES SUMMARY:")
print("="*60)
print("✓ Model versioning and registry")
print("✓ Health checks and monitoring")
print("✓ Load balancing and auto-scaling")
print("✓ Batch processing optimization")
print("✓ Performance tracking and alerting")
print("✓ Graceful degradation and error handling")
print("✓ A/B testing and canary deployments")
print("✓ Experiment tracking and reproducibility")

# %% [markdown]
# ## MLOps Best Practices and Production Checklist
#
# Successful MLOps requires comprehensive practices covering the entire ML lifecycle. This section provides a complete checklist and best practices for production ML systems.

# %%
print("MLOps and Production PyTorch: Comprehensive Best Practices Guide")
print("="*80)

mlops_checklist = {
    "Data Management": {
        "Data Quality": [
            "Data validation and schema enforcement",
            "Outlier detection and handling",
            "Missing data imputation strategies",
            "Data freshness and staleness monitoring",
            "Data lineage tracking"
        ],
        "Data Pipeline": [
            "Reproducible data preprocessing",
            "Feature store implementation",
            "Data versioning and snapshots",
            "Pipeline orchestration (Airflow, Kubeflow)",
            "Real-time and batch processing support"
        ],
        "Data Governance": [
            "Data privacy and compliance (GDPR, CCPA)",
            "Access control and audit trails",
            "Data retention policies",
            "Sensitive data anonymization",
            "Cross-region data regulations"
        ]
    },

    "Model Development": {
        "Experimentation": [
            "Experiment tracking (MLflow, Weights & Biases)",
            "Hyperparameter optimization",
            "Cross-validation strategies",
            "Reproducible random seeds",
            "Environment containerization"
        ],
        "Code Quality": [
            "Version control for all code",
            "Unit tests for data and model code",
            "Code review processes",
            "Linting and formatting standards",
            "Documentation and comments"
        ],
        "Model Validation": [
            "Cross-validation and holdout testing",
            "Statistical significance testing",
            "Fairness and bias evaluation",
            "Model interpretability analysis",
            "Stress testing and edge cases"
        ]
    },

    "Model Registry & Versioning": {
        "Model Management": [
            "Centralized model registry",
            "Semantic versioning (major.minor.patch)",
            "Model metadata and lineage",
            "Stage management (dev/staging/prod)",
            "Model approval workflows"
        ],
        "Artifacts": [
            "Model weights and architecture",
            "Training configuration and hyperparameters",
            "Performance metrics and validation results",
            "Dependencies and environment specs",
            "Data preprocessing pipelines"
        ],
        "Lifecycle Management": [
            "Model promotion criteria",
            "Rollback procedures",
            "Deprecation policies",
            "Model retirement workflows",
            "Compliance and audit trails"
        ]
    },

    "Deployment & Serving": {
        "Infrastructure": [
            "Containerization (Docker, Kubernetes)",
            "Auto-scaling policies",
            "Load balancing strategies",
            "Resource allocation and limits",
            "Multi-region deployment"
        ],
        "Serving Optimization": [
            "Model quantization and compression",
            "Batch prediction optimization",
            "GPU/CPU resource optimization",
            "Caching strategies",
            "Edge deployment considerations"
        ],
        "Deployment Strategies": [
            "Blue-green deployments",
            "Canary releases",
            "A/B testing frameworks",
            "Feature flags and toggles",
            "Gradual traffic migration"
        ]
    },

    "Monitoring & Observability": {
        "Performance Monitoring": [
            "Latency and throughput metrics",
            "Error rates and success rates",
            "Resource utilization (CPU, GPU, memory)",
            "Queue depths and processing times",
            "SLA compliance tracking"
        ],
        "Model Health": [
            "Data drift detection",
            "Model drift monitoring",
            "Prediction quality metrics",
            "Feature importance changes",
            "Concept drift identification"
        ],
        "Alerting & Response": [
            "Real-time alerting systems",
            "Escalation procedures",
            "Automated response actions",
            "Dashboard and visualization",
            "Incident response playbooks"
        ]
    },

    "Security & Compliance": {
        "Model Security": [
            "Adversarial attack protection",
            "Input validation and sanitization",
            "Model poisoning prevention",
            "Secure model storage",
            "API authentication and authorization"
        ],
        "Data Protection": [
            "Encryption at rest and in transit",
            "PII detection and masking",
            "Data access logging",
            "Secure data transmission",
            "Right to be forgotten compliance"
        ],
        "Regulatory Compliance": [
            "Model explainability requirements",
            "Audit trail maintenance",
            "Regulatory approval processes",
            "Documentation standards",
            "Risk assessment procedures"
        ]
    },

    "Testing & Validation": {
        "Testing Strategies": [
            "Unit tests for data processing",
            "Integration tests for pipelines",
            "Model validation tests",
            "Load testing and stress testing",
            "End-to-end system tests"
        ],
        "Continuous Testing": [
            "Automated testing in CI/CD",
            "Regression testing for model changes",
            "Data quality tests",
            "Performance benchmarking",
            "Shadow testing in production"
        ],
        "Validation Frameworks": [
            "Great Expectations for data validation",
            "TensorFlow Data Validation (TFDV)",
            "Custom validation rules",
            "Statistical testing frameworks",
            "Model comparison utilities"
        ]
    }
}

# Print the comprehensive checklist
for category, subcategories in mlops_checklist.items():
    print(f"\n{category}:")
    print("-" * (len(category) + 1))

    for subcategory, items in subcategories.items():
        print(f"\n  {subcategory}:")
        for item in items:
            print(f"    ☐ {item}")

print("\n" + "="*80)
print("PYTORCH-SPECIFIC PRODUCTION CONSIDERATIONS:")
print("="*80)

pytorch_specifics = {
    "Model Optimization": [
        "TorchScript for deployment (torch.jit.script/trace)",
        "Quantization with torch.quantization",
        "ONNX export for cross-platform deployment",
        "TensorRT integration for NVIDIA GPUs",
        "Mobile deployment with PyTorch Mobile"
    ],

    "Performance Tuning": [
        "DataLoader optimization (num_workers, pin_memory)",
        "Mixed precision training/inference (torch.amp)",
        "Gradient checkpointing for memory efficiency",
        "Model parallelism for large models",
        "Efficient data preprocessing pipelines"
    ],

    "Deployment Formats": [
        "PyTorch Lightning for structured training",
        "TorchServe for model serving",
        "Ray Serve for distributed serving",
        "Triton Inference Server integration",
        "Custom Flask/FastAPI serving endpoints"
    ],

    "Monitoring Tools": [
        "PyTorch Profiler for performance analysis",
        "TensorBoard for metrics visualization",
        "Weights & Biases for experiment tracking",
        "MLflow for model lifecycle management",
        "Custom logging with Python logging module"
    ]
}

for category, items in pytorch_specifics.items():
    print(f"\n{category}:")
    for item in items:
        print(f"  • {item}")

print("\n" + "="*80)
print("PRODUCTION READINESS ASSESSMENT:")
print("="*80)

readiness_criteria = {
    "Critical (Must Have)": [
        "Model performance meets business requirements",
        "Comprehensive testing coverage",
        "Monitoring and alerting in place",
        "Rollback procedures tested",
        "Security measures implemented",
        "Data quality validation active",
        "Disaster recovery plan exists"
    ],

    "Important (Should Have)": [
        "A/B testing capability",
        "Automated retraining pipeline",
        "Model interpretability tools",
        "Performance optimization applied",
        "Documentation complete",
        "Team training on operations",
        "Compliance requirements met"
    ],

    "Nice to Have (Could Have)": [
        "Advanced drift detection algorithms",
        "Automated model selection",
        "Multi-region deployment",
        "Real-time feature engineering",
        "Advanced model compression",
        "Custom hardware optimization",
        "MLOps platform integration"
    ]
}

for priority, criteria in readiness_criteria.items():
    print(f"\n{priority}:")
    for criterion in criteria:
        print(f"  ☐ {criterion}")

print("\n" + "="*80)
print("RECOMMENDED TOOLS AND FRAMEWORKS:")
print("="*80)

recommended_tools = {
    "Experiment Tracking": [
        "MLflow - Open-source ML lifecycle management",
        "Weights & Biases - Comprehensive experiment tracking",
        "Neptune - Collaborative ML experiment management",
        "Comet - ML experiment tracking and monitoring"
    ],

    "Model Serving": [
        "TorchServe - PyTorch native model serving",
        "BentoML - ML model serving framework",
        "Seldon Core - Kubernetes-native ML deployment",
        "KFServing - Kubernetes-based model serving"
    ],

    "Pipeline Orchestration": [
        "Apache Airflow - Workflow orchestration",
        "Kubeflow - ML workflows on Kubernetes",
        "Prefect - Modern workflow orchestration",
        "DVC - Data Version Control for ML pipelines"
    ],

    "Monitoring & Observability": [
        "Prometheus + Grafana - Metrics and dashboards",
        "ELK Stack - Logging and search",
        "Jaeger - Distributed tracing",
        "Evidently AI - ML model monitoring"
    ],

    "Infrastructure": [
        "Docker + Kubernetes - Containerization and orchestration",
        "AWS SageMaker - Managed ML platform",
        "Google Vertex AI - ML platform",
        "Azure ML - Microsoft's ML platform"
    ]
}

for category, tools in recommended_tools.items():
    print(f"\n{category}:")
    for tool in tools:
        print(f"  • {tool}")

print("\n" + "="*80)
print("FINAL RECOMMENDATIONS:")
print("="*80)
print("")
print("1. Start Simple, Scale Gradually:")
print("   • Begin with basic monitoring and logging")
print("   • Add complexity as you understand your needs")
print("   • Focus on reliability over features initially")
print("")
print("2. Automate Everything:")
print("   • Automate testing, deployment, and monitoring")
print("   • Use CI/CD pipelines for consistency")
print("   • Implement infrastructure as code")
print("")
print("3. Monitor Proactively:")
print("   • Set up comprehensive monitoring from day one")
print("   • Define clear SLAs and alert thresholds")
print("   • Monitor both technical and business metrics")
print("")
print("4. Plan for Failure:")
print("   • Design for graceful degradation")
print("   • Test rollback procedures regularly")
print("   • Have incident response procedures ready")
print("")
print("5. Maintain Documentation:")
print("   • Document architecture decisions")
print("   • Keep runbooks up to date")
print("   • Train team members on operations")
print("")
print("Remember: MLOps is a journey, not a destination. Start with the basics")
print("and continuously improve your practices as your system matures!")
print("="*80)