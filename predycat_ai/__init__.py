"""
PredycatAI Universal Optimizer
=========================
A production-grade AI optimization system for any neural network.
"""


class TargetPlatform:
    LAPTOP = "laptop"
    CLOUD = "cloud"
    MOBILE = "mobile"
    EDGE = "edge"


class ModelType:
    LLM = "llm"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    HYBRID = "hybrid"


class QuantizationType:
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    INT3 = "int3"


class ExportFormat:
    PYTORCH = "pt"
    ONNX = "onnx"
    GGUF = "gguf"


class OptimizationPreset:
    ULTRA_COMPRESSION = "ultra_compression"
    BALANCED = "balanced"
    MAX_ACCURACY = "max_accuracy"
    MOBILE_SAFE = "mobile_safe"


DEFAULT_CONFIG = {
    "max_accuracy_drop": 2.0,
    "quantization_priority": True,
    "safe_pruning_threshold": 0.3,
    "use_distillation_for_mobile": True,
    "enable_thermal_optimization": True,
    "batch_sizes": {
        "laptop": 8,
        "cloud": 16,
        "mobile": 1,
        "edge": 1
    },
    "memory_limits_mb": {
        "laptop": 8192,
        "cloud": 16384,
        "mobile": 2048,
        "edge": 1024
    }
}