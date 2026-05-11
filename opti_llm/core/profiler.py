"""Multi-Environment Profiler - Tests models across different hardware targets"""

import os
import time
import logging
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

import torch
import numpy as np

logger = logging.getLogger(__name__)


class Platform(Enum):
    LAPTOP = "laptop"
    CLOUD = "cloud"
    MOBILE = "mobile"
    EDGE = "edge"


@dataclass
class ProfileResult:
    platform: Platform
    latency_ms: float
    throughput_tokens_per_sec: float
    vram_mb: float
    ram_mb: float
    flops_estimate: float
    inference_count: int = 0
    warmup_runs: int = 3
    measured_runs: int = 10
    
    
@dataclass
class MobileProfileResult:
    platform: Platform
    latency_ms: float
    ram_mb: float
    cpu_usage_percent: float
    thermal_throttle_risk: float
    estimated_power_mw: float
    sustained_runtime_sec: float = 0.0
    
    
@dataclass
class FullProfile:
    laptop: Optional[ProfileResult] = None
    cloud: Optional[ProfileResult] = None
    mobile: Optional[MobileProfileResult] = None
    edge: Optional[MobileProfileResult] = None


class MultiProfiler:
    def __init__(self, model_wrapper: Any = None):
        self.model_wrapper = model_wrapper
        self.results = FullProfile()
        self._stop_monitoring = threading.Event()
        self._monitor_thread = None
        
    def profile_platform(
        self,
        model: Any,
        platform: Platform,
        input_shape: Optional[tuple] = None,
        sequence_length: int = 512,
        batch_size: int = 1,
        **kwargs
    ) -> ProfileResult:
        logger.info(f"Profiling for platform: {platform.value}")
        
        if platform in [Platform.LAPTOP, Platform.CLOUD]:
            return self._profile_gpu_platform(model, platform, input_shape, sequence_length, batch_size, **kwargs)
        else:
            return self._profile_cpu_only(model, platform, input_shape, sequence_length, batch_size, **kwargs)
    
    def _profile_gpu_platform(
        self,
        model: Any,
        platform: Platform,
        input_shape: Optional[tuple],
        sequence_length: int,
        batch_size: int,
        **kwargs
    ) -> ProfileResult:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_warmup = kwargs.get("warmup_runs", 3)
        num_runs = kwargs.get("measured_runs", 10)
        
        if hasattr(model, "device"):
            device = str(model.device)
        
        if input_shape is None:
            if hasattr(model, "config"):
                hidden_size = getattr(model.config, "hidden_size", 768)
                vocab_size = getattr(model.config, "vocab_size", 32000)
            else:
                hidden_size = 768
                vocab_size = 32000
            input_shape = (batch_size, sequence_length)
        
        if hasattr(model, "__call__"):
            dummy_input = torch.randint(0, vocab_size, input_shape, device=device)
            
            for _ in range(num_warmup):
                with torch.no_grad():
                    try:
                        _ = model(dummy_input)
                    except Exception as e:
                        logger.warning(f"Warmup failed: {e}")
                        break
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            
            latencies = []
            vram_before = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            
            for _ in range(num_runs):
                start = time.perf_counter()
                with torch.no_grad():
                    try:
                        _ = model(dummy_input)
                    except Exception as e:
                        logger.warning(f"Inference failed: {e}")
                        break
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
            
            vram_after = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
            
            avg_latency = np.mean(latencies) if latencies else 0
            std_latency = np.std(latencies) if latencies else 0
            
            throughput = batch_size / (avg_latency / 1000) if avg_latency > 0 else 0
            
            flops_estimate = self._estimate_flops(batch_size, sequence_length, hidden_size)
            
            return ProfileResult(
                platform=platform,
                latency_ms=avg_latency,
                throughput_tokens_per_sec=throughput,
                vram_mb=abs(vram_after - vram_before) + vram_before,
                ram_mb=psutil.Process().memory_info().rss / (1024**2),
                flops_estimate=flops_estimate,
                inference_count=len(latencies)
            )
        else:
            return ProfileResult(
                platform=platform,
                latency_ms=0,
                throughput_tokens_per_sec=0,
                vram_mb=0,
                ram_mb=0,
                flops_estimate=0
            )
    
    def _profile_cpu_only(
        self,
        model: Any,
        platform: Platform,
        input_shape: Optional[tuple],
        sequence_length: int,
        batch_size: int,
        **kwargs
    ) -> MobileProfileResult:
        latency_ms = 0
        cpu_usage = 0
        ram_usage = 0
        
        if hasattr(model, "device"):
            try:
                model = model.to("cpu")
            except:
                pass
        
        if input_shape is None:
            if hasattr(model, "config"):
                hidden_size = getattr(model.config, "hidden_size", 768)
                vocab_size = getattr(model.config, "vocab_size", 32000)
            else:
                hidden_size = 768
                vocab_size = 32000
            input_shape = (batch_size, sequence_length)
        
        num_warmup = 2
        num_runs = 5
        
        if hasattr(model, "__call__"):
            vocab_size = kwargs.get("vocab_size", 32000)
            dummy_input = torch.randint(0, vocab_size, input_shape)
            
            try:
                for _ in range(num_warmup):
                    with torch.no_grad():
                        _ = model(dummy_input)
            except:
                pass
            
            latencies = []
            for _ in range(num_runs):
                start = time.perf_counter()
                try:
                    with torch.no_grad():
                        _ = model(dummy_input)
                except:
                    pass
                latency = (time.perf_counter() - start) * 1000
                latencies.append(latency)
            
            avg_latency = np.mean(latencies) if latencies else 0
            
            try:
                process = psutil.Process()
                ram_usage = process.memory_info().rss / (1024**2)
                cpu_usage = process.cpu_percent(interval=0.1)
            except:
                pass
            
            estimated_power = self._estimate_power_consumption(avg_latency, ram_usage)
            thermal_risk = self._calculate_thermal_risk(avg_latency, cpu_usage)
            
            return MobileProfileResult(
                platform=platform,
                latency_ms=avg_latency,
                ram_mb=ram_usage,
                cpu_usage_percent=cpu_usage,
                thermal_throttle_risk=thermal_risk,
                estimated_power_mw=estimated_power
            )
        
        return MobileProfileResult(
            platform=platform,
            latency_ms=0,
            ram_mb=0,
            cpu_usage_percent=0,
            thermal_throttle_risk=0,
            estimated_power_mw=0
        )
    
    def profile_full(
        self,
        model: Any,
        input_shape: Optional[tuple] = None,
        **kwargs
    ) -> FullProfile:
        results = FullProfile()
        
        if torch.cuda.is_available():
            results.laptop = self.profile_platform(model, Platform.LAPTOP, input_shape, **kwargs)
            results.cloud = self.profile_platform(model, Platform.CLOUD, input_shape, **kwargs)
        
        results.mobile = self.profile_platform(model, Platform.MOBILE, input_shape, **kwargs)
        results.edge = self.profile_platform(model, Platform.EDGE, input_shape, **kwargs)
        
        self.results = results
        return results
    
    def _estimate_flops(self, batch_size: int, seq_len: int, hidden_size: int) -> float:
        n_layers = 12
        attention_flops = 4 * batch_size * seq_len**2 * hidden_size
        ffw_flops = 2 * batch_size * seq_len * hidden_size * (4 * hidden_size)
        total_flops_per_layer = attention_flops + ffw_flops
        total_flops = total_flops_per_layer * n_layers
        return total_flops
    
    def _estimate_power_consumption(self, latency_ms: float, ram_mb: float) -> float:
        base_power = 500
        ram_power = ram_mb * 0.1
        latency_factor = max(1, latency_ms / 100)
        return base_power + ram_power + latency_factor * 50
    
    def _calculate_thermal_risk(self, latency_ms: float, cpu_usage: float) -> float:
        if latency_ms > 500:
            return 0.9
        elif latency_ms > 200:
            return 0.6
        
        if cpu_usage > 80:
            return 0.8
        elif cpu_usage > 60:
            return 0.5
        
        return 0.3


def quick_benchmark(model: Any, input_shape: tuple = (1, 512), device: str = "cuda") -> Dict[str, float]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    try:
        if hasattr(model, "to"):
            model = model.to(device)
        model.eval()
    except:
        pass
    
    dummy_input = torch.randint(0, 32000, input_shape, device=device)
    
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    runs = 10
    for _ in range(runs):
        with torch.no_grad():
            _ = model(dummy_input)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    latency = (time.perf_counter() - start) / runs * 1000
    
    return {
        "latency_ms": latency,
        "throughput_tokens_per_sec": input_shape[0] / (latency / 1000),
        "device": device
    }


def get_available_memory() -> Dict[str, float]:
    mem = {}
    
    if torch.cuda.is_available():
        mem["vram_mb"] = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**2)
    else:
        mem["vram_mb"] = 0
    
    virtual = psutil.virtual_memory()
    mem["ram_available_mb"] = virtual.available / (1024**2)
    mem["ram_total_mb"] = virtual.total / (1024**2)
    
    return mem


def check_platform_compatibility(platform: Platform, memory_mb: float, require_gpu: bool = False) -> bool:
    available = get_available_memory()
    
    if require_gpu and platform != Platform.MOBILE:
        if "vram_mb" not in available or available["vram_mb"] < memory_mb:
            return False
    
    if platform == Platform.MOBILE:
        return available.get("ram_available_mb", 0) > memory_mb * 2
    
    return True