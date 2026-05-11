"""Thermal & Power Optimization for Mobile/Edge"""

import time
import threading
import psutil
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThermalState:
    temperature_c: float
    cpu_percent: float
    power_mw: int
    throttle_risk: float
    timestamp: float


class ThermalManager:
    """Manages thermal and power for mobile devices"""
    
    def __init__(self):
        self.monitoring = False
        self.history = deque(maxlen=100)
        self.throttle_threshold = 85.0
        self._monitor_thread = None
        
    def start_monitoring(self, interval_sec: float = 1.0):
        self.monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval_sec,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Thermal monitoring started")
        
    def stop_monitoring(self):
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
            
    def _monitor_loop(self, interval: float):
        while self.monitoring:
            state = self._read_thermal_state()
            self.history.append(state)
            time.sleep(interval)
            
    def _read_thermal_state(self) -> ThermalState:
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            
            try:
                temp = psutil.sensors_temperatures().get('cpu', [0])[0].current
            except:
                temp = 45.0
                
            power = int(500 + cpu * 10)
            throttle = self._calc_throttle_risk(temp, cpu)
            
            return ThermalState(
                temperature_c=temp or 45.0,
                cpu_percent=cpu,
                power_mw=power,
                throttle_risk=throttle,
                timestamp=time.time()
            )
            
        except Exception as e:
            return ThermalState(45.0, 0, 500, 0.0, time.time())
    
    def _calc_throttle_risk(self, temp: float, cpu: float) -> float:
        risk = 0.0
        
        if temp > 80:
            risk += 0.5
        elif temp > 70:
            risk += 0.25
            
        if cpu > 90:
            risk += 0.3
        elif cpu > 70:
            risk += 0.15
            
        return min(1.0, risk)
    
    def should_throttle(self) -> bool:
        if not self.history:
            return False
        recent = list(self.history)[-5:]
        avg_risk = sum(s.throttle_risk for s in recent) / len(recent)
        return avg_risk > self.throttle_threshold
    
    def get_stats(self) -> Dict:
        if not self.history:
            return {}
            
        temps = [s.temperature_c for s in self.history]
        cpus = [s.cpu_percent for s in self.history]
        
        return {
            "avg_temp_c": sum(temps) / len(temps),
            "max_temp_c": max(temps),
            "avg_cpu": sum(cpus) / len(cpus),
            "max_cpu": max(cpus),
            "samples": len(self.history)
        }


class DynamicThrottler:
    """Dynamically throttles based on thermal state"""
    
    def __init__(self, thermal_manager: ThermalManager):
        self.thermal = thermal_manager
        self.batch_size = 1
        self.max_batch = 8
        
    def adjust_batch_size(self) -> int:
        if self.thermal.should_throttle():
            self.batch_size = max(1, self.batch_size // 2)
            logger.info(f"Throttled batch size to {self.batch_size}")
        else:
            self.batch_size = min(self.max_batch, self.batch_size + 1)
            
        return self.batch_size
    
    def adaptive_inference(
        self, 
        model, 
        inputs, 
        max_tokens: int = 100
    ) -> List:
        results = []
        tokens_per_batch = max(1, max_tokens // self.batch_size)
        
        for i in range(self.batch_size):
            try:
                output = model(inputs)
                results.append(output)
            except Exception as e:
                logger.warning(f"Batch {i} failed: {e}")
                
        return results


def optimize_for_mobile(
    model,
    thermal_budget_c: float = 70.0,
    power_budget_mw: int = 2000
) -> dict:
    """Optimize model for mobile constraints"""
    
    thermal = ThermalManager()
    thermal.start_monitoring()
    
    time.sleep(2)
    
    stats = thermal.get_stats()
    thermal.stop_monitoring()
    
    return {
        "recommendations": {
            "batch_size": 1 if stats.get("max_temp_c", 0) > thermal_budget_c else 4,
            "use_fp16": True,
            "quantize": "int8" if stats.get("max_cpu", 0) > 50 else "int4",
        },
        "estimated_power_mw": stats.get("avg_cpu", 0) * 15 + 500,
        "thermal_stats": stats
    }