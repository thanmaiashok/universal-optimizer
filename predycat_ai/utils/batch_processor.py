"""Batch Processor - Process multiple models"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@dataclass
class BatchJob:
    job_id: str
    model_path: str
    target: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None


class BatchProcessor:
    """Process multiple models in batch"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.jobs: Dict[str, BatchJob] = {}
        
    def process(
        self,
        model_paths: List[str],
        target: str = "mobile",
        preset: str = "balanced",
        callback=None
    ) -> List[BatchJob]:
        """Process list of models"""
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for path in model_paths:
                job_id = f"batch_{len(self.jobs)}"
                job = BatchJob(job_id, path, target, "pending")
                self.jobs[job_id] = job
                
                future = executor.submit(
                    self._process_single,
                    job, preset, callback
                )
                futures[future] = job_id
                
            for future in as_completed(futures):
                job_id = futures[future]
                try:
                    job = future.result()
                    results.append(job)
                except Exception as e:
                    job = self.jobs[job_id]
                    job.status = "failed"
                    job.error = str(e)
                    results.append(job)
                    
        return results
    
    def _process_single(
        self,
        job: BatchJob,
        preset: str,
        callback
    ) -> BatchJob:
        """Process single model"""
        try:
            job.status = "running"
            
            from predycat_ai.core.optimizer import PredycatOptimizer, OptimizationConfig
            
            config = OptimizationConfig(
                target_platform=job.target,
                preset=preset
            )
            
            optimizer = PredycatOptimizer(config=config)
            result = optimizer.optimize(job.model_path)
            
            job.result = {
                "compression_ratio": result.compression_ratio,
                "latency_improvement": result.latency_improvement,
                "accuracy_drop": result.accuracy_drop,
            }
            job.status = "completed"
            
            if callback:
                callback(job)
                
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Batch job failed: {e}")
            
        return job
    
    def get_job(self, job_id: str) -> Optional[BatchJob]:
        return self.jobs.get(job_id)
    
    def get_status(self) -> Dict[str, int]:
        status = {"pending": 0, "running": 0, "completed": 0, "failed": 0}
        for job in self.jobs.values():
            status[job.status] = status.get(job.status, 0) + 1
        return status


def quick_batch(
    model_paths: List[str],
    target: str = "mobile",
    max_workers: int = 4
) -> List[BatchJob]:
    """Quick batch processing"""
    processor = BatchProcessor(max_workers=max_workers)
    return processor.process(model_paths, target)


class AutoRetry:
    """Auto-retry with backoff"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        
    def run(self, func, *args, **kwargs):
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait = self.backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt+1} failed: {e}, retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"All retries failed: {e}")
                    
        raise last_error