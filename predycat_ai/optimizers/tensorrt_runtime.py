"""TensorRT Runtime for GPU acceleration"""

import logging
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


class TensorRTRuntime:
    """TensorRT optimization and inference for NVIDIA GPUs"""
    
    def __init__(self):
        self.engine = None
        self.context = None
        self.trt = None
        
    def is_available(self) -> bool:
        try:
            import tensorrt as trt
            self.trt = trt
            return True
        except ImportError:
            logger.info("TensorRT not available, using fallback")
            return False
    
    def optimize(
        self,
        model,
        max_batch_size: int = 8,
        max_workspace: int = 1 << 30,
        fp16: bool = True,
        int8: bool = False
    ):
        if not self.is_available():
            logger.warning("TensorRT not available")
            return None
            
        logger.info("Building TensorRT engine...")
        
        builder = self.trt.Builder(self.trt.Logger)
        network = builder.create_network(1 << 1)  # EXPLICIT_BATCH
        config = builder.create_config()
        config.set_memory_pool(self.trt.MemoryPoolType.WORKSPACE, max_workspace)
        
        if fp16:
            config.set_flag(self.trt.BuilderFlag.FP16)
        if int8:
            config.set_flag(self.trt.BuilderFlag.INT8)
            
        return self._build_engine(builder, network, config, model)
    
    def _build_engine(self, builder, network, config, model):
        try:
            engine_bytes = builder.build_serialized_network(network, config)
            if engine_bytes:
                runtime = self.trt.Runtime(self.trt.Logger)
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
                self.context = self.engine.create_execution_context()
                logger.info("TensorRT engine built successfully")
                return self.engine
        except Exception as e:
            logger.warning(f"TensorRT build failed: {e}")
            return None
    
    def inference(self, input_data) -> Optional[Any]:
        if self.context is None:
            return None
            
        try:
            h_input = input_data.ravel().astype('float32')
            d_input = self._allocate_gpu(h_input)
            h_output = self._allocate_gpu_output(h_input.shape)
            d_output = self._allocate_gpu_output(h_input.shape)
            
            self.context.execute_v2([d_input, d_output])
            
            import pycuda.driver as cuda
            cuda.memcpy_dtoh(h_output, d_output)
            
            return h_output
            
        except Exception as e:
            logger.warning(f"Inference failed: {e}")
            return None
    
    def _allocate_gpu(self, array):
        try:
            import pycuda.driver as cuda
            import pycuda.gpuarray as gpuarray
            return gpuarray.to_gpu(array)
        except:
            return None
    
    def _allocate_gpu_output(self, shape):
        return None
    
    def get_optimal_config(self, device_info: dict) -> dict:
        vram_gb = device_info.get("vram_gb", 8)
        
        if vram_gb >= 24:
            return {"batch_size": 32, "fp16": True, "workspace": 1 << 31}
        elif vram_gb >= 12:
            return {"batch_size": 16, "fp16": True, "workspace": 1 << 30}
        else:
            return {"batch_size": 8, "fp16": True, "workspace": 1 << 29}


def is_tensorrt_available() -> bool:
    try:
        import tensorrt
        return True
    except:
        return False


def get_onnxruntime_providers() -> list:
    try:
        import onnxruntime as ort
        return ort.get_available_providers()
    except:
        return ["CPUExecutionProvider"]