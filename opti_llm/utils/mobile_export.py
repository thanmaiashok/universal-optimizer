"""Mobile Deployment - Android/iOS deployment helpers"""

import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AndroidExporter:
    """Export for Android (TFLite/ONNX)"""
    
    def __init__(self):
        self.tflite_available = self._check_tflite()
        
    def _check_tflite(self) -> bool:
        try:
            import tensorflow as tf
            return hasattr(tf, 'lite')
        except:
            return False
            
    def export_to_tflite(
        self,
        model,
        quantization: str = "int8",
        output_path: str = "model.tflite"
    ) -> str:
        if not self.tflite_available:
            raise ImportError("TensorFlow not installed")
            
        import tensorflow as tf
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        if quantization == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int8]
        elif quantization == "float16":
            converter.target_spec.supported_types = [tf.float16]
            
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        logger.info(f"Exported to TFLite: {output_path}")
        return output_path
    
    def export_to_onnx_mobile(
        self,
        model,
        output_path: str = "model_mobilenet.onnx"
    ) -> str:
        """Export with mobile optimizations"""
        from opti_llm.core.exporter import ExportEngine, ExportFormat

        engine = ExportEngine()
        result = engine.export(
            model,
            ExportFormat.ONNX,
            model_name="model_mobilenet",
            input_shape=(1, 224, 224, 3)
        )
        
        if result.success:
            return result.output_path
        raise RuntimeError(f"Export failed: {result.error}")


class iOSExporter:
    """Export for iOS (CoreML)"""
    
    def __init__(self):
        self.coreml_available = self._check_coreml()
        
    def _check_coreml(self) -> bool:
        try:
            import coremltools
            return True
        except:
            return False
            
    def export_to_coreml(
        self,
        model,
        output_path: str = "model.mlmodel",
        description: str = "OptiLLM Optimized Model"
    ) -> str:
        if not self.coreml_available:
            logger.warning("coremltools not installed, using fallback")
            return self._export_fallback(model, output_path)
            
        import coremltools as ct
        
        coreml_model = ct.convert(model)
        coreml_model.save(output_path)
        
        logger.info(f"Exported to CoreML: {output_path}")
        return output_path
    
    def _export_fallback(self, model, output_path: str) -> str:
        """Fallback to ONNX"""
        from opti_llm.core.exporter import ExportEngine, ExportFormat

        engine = ExportEngine()
        result = engine.export(model, ExportFormat.ONNX, model_name="model_ios")
        
        if result.success:
            logger.info(f"CoreML fallback to ONNX: {result.output_path}")
            return result.output_path
        raise RuntimeError("Export failed")


class MobileOptimizer:
    """Full mobile optimization pipeline"""
    
    def __init__(self):
        self.android = AndroidExporter()
        self.ios = iOSExporter()
        
    def optimize_for_android(
        self,
        model,
        target: str = "tflite_int8",
        output_dir: str = "./mobile_android"
    ) -> Dict[str, str]:
        """Optimize for Android"""
        
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        if target.startswith("tflite"):
            quant = target.replace("tflite_", "")
            tflite_path = output / "model.tflite"
            outputs["tflite"] = self.android.export_to_tflite(
                model, 
                quant, 
                str(tflite_path)
            )
            
        onnx_path = output / "model_android.onnx"
        outputs["onnx"] = self.android.export_to_onnx_mobile(
            model,
            str(onnx_path)
        )
        
        logger.info(f"Android export complete: {outputs}")
        return outputs
    
    def optimize_for_ios(
        self,
        model,
        output_dir: str = "./mobile_ios"
    ) -> Dict[str, str]:
        """Optimize for iOS"""
        
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        try:
            mlmodel_path = output / "model.mlmodel"
            outputs["coreml"] = self.ios.export_to_coreml(
                model,
                str(mlmodel_path)
            )
        except Exception as e:
            logger.warning(f"CoreML failed: {e}")
            
        onnx_path = output / "model_ios.onnx"
        outputs["onnx"] = self.android.export_to_onnx_mobile(
            model,
            str(onnx_path)
        )
        
        logger.info(f"iOS export complete: {outputs}")
        return outputs


def quick_mobile_export(
    model,
    platform: str = "android",
    output_dir: str = "./mobile"
) -> Dict[str, str]:
    """Quick mobile export"""
    
    optimizer = MobileOptimizer()
    
    if platform.lower() == "android":
        return optimizer.optimize_for_android(model, output_dir=output_dir)
    else:
        return optimizer.optimize_for_ios(model, output_dir=output_dir)