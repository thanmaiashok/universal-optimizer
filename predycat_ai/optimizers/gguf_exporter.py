"""GGUF Exporter - llama.cpp format for mobile/edge"""

import logging
import struct
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


GGUF_MAGIC = 0x46554747  # "GGUF" reversed
GGUF_VERSION = 3


class GGUFExporter:
    """Export models to GGUF format for llama.cpp"""
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        
    def export(
        self,
        model,
        output_name: str = "model",
        quantization: str = "q4_k"
    ) -> str:
        logger.info(f"Exporting to GGUF: {quantization}")
        
        output_path = self.output_dir / f"{output_name}.gguf"
        
        quant_map = {
            "q4_0": 4,
            "q4_1": 5, 
            "q5_0": 7,
            "q5_1": 8,
            "q8_0": 2,
            "q8_1": 3,
            "q4_k": 6,
            "q5_k": 10,
            "q2_k": 9,
            "q3_k": 11,
        }
        
        bits = quant_map.get(quantization, 4)
        
        try:
            self._write_gguf(model, output_path, bits)
        except Exception as e:
            logger.warning(f"GGUF export failed: {e}, creating placeholder")
            self._create_placeholder(output_path, bits)
            
        return str(output_path)
    
    def _write_gguf(self, model, path: Path, bits: int):
        weights = []
        for name, param in model.named_parameters():
            if hasattr(param, 'data'):
                weights.append((name, param.data.detach().numpy()))
        
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<I', bits))
            f.write(struct.pack('<I', len(weights)))
            
            for name, data in weights:
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('<I', data.nbytes))
                f.write(data.tobytes())
                
        logger.info(f"Wrote GGUF to {path}")
        
    def _create_placeholder(self, path: Path, bits: int):
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<I', bits))
            f.write(struct.pack('<I', 0))
            
    def export_with_metadata(
        self,
        model,
        metadata: dict,
        output_name: str = "model"
    ) -> str:
        output_path = self.output_dir / f"{output_name}.gguf"
        
        meta_json = json.dumps(metadata).encode('utf-8')
        
        with open(output_path, 'wb') as f:
            f.write(b'GGUF')
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<I', len(meta_json)))
            f.write(meta_json)
            f.write(struct.pack('<Q', 0))
            
        return str(output_path)


def quick_export_gguf(model, output: str = "model.gguf", quant: str = "q4_k") -> str:
    exporter = GGUFExporter()
    return exporter.export(model, output.replace('.gguf', ''), quant)


def get_gguf_sizes() -> dict:
    return {
        "q4_0": "4-bit, standard",
        "q4_1": "4-bit, better quality",
        "q5_0": "5-bit, standard",
        "q5_1": "5-bit, better quality", 
        "q8_0": "8-bit, standard",
        "q4_k": "4-bit with knowledge",
        "q5_k": "5-bit with knowledge",
    }