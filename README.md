<p align="center"><img src="docs/flow-3.svg" alt="Animated OptiLLM pipeline: Load → Profile → Strategy → Compress → Evaluate → Export" width="100%"/></p>

<p align="center"><sub>10-second tour: Load → Profile → Strategy → Compress → Evaluate → Export</sub></p>

<p align="center"><img src="docs/px3/intro.svg" width="100%" alt="Production-grade AI model optimization for any neural network. Compress, quantize, prune and deploy LLMs, Transformers and CNNs to laptop, cloud, mobile or edge in one command."/></p>

<p align="center"><img src="docs/px3/features.svg" width="100%" alt="Key features"/></p>

<a id="what-it-does"></a>
<h2><img src="docs/px3/h2-what-it-does.svg" width="100%" alt="What it does"/></h2>

<p align="center"><img src="docs/px3/t-01.svg" width="100%" alt="OptiLLM takes any neural network and automatically selects the best optimization pipeline for your target hardware - no manual tuning required. Optimization | What it does Quantization | FP16 / INT8 / INT4 / INT3 - shrinks model weight precision Pruning | Sensitivity-based removal of low-impact weights Distillation | Teacher-student compression for edge/mobile Strategy Selector | Auto-picks the right pipeline for your target Reasoning Evaluator | Multi-step logic scoring to guard accuracy Hardware Runtime | TensorRT, ONNX Runtime, GGUF export"/></p>

<a id="quick-start"></a>
<h2><img src="docs/px3/h2-quick-start.svg" width="100%" alt="Quick start"/></h2>

<p align="center"><img src="docs/px3/bar-bash.svg" width="100%" alt="bash code"/></p>

```bash
# Clone
git clone https://github.com/thanmaiashok/universal-optimizer.git
cd universal-optimizer

# Install (Python 3.9+)
pip install -e .

# Full install (GPU + mobile)
pip install -e ".[full]"

# Optimize a model
optillm optimize gpt2 --target mobile --preset balanced

# Profile hardware constraints
optillm profile ./model.pt

# Evaluate reasoning accuracy
optillm evaluate gpt2
```

<a id="python-api"></a>
<h2><img src="docs/px3/h2-python-api.svg" width="100%" alt="Python API"/></h2>

<p align="center"><img src="docs/px3/bar-python.svg" width="100%" alt="python code"/></p>

```python
from opti_llm.core.optimizer import OptiLLMOptimizer, OptimizationConfig

config = OptimizationConfig(
    target_platform="mobile",   # laptop | cloud | mobile | edge
    preset="balanced",          # balanced | ultra_compression | max_accuracy | mobile_safe
    max_accuracy_drop=2.0       # % accuracy loss allowed
)

optimizer = OptiLLMOptimizer(config=config)
result = optimizer.optimize(model)

print(f"Compression:   {result.compression_ratio:.1f}x")
print(f"Latency gain:  {result.latency_improvement:.1f}x")
print(f"Accuracy drop: {result.accuracy_drop:.2f}%")
```

<a id="presets"></a>
<h2><img src="docs/px3/h2-presets.svg" width="100%" alt="Presets"/></h2>

<p align="center"><img src="docs/px3/t-02.svg" width="100%" alt="Preset | Compression | Accuracy kept | Best for balanced | 2-4x | ~98% | General use ultra_compression | 4-10x | ~95% | Mobile / Edge max_accuracy | 1.5-2x | ~99% | Cloud / Production mobile_safe | 2-3x | ~97% | Mobile deployments"/></p>

<a id="target-platforms"></a>
<h2><img src="docs/px3/h2-target-platforms.svg" width="100%" alt="Target platforms"/></h2>

<p align="center"><img src="docs/px3/t-03.svg" width="100%" alt="Platform | Batch size | RAM limit | Default preset laptop | 8 | 8 GB | balanced cloud | 16 | 16 GB | max_accuracy mobile | 1 | 2 GB | balanced edge | 1 | 1 GB | ultra_compression"/></p>

<a id="rest-api-server"></a>
<h2><img src="docs/px3/h2-rest-api-server.svg" width="100%" alt="REST API server"/></h2>

<p align="center"><img src="docs/px3/bar-bash.svg" width="100%" alt="bash code"/></p>

```bash
cd opti_llm/api
python main.py
# → http://localhost:8080
```

<a id="project-structure"></a>
<h2><img src="docs/px3/h2-project-structure.svg" width="100%" alt="Project structure"/></h2>

<p align="center"><img src="docs/px3/bar-code.svg" width="100%" alt="code code"/></p>

```
opti_llm/
├── core/
│   ├── optimizer.py      # Main orchestrator (OptiLLMOptimizer)
│   ├── loader.py         # Universal model loader (HuggingFace, PyTorch, ONNX)
│   ├── profiler.py       # Multi-environment hardware profiler
│   ├── quantizer.py      # FP16/INT8/INT4/INT3 quantization engine
│   ├── pruner.py         # Sensitivity-based adaptive pruner
│   ├── distiller.py      # Knowledge distillation (teacher-student)
│   ├── evaluator.py      # Reasoning score + accuracy monitor
│   ├── strategy.py       # Auto pipeline selector
│   └── exporter.py       # PT / ONNX / GGUF export
├── optimizers/
│   ├── advanced_quantizer.py
│   ├── gguf_exporter.py
│   ├── tensorrt_runtime.py
│   └── thermal_manager.py
├── api/main.py           # FastAPI REST server
├── cli/main.py           # CLI entry point
└── utils/
    ├── batch_processor.py
    ├── mobile_export.py
    └── model_cards.py
examples/
└── run_examples.py
```

<a id="performance-targets"></a>
<h2><img src="docs/px3/h2-performance-targets.svg" width="100%" alt="Performance targets"/></h2>

<p align="center"><img src="docs/px3/t-04.svg" width="100%" alt="Compression: 3x-10x size reduction Latency: 2x-5x faster inference Mobile size: 100-300 MB RAM: &lt; 2 GB"/></p>

<a id="requirements"></a>
<h2><img src="docs/px3/h2-requirements.svg" width="100%" alt="Requirements"/></h2>

<p align="center"><img src="docs/px3/t-05.svg" width="100%" alt="Python 3.9+ PyTorch 2.0+ (Optional) CUDA for GPU quantization"/></p>

<a id="contributing"></a>
<h2><img src="docs/px3/h2-contributing.svg" width="100%" alt="Contributing"/></h2>

<p align="center"><img src="docs/px3/t-06.svg" width="100%" alt="Pull requests welcome. Open an issue first for large changes. Fork the repo Create a feature branch: git checkout -b feat/your-feature Commit: git commit -m &quot;feat: add your feature&quot; Push and open a PR"/></p>

<a id="license"></a>
<h2><img src="docs/px3/h2-license.svg" width="100%" alt="License"/></h2>

<p align="center"><img src="docs/px3/t-07.svg" width="100%" alt="MIT © OptiLLM contributors"/></p>

<p align="center"><a href="https://github.com/thanmaiashok"><img src="docs/px3/footer.svg" width="100%" alt="Built by Thanmai A, founder of FoxynAI"/></a></p>
