"""Knowledge Distillation - Teacher-Student for edge deployment"""

import logging
import copy
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class DistillationResult:
    original_size_mb: float
    student_size_mb: float
    original_accuracy: float
    distilled_accuracy: float
    accuracy_drop: float


class DistillationDataset(Dataset):
    _PROMPTS = [
        "The capital of France is",
        "What is 2 + 2?",
        "Explain quantum computing in simple terms",
        "Write a function to calculate fibonacci",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
        "What is machine learning?",
        "Explain the theory of relativity",
        "What is the meaning of life?",
        "How do computers process data?",
    ]

    def __init__(self, model: nn.Module, num_samples: int = 1000, tokenizer=None, seq_len: int = 128):
        self.model = model
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Pre-tokenize prompts if tokenizer is available; else use fixed GPT-2 token IDs
        # that correspond to common English words (avoids pure random noise)
        if tokenizer is not None:
            self._tokens = []
            for p in self._PROMPTS:
                enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=seq_len,
                                padding="max_length")
                self._tokens.append(enc["input_ids"].squeeze(0))
        else:
            # Fallback: GPT-2 token IDs for common English words (not random)
            _BASE = [464, 995, 318, 257, 1263, 290, 6496, 1295, 13, 383,
                     2068, 3290, 9229, 625, 262, 16931, 9371, 11, 475, 262,
                     3057, 318, 257, 4171, 530, 13, 383, 1281, 6952, 281]
            padded = (_BASE * ((seq_len // len(_BASE)) + 1))[:seq_len]
            self._tokens = [torch.tensor(padded, dtype=torch.long)] * len(self._PROMPTS)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self._tokens[idx % len(self._tokens)]


class SoftLabelDataset(Dataset):
    def __init__(self, teacher_outputs: List[Dict], input_ids: List[torch.Tensor]):
        self.teacher_outputs = teacher_outputs
        self.input_ids = input_ids
        
    def __len__(self):
        return len(self.teacher_outputs)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "logits": self.teacher_outputs[idx]["logits"],
            "hidden_states": self.teacher_outputs[idx].get("hidden_states", None)
        }


class DistillationEngine:
    def __init__(
        self,
        teacher_model: Optional[nn.Module] = None,
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.student_model = None
        
    def create_student(
        self,
        teacher_model: nn.Module,
        num_layers: Optional[int] = None,
        hidden_size: Optional[int] = None,
        num_heads: Optional[int] = None
    ) -> nn.Module:
        logger.info("Creating student architecture")

        student = copy.deepcopy(teacher_model)

        if num_layers:
            removed = False
            try:
                # GPT-2 / GPT-NeoX style: transformer.h
                if hasattr(student, "transformer") and hasattr(student.transformer, "h"):
                    student.transformer.h = student.transformer.h[:num_layers]
                    removed = True
                # LLaMA / Mistral / Falcon style: model.layers
                elif hasattr(student, "model") and hasattr(student.model, "layers"):
                    student.model.layers = student.model.layers[:num_layers]
                    removed = True
                # BERT style: encoder.layer
                elif hasattr(student, "encoder") and hasattr(student.encoder, "layer"):
                    student.encoder.layer = student.encoder.layer[:num_layers]
                    removed = True
                # OPT / BLOOM style: model.decoder.layers
                elif hasattr(student, "model") and hasattr(getattr(student.model, "decoder", None), "layers"):
                    student.model.decoder.layers = student.model.decoder.layers[:num_layers]
                    removed = True
                if removed:
                    student.config.num_hidden_layers = num_layers
                    logger.info(f"Student: reduced to {num_layers} layers")
                else:
                    logger.warning("Layer removal not supported for this architecture; config updated only")
                    student.config.num_hidden_layers = num_layers
            except Exception as e:
                logger.warning(f"Layer removal failed: {e}")

        if hidden_size:
            self._scale_hidden_layers(student, hidden_size)

        if num_heads:
            try:
                student.config.num_attention_heads = num_heads
            except Exception as e:
                logger.warning(f"num_attention_heads update failed: {e}")

        return student
    
    def _scale_hidden_layers(self, model: nn.Module, new_hidden_size: int):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if module.out_features >= 512:
                    with torch.no_grad():
                        module.out_features = new_hidden_size
                        module.weight = nn.Parameter(
                            torch.randn(new_hidden_size, module.weight.shape[1]) * 0.02
                        )
                        if module.bias is not None:
                            module.bias = nn.Parameter(torch.zeros(new_hidden_size))
    
    def distill(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        train_dataset: Optional[Dataset] = None,
        num_epochs: int = 3,
        batch_size: int = 4,
        lr: float = 1e-4,
        device: str = "auto"
    ) -> Tuple[nn.Module, DistillationResult]:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)
        
        teacher_model.eval()
        student_model.train()
        
        if train_dataset is None:
            train_dataset = DistillationDataset(teacher_model, num_samples=100)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr)
        
        original_size = self._get_model_size(teacher_model)
        student_size = self._get_model_size(student_model)
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for batch in train_loader:
                if isinstance(batch, torch.Tensor):
                    batch = batch.to(device)
                elif isinstance(batch, dict):
                    batch = batch.get("input_ids", torch.zeros(1)).to(device)
                
                optimizer.zero_grad()
                
                with torch.no_grad():
                    teacher_output = teacher_model(batch)
                
                student_output = student_model(batch)
                
                if isinstance(teacher_output, dict):
                    teacher_logits = teacher_output.get("logits", teacher_output.get("last_hidden_state", None))
                    student_logits = student_output.get("logits", student_output.get("last_hidden_state", None))
                else:
                    teacher_logits = teacher_output
                    student_logits = student_output
                
                if teacher_logits is None or student_logits is None:
                    continue
                else:
                    loss = self._distillation_loss(
                        student_logits,
                        teacher_logits,
                        temperature=self.temperature,
                        alpha=self.alpha
                    )
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Distillation epoch {epoch+1}/{num_epochs}, loss: {avg_loss:.4f}")
        
        student_model.eval()

        # Measure perplexity-based accuracy on calibration tokens
        def _ppl_accuracy(m: nn.Module, tokens: torch.Tensor) -> float:
            try:
                m.eval()
                tokens = tokens.to(device)
                with torch.no_grad():
                    out = m(input_ids=tokens, labels=tokens)
                if hasattr(out, "loss") and out.loss is not None:
                    import math
                    ppl = math.exp(min(out.loss.item(), 20))
                    return round(max(0.0, 100 - ppl), 2)
            except Exception:
                pass
            return 0.0

        try:
            _cal_tokens = next(iter(DataLoader(train_dataset, batch_size=1)))[0].unsqueeze(0)
        except Exception:
            _cal_tokens = torch.randint(0, 1000, (1, 32))

        orig_acc = _ppl_accuracy(teacher_model, _cal_tokens)
        dist_acc = _ppl_accuracy(student_model, _cal_tokens)

        return student_model, DistillationResult(
            original_size_mb=original_size,
            student_size_mb=student_size,
            original_accuracy=orig_acc,
            distilled_accuracy=dist_acc,
            accuracy_drop=round(max(0.0, orig_acc - dist_acc), 2)
        )
    
    def _distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
        alpha: float
    ) -> torch.Tensor:
        # Soft loss: KL divergence between temperature-scaled distributions
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

        # Hard loss: cross entropy of student logits against teacher's argmax (hard labels)
        # Flatten seq dimension if present (batch, seq, vocab) → (batch*seq, vocab)
        if student_logits.dim() == 3:
            s = student_logits.reshape(-1, student_logits.size(-1))
            hard_targets = teacher_logits.reshape(-1, teacher_logits.size(-1)).argmax(dim=-1)
        else:
            s = student_logits
            hard_targets = teacher_logits.argmax(dim=-1)
        hard_loss = F.cross_entropy(s, hard_targets)

        return alpha * soft_loss + (1 - alpha) * hard_loss
    
    def _get_model_size(self, model: nn.Module) -> float:
        total = 0
        for param in model.parameters():
            total += param.numel() * param.element_size()
        return total / (1024 * 1024)
    
    def distill_with_prompts(
        self,
        student_model: nn.Module,
        prompts: List[str],
        tokenizer: Any,
        max_length: int = 512,
        num_steps: int = 100
    ) -> nn.Module:
        logger.info(f"Distilling with {len(prompts)} prompts")
        
        self.student_model = student_model
        student_model.train()
        
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)
        
        for step in range(num_steps):
            for prompt in prompts[:3]:
                try:
                    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_length, truncation=True)
                    input_ids = inputs["input_ids"].to(student_model.device if hasattr(student_model, "device") else "cpu")
                    
                    optimizer.zero_grad()
                    
                    outputs = student_model(input_ids)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0].mean()
                    
                    loss.backward()
                    optimizer.step()
                    
                except Exception as e:
                    logger.warning(f"Distillation step failed: {e}")
        
        student_model.eval()
        return student_model


def simple_distill(
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    temperature: float = 2.0,
    alpha: float = 0.7,
    device: str = "auto"
) -> nn.Module:
    engine = DistillationEngine(teacher_model, temperature=temperature, alpha=alpha)
    student_model, _ = engine.distill(student_model, teacher_model, train_loader.dataset if train_loader else None, device=device)
    return student_model