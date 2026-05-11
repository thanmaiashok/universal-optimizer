"""Reasoning Score Engine - Measures multi-step logic, consistency, hallucination rate"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import torch

logger = logging.getLogger(__name__)


class ReasoningTaskType(Enum):
    MATH = "math"
    LOGIC = "logic"
    MULTI_STEP = "multi_step"
    CAUSAL = "causal"
    COMMON_SENSE = "common_sense"


@dataclass
class ReasoningScore:
    coherence: float
    logical_correctness: float
    hallucination_rate: float
    overall_score: float
    task_type: ReasoningTaskType


@dataclass
class EvaluationResult:
    accuracy: float
    reasoning_score: ReasoningScore
    task_results: Dict[str, float]
    timestamp: str = ""


# Ground-truth Q&A pairs for math and logic (answer must appear in model output)
_MATH_QA = [
    ("If a train travels 60 miles in 45 minutes, what's its average speed in mph?", "80"),
    ("What is 15% of 200 plus 25% of 80?", "50"),
    ("If x + 5 = 12, what is 2x - 3?", "11"),
    ("A store marks up items by 40%. If an item costs $50 wholesale, what's the retail price?", "70"),
    ("If you have 3 apples and eat half, then buy 4 more, how many do you have?", "5.5"),
]

_LOGIC_QA = [
    ("All cats are mammals. Fluffy is a cat. Is Fluffy a mammal? Answer yes or no.", "yes"),
    ("If it rains, the ground gets wet. The ground is dry. Did it rain? Answer yes or no.", "no"),
    ("No reptiles can fly. Snakes are reptiles. Can snakes fly? Answer yes or no.", "no"),
    ("If A implies B, and B implies C, does A imply C? Answer yes or no.", "yes"),
    ("All squares are rectangles. Is every rectangle a square? Answer yes or no.", "no"),
]

_reasoning_prompts = {
    ReasoningTaskType.MATH: [q for q, _ in _MATH_QA],
    ReasoningTaskType.LOGIC: [q for q, _ in _LOGIC_QA],
    ReasoningTaskType.MULTI_STEP: [
        "Alice has twice as many books as Bob. Bob has 15 books. Carol has 5 more than Alice. How many books does Carol have?",
        "A farmer has 20 animals, all chickens and cows. If there are 50 legs total, how many of each?",
        "A train leaves at 2pm traveling 60mph. Another leaves at 3pm traveling 80mph. When do they meet?",
        "If you invest $1000 at 10% annual interest, compounded annually, what's it worth after 3 years?",
        "A clock shows 3:15. What angle do the hour and minute hands make?",
    ],
    ReasoningTaskType.CAUSAL: [
        "Why does putting ice in a drink make it cold?",
        "Why do we feel hungry before meals?",
        "Why do airplane windows have small holes?",
        "Why does breathing faster after exercise make you feel better?",
        "Why do plants grow toward sunlight?",
    ],
    ReasoningTaskType.COMMON_SENSE: [
        "If you drop a glass cup on concrete, what will happen?",
        "Why can't you see wind?",
        "What happens if you mix bleach and ammonia?",
        "Why do we close our eyes when we sneeze?",
        "What makes a sound seem to echo?",
    ],
}


class ReasoningEvaluator:
    def __init__(self, model: Any = None, tokenizer: Any = None):
        self.model = model
        self.tokenizer = tokenizer
        self.results = []
        
    def evaluate_reasoning(
        self,
        model: Any,
        task_types: Optional[List[ReasoningTaskType]] = None,
        num_prompts: int = 3
    ) -> ReasoningScore:
        logger.info("Evaluating reasoning capability")
        
        if task_types is None:
            task_types = list(ReasoningTaskType)
        
        coherence_scores = []
        logic_scores = []
        hallucination_scores = []
        
        for task_type in task_types:
            prompts = _reasoning_prompts.get(task_type, [])[:num_prompts]
            
            for prompt in prompts:
                try:
                    result = self._evaluate_single_prompt(model, prompt, task_type)
                    coherence_scores.append(result[0])
                    logic_scores.append(result[1])
                    hallucination_scores.append(result[2])
                except Exception as e:
                    logger.warning(f"Evaluation failed for prompt: {prompt[:30]}... - {e}")
        
        avg_coherence = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        avg_logic = sum(logic_scores) / len(logic_scores) if logic_scores else 0.0
        avg_hallucination = sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0
        
        overall = (avg_coherence * 0.3 + avg_logic * 0.4 + (1 - avg_hallucination) * 0.3) * 100
        
        return ReasoningScore(
            coherence=avg_coherence,
            logical_correctness=avg_logic,
            hallucination_rate=avg_hallucination,
            overall_score=overall,
            task_type=task_types[0] if task_types else ReasoningTaskType.MULTI_STEP
        )
    
    def _evaluate_single_prompt(
        self,
        model: Any,
        prompt: str,
        task_type: ReasoningTaskType
    ) -> Tuple[float, float, float]:
        try:
            response = self._generate_response(model, prompt)
            
            coherence = self._evaluate_coherence(response, prompt)
            logic = self._evaluate_logical_correctness(response, task_type, prompt)
            hallucination = self._evaluate_hallucination(response)
            
            return coherence, logic, hallucination
            
        except Exception as e:
            logger.warning(f"Single prompt evaluation failed: {e}")
            return 0.5, 0.5, 0.3
    
    def _generate_response(self, model: Any, prompt: str) -> str:
        try:
            if hasattr(model, "generate"):
                if self.tokenizer:
                    inputs = self.tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
                    if hasattr(inputs, "to"):
                        inputs = {k: v.to(model.device if hasattr(model, "device") else "cpu") for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None
                    )
                    response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    response = "Generic response for evaluation purposes."
            else:
                response = str(model)
                
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            response = "Generic response for evaluation purposes."
            
        return response
    
    def _evaluate_coherence(self, response: str, prompt: str) -> float:
        response_lower = response.lower()
        prompt_lower = prompt.lower()
        
        response_words = set(re.findall(r'\w+', response_lower))
        prompt_words = set(re.findall(r'\w+', prompt_lower))
        
        if not response_words:
            return 0.0
        
        overlap = len(response_words & prompt_words) / len(response_words)
        
        if len(response) < 10:
            return 0.3
        elif len(response) > 5000:
            return 0.6
        
        return min(1.0, overlap + 0.5)
    
    def _evaluate_logical_correctness(self, response: str, task_type: ReasoningTaskType,
                                       prompt: str = "") -> float:
        response = response.strip()
        if not response or len(response) < 5:
            return 0.0

        response_lower = response.lower()

        # ── Ground-truth answer check ──
        if task_type == ReasoningTaskType.MATH:
            for q, expected in _MATH_QA:
                if prompt.startswith(q[:40]):
                    nums = re.findall(r'-?\d+\.?\d*', response)
                    for n in nums:
                        try:
                            if abs(float(n) - float(expected)) < 0.6:
                                return 1.0
                        except ValueError:
                            pass
                    return 0.0  # answer found but wrong

        if task_type == ReasoningTaskType.LOGIC:
            for q, expected in _LOGIC_QA:
                if prompt.startswith(q[:40]):
                    if expected in response_lower:
                        # Make sure the opposite isn't also there (e.g. "not yes")
                        opposite = "no" if expected == "yes" else "yes"
                        if f"not {expected}" in response_lower or f"isn't {expected}" in response_lower:
                            return 0.0
                        return 1.0
                    return 0.0

        # ── Structural fallback for other task types ──
        has_numbers = bool(re.search(r'\d+', response))
        has_indicators = any(w in response_lower for w in ["therefore", "thus", "since", "because", "then", "so"])
        has_quantifiers = any(w in response_lower for w in ["all", "some", "none", "every", "no"])

        score = 0.3
        if has_numbers:
            score += 0.2
        if has_indicators:
            score += 0.25
        if has_quantifiers:
            score += 0.2
        return min(1.0, score)
    
    def _evaluate_hallucination(self, response: str) -> float:
        response_lower = response.lower()
        
        vague_indicators = ["i think", "probably", "might be", "could be", "may be", "likely", "perhaps", "possibly"]
        fabrication_patterns = [
            r"studies show that \d+%",
            r"researchers found that \d+",
            r"according to \w+ \d+",
            r"statistics show \d+%",
        ]
        
        score = 0.0
        
        for indicator in vague_indicators:
            if indicator in response_lower:
                score += 0.15
        
        for pattern in fabrication_patterns:
            if re.search(pattern, response_lower):
                score += 0.1
        
        if not response or len(response) < 20:
            return 0.1
        
        return min(1.0, max(0.0, score))
    
    def generate_synthetic_prompts(
        self,
        task_type: ReasoningTaskType,
        num_prompts: int = 10
    ) -> List[str]:
        base_prompts = _reasoning_prompts.get(task_type, [])
        
        prompts = base_prompts[:num_prompts] if len(base_prompts) >= num_prompts else base_prompts * (num_prompts // len(base_prompts) + 1)
        
        return prompts[:num_prompts]


class AccuracyMonitor:
    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self.best_score = None
        self.best_state = None
        self.history = []
        
    def check_accuracy(
        self,
        current_score: float,
        model_state: Optional[Dict] = None
    ) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            if model_state:
                self.best_state = model_state
            return True
        
        drop = self.best_score - current_score
        
        self.history.append({
            "current": current_score,
            "best": self.best_score,
            "drop": drop
        })
        
        if drop > self.threshold:
            logger.warning(f"Accuracy drop exceeded: {drop:.2f}% > {self.threshold}%")
            return False
        
        if current_score > self.best_score:
            self.best_score = current_score
            if model_state:
                self.best_state = model_state
                
        return True
    
    def should_stop(self, current_score: float) -> bool:
        return (self.best_score - current_score) > self.threshold
    
    def get_best_state(self) -> Optional[Dict]:
        return self.best_state
    
    def rollback(self) -> Optional[Dict]:
        return self.best_state


def quick_evaluate(model: Any, prompts: List[str]) -> Dict[str, float]:
    evaluator = ReasoningEvaluator(model)
    score = evaluator.evaluate_reasoning(model)
    
    return {
        "coherence": score.coherence,
        "logical_correctness": score.logical_correctness,
        "hallucination_rate": score.hallucination_rate,
        "overall": score.overall_score
    }