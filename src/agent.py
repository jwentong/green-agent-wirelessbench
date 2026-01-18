# -*- coding: utf-8 -*-
"""
WCHW Green Agent - Wireless Communication Homework Benchmark Evaluator
UC Berkeley RDI Foundation AgentBeats Competition

This green agent evaluates purple agents on the WCHW (Wireless Communication Homework)
benchmark, supporting multiple answer types and providing comprehensive scoring.

Author: Jingwen Tong
Date: 2026-01-18
"""

import json
import re
import random
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


# ============================================================================
# Data Models
# ============================================================================

class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    participants: Dict[str, HttpUrl]  # role -> agent URL
    config: Dict[str, Any]


class WCHWProblem(BaseModel):
    """A single WCHW problem."""
    question: str
    answer: str
    cot: Optional[str] = None  # Chain of thought (optional)
    id: str


class EvaluationResult(BaseModel):
    """Result of evaluating a single problem."""
    problem_id: str
    question: str
    expected_answer: str
    agent_answer: str
    score: float
    answer_type: str
    details: Dict[str, Any] = {}


class BenchmarkSummary(BaseModel):
    """Summary of benchmark evaluation."""
    total_problems: int
    evaluated: int
    average_score: float
    max_score: float
    min_score: float
    by_category: Dict[str, float] = {}
    by_answer_type: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# WCHW Evaluator
# ============================================================================

class WCHWEvaluator:
    """
    Enhanced evaluator for WCHW benchmark with multi-type answer evaluation.
    
    Supports:
    1. Numeric with units (e.g., "16 kbit/s", "44.8 kHz")
    2. Pure numeric (e.g., "1.54", "4800")
    3. Mathematical formulas/expressions (e.g., "1/(2τ_0)", "A^2 T")
    4. Scientific notation (e.g., "5.42e-6", "2.2×10^-8")
    5. Text/descriptive answers
    6. LaTeX expressions
    """
    
    # Unit conversion factors to base units
    UNIT_MULTIPLIERS: Dict[str, float] = {
        # Frequency
        'ghz': 1e9, 'mhz': 1e6, 'khz': 1e3, 'hz': 1,
        # Data rate
        'gbps': 1e9, 'gbit/s': 1e9, 'gbits/s': 1e9,
        'mbps': 1e6, 'mbit/s': 1e6, 'mbits/s': 1e6,
        'kbps': 1e3, 'kbit/s': 1e3, 'kbits/s': 1e3,
        'bps': 1, 'bit/s': 1, 'bits/s': 1,
        'baud': 1, 'kbaud': 1e3, 'mbaud': 1e6,
        # Power
        'kw': 1e3, 'w': 1, 'mw': 1e-3, 'uw': 1e-6, 'μw': 1e-6,
        'dbm': 1, 'dbw': 1, 'db': 1,  # dB values kept as-is
        # Time
        's': 1, 'ms': 1e-3, 'us': 1e-6, 'μs': 1e-6, 'ns': 1e-9,
        # Distance
        'km': 1e3, 'm': 1, 'cm': 1e-2, 'mm': 1e-3,
        # Spectral efficiency
        'bit/(s·hz)': 1, 'bit/s/hz': 1, 'bps/hz': 1,
        # Angle
        'deg': 1, 'rad': 1,
    }
    
    # Patterns for formula detection
    FORMULA_INDICATORS = [
        r'\\tau', r'\\omega', r'\\pi', r'\\alpha', r'\\beta', r'\\phi',
        r'\\cos', r'\\sin', r'\\log', r'\\exp', r'\\sqrt',
        r'_\{', r'\^', r'\\frac', r'\\le', r'\\ge',
        r'\$', r'tau_0', r'T_s', r'f_', r'R_',
    ]
    
    def classify_answer_type(self, answer: str) -> str:
        """Classify the answer type to determine scoring strategy."""
        if answer is None:
            return 'unknown'
        
        answer_str = str(answer).strip()
        
        # Check for formula indicators
        for pattern in self.FORMULA_INDICATORS:
            if re.search(pattern, answer_str):
                numbers = re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', answer_str)
                if len(numbers) == 1 and len(answer_str) < 20:
                    return 'numeric'
                return 'formula'
        
        # Check for scientific notation
        if re.search(r'[-+]?\d+\.?\d*[eE][-+]?\d+', answer_str):
            return 'scientific'
        if re.search(r'[-+]?\d+\.?\d*\s*[×x\*]\s*10', answer_str):
            return 'scientific'
        
        # Check if primarily numeric
        numbers = re.findall(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?', answer_str)
        if numbers:
            total_num_chars = sum(len(n) for n in numbers)
            cleaned = answer_str.lower()
            for unit in self.UNIT_MULTIPLIERS.keys():
                cleaned = cleaned.replace(unit, '')
            cleaned = re.sub(r'[,\s\.\-\+]', '', cleaned)
            
            if len(cleaned) < 5 or total_num_chars > len(cleaned) * 0.3:
                return 'numeric'
        
        # Long text answer
        if len(answer_str) > 100 or answer_str.count(' ') > 20:
            return 'text'
        
        # Formula-like expressions
        if re.search(r'[a-zA-Z]+\s*=', answer_str) or '(' in answer_str:
            return 'formula'
        
        if numbers:
            return 'numeric'
        
        return 'text'
    
    def extract_unit(self, text: str) -> Tuple[Optional[str], float]:
        """Extract unit from text and return (unit_name, multiplier)."""
        text_lower = text.lower()
        unit_patterns = sorted(self.UNIT_MULTIPLIERS.keys(), key=len, reverse=True)
        
        for unit in unit_patterns:
            if re.search(rf'\b{re.escape(unit)}\b', text_lower):
                return unit, self.UNIT_MULTIPLIERS[unit]
        
        return None, 1.0
    
    def extract_number_with_unit(self, text: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract number from text considering units and scientific notation."""
        if text is None:
            return None, None
        
        text_str = str(text)
        
        # Unicode superscript mapping
        superscript_map = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
            '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
            '⁻': '-', '⁺': '+'
        }
        
        normalized_text = text_str
        for sup, normal in superscript_map.items():
            normalized_text = normalized_text.replace(sup, normal)
        
        # Scientific notation patterns
        sci_patterns = [
            r'([-+]?\d+\.?\d*)\s*[×x\*]\s*10\s*\^?\s*[{\(]?\s*([-+]?\d+)\s*[}\)]?',
            r'([-+]?\d+\.?\d*)[eE]([-+]?\d+)',
        ]
        
        for pattern in sci_patterns:
            matches = re.findall(pattern, normalized_text)
            if matches:
                mantissa_str, exp_str = matches[-1]
                try:
                    mantissa = float(mantissa_str)
                    exponent = int(exp_str)
                    value = mantissa * (10 ** exponent)
                    
                    full_match = re.search(pattern, normalized_text)
                    if full_match:
                        text_after = normalized_text[full_match.end():].strip()[:20]
                        unit, multiplier = self.extract_unit(text_after)
                        if unit and unit not in ['db', 'dbm', 'dbw']:
                            value = value * multiplier
                        return value, unit
                    return value, None
                except (ValueError, OverflowError):
                    continue
        
        # Regular number pattern
        number_pattern = r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?'
        matches = re.findall(number_pattern, text_str)
        
        if not matches:
            return None, None
        
        last_number_str = matches[-1].replace(",", "")
        
        try:
            value = float(last_number_str)
        except ValueError:
            return None, None
        
        last_match_pos = text_str.rfind(last_number_str)
        text_after_number = text_str[last_match_pos + len(last_number_str):].strip()[:20]
        
        unit, multiplier = self.extract_unit(text_after_number)
        if unit and unit not in ['db', 'dbm', 'dbw']:
            value = value * multiplier
        
        return value, unit
    
    def normalize_formula(self, formula: str) -> str:
        """Normalize a mathematical formula for comparison."""
        if not formula:
            return ""
        
        result = str(formula)
        result = re.sub(r'\$+', '', result)
        
        replacements = [
            (r'\\tau_0', 'tau0'), (r'\\tau', 'tau'),
            (r'\\omega', 'w'), (r'\\pi', 'pi'),
            (r'\\alpha', 'alpha'), (r'\\beta', 'beta'),
            (r'\\phi', 'phi'), (r'\\cos', 'cos'),
            (r'\\sin', 'sin'), (r'\\log', 'log'),
            (r'\\exp', 'exp'), (r'\\sqrt', 'sqrt'),
            (r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)'),
            (r'\\cdot', '*'), (r'\\times', '*'),
            (r'\\,', ''), (r'\\;', ''),
            (r'\\text\{[^}]*\}', ''), (r'\\mathrm\{[^}]*\}', ''),
        ]
        
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        
        result = re.sub(r'\s+', '', result).lower()
        result = result.replace('[', '(').replace(']', ')')
        result = result.replace('{', '(').replace('}', ')')
        
        return result
    
    def compare_formulas(self, expected: str, predicted: str) -> float:
        """Compare two formula strings after normalization."""
        norm_expected = self.normalize_formula(expected)
        norm_predicted = self.normalize_formula(predicted)
        
        if not norm_expected or not norm_predicted:
            return 0.0
        
        if norm_expected == norm_predicted:
            return 1.0
        
        if norm_expected in norm_predicted or norm_predicted in norm_expected:
            return 0.8
        
        set_expected = set(norm_expected)
        set_predicted = set(norm_predicted)
        intersection = len(set_expected & set_predicted)
        union = len(set_expected | set_predicted)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        expected_vars = set(re.findall(r'[a-z]+[0-9]*', norm_expected))
        predicted_vars = set(re.findall(r'[a-z]+[0-9]*', norm_predicted))
        var_match = len(expected_vars & predicted_vars) / max(len(expected_vars), 1)
        
        final_score = 0.5 * similarity + 0.5 * var_match
        
        if final_score > 0.7:
            return 0.8
        elif final_score > 0.5:
            return 0.5
        return 0.0
    
    def compare_text_answers(self, expected: str, predicted: str) -> float:
        """Compare text-based answers using keyword matching."""
        if not expected or not predicted:
            return 0.0
        
        expected_lower = expected.lower()
        predicted_lower = predicted.lower()
        
        if expected_lower.strip() == predicted_lower.strip():
            return 1.0
        
        def extract_key_terms(text):
            numbers = set(re.findall(r'[-+]?\d+\.?\d*', text))
            words = set(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower()))
            stopwords = {'the', 'is', 'are', 'and', 'or', 'for', 'to', 'of', 'in', 'at', 'with', 'that', 'this'}
            words = words - stopwords
            return numbers, words
        
        exp_nums, exp_words = extract_key_terms(expected)
        pred_nums, pred_words = extract_key_terms(predicted)
        
        num_match = len(exp_nums & pred_nums) / max(len(exp_nums), 1) if exp_nums else 0.5
        word_match = len(exp_words & pred_words) / max(len(exp_words), 1) if exp_words else 0.5
        
        score = 0.6 * num_match + 0.4 * word_match
        
        if score > 0.8:
            return 1.0
        elif score > 0.6:
            return 0.8
        elif score > 0.4:
            return 0.5
        return 0.0
    
    def calculate_numeric_score(self, expected: float, predicted: float) -> float:
        """Calculate score for numeric answers with relative tolerance."""
        if predicted is None or expected is None:
            return 0.0
        
        if abs(expected) < 1e-15:
            return 1.0 if abs(predicted) < 1e-15 else 0.0
        
        relative_error = abs(expected - predicted) / abs(expected)
        
        if relative_error < 0.01:
            return 1.0
        if relative_error < 0.05:
            return 0.9
        if relative_error < 0.10:
            return 0.7
        
        ratio = predicted / expected if expected != 0 else float('inf')
        
        # Off by factor of 1000
        if 0.0009 < ratio < 0.0011 or 900 < ratio < 1100:
            return 0.5
        # Off by factor of 1e6
        if 0.9e-6 < ratio < 1.1e-6 or 0.9e6 < ratio < 1.1e6:
            return 0.5
        # Off by factor of 2
        if 0.45 < ratio < 0.55 or 1.9 < ratio < 2.1:
            return 0.3
        
        return 0.0
    
    def evaluate(self, expected_answer: str, agent_answer: str) -> Tuple[float, str, Dict]:
        """
        Evaluate an agent's answer against the expected answer.
        
        Returns: (score, answer_type, details)
        """
        answer_type = self.classify_answer_type(expected_answer)
        details = {
            "answer_type": answer_type,
            "expected": expected_answer,
            "predicted": agent_answer,
        }
        
        if answer_type in ['numeric', 'scientific']:
            expected_val, exp_unit = self.extract_number_with_unit(expected_answer)
            predicted_val, pred_unit = self.extract_number_with_unit(agent_answer)
            
            details["expected_value"] = expected_val
            details["predicted_value"] = predicted_val
            details["expected_unit"] = exp_unit
            details["predicted_unit"] = pred_unit
            
            score = self.calculate_numeric_score(expected_val, predicted_val)
            
        elif answer_type == 'formula':
            # Try numeric first
            exp_val, _ = self.extract_number_with_unit(expected_answer)
            pred_val, _ = self.extract_number_with_unit(agent_answer)
            
            if exp_val is not None and pred_val is not None:
                numeric_score = self.calculate_numeric_score(exp_val, pred_val)
                if numeric_score >= 0.7:
                    details["method"] = "numeric_extraction"
                    return numeric_score, answer_type, details
            
            score = self.compare_formulas(expected_answer, agent_answer)
            details["method"] = "formula_comparison"
            
        elif answer_type == 'text':
            score = self.compare_text_answers(expected_answer, agent_answer)
            details["method"] = "text_comparison"
            
        else:
            # Unknown type, try numeric then text
            exp_val, _ = self.extract_number_with_unit(expected_answer)
            pred_val, _ = self.extract_number_with_unit(agent_answer)
            
            if exp_val is not None and pred_val is not None:
                score = self.calculate_numeric_score(exp_val, pred_val)
                details["method"] = "numeric_fallback"
            else:
                score = self.compare_text_answers(expected_answer, agent_answer)
                details["method"] = "text_fallback"
        
        return score, answer_type, details


# ============================================================================
# Agent Implementation
# ============================================================================

class Agent:
    """
    WCHW Green Agent for AgentBeats Competition.
    
    This agent:
    1. Loads WCHW test problems
    2. Sends problems to purple agents (participants)
    3. Evaluates responses using the WCHW evaluator
    4. Reports comprehensive scoring results
    """
    
    # Required participant roles
    required_roles: List[str] = ["wireless_solver"]
    
    # Required config keys
    required_config_keys: List[str] = ["num_problems"]
    
    def __init__(self):
        self.messenger = Messenger()
        self.evaluator = WCHWEvaluator()
        self.problems: List[WCHWProblem] = []
        self.data_dir = Path(__file__).parent.parent / "data"
        self._load_problems()
    
    def _load_problems(self):
        """Load WCHW test problems from the dataset file."""
        dataset_path = self.data_dir / "wchw_test.jsonl"
        
        if not dataset_path.exists():
            print(f"Warning: Dataset not found at {dataset_path}")
            return
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        self.problems.append(WCHWProblem(**data))
            print(f"Loaded {len(self.problems)} WCHW problems")
        except Exception as e:
            print(f"Error loading dataset: {e}")
    
    def validate_request(self, request: EvalRequest) -> Tuple[bool, str]:
        """Validate the incoming evaluation request."""
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        
        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"
        
        num_problems = request.config.get("num_problems", 0)
        if num_problems <= 0:
            return False, "num_problems must be positive"
        
        if num_problems > len(self.problems):
            return False, f"Requested {num_problems} problems but only {len(self.problems)} available"
        
        return True, "ok"
    
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """
        Main evaluation loop.
        
        1. Parse and validate the request
        2. Send problems to the purple agent
        3. Evaluate responses
        4. Report results
        """
        input_text = get_message_text(message)
        
        # Parse request
        try:
            request = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return
        
        # Get configuration
        num_problems = request.config.get("num_problems", 10)
        timeout = request.config.get("timeout", 60)
        purple_agent_url = str(request.participants["wireless_solver"])
        
        # Select problems
        selected_indices = request.config.get("problem_indices")
        if selected_indices is None:
            selected_indices = random.sample(range(len(self.problems)), min(num_problems, len(self.problems)))
        
        selected_problems = [self.problems[i] for i in selected_indices]
        
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting WCHW evaluation with {len(selected_problems)} problems...")
        )
        
        # Evaluate each problem
        results: List[EvaluationResult] = []
        total_score = 0.0
        scores_by_type: Dict[str, List[float]] = {}
        
        for idx, problem in enumerate(selected_problems):
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Evaluating problem {idx + 1}/{len(selected_problems)}: {problem.id}")
            )
            
            try:
                # Send question to purple agent
                agent_answer = await self.messenger.talk_to_agent(
                    message=problem.question,
                    url=purple_agent_url,
                    new_conversation=True,
                    timeout=timeout
                )
                
                # Evaluate the response
                score, answer_type, details = self.evaluator.evaluate(
                    expected_answer=problem.answer,
                    agent_answer=agent_answer
                )
                
                result = EvaluationResult(
                    problem_id=problem.id,
                    question=problem.question,
                    expected_answer=problem.answer,
                    agent_answer=agent_answer,
                    score=score,
                    answer_type=answer_type,
                    details=details
                )
                
            except Exception as e:
                # Handle errors gracefully
                result = EvaluationResult(
                    problem_id=problem.id,
                    question=problem.question,
                    expected_answer=problem.answer,
                    agent_answer=f"ERROR: {str(e)}",
                    score=0.0,
                    answer_type="error",
                    details={"error": str(e)}
                )
            
            results.append(result)
            total_score += result.score
            
            # Track scores by answer type
            if result.answer_type not in scores_by_type:
                scores_by_type[result.answer_type] = []
            scores_by_type[result.answer_type].append(result.score)
        
        # Compute summary statistics
        average_score = total_score / len(results) if results else 0.0
        scores = [r.score for r in results]
        
        by_answer_type = {}
        for atype, type_scores in scores_by_type.items():
            by_answer_type[atype] = {
                "count": len(type_scores),
                "average": sum(type_scores) / len(type_scores) if type_scores else 0,
                "max": max(type_scores) if type_scores else 0,
                "min": min(type_scores) if type_scores else 0,
            }
        
        summary = BenchmarkSummary(
            total_problems=len(self.problems),
            evaluated=len(results),
            average_score=average_score,
            max_score=max(scores) if scores else 0,
            min_score=min(scores) if scores else 0,
            by_answer_type=by_answer_type
        )
        
        # Prepare output
        result_text = f"""
## WCHW Benchmark Evaluation Complete

### Summary
- **Total Problems**: {summary.total_problems}
- **Evaluated**: {summary.evaluated}
- **Average Score**: {summary.average_score:.4f}
- **Max Score**: {summary.max_score:.4f}
- **Min Score**: {summary.min_score:.4f}

### By Answer Type
"""
        for atype, stats in by_answer_type.items():
            result_text += f"- **{atype}**: {stats['count']} problems, avg={stats['average']:.4f}\n"
        
        result_text += "\n### Detailed Results\n"
        for r in results[:10]:  # Show first 10 results
            result_text += f"- [{r.problem_id}] Score: {r.score:.2f} ({r.answer_type})\n"
        
        if len(results) > 10:
            result_text += f"\n... and {len(results) - 10} more problems\n"
        
        # Create artifact with results
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=result_text)),
                Part(root=DataPart(data={
                    "summary": summary.model_dump(),
                    "results": [r.model_dump() for r in results],
                    "benchmark": "WCHW",
                    "version": "1.0.0"
                }))
            ],
            name="WCHW Evaluation Results",
        )
