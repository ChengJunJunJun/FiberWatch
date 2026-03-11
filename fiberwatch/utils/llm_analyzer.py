"""
LLM-based risk warning analysis for OTDR detection results.

When no events are detected, uses Qwen to analyze potential risks:
- Signal fluctuations that may indicate dirty connectors
- Excludes fiber-end noise (normal phenomenon after last reflection peak)
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from ..config.llm_config import LLMConfig

if TYPE_CHECKING:
    from ..core.models import DetectionResult


SYSTEM_PROMPT = """\
你是一位OTDR（光时域反射仪）数据分析专家。
你的任务是对光纤链路进行隐患预警分析。
请严格按JSON格式回复，不要包含其他内容。"""

RISK_PROMPT_TEMPLATE = """\
算法未检测到明确事件，请根据以下光纤信号特征分析是否存在潜在隐患。

预警规则：
1. 判断光纤是否有异常信号波动（相邻分段间均值或标准差突变），如果有信号波动，\
则该波动之前最近的反射峰可能存在脏污问题。
2. 特别注意：最后一个反射峰之后的光纤末端信号（进入噪声区）是正常现象，不算隐患，\
不要对末端噪声区发出预警。

光纤总长: {fiber_length_km:.3f} km, 采样点数: {n_samples}

信号统计:
- 整体均值: {mean:.2f} dB, 标准差: {std:.3f} dB
- 最大值: {max_val:.2f} dB, 最小值: {min_val:.2f} dB
- 衰减斜率: {slope:.4f} dB/km

分段特征（将光纤等分为 {n_segments} 段）:
{segments_text}

反射峰（按位置排列）: {reflection_text}
（最后一个反射峰之后为光纤末端噪声区，不在分析范围内）

请严格按以下JSON格式回复：
```json
[
  {{
    "position_km": 0.5,
    "risk": "低/中/高",
    "description": "一句话描述隐患，例如：该位置前的反射峰可能存在脏污"
  }}
]
```
如果光纤状态良好无隐患，返回空数组 []。"""


def _build_risk_context(result: DetectionResult, n_segments: int = 10) -> dict:
    """Build signal statistics for risk analysis when no events detected."""
    trace = result.trace_db
    distance = result.distance_km
    n = len(trace)

    # Overall stats
    overall_mean = float(np.mean(trace))
    overall_std = float(np.std(trace))
    max_val = float(np.max(trace))
    min_val = float(np.min(trace))

    # Slope via linear fit
    coeffs = np.polyfit(distance, trace, 1)
    slope = float(coeffs[0])

    # Segment stats
    seg_size = n // n_segments
    seg_lines = []
    for i in range(n_segments):
        start = i * seg_size
        end = min(start + seg_size, n)
        seg = trace[start:end]
        d_start = float(distance[start])
        d_end = float(distance[min(end - 1, n - 1)])
        seg_lines.append(
            f"  [{d_start:.3f}~{d_end:.3f}km] "
            f"均值={np.mean(seg):.2f} std={np.std(seg):.3f} "
            f"范围={np.ptp(seg):.3f}dB"
        )

    # Reflection peaks
    refl_lines = []
    for peak in result.reflection_peaks:
        peak_z = float(distance[peak["index"]])
        height = peak.get("peak_height_db", float("nan"))
        refl_lines.append(f"  {peak_z:.4f}km 峰高={height:.1f}dB")

    return {
        "mean": overall_mean,
        "std": overall_std,
        "max_val": max_val,
        "min_val": min_val,
        "slope": slope,
        "n_segments": n_segments,
        "segments_text": "\n".join(seg_lines),
        "reflection_text": "\n".join(refl_lines) if refl_lines else "无",
    }


class LLMAnalyzer:
    """Risk warning analyzer using Qwen via OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        config.validate()
        self.config = config
        from openai import OpenAI

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def _chat(self, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content

    def analyze_risks(self, result: DetectionResult) -> list[dict]:
        """
        Analyze potential risks when no events are detected.

        Focuses on:
        - Signal fluctuations that may indicate dirty connectors
        - Ignores fiber-end noise after last reflection peak

        Returns list of risk dicts with keys: position_km, risk, description.
        """
        ctx = _build_risk_context(result)

        prompt = RISK_PROMPT_TEMPLATE.format(
            fiber_length_km=result.distance_km[-1],
            n_samples=len(result.trace_db),
            **ctx,
        )

        print("  正在调用千问分析潜在隐患...")
        response_text = self._chat(prompt)

        try:
            json_text = response_text
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]
            return json.loads(json_text.strip())
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  LLM 响应解析失败: {e}")
            return [{"position_km": -1, "risk": "N/A", "description": f"解析失败: {e}"}]
