"""
OTDR visualization with LLM risk warning analysis.

CNN detection is handled by visualize.py; LLM only provides risk warnings
when no events are detected by the algorithm.

Usage:
    python -m fiberwatch.scripts.visualize_with_llm \
        --input_file data/test.txt \
        --cnn-file data/cnn/bend/output/orgChip1_after_cnn.txt

Environment:
    DASHSCOPE_API_KEY: Your Qwen/DashScope API key
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fiberwatch.config.llm_config import LLMConfig, load_llm_config
from fiberwatch.scripts.visualize import run_visualization
from fiberwatch.utils.llm_analyzer import LLMAnalyzer


def run_with_llm(
    input_file: Path,
    baseline_file: Path | None = None,
    cnn_file: Path | None = None,
    output_dir: Path = Path("output"),
    sample_spacing_km: float = 0.0025545,
    save_plots: bool = True,
    llm_config: LLMConfig | None = None,
):
    """Run standard visualization, then LLM risk warning if no events detected."""
    # Phase 1: detection pipeline (includes CNN local bend detection if cnn_file provided)
    vis_result = run_visualization(
        input_file=input_file,
        baseline_file=baseline_file,
        cnn_file=cnn_file,
        output_dir=output_dir,
        sample_spacing_km=sample_spacing_km,
        save_plots=save_plots,
    )

    detection_result = vis_result["detection_result"]
    clustered_events = vis_result["clustered_events"]

    # Phase 2: LLM risk warning (only when no events detected)
    if clustered_events:
        print("\n已检测到事件，跳过LLM预警分析。")
        return vis_result

    if llm_config is None:
        llm_config = LLMConfig()

    try:
        llm_config.validate()
    except ValueError as e:
        print(f"\nLLM 配置错误: {e}")
        return vis_result

    analyzer = LLMAnalyzer(llm_config)

    print("\n--- LLM 预警分析 (Qwen) ---")
    risks = analyzer.analyze_risks(detection_result)

    if not risks:
        print("\n光纤状态良好，未发现潜在隐患。")
    else:
        print(f"\n{'位置(km)':<12} {'风险':<6} {'描述'}")
        print("-" * 60)
        for r in risks:
            pos = r.get("position_km", -1)
            pos_str = f"{pos:.4f}" if isinstance(pos, (int, float)) and pos >= 0 else "整体"
            print(f"{pos_str:<12} {r.get('risk', '?'):<6} {r.get('description', '')}")

    if save_plots:
        _save_risk_report(input_file, output_dir, risks)

    return {**vis_result, "risks": risks}


def _save_risk_report(input_file: Path, output_dir: Path, risks: list) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / f"{input_file.stem}_llm_risks.txt"
    lines = [f"{'位置(km)':<12} {'风险':<6} {'描述'}"]
    lines.append("-" * 60)
    for r in risks:
        pos = r.get("position_km", -1)
        pos_str = f"{pos:.4f}" if isinstance(pos, (int, float)) and pos >= 0 else "整体"
        lines.append(f"{pos_str:<12} {r.get('risk', '?'):<6} {r.get('description', '')}")
    report_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n预警报告已保存: {report_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="FiberWatch OTDR + LLM Risk Warning")
    parser.add_argument("--input_file", type=Path, required=True, help="Input OTDR data file")
    parser.add_argument("--baseline", type=Path, help="Baseline reference file")
    parser.add_argument("--cnn-file", type=Path, help="CNN-denoised trace file for bend refinement")
    parser.add_argument("--output", "-o", type=Path, default="output", help="Output directory")
    parser.add_argument("--sample-spacing", type=float, default=0.0025545, help="Sample spacing in km")
    parser.add_argument("--no-save", action="store_true", help="Don't save plots/reports")
    parser.add_argument("--llm-model", type=str, default=None, help="Model name (default: qwen-max)")

    args = parser.parse_args()

    llm_config = load_llm_config(model=args.llm_model)

    run_with_llm(
        input_file=args.input_file,
        baseline_file=args.baseline,
        cnn_file=args.cnn_file,
        output_dir=args.output,
        sample_spacing_km=args.sample_spacing,
        save_plots=not args.no_save,
        llm_config=llm_config,
    )


if __name__ == "__main__":
    main()
