"""
UI components for the Streamlit web interface.

This module provides reusable UI components for the web interface,
including header, sidebar, results display, and download sections.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

from ..utils.visualization import (
    create_streamlit_analysis_figure,
    add_event_markers,
    fig_to_bytes,
)


def render_header():
    """Render the application header with title and description."""
    st.markdown(
        '<div class="main-title">🔬 FiberWatch OTDR 智能分析</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">专业的光纤时域反射测量分析工具 • 上传数据文件开始分析</div>',
        unsafe_allow_html=True,
    )


def render_sidebar(config) -> Dict[str, Any]:
    """
    Render the sidebar with data input and parameter controls.

    Args:
        config: Application configuration

    Returns:
        Dictionary of parameters from the sidebar
    """
    with st.sidebar:
        st.markdown("### 📁 数据输入")

        # File upload section
        with st.container():
            test_upload = st.file_uploader(
                "上传测试曲线",
                type=config.web.allowed_extensions,
                help="单列dB值的文本文件",
            )

            baseline_upload = st.file_uploader(
                "上传基线 (可选)",
                type=config.web.allowed_extensions,
                help="参考基线文件，格式与测试曲线相同",
            )

        st.markdown("### ⚙️ 基本参数")

        # Basic parameters
        col1, col2 = st.columns(2)
        with col1:
            total_distance_km = st.number_input(
                "总长度 (km)",
                min_value=0.1,
                max_value=200.0,
                value=config.web.default_distance_km,
                step=0.1,
                help="光纤总长度",
            )

        with col2:
            series_name = st.text_input(
                "文件前缀",
                value=config.web.default_series_name,
                help="保存文件的名称前缀",
            )

        st.markdown("### 💾 输出设置")

        # Output settings
        save_images = st.toggle(
            "保存图片到本地",
            value=False,
            help="是否将分析图片保存到指定目录",
        )

        if save_images:
            output_dir = st.text_input(
                "输出目录",
                value=config.web.default_output_dir,
                help="图片保存的目录路径",
            )
        else:
            output_dir = config.web.default_output_dir

        # Run button
        run_analysis = st.button("运行检测", type="primary", width=True)

    return {
        "test_data": test_upload,
        "baseline_data": baseline_upload,
        "total_distance_km": total_distance_km,
        "series_name": series_name,
        "save_images": save_images,
        "output_dir": output_dir,
        "run_analysis": run_analysis,
    }


def render_analysis_results(results: Dict[str, Any], params: Dict[str, Any]):
    """
    Render the analysis results section.

    Args:
        results: Analysis results dictionary
        params: Parameters used for analysis
    """
    st.markdown("## 📊 分析结果")

    # Render statistics cards
    _render_statistics_cards(results, params)

    st.markdown("---")

    # Create and render plots
    _render_analysis_plots(results, params)


def _render_statistics_cards(results: Dict[str, Any], params: Dict[str, Any]):
    """Render statistics cards showing key metrics."""
    event_stats = results["event_stats"]
    clustered_events = results["clustered_events"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="color: #2563eb; margin: 0; font-size: 0.9rem;">📏 Total Length</h4>
                <h3 style="margin: 0.3rem 0; font-size: 1.3rem;">{params["total_distance_km"]:.1f} km</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="color: #dc2626; margin: 0; font-size: 0.9rem;">⚠️ Events Detected</h4>
                <h3 style="margin: 0.3rem 0; font-size: 1.3rem;">{event_stats["total_events"]}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        break_count = event_stats["event_types"].get("break", 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="color: #ef4444; margin: 0; font-size: 0.9rem;">💥 Breaks</h4>
                <h3 style="margin: 0.3rem 0; font-size: 1.3rem;">{break_count}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        connector_count = event_stats["event_types"].get(
            "dirty_connector", 0
        ) + event_stats["event_types"].get("clean_connector", 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <h4 style="color: #f59e0b; margin: 0; font-size: 0.9rem;">🔗 Connectors</h4>
                <h3 style="margin: 0.3rem 0; font-size: 1.3rem;">{connector_count}</h3>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_analysis_plots(results: Dict[str, Any], params: Dict[str, Any]):
    """Render analysis plots in tabs."""
    detection_result = results["detection_result"]
    clustered_events = results["clustered_events"]
    distance_km = results["distance_km"]
    test_data = results["test_data"]
    baseline_provided = results["baseline_provided"]

    # Create all figures
    plt.style.use("default")

    # 1. Baseline figure
    fig_baseline, ax_baseline = plt.subplots(figsize=(10, 4))
    ax_baseline.plot(
        distance_km * 1000,
        detection_result.baseline_db,
        color="#dc2626",
        lw=2,
        label="Baseline",
    )
    ax_baseline.set_xlabel("Distance (m)", fontsize=11)
    ax_baseline.set_ylabel("Return Power (dB)", fontsize=11)
    ax_baseline.set_title("Baseline Analysis", fontsize=13, fontweight="bold")
    ax_baseline.grid(True, alpha=0.3)
    ax_baseline.legend()
    ax_baseline.set_facecolor("#fafafa")

    # 2. OTDR trace figure
    fig_otdr, ax_otdr = plt.subplots(figsize=(10, 4))
    ax_otdr.plot(
        distance_km * 1000,
        test_data,
        color="#94a3b8",
        lw=1,
        alpha=0.7,
        label="Raw Data",
    )
    ax_otdr.plot(
        distance_km * 1000,
        detection_result.trace_smooth_db,
        color="#2563eb",
        lw=2,
        label="Smoothed Data",
    )
    add_event_markers(ax_otdr, clustered_events)
    ax_otdr.set_xlabel("Distance (m)", fontsize=11)
    ax_otdr.set_ylabel("Return Power (dB)", fontsize=11)
    ax_otdr.set_title("OTDR Measurement Trace", fontsize=13, fontweight="bold")
    ax_otdr.grid(True, alpha=0.3)
    ax_otdr.legend()
    ax_otdr.set_facecolor("#fafafa")

    # 3. Residual figure
    fig_residual, ax_residual = plt.subplots(figsize=(10, 4))
    ax_residual.plot(
        distance_km * 1000,
        detection_result.residual_db,
        color="#059669",
        lw=2,
        label="Residual (Measurement - Baseline)",
    )
    add_event_markers(ax_residual, clustered_events)
    ax_residual.axhline(0, color="black", ls="-", alpha=0.3)
    ax_residual.set_xlabel("Distance (m)", fontsize=11)
    ax_residual.set_ylabel("Difference (dB)", fontsize=11)
    ax_residual.set_title("Residual Analysis", fontsize=13, fontweight="bold")
    ax_residual.grid(True, alpha=0.3)
    ax_residual.legend()
    ax_residual.set_facecolor("#fafafa")

    # 4. Comprehensive analysis figure
    fig_analysis = create_streamlit_analysis_figure(
        detection_result,
        clustered_events,
        params["series_name"],
        reference_provided=baseline_provided,
    )

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Analysis", "📏 Baseline", "📊 OTDR Trace", "📉 Residual", "📋 Events"]
    )

    with tab1:
        st.markdown("#### 🔬 Comprehensive Analysis")
        st.markdown(
            "Multi-dimensional analysis: data comparison, residual analysis, derivative changes"
        )
        st.pyplot(fig_analysis, clear_figure=False)

    with tab2:
        st.markdown("#### 📏 Baseline Reference")
        st.markdown("Reference baseline curve for comparison")
        st.pyplot(fig_baseline, clear_figure=False)

    with tab3:
        st.markdown("#### 📊 OTDR Measurement")
        st.markdown("Raw and smoothed measurement data with detected events marked")
        st.pyplot(fig_otdr, clear_figure=False)

    with tab4:
        st.markdown("#### 📉 Residual Analysis")
        st.markdown(
            "Difference analysis between measurement and baseline for anomaly detection"
        )
        st.pyplot(fig_residual, clear_figure=False)

    with tab5:
        _render_events_table(clustered_events)

    # Store figures in session state for download
    st.session_state.figures = {
        "analysis": fig_analysis,
        "baseline": fig_baseline,
        "otdr": fig_otdr,
        "residual": fig_residual,
    }


def _render_events_table(clustered_events):
    """Render the events detail table."""
    st.markdown("#### 📋 Event Details")

    if clustered_events:
        # Create event dataframe
        df = pd.DataFrame(
            [
                {
                    "Event Type": ev.kind,
                    "Position (km)": round(ev.z_km, 4),
                    "Position (m)": round(ev.z_km * 1000, 1),
                    "Loss (dB)": round(ev.magnitude_db, 3),
                    "Reflection (dB)": round(ev.reflect_db, 3),
                }
                for ev in clustered_events
            ]
        )

        # Event type statistics
        event_counts = df["Event Type"].value_counts()
        st.markdown("##### 📊 Event Statistics")

        stats_cols = st.columns(len(event_counts))
        for i, (event_type, count) in enumerate(event_counts.items()):
            with stats_cols[i]:
                st.metric(f"{event_type}", f"{count}")

        st.markdown("##### 📝 Detailed List")
        st.dataframe(
            df,
            width=True,
            height=400,
            column_config={
                "Event Type": st.column_config.TextColumn(
                    "Event Type",
                    help="Type of detected event",
                    width="small",
                ),
                "Position (km)": st.column_config.NumberColumn(
                    "Position (km)",
                    help="Event position in fiber (kilometers)",
                    format="%.3f",
                ),
                "Position (m)": st.column_config.NumberColumn(
                    "Position (m)",
                    help="Event position in fiber (meters)",
                    format="%.1f",
                ),
                "Loss (dB)": st.column_config.NumberColumn(
                    "Loss (dB)",
                    help="Magnitude of loss caused by event",
                    format="%.3f",
                ),
                "Reflection (dB)": st.column_config.NumberColumn(
                    "Reflection (dB)",
                    help="Reflection intensity of event",
                    format="%.3f",
                ),
            },
        )
    else:
        st.info("🎉 No anomalous events detected - fiber is in good condition!")


def render_download_section(results: Dict[str, Any], params: Dict[str, Any]):
    """
    Render the download section with export options.

    Args:
        results: Analysis results dictionary
        params: Parameters used for analysis
    """
    st.markdown("---")
    st.markdown("## 💾 Export Functions")

    # Save images locally if requested
    if params["save_images"]:
        _save_images_locally(results, params)

    # Download buttons section
    st.markdown("### 📥 Download Images")
    _render_download_buttons(params)

    # Download data table
    _render_data_download(results, params)


def _save_images_locally(results: Dict[str, Any], params: Dict[str, Any]):
    """Save images to local directory."""
    from pathlib import Path

    with st.spinner("正在保存图片..."):
        output_dir = Path(params["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        series_name = params["series_name"]
        figures = st.session_state.get("figures", {})

        saved_files = []
        for fig_type, fig in figures.items():
            output_path = output_dir / f"{series_name}_{fig_type}.png"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            saved_files.append(str(output_path))

        if saved_files:
            st.success(
                f"✅ Successfully saved {len(saved_files)} files to `{params['output_dir']}` directory:\n"
                + "\n".join([f"• {Path(f).name}" for f in saved_files])
            )


def _render_download_buttons(params: Dict[str, Any]):
    """Render download buttons for plots."""
    figures = st.session_state.get("figures", {})
    series_name = params["series_name"]

    col1, col2, col3, col4 = st.columns(4)

    button_configs = [
        (
            "analysis",
            "📈 Analysis Plot",
            "Download comprehensive multi-dimensional analysis chart",
        ),
        ("baseline", "📏 Baseline Plot", "Download baseline reference chart"),
        ("otdr", "📊 OTDR Plot", "Download OTDR measurement trace chart"),
        ("residual", "📉 Residual Plot", "Download residual analysis chart"),
    ]

    for i, (fig_key, label, help_text) in enumerate(button_configs):
        with [col1, col2, col3, col4][i]:
            if fig_key in figures:
                st.download_button(
                    label,
                    data=fig_to_bytes(figures[fig_key]),
                    file_name=f"{series_name}_{fig_key}.png",
                    mime="image/png",
                    width=True,
                    help=help_text,
                )


def _render_data_download(results: Dict[str, Any], params: Dict[str, Any]):
    """Render data download section."""
    clustered_events = results["clustered_events"]

    if clustered_events:
        st.markdown("### 📊 Download Data")

        # Create CSV data
        csv_data = pd.DataFrame(
            [
                {
                    "Event_Type": ev.kind,
                    "Position_km": ev.z_km,
                    "Position_m": ev.z_km * 1000,
                    "Loss_dB": ev.magnitude_db,
                    "Reflection_dB": ev.reflect_db,
                }
                for ev in clustered_events
            ]
        ).to_csv(index=False, encoding="utf-8-sig")

        st.download_button(
            "📋 Download Event Data (CSV)",
            data=csv_data,
            file_name=f"{params['series_name']}_events.csv",
            mime="text/csv",
            help="Download detailed event detection data table",
        )
