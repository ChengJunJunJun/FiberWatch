"""
Streamlit web application for FiberWatch OTDR analysis.

This module provides an interactive web interface for uploading OTDR data,
configuring analysis parameters, and visualizing results.
"""

import numpy as np
import streamlit as st

from fiberwatch.config import get_default_config

# Import FiberWatch components
from fiberwatch.core import Detector, DetectorConfig
from fiberwatch.utils.data_io import create_distance_axis, parse_uploaded_file
from fiberwatch.utils.event_processing import cluster_events, get_event_statistics
from fiberwatch.web.ui_components import (
    render_analysis_results,
    render_download_section,
    render_header,
    render_sidebar,
)


def main():
    """Main Streamlit application entry point."""
    # Load configuration
    config = get_default_config()
    web_config = config.web

    # Page configuration
    st.set_page_config(
        page_title=web_config.page_title,
        layout=web_config.layout,
        initial_sidebar_state=web_config.initial_sidebar_state,
    )

    # Apply custom CSS styling
    _apply_custom_styling()

    # Render header
    render_header()

    # Render sidebar for parameters and data input
    params = render_sidebar(config)

    # Check if analysis should run
    if not params.get("run_analysis", False):
        st.info("请上传数据并点击运行检测")
        st.stop()

    # Validate required data
    if params["test_data"] is None:
        st.error("请上传测试曲线 .txt 文件。")
        st.stop()

    # Run analysis
    try:
        results = _run_analysis(params, config)

        # Render results
        render_analysis_results(results, params)

        # Render download section
        render_download_section(results, params)

    except Exception as e:
        st.error(f"分析过程中发生错误: {e!s}")
        if config.debug:
            st.exception(e)


def _run_analysis(params: dict, config) -> dict:
    """
    Run OTDR analysis with given parameters.

    Args:
        params: Analysis parameters from UI
        config: Application configuration

    Returns:
        Dictionary containing analysis results

    """
    with st.spinner("正在进行OTDR分析..."):
        # Parse uploaded data
        test_data = parse_uploaded_file(params["test_data"])

        if test_data.size == 0:
            raise ValueError("测试曲线为空或无法解析")

        # Create distance axis
        distance_km = create_distance_axis(len(test_data), params["total_distance_km"])

        # Parse baseline if provided
        baseline_data = None
        if params["baseline_data"] is not None:
            baseline_data = parse_uploaded_file(params["baseline_data"])
            # Interpolate baseline to match test data length
            if baseline_data.size != test_data.size:
                baseline_distance = create_distance_axis(
                    len(baseline_data),
                    params["total_distance_km"],
                )
                baseline_data = np.interp(distance_km, baseline_distance, baseline_data)

        # Create detector configuration
        detector_config = DetectorConfig(
            refl_min_db=params["refl_min_db"],
            step_min_db=params["step_min_db"],
            slope_min_db_per_km=params["slope_min_db"],
            min_event_separation=params["min_event_separation"],
        )

        # Run detection
        detector = Detector(
            distance_km=distance_km,
            baseline=baseline_data,
            config=detector_config,
        )

        detection_result = detector.detect(test_data)

        # Cluster events
        clustered_events = cluster_events(
            detection_result.events,
            distance_threshold_m=params["distance_cluster_m"],
        )

        # Calculate statistics
        event_stats = get_event_statistics(clustered_events)

        return {
            "detection_result": detection_result,
            "clustered_events": clustered_events,
            "event_stats": event_stats,
            "distance_km": distance_km,
            "test_data": test_data,
            "baseline_provided": baseline_data is not None,
        }


def _apply_custom_styling():
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Global settings */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        /* Main title */
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.3rem;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        }
        
        /* Subtitle */
        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.5rem;
            font-weight: 400;
        }
        
        /* Compact layout */
        .stMarkdown h2 {
            margin-top: 1rem !important;
            margin-bottom: 0.8rem !important;
            font-size: 1.4rem !important;
        }
        
        .stMarkdown h3 {
            margin-top: 0.8rem !important;
            margin-bottom: 0.5rem !important;
            font-size: 1.1rem !important;
        }
        
        /* Card containers */
        .card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        /* Sidebar styling */
        .stSidebar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255, 255, 255, 0.5);
            border-radius: 12px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            color: #6b7280;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(102, 126, 234, 0.1);
        }
        
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        /* File uploader styling */
        .stFileUploader > div {
            background: #f8fafc;
            border: 2px dashed #d1d5db;
            border-radius: 12px;
            transition: all 0.3s ease;
        }
        
        .stFileUploader > div:hover {
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.05);
        }
        
        /* Metric card styling */
        .metric-card {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 0.75rem;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.3);
            margin-bottom: 0.5rem;
        }
        
        /* Success/Error/Info message styling */
        .stSuccess {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 12px;
            border: none;
        }
        
        .stError {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            border-radius: 12px;
            border: none;
        }
        
        .stInfo {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            border-radius: 12px;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
