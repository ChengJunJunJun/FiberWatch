"""
Tests for core detection functionality.

This module tests the main OTDR event detection algorithms
and data structures.
"""

import numpy as np
import pytest

from fiberwatch.core import Detector, DetectorConfig, DetectedEvent, DetectionResult
from fiberwatch.utils.data_io import create_distance_axis


class TestDetectorConfig:
    """Test DetectorConfig validation and functionality."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DetectorConfig()
        assert config.smooth_win == 21
        assert config.smooth_poly == 3
        assert config.refl_min_db == 1.0
        assert config.step_min_db == 0.05

    def test_config_validation(self):
        """Test configuration parameter validation."""
        # Valid config should not raise
        config = DetectorConfig(smooth_win=21, smooth_poly=3)
        config.validate()

        # Invalid smooth_win (even number)
        config = DetectorConfig(smooth_win=20)
        with pytest.raises(ValueError, match="smooth_win must be positive and odd"):
            config.validate()

        # Invalid refl_min_db (negative)
        config = DetectorConfig(refl_min_db=-1.0)
        with pytest.raises(ValueError, match="refl_min_db must be positive"):
            config.validate()


class TestDetectedEvent:
    """Test DetectedEvent data structure."""

    def test_event_creation(self):
        """Test event creation and attributes."""
        event = DetectedEvent(
            kind="break", z_km=1.5, magnitude_db=5.2, reflect_db=10.0, index=150
        )

        assert event.kind == "break"
        assert event.z_km == 1.5
        assert event.magnitude_db == 5.2
        assert event.reflect_db == 10.0
        assert event.index == 150
        assert event.extra == {}

    def test_event_with_extra_data(self):
        """Test event with extra metadata."""
        event = DetectedEvent(
            kind="splice", z_km=0.8, extra={"confidence": 0.95, "method": "gradient"}
        )

        assert event.extra["confidence"] == 0.95
        assert event.extra["method"] == "gradient"


class TestDetector:
    """Test main Detector functionality."""

    def test_detector_creation(self):
        """Test detector initialization."""
        distance_km = create_distance_axis(1000, 10.0)
        config = DetectorConfig()

        detector = Detector(distance_km=distance_km, config=config)

        assert len(detector.distance_km) == 1000
        assert detector.config.smooth_win == 21
        assert detector.baseline is None

    def test_detector_with_baseline(self):
        """Test detector with baseline data."""
        distance_km = create_distance_axis(1000, 10.0)
        baseline = np.linspace(-10, -50, 1000)  # Typical OTDR baseline

        detector = Detector(distance_km=distance_km, baseline=baseline)

        assert detector.baseline is not None
        assert len(detector.baseline) == 1000

    def test_simple_detection(self):
        """Test basic event detection on synthetic data."""
        # Create simple synthetic OTDR trace
        n_samples = 1000
        distance_km = create_distance_axis(n_samples, 10.0)

        # Create trace with exponential decay (normal fiber loss)
        trace = -10 - 0.2 * distance_km * 1000  # 0.2 dB/km loss

        # Add some noise
        trace += np.random.normal(0, 0.1, n_samples)

        # Add a break at 5 km
        break_idx = int(n_samples * 0.5)
        trace[break_idx:] -= 20  # 20 dB drop
        trace[break_idx:] += np.random.normal(
            0, 2, n_samples - break_idx
        )  # Noise after break

        config = DetectorConfig(min_signal_drop_db=10.0)
        detector = Detector(distance_km=distance_km, config=config)

        result = detector.detect(trace)

        assert isinstance(result, DetectionResult)
        assert len(result.events) > 0
        assert result.distance_km is not None
        assert result.trace_smooth_db is not None
        assert result.baseline_db is not None
        assert result.residual_db is not None

    def test_no_events_detection(self):
        """Test detection on clean fiber with no events."""
        n_samples = 1000
        distance_km = create_distance_axis(n_samples, 10.0)

        # Create clean trace with only normal fiber loss
        trace = -10 - 0.2 * distance_km * 1000
        trace += np.random.normal(0, 0.05, n_samples)  # Small amount of noise

        detector = Detector(distance_km=distance_km)
        result = detector.detect(trace)

        # Should detect few or no events on clean fiber
        assert len(result.events) <= 2  # Allow for minor noise artifacts


class TestDetectionResult:
    """Test DetectionResult functionality."""

    def test_result_plot(self):
        """Test that result plotting works without errors."""
        # Create minimal detection result
        distance_km = create_distance_axis(100, 1.0)
        events = [
            DetectedEvent(kind="reflection", z_km=0.5, magnitude_db=2.0, reflect_db=5.0)
        ]

        result = DetectionResult(
            events=events,
            distance_km=distance_km,
            trace_smooth_db=np.linspace(-10, -20, 100),
            baseline_db=np.linspace(-10, -20, 100),
            residual_db=np.zeros(100),
        )

        # Should not raise an exception
        fig = result.plot()
        assert fig is not None


@pytest.mark.integration
class TestIntegrationDetection:
    """Integration tests for full detection pipeline."""

    def test_full_pipeline_with_multiple_events(self):
        """Test detection pipeline with multiple event types."""
        n_samples = 2000
        distance_km = create_distance_axis(n_samples, 20.0)

        # Create base trace
        trace = -10 - 0.2 * distance_km * 1000

        # Add events at different locations
        # Splice at 5 km (small step loss)
        splice_idx = int(n_samples * 0.25)
        trace[splice_idx:] -= 0.3

        # Connector at 10 km (reflection + small loss)
        conn_idx = int(n_samples * 0.5)
        trace[conn_idx - 2 : conn_idx + 2] += [0, 3, -1, 0]  # Reflection spike
        trace[conn_idx:] -= 0.1

        # Break at 15 km
        break_idx = int(n_samples * 0.75)
        trace[break_idx:] -= 25
        trace[break_idx:] += np.random.normal(0, 3, n_samples - break_idx)

        # Add general noise
        trace += np.random.normal(0, 0.1, n_samples)

        detector = Detector(distance_km=distance_km)
        result = detector.detect(trace)

        # Should detect multiple events
        assert len(result.events) >= 1  # At least the break should be detected

        # Check that break is detected (should be the most significant)
        break_events = [e for e in result.events if e.kind == "break"]
        assert len(break_events) >= 1

        # Break should be near 15 km position
        break_event = break_events[0]
        assert 14.0 < break_event.z_km < 16.0  # Allow some tolerance
