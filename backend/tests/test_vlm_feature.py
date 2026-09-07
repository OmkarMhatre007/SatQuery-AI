import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.services.geo_compat import HAS_RASTERIO
from app.services.vlm_feature import (
    CalibratedConfidenceEstimator,
    FeatureTensorPackage,
    GatedFusionModule,
    SARSpecFeatExtractor,
    VLMInferenceGuard,
    vlm_feature_pipeline,
)
from app.tools.tool1_vlm import vlm_tool


@pytest.fixture
def test_rasters(tmp_path):
    opt_path = tmp_path / "opt_scene.tif"
    sar_path = tmp_path / "sar_scene.tif"

    opt_data = np.random.randint(40, 220, (100, 100, 3), dtype=np.uint8)
    sar_data = np.random.randint(15, 190, (100, 100), dtype=np.uint8)

    if HAS_RASTERIO:
        import rasterio
        from rasterio.transform import from_origin
        transform = from_origin(500000, 3000000, 1.0, 1.0)
        with rasterio.open(
            opt_path, 'w', driver='GTiff', height=100, width=100, count=3,
            dtype=np.uint8, crs='EPSG:32643', transform=transform
        ) as dst:
            dst.write(np.transpose(opt_data, (2, 0, 1)))

        with rasterio.open(
            sar_path, 'w', driver='GTiff', height=100, width=100, count=1,
            dtype=np.uint8, crs='EPSG:32643', transform=transform
        ) as dst:
            dst.write(sar_data, 1)
    else:
        Image.fromarray(opt_data).save(opt_path)
        Image.fromarray(sar_data).save(sar_path)

    return opt_path, sar_path


def test_feature_package_validation():
    # Valid shape
    valid_tokens = np.ones((1, 576, 1024), dtype=np.float32)
    valid_patch = np.ones((1, 576, 1), dtype=np.float32) * 0.8
    pkg = FeatureTensorPackage(tokens=valid_tokens, confidence=0.85, patch_confidence=valid_patch)
    pkg.validate(1, 576, 1024)

    # Invalid shape
    with pytest.raises(ValueError, match="Shape invariant violation"):
        pkg.validate(1, 256, 1024)

    # Invalid confidence
    with pytest.raises(ValueError, match="out of bounds"):
        bad_conf_pkg = FeatureTensorPackage(tokens=valid_tokens, confidence=1.4, patch_confidence=valid_patch)
        bad_conf_pkg.validate(1, 576, 1024)

    # NaN detection
    with pytest.raises(FloatingPointError, match="NaN or Inf"):
        nan_tokens = valid_tokens.copy()
        nan_tokens[0, 10, 10] = np.nan
        nan_pkg = FeatureTensorPackage(tokens=nan_tokens, confidence=0.5, patch_confidence=valid_patch)
        nan_pkg.validate(1, 576, 1024)


def test_sar_spec_feat_extractor():
    extractor = SARSpecFeatExtractor(target_dim=1024, num_patches_side=24)
    opt_sample = np.random.randint(50, 200, (3, 80, 80), dtype=np.uint8)
    sar_sample = np.random.randint(20, 150, (80, 80), dtype=np.uint8)

    # With SAR
    tokens, meta = extractor.extract(opt_sample, sar_sample)
    assert tokens.shape == (1, 576, 1024)
    assert meta["sar_engaged"] is True
    assert "radar_enl" in meta
    assert "cloud_fraction" in meta

    # Without SAR
    tokens_no_sar, meta_no_sar = extractor.extract(opt_sample, None)
    assert tokens_no_sar.shape == (1, 576, 1024)
    assert meta_no_sar["sar_engaged"] is False


def test_calibrated_confidence_estimator():
    estimator = CalibratedConfidenceEstimator(temperature=1.2)
    dummy_tokens = np.random.randn(1, 576, 1024).astype(np.float32)

    meta_cloudy = {"cloud_fraction": 0.85, "sar_engaged": False, "radar_enl": 0.0}
    meta_clear_sar = {"cloud_fraction": 0.05, "sar_engaged": True, "radar_enl": 4.5}

    conf_cloudy, patch_cloudy = estimator.estimate(dummy_tokens, meta_cloudy)
    conf_clear, patch_clear = estimator.estimate(dummy_tokens, meta_clear_sar)

    assert 0.0 < conf_cloudy < 1.0
    assert 0.0 < conf_clear < 1.0
    assert conf_clear > conf_cloudy  # Clear scene with radar has higher calibrated confidence
    assert patch_cloudy.shape == (1, 576, 1)


def test_gated_fusion_boundary_conditions():
    fuser = GatedFusionModule(dim=1024)
    base_tokens = np.ones((1, 576, 1024), dtype=np.float32) * 2.0
    feat_tokens = np.ones((1, 576, 1024), dtype=np.float32) * 5.0

    fused, gate, telemetry = fuser.fuse(base_tokens, feat_tokens, confidence=0.85)

    assert fused.shape == (1, 576, 1024)
    assert gate.shape == (1, 576, 1)
    assert 0.0 <= telemetry["gate_mean_activation"] <= 1.0
    assert 0.0 <= telemetry["fusion_contribution_ratio"] <= 1.0
    # Values should strictly lie within bounds of inputs
    assert np.all(fused >= 2.0)
    assert np.all(fused <= 5.0)


def test_circuit_breaker_nan_and_fault_tolerance():
    guard = VLMInferenceGuard(max_consecutive_errors=2)
    extractor = SARSpecFeatExtractor()
    conf_estimator = CalibratedConfidenceEstimator()
    fuser = GatedFusionModule()

    nan_base = np.ones((1, 576, 1024), dtype=np.float32)
    nan_base[0, 0, 0] = np.nan
    opt = np.random.randint(0, 255, (3, 60, 60))

    # Error 1
    tokens, conf, tele = guard.execute_safe(nan_base, extractor, conf_estimator, fuser, opt)
    assert "exception_fallback" in tele["circuit_breaker_status"]
    assert guard.error_count == 1
    assert not guard.circuit_open

    # Error 2 -> opens circuit
    tokens, conf, tele = guard.execute_safe(nan_base, extractor, conf_estimator, fuser, opt)
    assert guard.circuit_open is True

    # Error 3 -> circuit open fallback
    clean_base = np.ones((1, 576, 1024), dtype=np.float32)
    tokens, conf, tele = guard.execute_safe(clean_base, extractor, conf_estimator, fuser, opt)
    assert tele["circuit_breaker_status"] == "circuit_open_fallback"
    assert tele["gate_mean_activation"] == 1.0

    # Reset
    guard.reset()
    assert not guard.circuit_open


def test_vlm_tool_integration_with_sar_pair(test_rasters):
    opt_path, sar_path = test_rasters
    output = vlm_tool.execute([opt_path, sar_path], {"query": "Identify water bodies and flooded zones"})

    assert output.tool == "vqa_grounding"
    assert output.bboxes is not None
    assert len(output.bboxes) > 0
    assert "water" in output.answer.lower()
    assert "SAR backscatter" in output.answer

    # Telemetry verification
    metrics = output.metrics
    assert metrics.get("sar_spec_feat_engaged") is True
    assert "gate_mean_activation" in metrics
    assert "fusion_contribution_ratio" in metrics
    assert metrics["circuit_breaker_status"] == "success"
    assert 0.05 <= output.confidence <= 1.0


def test_vlm_tool_feature_flag_disabled(test_rasters):
    opt_path, _ = test_rasters
    original_state = settings.VLM_FEATURE_ENABLED
    try:
        settings.VLM_FEATURE_ENABLED = False
        output = vlm_tool.execute([opt_path], {"query": "Find urban settlements"})
        assert output.tool == "vqa_grounding"
        assert output.metrics["circuit_breaker_status"] == "feature_flag_disabled"
        assert output.metrics["gate_mean_activation"] == 1.0
    finally:
        settings.VLM_FEATURE_ENABLED = original_state


def test_vlm_tool_captioning_capability(test_rasters):
    opt_path, _ = test_rasters
    output = vlm_tool.execute([opt_path], {"query": "Provide a comprehensive caption and scene summary."})
    assert output.tool == "vqa_grounding"
    assert output.metrics["vlm_capability_executed"] == "captioning"
    assert len(output.answer) > 20
    assert output.mask_geojson is not None
    assert output.mask_geojson["type"] == "FeatureCollection"


def test_vlm_tool_scene_understanding_capability(test_rasters):
    opt_path, _ = test_rasters
    output = vlm_tool.execute([opt_path], {"query": "Analyze scene understanding and spatial relationships between entities."})
    assert output.tool == "vqa_grounding"
    assert output.metrics["vlm_capability_executed"] == "scene_understanding"
    assert "scene_relationships" in output.metrics
    assert len(output.metrics["scene_relationships"]) > 0


def test_vlm_tool_structured_reasoning_capability(test_rasters):
    opt_path, _ = test_rasters
    output = vlm_tool.execute([opt_path], {"query": "Execute structured reasoning and extract observations."})
    assert output.tool == "vqa_grounding"
    assert output.metrics["vlm_capability_executed"] == "structured_reasoning"
    assert "structured_observations" in output.metrics
    assert len(output.metrics["structured_observations"]) > 0


def test_vlm_georeferenced_wgs84_geojson_grounding(test_rasters):
    opt_path, sar_path = test_rasters
    output = vlm_tool.execute([opt_path, sar_path], {"query": "Locate water reservoir"})
    assert output.mask_geojson is not None
    assert output.mask_geojson["type"] == "FeatureCollection"
    features = output.mask_geojson["features"]
    assert len(features) > 0
    geom = features[0]["geometry"]
    assert geom["type"] == "Polygon"
    coords = geom["coordinates"][0]
    assert len(coords) >= 5  # Closed 4-point polygon has at least 5 coordinates
    # Coordinates should be valid lon/lat
    for pt in coords:
        assert len(pt) == 2


def test_geotiff_preprocessing_and_tiling():
    from app.services.rs_vlm_backend import GeoTIFFPreprocessPipeline
    data = np.random.randint(0, 10000, (3, 200, 200), dtype=np.uint16)
    clipped = GeoTIFFPreprocessPipeline.percentile_clip(data, 2.0, 98.0)
    assert clipped.shape == (3, 200, 200)
    assert 0.0 <= np.min(clipped) <= np.max(clipped) <= 1.0

    tiles = GeoTIFFPreprocessPipeline.generate_tiles(img_width=1200, img_height=1200, tile_size=512, overlap=64)
    assert len(tiles) > 1

    mapped_box = GeoTIFFPreprocessPipeline.map_tile_bbox_to_global(
        tile_bbox=[0.1, 0.1, 0.9, 0.9],
        col_off=100,
        row_off=100,
        tile_w=512,
        tile_h=512,
        global_w=1200,
        global_h=1200
    )
    assert len(mapped_box) == 4
    for val in mapped_box:
        assert 0.0 <= val <= 1.0


def test_vlm_backend_health_check_and_model_info():
    health = vlm_tool.health_check()
    assert "status" in health
    assert health["status"] in ("healthy", "fallback_active")

    info = vlm_tool.model_info()
    assert "name" in info
    assert "capabilities" in info or "backend" in info


def test_vlm_adapter_normalization(test_rasters):
    """Verify that Tool 1 output correctly adapts into standard EvaluationRecord schema."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from satquery.evaluation.adapters import VLMAdapter
    from satquery.evaluation.schemas import EvaluationRecord

    opt_path, sar_path = test_rasters
    output = vlm_tool.execute([opt_path, sar_path], {"query": "Detect water body"})
    raw_dict = {
        "text": output.answer,
        "mask_geojson": output.mask_geojson,
        "confidence": output.confidence,
        "metrics": output.metrics,
        "execution_mode": output.execution_mode,
        "capability": "grounding",
    }
    rec = VLMAdapter.parse(raw_dict, task_id="TEST-VLM-01")
    assert isinstance(rec, EvaluationRecord)
    assert rec.task_id == "TEST-VLM-01"
    assert rec.schema_version == "0.2"
    assert rec.confidence_score > 0.0
    assert rec.confidence_breakdown is not None
    assert 0.0 <= rec.confidence_breakdown.input_quality <= 1.0
    assert rec.geojson_geometry is not None
    assert "GeospatialGroundingEngine" in rec.tool_trace


