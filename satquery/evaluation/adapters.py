"""
Pipeline Adapters (Co-owned with Kartika)
Normalizes outputs from diverse lead pipelines into the standard EvaluationRecord schema (v0.2).
Provides versioned parsers to insulate the evaluation harness from API drift.
"""

from typing import Dict, Any, Optional
from .schemas import EvaluationRecord, ConfidenceBreakdown


class PipelineAdapter:
    """
    Base adapter defining the normalization contract for specialist and agent pipelines.
    """

    @classmethod
    def parse(cls, raw_output: Dict[str, Any], task_id: str, version: str = "0.2") -> EvaluationRecord:
        """Dispatches to version-specific parser."""
        if version == "0.1":
            return cls.parse_v01(raw_output, task_id)
        return cls.parse_v02(raw_output, task_id)

    @classmethod
    def parse_v01(cls, raw_output: Dict[str, Any], task_id: str) -> EvaluationRecord:
        """Legacy v0.1 parser."""
        return cls.parse_v02(raw_output, task_id)

    @classmethod
    def parse_v02(cls, raw_output: Dict[str, Any], task_id: str) -> EvaluationRecord:
        """Must be implemented by subclasses."""
        raise NotImplementedError


class ChangeDetectionAdapter(PipelineAdapter):
    """
    Normalizes Tool 3 (ChangeDetector / Bi-temporal Engine) outputs into EvaluationRecord.
    """

    @classmethod
    def parse_v02(cls, raw_output: Dict[str, Any], task_id: str) -> EvaluationRecord:
        status_raw = raw_output.get("status", "ok")
        status_mapped = (
            "classical_algorithm" if status_raw in ["ok", "partial"]
            else "failed" if status_raw == "error"
            else "heuristic_fallback"
        )

        area_metrics = raw_output.get("area_metrics", {})
        det_nums: Dict[str, float] = {}
        if "area_ha" in area_metrics:
            det_nums["area_ha"] = float(area_metrics["area_ha"])
        elif "changed_area_ha" in area_metrics:
            det_nums["area_ha"] = float(area_metrics["changed_area_ha"])

        if "area_m2" in area_metrics:
            det_nums["area_m2"] = float(area_metrics["area_m2"])
        elif "changed_area_m2" in area_metrics:
            det_nums["area_m2"] = float(area_metrics["changed_area_m2"])

        if "pct_changed" in area_metrics:
            det_nums["change_pct"] = float(area_metrics["pct_changed"])
        elif "change_percentage" in area_metrics:
            det_nums["change_pct"] = float(area_metrics["change_percentage"])

        if "n_changed_pixels" in raw_output:
            det_nums["pixel_count"] = float(raw_output["n_changed_pixels"])
        elif "n_changed_pixels" in area_metrics:
            det_nums["pixel_count"] = float(area_metrics["n_changed_pixels"])


        conf = float(raw_output.get("confidence", 0.0))
        # Build 6-factor decomposed confidence
        breakdown = ConfidenceBreakdown(
            input_quality=0.95,
            model_confidence=conf,
            evidence_agreement=0.90,
            geospatial_validity=1.0 if not raw_output.get("warnings") else 0.8,
            temporal_validity=1.0,
            contradiction_penalty=0.0,
        )

        # Extract trace tool names
        raw_trace = raw_output.get("execution_trace", [])
        tool_names = [step.get("tool", "unknown") for step in raw_trace if isinstance(step, dict)]

        return EvaluationRecord(
            task_id=task_id,
            schema_version="0.2",
            status=status_mapped,
            text_response=raw_output.get("summary", ""),
            deterministic_numbers=det_nums,
            geojson_geometry=raw_output.get("geojson"),
            confidence_score=conf,
            confidence_breakdown=breakdown,
            execution_time_ms=float(raw_output.get("total_processing_ms", 0.0)),
            tool_trace=tool_names,
        )


class AgentPlannerAdapter(PipelineAdapter):
    """
    Normalizes Kartika's agent orchestrator execution traces and final mission responses.
    """

    @classmethod
    def parse_v02(cls, raw_output: Dict[str, Any], task_id: str) -> EvaluationRecord:
        status_label = raw_output.get("status", "real_model")
        
        # Parse decomposed confidence if provided by Kartika's fusion layer
        raw_breakdown = raw_output.get("confidence_breakdown")
        if isinstance(raw_breakdown, dict):
            breakdown = ConfidenceBreakdown(**raw_breakdown)
        else:
            breakdown = ConfidenceBreakdown(
                input_quality=raw_output.get("input_quality", 0.9),
                model_confidence=raw_output.get("model_confidence", 0.85),
                evidence_agreement=raw_output.get("evidence_agreement", 0.88),
                geospatial_validity=raw_output.get("geospatial_validity", 0.95),
                temporal_validity=raw_output.get("temporal_validity", 0.95),
                contradiction_penalty=raw_output.get("contradiction_penalty", 0.0),
            )

        conf = float(raw_output.get("confidence", breakdown.calculate_composite()))

        return EvaluationRecord(
            task_id=task_id,
            schema_version="0.2",
            status=status_label,
            text_response=raw_output.get("response_text", raw_output.get("text", "")),
            deterministic_numbers=raw_output.get("deterministic_numbers", {}),
            geojson_geometry=raw_output.get("geojson"),
            confidence_score=conf,
            confidence_breakdown=breakdown,
            execution_time_ms=float(raw_output.get("latency_ms", 0.0)),
            tool_trace=raw_output.get("tool_sequence", []),
        )


class VLMAdapter(PipelineAdapter):
    """
    Normalizes Tool 1 (VLMGroundingTool / RSModelBackend) outputs into EvaluationRecord.
    Supports all 5 VLM capabilities: VQA, Captioning, Grounding, Scene Understanding, Structured Reasoning.
    """

    @classmethod
    def parse_v02(cls, raw_output: Dict[str, Any], task_id: str) -> EvaluationRecord:
        metrics = raw_output.get("metrics", {})
        det_nums: Dict[str, float] = {}

        # Detections count or areas
        detections = raw_output.get("detections", [])
        if isinstance(detections, list):
            det_nums["detection_count"] = float(len(detections))

        # Metrics from VLMFeaturePipeline
        if "feature_confidence" in metrics:
            det_nums["feature_confidence"] = float(metrics["feature_confidence"])
        if "gate_mean_activation" in metrics:
            det_nums["gate_mean_activation"] = float(metrics["gate_mean_activation"])
        if "gate_saturation_rate" in metrics:
            det_nums["gate_saturation_rate"] = float(metrics["gate_saturation_rate"])

        conf = float(raw_output.get("confidence", metrics.get("feature_confidence", 0.85)))
        provenance = raw_output.get("execution_mode", raw_output.get("provenance", "heuristic_fallback"))
        status_label = "real_model" if provenance in ("real_model", "gpu_active") else "heuristic_fallback"

        # Construct 6-factor decomposed confidence
        input_quality = 0.95
        if metrics.get("gate_mean_activation", 0.0) > 0.8:
            input_quality = 0.70  # Higher cloud/noise suppressed by gate

        breakdown = ConfidenceBreakdown(
            input_quality=input_quality,
            model_confidence=min(1.0, max(0.0, conf)),
            evidence_agreement=float(metrics.get("fusion_contribution_ratio", 0.90)),
            geospatial_validity=1.0 if raw_output.get("mask_geojson") else 0.85,
            temporal_validity=1.0,
            contradiction_penalty=0.0,
        )

        tool_trace = ["SARSpecFeatExtractor", "GatedFusionModule", "GeospatialGroundingEngine"]
        if raw_output.get("capability"):
            tool_trace.append(f"VLM.{raw_output['capability']}")

        return EvaluationRecord(
            task_id=task_id,
            schema_version="0.2",
            status=status_label,
            text_response=raw_output.get("response_text", raw_output.get("text", raw_output.get("answer", raw_output.get("summary", "")))),
            deterministic_numbers=det_nums,
            geojson_geometry=raw_output.get("mask_geojson"),
            confidence_score=min(1.0, max(0.0, conf)),
            confidence_breakdown=breakdown,
            execution_time_ms=float(raw_output.get("latency_ms", 0.0)),
            tool_trace=tool_trace,
        )


class GenericSpecialistAdapter(PipelineAdapter):
    """
    Fallback adapter for Omkar's VLM or Moiz's Fusion raw dictionaries.
    """

    @classmethod
    def parse_v02(cls, raw_output: Dict[str, Any], task_id: str) -> EvaluationRecord:
        conf = float(raw_output.get("confidence", raw_output.get("score", 0.8)))
        return EvaluationRecord(
            task_id=task_id,
            schema_version="0.2",
            status=raw_output.get("status", "real_model"),
            text_response=raw_output.get("answer", raw_output.get("text", "")),
            deterministic_numbers=raw_output.get("numbers", {}),
            geojson_geometry=raw_output.get("geojson", raw_output.get("geometry")),
            confidence_score=min(1.0, max(0.0, conf)),
            confidence_breakdown=ConfidenceBreakdown(model_confidence=conf),
            execution_time_ms=float(raw_output.get("latency_ms", 0.0)),
            tool_trace=raw_output.get("tools", []),
        )

