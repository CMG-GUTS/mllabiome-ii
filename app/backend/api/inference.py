import io
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.backend.services.production_inference import production_service

router = APIRouter()


def _read_uploaded_table(content: bytes, filename: str | None) -> pd.DataFrame:
    """Read uploaded microbiome table while preserving sample_id as data.

    LAMPP benchmark test sets are CSV files with sample_id + lineage columns.
    The original single-sample IBS example is TSV/profile-style.
    """
    name = (filename or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    return pd.read_csv(io.BytesIO(content), sep="\t")


class InferenceRequest(BaseModel):
    model_id: str
    features: dict[str, float]


class InferenceResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    confidence: float
    model_id: str
    n_models_used: int
    sample_id: Optional[str] = None


class BatchInferenceResponse(BaseModel):
    predictions: list[InferenceResponse]
    model_id: str
    n_samples: int


class FeatureExplanation(BaseModel):
    feature: str
    importance: float
    std: float
    n_models: int
    abs_importance: Optional[float] = None
    support: Optional[float] = None
    direction: Optional[str] = None
    method: Optional[str] = None
    value: Optional[float] = None
    baseline: Optional[float] = None


class InteractionEdge(BaseModel):
    source: str
    target: str
    strength: float
    abs_strength: Optional[float] = None
    support: Optional[float] = None
    direction: Optional[str] = None
    method: Optional[str] = None


class SampleExplanation(BaseModel):
    sample_id: str
    prediction: int
    label: str
    probability: float
    top_features: list[FeatureExplanation]
    shap_features: list[FeatureExplanation] = []
    lime_features: list[FeatureExplanation] = []
    interactions: list[InteractionEdge] = []
    interaction_method: Optional[str] = None
    n_models_explained: int
    explanation_method: Optional[str] = None
    explanation_basis: Optional[str] = None


class ExplainResponse(BaseModel):
    explanations: list[SampleExplanation]
    model_id: str
    n_samples: int
    method: Optional[str] = None
    dependencies: Optional[dict[str, Any]] = None


@router.get("/inference/models")
async def list_available_models():
    return {"models": production_service.list_models()}


@router.get("/inference/explainability/status")
async def explainability_status():
    """Report explainability dependency availability."""
    return production_service.explainability_status()


@router.get("/inference/effect-curves")
async def get_effect_curves(model_id: str, features: Optional[str] = None):
    """Return precomputed package curves if present."""
    try:
        return {
            "model_id": model_id,
            "curves": production_service.effect_curves(model_id, features=features),
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/effect-curves")
async def compute_effect_curves(
    model_id: str,
    data: UploadFile = File(...),
    sample_id_column: Optional[str] = "sample_id",
    features: Optional[str] = None,
    num_points: int = 25,
):
    """Compute feature sensitivity curves."""
    try:
        content = await data.read()

        df = _read_uploaded_table(content, data.filename)

        curves = production_service.effect_curves(
            model_id=model_id,
            data=df,
            sample_id_column=sample_id_column,
            features=features,
            num_points=num_points,
        )
        return {"model_id": model_id, "curves": curves}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/predict", response_model=InferenceResponse)
async def predict_single(request: InferenceRequest):
    try:
        result = production_service.predict(request.model_id, request.features)
        return InferenceResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/predict/batch", response_model=BatchInferenceResponse)
async def predict_batch(
    model_id: str,
    data: UploadFile = File(...),
    sample_id_column: Optional[str] = "sample_id",
):
    try:
        content = await data.read()

        df = _read_uploaded_table(content, data.filename)

        results = production_service.predict_batch(
            model_id=model_id,
            data=df,
            sample_id_column=sample_id_column,
        )

        return BatchInferenceResponse(
            predictions=[InferenceResponse(**r) for r in results],
            model_id=model_id,
            n_samples=len(results),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/predict/submission")
async def predict_submission_csv(
    model_id: str,
    data: UploadFile = File(...),
    sample_id_column: Optional[str] = "sample_id",
):
    """Return benchmark submission CSV: sample_id,prediction.

    The prediction values are probabilities for class 1 / the positive class.
    """
    try:
        content = await data.read()
        df = _read_uploaded_table(content, data.filename)
        submission = production_service.predict_submission_dataframe(
            model_id=model_id,
            data=df,
            sample_id_column=sample_id_column,
        )
        filename = production_service.submission_filename(
            model_id, uploaded_filename=data.filename
        )
        buf = io.StringIO()
        submission.to_csv(buf, index=False, float_format="%.10g")
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(
            iter([buf.getvalue()]), media_type="text/csv", headers=headers
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/explain", response_model=ExplainResponse)
async def explain_predictions(
    model_id: str,
    data: UploadFile = File(...),
    sample_id_column: Optional[str] = "sample_id",
    sample_id: Optional[str] = None,
    num_features: int = 8,
    num_samples: int = 96,
):
    """Generate bounded local SHAP and LIME explanations for one uploaded sample."""
    try:
        content = await data.read()

        df = _read_uploaded_table(content, data.filename)

        results = production_service.explain_instance(
            model_id=model_id,
            data=df,
            sample_id_column=sample_id_column,
            sample_id=sample_id,
            num_features=num_features,
            num_samples=num_samples,
        )

        return ExplainResponse(
            explanations=[SampleExplanation(**r) for r in results],
            model_id=model_id,
            n_samples=len(results),
            method="local_shap_lime",
            dependencies=production_service.explainability_status().get(
                "dependencies", {}
            ),
        )
    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inference/explain/plot")
async def explain_plot(
    model_id: str,
    data: UploadFile = File(...),
    sample_id_column: Optional[str] = "sample_id",
    sample_index: int = 0,
    num_features: int = 8,
):
    """
    Generate a directional local-ablation explanation plot showing how each feature
    contributes positively or negatively to the prediction.
    """
    try:
        import base64
        import io as stdio

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        content = await data.read()

        df = _read_uploaded_table(content, data.filename)

        results = production_service.explain_instance(
            model_id=model_id,
            data=df,
            sample_id_column=sample_id_column,
            num_features=num_features,
            num_samples=96,
        )

        if not results or sample_index >= len(results):
            raise HTTPException(status_code=404, detail="Sample not found")

        exp = results[sample_index]
        features = exp["top_features"][:num_features]
        prediction_label = exp["label"]
        probability = exp["probability"]

        # Prepare data for directional horizontal bar chart
        names = []
        values = []
        for f in reversed(features):  # Reverse for proper display order
            name = f["feature"]
            if "___" in name:
                name = name.split("___")[-1]
            if len(name) > 25:
                name = name[:22] + "..."
            names.append(name)
            values.append(f["importance"])

        # Create directional LIME plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Color bars by direction: positive (toward predicted class) vs negative
        colors = ["#2563eb" if v >= 0 else "#64748b" for v in values]

        # Create horizontal bars centered at 0
        bars = ax.barh(names, values, color=colors, height=0.65, edgecolor="none")

        # Add a vertical line at 0
        ax.axvline(x=0, color="#0f172a", linewidth=1, linestyle="-", alpha=0.8)

        # Styling
        ax.set_xlabel(
            "Local baseline-ablation contribution (Δ probability)",
            fontsize=11,
            fontweight="500",
            color="#0f172a",
        )
        ax.set_title(
            f"Local Explanation: {exp['sample_id']}\n{prediction_label} ({probability * 100:.1f}% probability)",
            fontsize=12,
            fontweight="600",
            color="#0f172a",
            pad=15,
        )

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#2563eb", label=f"Pushes toward {prediction_label}"),
            Patch(facecolor="#64748b", label=f"Pushes against {prediction_label}"),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower right",
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            fontsize=9,
        )

        # Clean up spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e2e8f0")
        ax.spines["bottom"].set_color("#e2e8f0")

        ax.tick_params(axis="y", labelsize=9, colors="#0f172a")
        ax.tick_params(axis="x", labelsize=9, colors="#0f172a")

        # Add subtle grid
        ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#cbd5e1")
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save to buffer
        buf = stdio.BytesIO()
        plt.savefig(
            buf,
            format="png",
            dpi=150,
            bbox_inches="tight",
            facecolor="#ffffff",
            edgecolor="none",
        )
        plt.close(fig)
        buf.seek(0)

        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "sample_id": exp["sample_id"],
            "prediction": exp["label"],
            "probability": exp["probability"],
            "image": f"data:image/png;base64,{img_base64}",
            "features": [
                {"name": n, "contribution": v}
                for n, v in zip(reversed(names), reversed(values))
            ],
        }

    except ImportError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
