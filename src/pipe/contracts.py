"""Contrat v0.1 proposé dans le vault ; validation collective G1 encore requise."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModelName = Literal["tslm", "baseline"]
ExecutionMode = Literal["live", "replay", "development_fixture"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Observation(Contract):
    name: str
    value: float | str
    unit: str | None
    method: str


class Prediction(Contract):
    schema_version: Literal["0.1"] = "0.1"
    sample_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_name: ModelName
    model_version: str
    preprocessing_version: str
    prediction: Literal["leak", "no_leak"] | None
    class_scores: dict[str, float] | None = None
    score_type: Literal["raw", "calibrated", "none"]
    abstained: bool
    abstention_reason: str | None = None
    observations: list[Observation] = Field(default_factory=list)
    description: str | None = None
    latency_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    execution_mode: ExecutionMode


class PredictRequest(Contract):
    sample_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    model_name: ModelName
    request_id: str = Field(min_length=1, max_length=80)


class Sample(Contract):
    sample_id: str
    input_sha256: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    source: str
    execution_mode: ExecutionMode
    warnings: list[str]
    label_available: bool = False
