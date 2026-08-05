"""
PHK Studio
Clinical Pharmacy Engine

Assessment Models.

Modelos compartilhados pelos motores de avaliação clínica:

- função renal;
- função hepática;
- geriatria;
- pediatria;
- gestação;
- alergias;
- farmacogenômica.

Todos os resultados representam suporte à decisão e exigem
revisão do farmacêutico responsável.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Retorna a data e hora atual em UTC."""

    return datetime.now(
        timezone.utc,
    )


class AssessmentStatus(str, Enum):
    """Estado de execução da avaliação clínica."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_DATA = "invalid_data"
    FAILED = "failed"


class ClinicalRiskLevel(str, Enum):
    """Nível geral de risco clínico."""

    UNDETERMINED = "undetermined"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationPriority(str, Enum):
    """Prioridade operacional de uma recomendação."""

    INFORMATIONAL = "informational"
    ROUTINE = "routine"
    PRIORITY = "priority"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class RecommendationCategory(str, Enum):
    """Categoria da recomendação clínica."""

    MONITORING = "monitoring"
    DOSE_REVIEW = "dose_review"
    INTERVAL_REVIEW = "interval_review"
    CONTRAINDICATION_REVIEW = (
        "contraindication_review"
    )
    LABORATORY_REVIEW = "laboratory_review"
    PRESCRIBER_CONTACT = "prescriber_contact"
    REFERRAL = "referral"
    EMERGENCY_REFERRAL = "emergency_referral"
    PATIENT_EDUCATION = "patient_education"
    DOCUMENTATION = "documentation"
    OTHER = "other"


@dataclass(slots=True)
class ClinicalRecommendation:
    """
    Recomendação gerada por um motor clínico.

    A recomendação não altera automaticamente a prescrição.
    """

    title: str
    description: str

    category: RecommendationCategory
    priority: RecommendationPriority

    rationale: str | None = None

    related_medications: list[str] = field(
        default_factory=list,
    )

    monitoring_parameters: list[str] = field(
        default_factory=list,
    )

    references: list[str] = field(
        default_factory=list,
    )

    requires_pharmacist_review: bool = True
    requires_prescriber_contact: bool = False
    requires_immediate_action: bool = False

    id: str = field(
        default_factory=lambda: str(
            uuid4(),
        )
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class AssessmentAlert:
    """Alerta identificado durante a avaliação clínica."""

    code: str
    title: str
    description: str

    risk_level: ClinicalRiskLevel

    related_medications: list[str] = field(
        default_factory=list,
    )

    evidence: list[str] = field(
        default_factory=list,
    )

    requires_acknowledgement: bool = True
    requires_immediate_action: bool = False

    id: str = field(
        default_factory=lambda: str(
            uuid4(),
        )
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class AssessmentDataQuality:
    """Qualidade e completude dos dados usados."""

    complete: bool = False

    missing_fields: list[str] = field(
        default_factory=list,
    )

    invalid_fields: list[str] = field(
        default_factory=list,
    )

    warnings: list[str] = field(
        default_factory=list,
    )

    confidence: float = 0.0

    def __post_init__(self) -> None:
        self.confidence = max(
            0.0,
            min(
                1.0,
                float(self.confidence),
            ),
        )


@dataclass(slots=True)
class BaseAssessmentResult:
    """Resultado base de qualquer avaliação clínica."""

    assessment_type: str
    status: AssessmentStatus

    risk_level: ClinicalRiskLevel = (
        ClinicalRiskLevel.UNDETERMINED
    )

    summary: str = ""

    alerts: list[AssessmentAlert] = field(
        default_factory=list,
    )

    recommendations: list[
        ClinicalRecommendation
    ] = field(
        default_factory=list,
    )

    data_quality: AssessmentDataQuality = field(
        default_factory=AssessmentDataQuality,
    )

    references: list[str] = field(
        default_factory=list,
    )

    requires_pharmacist_review: bool = True
    requires_prescriber_contact: bool = False
    requires_referral: bool = False
    requires_emergency_referral: bool = False

    calculated_at: datetime = field(
        default_factory=utc_now,
    )

    id: str = field(
        default_factory=lambda: str(
            uuid4(),
        )
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    @property
    def alert_count(self) -> int:
        """Quantidade total de alertas."""

        return len(self.alerts)

    @property
    def recommendation_count(self) -> int:
        """Quantidade total de recomendações."""

        return len(
            self.recommendations,
        )

    @property
    def has_critical_alert(self) -> bool:
        """Indica presença de alerta crítico."""

        return any(
            alert.risk_level
            == ClinicalRiskLevel.CRITICAL
            for alert in self.alerts
        )