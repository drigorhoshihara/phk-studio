"""
Modelos internos do mecanismo de interações clínicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

from app.clinical_pharmacy_engine.models import (
    InteractionType,
    SeverityLevel,
)


class InteractionEvidenceLevel(str, Enum):
    """Força da evidência da interação."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    NOT_ASSESSED = "not_assessed"


class InteractionManagement(str, Enum):
    """Conduta clínica geral sugerida."""

    INFORMATION_ONLY = "information_only"
    MONITOR = "monitor"
    ADJUST_DOSE = "adjust_dose"
    SEPARATE_ADMINISTRATION = "separate_administration"
    CONSIDER_ALTERNATIVE = "consider_alternative"
    AVOID_COMBINATION = "avoid_combination"
    URGENT_REVIEW = "urgent_review"


@dataclass
class InteractionRule:
    """
    Regra declarativa de interação.

    As regras são preliminares e não substituem
    bases clínicas licenciadas ou revisão profissional.
    """

    agent_a: str
    agent_b: str

    interaction_type: InteractionType
    severity: SeverityLevel

    mechanism: str
    clinical_effect: str
    recommendation: str

    evidence_level: InteractionEvidenceLevel = (
        InteractionEvidenceLevel.NOT_ASSESSED
    )

    management: InteractionManagement = (
        InteractionManagement.MONITOR
    )

    monitoring_parameters: list[str] = field(
        default_factory=list,
    )

    onset: Optional[str] = None
    documentation: Optional[str] = None

    contraindicated: bool = False
    requires_pharmacist_review: bool = True

    references: list[str] = field(
        default_factory=list,
    )

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )


@dataclass
class InteractionAnalysisContext:
    """Contexto adicional do paciente."""

    patient_id: str

    age: Optional[int] = None

    pregnancy: bool = False
    breastfeeding: bool = False

    renal_function: Optional[float] = None
    hepatic_function: Optional[str] = None

    conditions: list[str] = field(
        default_factory=list,
    )

    foods: list[str] = field(
        default_factory=list,
    )

    alcohol_use: bool = False

    herbal_products: list[str] = field(
        default_factory=list,
    )

    supplements: list[str] = field(
        default_factory=list,
    )

    laboratory_tests: list[str] = field(
        default_factory=list,
    )


@dataclass
class InteractionAnalysisResult:
    """Resposta consolidada do analisador."""

    patient_id: str

    interactions: list = field(
        default_factory=list,
    )

    total_interactions: int = 0
    critical_count: int = 0
    high_count: int = 0
    moderate_count: int = 0
    low_count: int = 0

    overall_risk: SeverityLevel = (
        SeverityLevel.INFORMATIONAL
    )

    risk_score: float = 0.0

    requires_urgent_review: bool = False
    requires_prescriber_contact: bool = False
    requires_pharmacist_review: bool = True

    warnings: list[str] = field(
        default_factory=list,
    )