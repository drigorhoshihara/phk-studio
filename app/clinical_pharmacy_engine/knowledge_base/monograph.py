"""
PHK Studio
Clinical Pharmacy Engine

Drug Monograph Models.

Modelos normalizados para fichas clínicas de medicamentos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from app.clinical_pharmacy_engine.knowledge_base.models import (
    KnowledgeEvidence,
    KnowledgeReference,
)


def utc_now() -> datetime:
    """Retorna data e hora em UTC."""
    return datetime.now(timezone.utc)


class MonographStatus(str, Enum):
    """Estado de completude da monografia."""

    EMPTY = "empty"
    PARTIAL = "partial"
    COMPLETE = "complete"
    REVIEW_REQUIRED = "review_required"
    VALIDATED = "validated"


@dataclass
class DosageRecommendation:
    """Recomendação posológica normalizada."""

    indication: str
    population: str = "adult"

    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    maximum_dose: str | None = None

    notes: list[str] = field(
        default_factory=list,
    )

    references: list[KnowledgeReference] = field(
        default_factory=list,
    )


@dataclass
class DoseAdjustment:
    """Ajuste posológico por condição clínica."""

    condition: str
    recommendation: str

    threshold: str | None = None
    parameter: str | None = None

    severity: str | None = None

    references: list[KnowledgeReference] = field(
        default_factory=list,
    )


@dataclass
class MonographInteraction:
    """Interação clínica associada ao medicamento."""

    interacting_agent: str
    interaction_type: str

    severity: str | None = None
    mechanism: str | None = None
    clinical_effect: str | None = None
    recommendation: str | None = None

    evidence_level: str | None = None

    references: list[KnowledgeReference] = field(
        default_factory=list,
    )


@dataclass
class AdverseReactionEntry:
    """Reação adversa associada ao medicamento."""

    reaction: str

    frequency: str | None = None
    severity: str | None = None
    seriousness: str | None = None
    onset: str | None = None
    outcome: str | None = None

    source_code: str | None = None

    references: list[KnowledgeReference] = field(
        default_factory=list,
    )


@dataclass
class MonitoringParameter:
    """Parâmetro de monitorização clínica."""

    parameter: str
    recommendation: str

    timing: str | None = None
    target: str | None = None
    rationale: str | None = None


@dataclass
class DrugMonograph:
    """
    Monografia clínica consolidada de um medicamento.

    A monografia combina dados químicos, farmacológicos,
    regulatórios e clínicos provenientes de múltiplas fontes.
    """

    preferred_name: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    generic_name: str | None = None
    brand_names: list[str] = field(
        default_factory=list,
    )

    synonyms: list[str] = field(
        default_factory=list,
    )

    active_ingredients: list[str] = field(
        default_factory=list,
    )

    therapeutic_classes: list[str] = field(
        default_factory=list,
    )

    pharmacological_classes: list[str] = field(
        default_factory=list,
    )

    atc_codes: list[str] = field(
        default_factory=list,
    )

    rxnorm_ids: list[str] = field(
        default_factory=list,
    )

    pubchem_cids: list[str] = field(
        default_factory=list,
    )

    molecular_formula: str | None = None
    molecular_weight: float | str | None = None
    canonical_smiles: str | None = None
    isomeric_smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None

    description: str | None = None
    mechanism_of_action: str | None = None

    pharmacodynamics: list[str] = field(
        default_factory=list,
    )

    pharmacokinetics: dict[str, Any] = field(
        default_factory=dict,
    )

    indications: list[str] = field(
        default_factory=list,
    )

    off_label_uses: list[str] = field(
        default_factory=list,
    )

    dosage_recommendations: list[
        DosageRecommendation
    ] = field(
        default_factory=list,
    )

    renal_adjustments: list[
        DoseAdjustment
    ] = field(
        default_factory=list,
    )

    hepatic_adjustments: list[
        DoseAdjustment
    ] = field(
        default_factory=list,
    )

    contraindications: list[str] = field(
        default_factory=list,
    )

    precautions: list[str] = field(
        default_factory=list,
    )

    warnings: list[str] = field(
        default_factory=list,
    )

    boxed_warnings: list[str] = field(
        default_factory=list,
    )

    interactions: list[
        MonographInteraction
    ] = field(
        default_factory=list,
    )

    adverse_reactions: list[
        AdverseReactionEntry
    ] = field(
        default_factory=list,
    )

    pregnancy_information: list[str] = field(
        default_factory=list,
    )

    lactation_information: list[str] = field(
        default_factory=list,
    )

    pediatric_information: list[str] = field(
        default_factory=list,
    )

    geriatric_information: list[str] = field(
        default_factory=list,
    )

    monitoring_parameters: list[
        MonitoringParameter
    ] = field(
        default_factory=list,
    )

    patient_counseling: list[str] = field(
        default_factory=list,
    )

    storage_information: list[str] = field(
        default_factory=list,
    )

    regulatory_alerts: list[str] = field(
        default_factory=list,
    )

    source_codes: list[str] = field(
        default_factory=list,
    )

    evidences: list[KnowledgeEvidence] = field(
        default_factory=list,
    )

    references: list[KnowledgeReference] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    status: MonographStatus = MonographStatus.EMPTY

    requires_professional_review: bool = True
    pharmacist_validated: bool = False

    created_at: datetime = field(
        default_factory=utc_now,
    )

    updated_at: datetime = field(
        default_factory=utc_now,
    )

    validated_at: datetime | None = None
    validated_by: str | None = None

    @property
    def display_name(self) -> str:
        """Nome preferencial para exibição."""

        return (
            self.generic_name
            or self.preferred_name
        )

    @property
    def evidence_count(self) -> int:
        return len(self.evidences)

    @property
    def source_count(self) -> int:
        return len(set(self.source_codes))