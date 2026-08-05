"""
PHK Studio
Clinical Pharmacy Engine

Modelos compartilhados da Clinical Knowledge Base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Retorna data e hora UTC com timezone explícito."""
    return datetime.now(timezone.utc)


class KnowledgeDomain(str, Enum):
    """Domínio clínico coberto por uma evidência."""

    DRUG_MONOGRAPH = "drug_monograph"
    INDICATION = "indication"
    CONTRAINDICATION = "contraindication"
    DOSAGE = "dosage"
    RENAL_ADJUSTMENT = "renal_adjustment"
    HEPATIC_ADJUSTMENT = "hepatic_adjustment"
    DRUG_INTERACTION = "drug_interaction"
    FOOD_INTERACTION = "food_interaction"
    ALCOHOL_INTERACTION = "alcohol_interaction"
    HERBAL_INTERACTION = "herbal_interaction"
    SUPPLEMENT_INTERACTION = "supplement_interaction"
    ADVERSE_REACTION = "adverse_reaction"
    PHARMACOVIGILANCE = "pharmacovigilance"
    TOXICOLOGY = "toxicology"
    PREGNANCY = "pregnancy"
    LACTATION = "lactation"
    PHARMACOKINETICS = "pharmacokinetics"
    PHARMACODYNAMICS = "pharmacodynamics"
    PHARMACOGENOMICS = "pharmacogenomics"
    LABORATORY_INTERFERENCE = "laboratory_interference"
    CLINICAL_GUIDELINE = "clinical_guideline"
    REGULATORY_ALERT = "regulatory_alert"
    CLINICAL_TRIAL = "clinical_trial"
    SCIENTIFIC_ARTICLE = "scientific_article"
    OTHER = "other"


class EvidenceStrength(str, Enum):
    """Força ou confiabilidade geral da evidência."""

    VERY_HIGH = "very_high"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    UNASSESSED = "unassessed"


class SourceAccessType(str, Enum):
    """Forma de acesso à fonte."""

    PUBLIC_API = "public_api"
    PUBLIC_WEB = "public_web"
    OPEN_DATASET = "open_dataset"
    LICENSED_API = "licensed_api"
    LOCAL_DATASET = "local_dataset"
    MANUAL_IMPORT = "manual_import"
    NOT_CONFIGURED = "not_configured"


class SourceStatus(str, Enum):
    """Estado operacional de uma fonte."""

    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"
    REQUIRES_CREDENTIALS = "requires_credentials"


@dataclass
class KnowledgeReference:
    """Referência associada a uma evidência clínica."""

    title: str

    url: str | None = None
    doi: str | None = None
    pmid: str | None = None

    authors: list[str] = field(
        default_factory=list,
    )

    publication_year: int | None = None
    publisher: str | None = None
    citation: str | None = None


@dataclass
class KnowledgeEvidence:
    """
    Unidade normalizada de conhecimento clínico.

    Cada conector converte sua resposta nativa para este formato.
    """

    source_code: str
    domain: KnowledgeDomain
    subject: str
    title: str
    summary: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    related_agents: list[str] = field(
        default_factory=list,
    )

    recommendation: str | None = None
    clinical_effect: str | None = None
    mechanism: str | None = None

    severity: str | None = None
    evidence_strength: EvidenceStrength = (
        EvidenceStrength.UNASSESSED
    )

    confidence: float = 0.0

    country: str | None = None
    language: str | None = None

    published_at: datetime | None = None
    updated_at: datetime | None = None

    references: list[KnowledgeReference] = field(
        default_factory=list,
    )

    raw_identifiers: dict[str, str] = field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    retrieved_at: datetime = field(
        default_factory=utc_now,
    )

    requires_professional_review: bool = True


@dataclass
class KnowledgeQuery:
    """Consulta padronizada à base de conhecimento."""

    term: str

    domains: list[KnowledgeDomain] = field(
        default_factory=list,
    )

    source_codes: list[str] = field(
        default_factory=list,
    )

    related_agents: list[str] = field(
        default_factory=list,
    )

    language: str | None = None
    country: str | None = None

    limit_per_source: int = 20
    include_raw_data: bool = False

    def normalized_term(self) -> str:
        return " ".join(
            self.term.strip().casefold().split()
        )


@dataclass
class SourceExecutionResult:
    """Resultado individual de uma fonte."""

    source_code: str
    success: bool

    evidences: list[KnowledgeEvidence] = field(
        default_factory=list,
    )

    error: str | None = None
    elapsed_seconds: float = 0.0
    from_cache: bool = False


@dataclass
class KnowledgeSearchResult:
    """Resultado consolidado do agregador."""

    query: KnowledgeQuery

    evidences: list[KnowledgeEvidence] = field(
        default_factory=list,
    )

    successful_sources: list[str] = field(
        default_factory=list,
    )

    failed_sources: dict[str, str] = field(
        default_factory=dict,
    )

    unavailable_sources: list[str] = field(
        default_factory=list,
    )

    total_before_deduplication: int = 0
    total_results: int = 0

    generated_at: datetime = field(
        default_factory=utc_now,
    )

    warnings: list[str] = field(
        default_factory=list,
    )


@dataclass
class KnowledgeSourceDescriptor:
    """Metadados públicos de uma fonte."""

    code: str
    name: str

    access_type: SourceAccessType
    status: SourceStatus

    description: str | None = None
    base_url: str | None = None

    supported_domains: list[KnowledgeDomain] = field(
        default_factory=list,
    )

    requires_credentials: bool = False
    license_notes: str | None = None

    country: str | None = None
    language: str | None = None

    version: str | None = None