"""
PHK Studio
Clinical Pharmacy Engine

Public API da Clinical Knowledge Base.

Centraliza as exportações públicas dos componentes de:

- agregação de conhecimento;
- cache clínico;
- registro de fontes;
- modelos de evidência;
- conectores externos;
- monografias clínicas;
- construção automática de monografias.
"""

from app.clinical_pharmacy_engine.knowledge_base.aggregator import (
    ClinicalKnowledgeAggregator,
)
from app.clinical_pharmacy_engine.knowledge_base.cache import (
    ClinicalKnowledgeCache,
)
from app.clinical_pharmacy_engine.knowledge_base.dailymed import (
    DailyMedKnowledgeSource,
)
from app.clinical_pharmacy_engine.knowledge_base.models import (
    EvidenceStrength,
    KnowledgeDomain,
    KnowledgeEvidence,
    KnowledgeQuery,
    KnowledgeReference,
    KnowledgeSearchResult,
    KnowledgeSourceDescriptor,
    SourceAccessType,
    SourceExecutionResult,
    SourceStatus,
)
from app.clinical_pharmacy_engine.knowledge_base.monograph import (
    AdverseReactionEntry,
    DosageRecommendation,
    DoseAdjustment,
    DrugMonograph,
    MonographInteraction,
    MonographStatus,
    MonitoringParameter,
)
from app.clinical_pharmacy_engine.knowledge_base.monograph_builder import (
    DrugMonographBuilder,
)
from app.clinical_pharmacy_engine.knowledge_base.openfda import (
    OpenFDAKnowledgeSource,
)
from app.clinical_pharmacy_engine.knowledge_base.pubchem import (
    PubChemKnowledgeSource,
)
from app.clinical_pharmacy_engine.knowledge_base.registry import (
    ClinicalKnowledgeRegistry,
)
from app.clinical_pharmacy_engine.knowledge_base.rxnorm import (
    RxNormKnowledgeSource,
)
from app.clinical_pharmacy_engine.knowledge_base.source import (
    ClinicalKnowledgeSource,
)

__all__ = [
    # Infraestrutura
    "ClinicalKnowledgeAggregator",
    "ClinicalKnowledgeCache",
    "ClinicalKnowledgeRegistry",
    "ClinicalKnowledgeSource",

    # Modelos de conhecimento
    "EvidenceStrength",
    "KnowledgeDomain",
    "KnowledgeEvidence",
    "KnowledgeQuery",
    "KnowledgeReference",
    "KnowledgeSearchResult",
    "KnowledgeSourceDescriptor",
    "SourceAccessType",
    "SourceExecutionResult",
    "SourceStatus",

    # Fontes externas
    "DailyMedKnowledgeSource",
    "OpenFDAKnowledgeSource",
    "PubChemKnowledgeSource",
    "RxNormKnowledgeSource",

    # Monografias clínicas
    "AdverseReactionEntry",
    "DosageRecommendation",
    "DoseAdjustment",
    "DrugMonograph",
    "DrugMonographBuilder",
    "MonographInteraction",
    "MonographStatus",
    "MonitoringParameter",
]