"""
PHK Studio
Clinical Pharmacy Engine

Interaction Engine

Exporta os principais componentes do mecanismo de
interações medicamentosas.
"""

from .drug_drug import DrugDrugInteractionDetector
from .drug_food import DrugFoodInteractionDetector

from .models import (
    InteractionRule,
    InteractionAnalysisContext,
    InteractionAnalysisResult,
    InteractionEvidenceLevel,
    InteractionManagement,
)

__all__ = [
    # Detectores
    "DrugDrugInteractionDetector",
    "DrugFoodInteractionDetector",

    # Modelos
    "InteractionRule",
    "InteractionAnalysisContext",
    "InteractionAnalysisResult",

    # Enums
    "InteractionEvidenceLevel",
    "InteractionManagement",
]