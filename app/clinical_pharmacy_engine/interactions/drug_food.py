"""
PHK Studio
Clinical Pharmacy Engine

Detector de interações medicamento-alimento.
"""

from __future__ import annotations

from app.clinical_pharmacy_engine.interactions.models import (
    InteractionRule,
)
from app.clinical_pharmacy_engine.interactions.rules import (
    DRUG_FOOD_RULES,
)
from app.clinical_pharmacy_engine.models import (
    DrugInteraction,
    MedicationRecord,
)


class DrugFoodInteractionDetector:
    """
    Detecta interações entre medicamentos e alimentos
    informados no contexto clínico.
    """

    def __init__(
        self,
        rules: list[InteractionRule] | None = None,
    ) -> None:
        self.rules = rules or DRUG_FOOD_RULES

    def analyze(
        self,
        medications: list[MedicationRecord],
        foods: list[str],
    ) -> list[DrugInteraction]:
        interactions: list[DrugInteraction] = []

        normalized_foods = {
            self._normalize(food)
            for food in foods
            if food and food.strip()
        }

        if not normalized_foods:
            return interactions

        for medication in medications:
            medication_terms = self._medication_terms(
                medication,
            )

            for rule in self.rules:
                agent_a = self._normalize(
                    rule.agent_a,
                )
                agent_b = self._normalize(
                    rule.agent_b,
                )

                if (
                    agent_a not in medication_terms
                    or agent_b not in normalized_foods
                ):
                    continue

                interactions.append(
                    DrugInteraction(
                        medication_a=medication.name,
                        medication_b=rule.agent_b,
                        interaction_type=rule.interaction_type,
                        severity=rule.severity,
                        mechanism=rule.mechanism,
                        clinical_effect=rule.clinical_effect,
                        recommendation=rule.recommendation,
                        evidence_level=(
                            rule.evidence_level.value
                        ),
                        confidence=self._confidence(
                            rule,
                        ),
                        monitoring_parameters=list(
                            rule.monitoring_parameters,
                        ),
                        requires_pharmacist_review=True,
                    )
                )

        return interactions

    def _medication_terms(
        self,
        medication: MedicationRecord,
    ) -> set[str]:
        terms = {
            self._normalize(
                medication.name,
            ),
        }

        if medication.active_ingredient:
            terms.add(
                self._normalize(
                    medication.active_ingredient,
                )
            )

        return {
            term
            for term in terms
            if term
        }

    @staticmethod
    def _normalize(
        value: str,
    ) -> str:
        return " ".join(
            value.casefold().strip().split()
        )

    @staticmethod
    def _confidence(
        rule: InteractionRule,
    ) -> float:
        mapping = {
            "very_high": 0.98,
            "high": 0.90,
            "moderate": 0.75,
            "low": 0.55,
            "very_low": 0.35,
            "not_assessed": 0.20,
        }

        return mapping.get(
            rule.evidence_level.value,
            0.20,
        )