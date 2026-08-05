"""
Detector de interação medicamento-medicamento.
"""

from __future__ import annotations

from itertools import combinations

from app.clinical_pharmacy_engine.interactions.models import (
    InteractionRule,
)
from app.clinical_pharmacy_engine.interactions.rules import (
    DRUG_DRUG_RULES,
)
from app.clinical_pharmacy_engine.models import (
    DrugInteraction,
    MedicationRecord,
)


class DrugDrugInteractionDetector:
    """Analisa pares de medicamentos."""

    def __init__(
        self,
        rules: list[InteractionRule] | None = None,
    ) -> None:
        self.rules = rules or DRUG_DRUG_RULES

    def analyze(
        self,
        medications: list[MedicationRecord],
    ) -> list[DrugInteraction]:
        interactions: list[DrugInteraction] = []

        normalized = [
            (
                medication,
                self._medication_terms(medication),
            )
            for medication in medications
        ]

        for first, second in combinations(normalized, 2):
            medication_a, terms_a = first
            medication_b, terms_b = second

            rule = self._find_rule(
                terms_a=terms_a,
                terms_b=terms_b,
            )

            if rule is None:
                continue

            interactions.append(
                DrugInteraction(
                    medication_a=medication_a.name,
                    medication_b=medication_b.name,
                    interaction_type=rule.interaction_type,
                    severity=rule.severity,
                    mechanism=rule.mechanism,
                    clinical_effect=rule.clinical_effect,
                    recommendation=rule.recommendation,
                    evidence_level=rule.evidence_level.value,
                    confidence=self._confidence(rule),
                    monitoring_parameters=list(
                        rule.monitoring_parameters
                    ),
                    requires_pharmacist_review=True,
                )
            )

        return interactions

    def _find_rule(
        self,
        terms_a: set[str],
        terms_b: set[str],
    ) -> InteractionRule | None:
        for rule in self.rules:
            first = self._normalize(rule.agent_a)
            second = self._normalize(rule.agent_b)

            direct_match = (
                first in terms_a
                and second in terms_b
            )

            reverse_match = (
                first in terms_b
                and second in terms_a
            )

            if direct_match or reverse_match:
                return rule

        return None

    def _medication_terms(
        self,
        medication: MedicationRecord,
    ) -> set[str]:
        terms = {
            self._normalize(medication.name),
        }

        if medication.active_ingredient:
            terms.add(
                self._normalize(
                    medication.active_ingredient
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