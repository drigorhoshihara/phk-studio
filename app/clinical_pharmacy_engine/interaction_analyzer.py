"""
PHK Studio
Clinical Pharmacy Engine

Orquestrador central de análise de interações.
"""

from __future__ import annotations

from app.clinical_pharmacy_engine.interactions.drug_drug import (
    DrugDrugInteractionDetector,
)
from app.clinical_pharmacy_engine.interactions.models import (
    InteractionAnalysisContext,
    InteractionAnalysisResult,
)
from app.clinical_pharmacy_engine.models import (
    MedicationRecord,
    SeverityLevel,
)


class InteractionAnalyzer:
    """
    Coordena os detectores de interações clínicas.

    Esta etapa inicial implementa interação
    medicamento-medicamento. Os demais detectores
    serão conectados progressivamente.
    """

    def __init__(
        self,
        drug_drug_detector: (
            DrugDrugInteractionDetector | None
        ) = None,
    ) -> None:
        self.drug_drug_detector = (
            drug_drug_detector
            or DrugDrugInteractionDetector()
        )

    def analyze(
        self,
        medications: list[MedicationRecord],
        context: InteractionAnalysisContext,
    ) -> InteractionAnalysisResult:
        interactions = (
            self.drug_drug_detector.analyze(
                medications,
            )
        )

        critical_count = sum(
            interaction.severity
            == SeverityLevel.CRITICAL
            for interaction in interactions
        )

        high_count = sum(
            interaction.severity
            == SeverityLevel.HIGH
            for interaction in interactions
        )

        moderate_count = sum(
            interaction.severity
            == SeverityLevel.MODERATE
            for interaction in interactions
        )

        low_count = sum(
            interaction.severity
            == SeverityLevel.LOW
            for interaction in interactions
        )

        overall_risk = self._overall_risk(
            interactions,
        )

        risk_score = self._risk_score(
            critical_count=critical_count,
            high_count=high_count,
            moderate_count=moderate_count,
            low_count=low_count,
        )

        return InteractionAnalysisResult(
            patient_id=context.patient_id,
            interactions=interactions,
            total_interactions=len(interactions),
            critical_count=critical_count,
            high_count=high_count,
            moderate_count=moderate_count,
            low_count=low_count,
            overall_risk=overall_risk,
            risk_score=risk_score,
            requires_urgent_review=(
                critical_count > 0
            ),
            requires_prescriber_contact=(
                critical_count > 0
                or high_count > 0
            ),
            requires_pharmacist_review=True,
            warnings=self._warnings(
                interactions=interactions,
                context=context,
            ),
        )

    @staticmethod
    def _overall_risk(
        interactions: list,
    ) -> SeverityLevel:
        order = {
            SeverityLevel.INFORMATIONAL: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MODERATE: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4,
        }

        highest = SeverityLevel.INFORMATIONAL

        for interaction in interactions:
            if (
                order[interaction.severity]
                > order[highest]
            ):
                highest = interaction.severity

        return highest

    @staticmethod
    def _risk_score(
        critical_count: int,
        high_count: int,
        moderate_count: int,
        low_count: int,
    ) -> float:
        raw_score = (
            critical_count * 35
            + high_count * 20
            + moderate_count * 8
            + low_count * 2
        )

        return float(
            min(raw_score, 100)
        )

    @staticmethod
    def _warnings(
        interactions: list,
        context: InteractionAnalysisContext,
    ) -> list[str]:
        warnings: list[str] = []

        if context.age is not None and context.age >= 65:
            warnings.append(
                "Paciente idoso: revisar risco de quedas, "
                "sangramento, sedação e carga anticolinérgica."
            )

        if context.pregnancy:
            warnings.append(
                "Gestação informada: validar segurança fetal "
                "e protocolos obstétricos."
            )

        if context.breastfeeding:
            warnings.append(
                "Lactação informada: avaliar exposição do lactente."
            )

        if (
            context.renal_function is not None
            and context.renal_function < 60
        ):
            warnings.append(
                "Função renal reduzida: revisar doses, "
                "intervalos e medicamentos nefrotóxicos."
            )

        if not interactions:
            warnings.append(
                "Nenhuma interação cadastrada foi identificada. "
                "Isso não exclui interações ausentes da base local."
            )

        return warnings