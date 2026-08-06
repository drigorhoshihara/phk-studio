"""
PHK Studio
Clinical Pharmacy Engine

Hypertension Assessment Engine.

Responsabilidades:

- validar medições de pressão arterial;
- calcular médias por contexto;
- classificar pressão arterial por diretriz;
- identificar hipotensão;
- identificar hipertensão sistólica isolada;
- sinalizar pressão arterial severamente elevada;
- diferenciar elevação assintomática de possível emergência;
- sugerir confirmação diagnóstica e revisão clínica.

O módulo fornece suporte à decisão clínica.
Uma leitura isolada não estabelece diagnóstico definitivo.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import isfinite
from typing import Iterable

from app.clinical_pharmacy_engine.assessment.cardiovascular.models import (
    BloodPressureClassification,
    BloodPressureContext,
    BloodPressureMeasurement,
    HypertensionAssessmentResult,
    HypertensionGuideline,
    HypertensionPhenotype,
    HypertensiveEventType,
)


# ============================================================
# Configurações
# ============================================================


@dataclass(slots=True)
class HypertensionAssessmentConfig:
    """Configurações operacionais do motor."""

    guideline: HypertensionGuideline = (
        HypertensionGuideline.AHA_ACC
    )

    preferred_context: BloodPressureContext = (
        BloodPressureContext.OFFICE
    )

    minimum_measurements_for_confirmation: int = 2

    hypotension_systolic_threshold: float = 90.0
    hypotension_diastolic_threshold: float = 60.0

    severe_systolic_threshold: float = 180.0
    severe_diastolic_threshold: float = 120.0

    isolated_systolic_minimum: float = 140.0
    isolated_systolic_diastolic_maximum: float = 89.0

    target_systolic_mm_hg: float = 130.0
    target_diastolic_mm_hg: float = 80.0

    emergency_symptoms: set[str] = field(
        default_factory=lambda: {
            "chest pain",
            "dor torácica",
            "shortness of breath",
            "dispneia",
            "back pain",
            "dor nas costas",
            "weakness",
            "fraqueza",
            "numbness",
            "dormência",
            "vision change",
            "alteração visual",
            "difficulty speaking",
            "dificuldade para falar",
            "confusion",
            "confusão",
            "syncope",
            "síncope",
            "seizure",
            "convulsão",
            "focal neurological deficit",
            "déficit neurológico focal",
        }
    )


@dataclass(slots=True)
class ValidatedBloodPressure:
    """Medição validada para análise."""

    measurement: BloodPressureMeasurement
    valid: bool
    warnings: list[str] = field(
        default_factory=list,
    )


# ============================================================
# Motor principal
# ============================================================


class HypertensionAssessmentEngine:
    """Motor de avaliação da pressão arterial."""

    def __init__(
        self,
        config: HypertensionAssessmentConfig | None = None,
    ) -> None:
        self.config = (
            config
            or HypertensionAssessmentConfig()
        )

    def assess(
        self,
        measurements: list[BloodPressureMeasurement],
        *,
        treated_hypertension: bool = False,
        hypertension_history: bool = False,
        symptoms: Iterable[str] | None = None,
        possible_target_organ_damage: bool = False,
        office_measurements: (
            list[BloodPressureMeasurement] | None
        ) = None,
        home_measurements: (
            list[BloodPressureMeasurement] | None
        ) = None,
    ) -> HypertensionAssessmentResult:
        """
        Executa avaliação pressórica integrada.

        O diagnóstico deve ser confirmado por medições
        repetidas e, quando indicado, por monitorização
        fora do consultório.
        """

        warnings: list[str] = []

        validated = [
            self.validate_measurement(measurement)
            for measurement in measurements
        ]

        valid_measurements = [
            item.measurement
            for item in validated
            if item.valid
        ]

        for item in validated:
            warnings.extend(item.warnings)

        if not valid_measurements:
            return HypertensionAssessmentResult(
                guideline=self.config.guideline,
                valid=False,
                measurements_used=0,
                requires_confirmation=True,
                warnings=self._unique_strings(
                    warnings
                    + [
                        "Nenhuma medição válida foi "
                        "disponibilizada para avaliação."
                    ]
                ),
            )

        selected = self._select_measurements(
            valid_measurements,
        )

        average_systolic = self._average(
            measurement.systolic_mm_hg
            for measurement in selected
        )

        average_diastolic = self._average(
            measurement.diastolic_mm_hg
            for measurement in selected
        )

        classification = self.classify(
            systolic_mm_hg=average_systolic,
            diastolic_mm_hg=average_diastolic,
            guideline=self.config.guideline,
        )

        acute_event = self._classify_acute_event(
            systolic_mm_hg=average_systolic,
            diastolic_mm_hg=average_diastolic,
            symptoms=list(symptoms or []),
            possible_target_organ_damage=(
                possible_target_organ_damage
            ),
        )

        phenotype = self._determine_phenotype(
            average_systolic=average_systolic,
            average_diastolic=average_diastolic,
            classification=classification,
            treated_hypertension=treated_hypertension,
            hypertension_history=hypertension_history,
            office_measurements=office_measurements,
            home_measurements=home_measurements,
        )

        requires_confirmation = (
            len(selected)
            < self.config.minimum_measurements_for_confirmation
        )

        if requires_confirmation:
            warnings.append(
                "Quantidade de medições insuficiente para "
                "confirmação diagnóstica."
            )

        if any(
            measurement.validated_device is False
            for measurement in selected
        ):
            warnings.append(
                "Uma ou mais medições foram realizadas "
                "com dispositivo não confirmado como validado."
            )

        if any(
            measurement.rested_before_measurement is False
            for measurement in selected
        ):
            warnings.append(
                "Uma ou mais medições podem ter sido obtidas "
                "sem repouso prévio adequado."
            )

        if (
            classification
            == BloodPressureClassification.ISOLATED_SYSTOLIC_HYPERTENSION
        ):
            warnings.append(
                "Foi identificado padrão compatível com "
                "hipertensão sistólica isolada."
            )

        if (
            acute_event
            == HypertensiveEventType.SEVERE_ASYMPTOMATIC
        ):
            warnings.append(
                "Pressão severamente elevada sem evidência "
                "informada de dano agudo. Repetir a medição "
                "com técnica adequada e realizar avaliação "
                "clínica oportuna."
            )

        if (
            acute_event
            == HypertensiveEventType.POSSIBLE_EMERGENCY
        ):
            warnings.append(
                "Pressão severamente elevada associada a "
                "sintomas ou possível dano agudo de órgão-alvo."
            )

        return HypertensionAssessmentResult(
            classification=classification,
            guideline=self.config.guideline,
            phenotype=phenotype,
            acute_event_type=acute_event,
            average_systolic_mm_hg=round(
                average_systolic,
                2,
            ),
            average_diastolic_mm_hg=round(
                average_diastolic,
                2,
            ),
            target_systolic_mm_hg=(
                self.config.target_systolic_mm_hg
            ),
            target_diastolic_mm_hg=(
                self.config.target_diastolic_mm_hg
            ),
            valid=True,
            measurements_used=len(selected),
            possible_target_organ_damage=(
                possible_target_organ_damage
            ),
            requires_confirmation=requires_confirmation,
            requires_immediate_evaluation=(
                acute_event
                == HypertensiveEventType.POSSIBLE_EMERGENCY
            ),
            warnings=self._unique_strings(warnings),
            metadata={
                "preferred_context": (
                    self.config.preferred_context.value
                ),
                "total_measurements_received": len(
                    measurements
                ),
                "valid_measurements": len(
                    valid_measurements
                ),
                "selected_measurements": len(selected),
                "treated_hypertension": (
                    treated_hypertension
                ),
                "hypertension_history": (
                    hypertension_history
                ),
            },
        )

    # ========================================================
    # Validação
    # ========================================================

    def validate_measurement(
        self,
        measurement: BloodPressureMeasurement,
    ) -> ValidatedBloodPressure:
        """Valida plausibilidade de uma medição."""

        warnings: list[str] = []

        systolic = measurement.systolic_mm_hg
        diastolic = measurement.diastolic_mm_hg

        if not self._valid_number(systolic):
            return ValidatedBloodPressure(
                measurement=measurement,
                valid=False,
                warnings=[
                    "Pressão sistólica inválida.",
                ],
            )

        if not self._valid_number(diastolic):
            return ValidatedBloodPressure(
                measurement=measurement,
                valid=False,
                warnings=[
                    "Pressão diastólica inválida.",
                ],
            )

        if systolic <= 0 or systolic > 300:
            return ValidatedBloodPressure(
                measurement=measurement,
                valid=False,
                warnings=[
                    "Pressão sistólica fora do intervalo "
                    "fisiologicamente plausível."
                ],
            )

        if diastolic <= 0 or diastolic > 200:
            return ValidatedBloodPressure(
                measurement=measurement,
                valid=False,
                warnings=[
                    "Pressão diastólica fora do intervalo "
                    "fisiologicamente plausível."
                ],
            )

        if systolic <= diastolic:
            return ValidatedBloodPressure(
                measurement=measurement,
                valid=False,
                warnings=[
                    "Pressão sistólica deve ser maior que "
                    "a pressão diastólica."
                ],
            )

        pulse_pressure = systolic - diastolic

        if pulse_pressure < 10:
            warnings.append(
                "Pressão de pulso muito estreita; confirmar "
                "técnica, equipamento e contexto clínico."
            )

        if pulse_pressure > 100:
            warnings.append(
                "Pressão de pulso muito ampla; confirmar "
                "medição e contexto clínico."
            )

        if measurement.heart_rate_bpm is not None:
            if (
                not self._valid_number(
                    measurement.heart_rate_bpm
                )
                or measurement.heart_rate_bpm <= 0
                or measurement.heart_rate_bpm > 300
            ):
                warnings.append(
                    "Frequência cardíaca informada é inválida."
                )

        return ValidatedBloodPressure(
            measurement=measurement,
            valid=True,
            warnings=warnings,
        )

    # ========================================================
    # Classificações
    # ========================================================

    def classify(
        self,
        *,
        systolic_mm_hg: float,
        diastolic_mm_hg: float,
        guideline: HypertensionGuideline,
    ) -> BloodPressureClassification:
        """Classifica conforme a diretriz selecionada."""

        if (
            systolic_mm_hg
            < self.config.hypotension_systolic_threshold
            or diastolic_mm_hg
            < self.config.hypotension_diastolic_threshold
        ):
            return BloodPressureClassification.HYPOTENSION

        if (
            systolic_mm_hg
            > self.config.severe_systolic_threshold
            or diastolic_mm_hg
            > self.config.severe_diastolic_threshold
        ):
            return (
                BloodPressureClassification.HYPERTENSIVE_CRISIS
            )

        if (
            systolic_mm_hg
            >= self.config.isolated_systolic_minimum
            and diastolic_mm_hg
            <= self.config.isolated_systolic_diastolic_maximum
        ):
            return (
                BloodPressureClassification
                .ISOLATED_SYSTOLIC_HYPERTENSION
            )

        if guideline == HypertensionGuideline.AHA_ACC:
            return self._classify_aha_acc(
                systolic_mm_hg,
                diastolic_mm_hg,
            )

        if guideline == HypertensionGuideline.ESC_ESH:
            return self._classify_esc_2024(
                systolic_mm_hg,
                diastolic_mm_hg,
            )

        if guideline == HypertensionGuideline.SBC:
            return self._classify_sbc(
                systolic_mm_hg,
                diastolic_mm_hg,
            )

        if guideline == HypertensionGuideline.WHO:
            return self._classify_who(
                systolic_mm_hg,
                diastolic_mm_hg,
            )

        return BloodPressureClassification.UNDETERMINED

    @staticmethod
    def _classify_aha_acc(
        systolic: float,
        diastolic: float,
    ) -> BloodPressureClassification:
        """Classificação AHA/ACC 2025."""

        if systolic < 120 and diastolic < 80:
            return BloodPressureClassification.NORMAL

        if 120 <= systolic <= 129 and diastolic < 80:
            return BloodPressureClassification.ELEVATED

        if (
            130 <= systolic <= 139
            or 80 <= diastolic <= 89
        ):
            return (
                BloodPressureClassification
                .HYPERTENSION_STAGE_1
            )

        if systolic >= 140 or diastolic >= 90:
            return (
                BloodPressureClassification
                .HYPERTENSION_STAGE_2
            )

        return BloodPressureClassification.UNDETERMINED

    @staticmethod
    def _classify_esc_2024(
        systolic: float,
        diastolic: float,
    ) -> BloodPressureClassification:
        """Classificação simplificada ESC 2024."""

        if systolic < 120 and diastolic < 70:
            return BloodPressureClassification.NORMAL

        if (
            120 <= systolic <= 139
            or 70 <= diastolic <= 89
        ):
            return BloodPressureClassification.ELEVATED

        if systolic >= 140 or diastolic >= 90:
            return (
                BloodPressureClassification
                .HYPERTENSION_STAGE_1
            )

        return BloodPressureClassification.UNDETERMINED

    @staticmethod
    def _classify_sbc(
        systolic: float,
        diastolic: float,
    ) -> BloodPressureClassification:
        """
        Classificação brasileira convencional.

        A versão exata deverá permanecer associada aos
        metadados do protocolo institucional.
        """

        if systolic < 120 and diastolic < 80:
            return BloodPressureClassification.OPTIMAL

        if systolic < 130 and diastolic < 85:
            return BloodPressureClassification.NORMAL

        if systolic < 140 and diastolic < 90:
            return BloodPressureClassification.HIGH_NORMAL

        if systolic < 160 and diastolic < 100:
            return (
                BloodPressureClassification
                .HYPERTENSION_STAGE_1
            )

        if systolic < 180 and diastolic < 110:
            return (
                BloodPressureClassification
                .HYPERTENSION_STAGE_2
            )

        return (
            BloodPressureClassification
            .HYPERTENSION_STAGE_3
        )

    @staticmethod
    def _classify_who(
        systolic: float,
        diastolic: float,
    ) -> BloodPressureClassification:
        """Classificação operacional baseada em ≥140/90."""

        if systolic < 140 and diastolic < 90:
            return BloodPressureClassification.NORMAL

        return (
            BloodPressureClassification
            .HYPERTENSION_STAGE_1
        )

    # ========================================================
    # Eventos agudos
    # ========================================================

    def _classify_acute_event(
        self,
        *,
        systolic_mm_hg: float,
        diastolic_mm_hg: float,
        symptoms: list[str],
        possible_target_organ_damage: bool,
    ) -> HypertensiveEventType:
        """Classifica elevação pressórica aguda."""

        severe = (
            systolic_mm_hg
            > self.config.severe_systolic_threshold
            or diastolic_mm_hg
            > self.config.severe_diastolic_threshold
        )

        if not severe:
            return HypertensiveEventType.NONE

        normalized_symptoms = {
            self._normalize_text(symptom)
            for symptom in symptoms
            if symptom
        }

        emergency_symptoms = {
            self._normalize_text(symptom)
            for symptom in self.config.emergency_symptoms
        }

        concerning_symptom = bool(
            normalized_symptoms
            & emergency_symptoms
        )

        if (
            possible_target_organ_damage
            or concerning_symptom
        ):
            return (
                HypertensiveEventType.POSSIBLE_EMERGENCY
            )

        return (
            HypertensiveEventType.SEVERE_ASYMPTOMATIC
        )

    # ========================================================
    # Fenótipos
    # ========================================================

    def _determine_phenotype(
        self,
        *,
        average_systolic: float,
        average_diastolic: float,
        classification: BloodPressureClassification,
        treated_hypertension: bool,
        hypertension_history: bool,
        office_measurements: (
            list[BloodPressureMeasurement] | None
        ),
        home_measurements: (
            list[BloodPressureMeasurement] | None
        ),
    ) -> HypertensionPhenotype:
        """Determina fenótipo pressórico simplificado."""

        if (
            average_systolic
            >= self.config.isolated_systolic_minimum
            and average_diastolic
            <= self.config.isolated_systolic_diastolic_maximum
        ):
            return HypertensionPhenotype.ISOLATED_SYSTOLIC

        if office_measurements and home_measurements:
            office_average = self._measurement_average(
                office_measurements
            )

            home_average = self._measurement_average(
                home_measurements
            )

            if (
                office_average is not None
                and home_average is not None
            ):
                office_high = (
                    office_average[0] >= 140
                    or office_average[1] >= 90
                )

                home_high = (
                    home_average[0] >= 135
                    or home_average[1] >= 85
                )

                if office_high and not home_high:
                    return HypertensionPhenotype.WHITE_COAT

                if not office_high and home_high:
                    return HypertensionPhenotype.MASKED

                if office_high and home_high:
                    return HypertensionPhenotype.SUSTAINED

        hypertensive_classes = {
            BloodPressureClassification
            .HYPERTENSION_STAGE_1,
            BloodPressureClassification
            .HYPERTENSION_STAGE_2,
            BloodPressureClassification
            .HYPERTENSION_STAGE_3,
            BloodPressureClassification
            .ISOLATED_SYSTOLIC_HYPERTENSION,
            BloodPressureClassification
            .HYPERTENSIVE_CRISIS,
        }

        if classification in hypertensive_classes:
            return HypertensionPhenotype.SUSTAINED

        if treated_hypertension or hypertension_history:
            return HypertensionPhenotype.UNDETERMINED

        return HypertensionPhenotype.NONE

    # ========================================================
    # Utilidades
    # ========================================================

    def _select_measurements(
        self,
        measurements: list[BloodPressureMeasurement],
    ) -> list[BloodPressureMeasurement]:
        """Prioriza medições do contexto configurado."""

        by_context: dict[
            BloodPressureContext,
            list[BloodPressureMeasurement],
        ] = defaultdict(list)

        for measurement in measurements:
            by_context[measurement.context].append(
                measurement
            )

        preferred = by_context.get(
            self.config.preferred_context,
            [],
        )

        if preferred:
            return preferred

        return measurements

    def _measurement_average(
        self,
        measurements: list[BloodPressureMeasurement],
    ) -> tuple[float, float] | None:
        """Calcula média de uma coleção válida."""

        valid = [
            item.measurement
            for item in (
                self.validate_measurement(measurement)
                for measurement in measurements
            )
            if item.valid
        ]

        if not valid:
            return None

        systolic = self._average(
            measurement.systolic_mm_hg
            for measurement in valid
        )

        diastolic = self._average(
            measurement.diastolic_mm_hg
            for measurement in valid
        )

        return systolic, diastolic

    @staticmethod
    def _average(
        values: Iterable[float],
    ) -> float:
        """Calcula média aritmética."""

        data = list(values)

        if not data:
            raise ValueError(
                "Não existem valores para cálculo da média."
            )

        return sum(data) / len(data)

    @staticmethod
    def _valid_number(
        value: object,
    ) -> bool:
        """Verifica se o valor é numérico e finito."""

        try:
            return isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        """Normaliza texto para comparação."""

        return " ".join(
            value.strip().casefold().split()
        )

    @classmethod
    def _unique_strings(
        cls,
        values: list[str],
    ) -> list[str]:
        """Remove duplicações preservando ordem."""

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(
                value.strip().split()
            )

            key = normalized.casefold()

            if not normalized or key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result