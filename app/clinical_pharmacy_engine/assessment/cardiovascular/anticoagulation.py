"""
PHK Studio
Clinical Pharmacy Engine

Anticoagulation Assessment Engine.

Responsabilidades:

- calcular CHA2DS2-VASc;
- calcular CHA2DS2-VA;
- calcular HAS-BLED;
- estratificar risco tromboembólico;
- identificar fatores hemorrágicos modificáveis;
- detectar situações que exigem avaliação especializada;
- gerar alertas clínicos e farmacoterapêuticos;
- preservar rastreabilidade das regras utilizadas.

O módulo não:

- inicia anticoagulante automaticamente;
- seleciona dose automaticamente;
- substitui avaliação médica ou farmacêutica;
- usa HAS-BLED isoladamente para negar anticoagulação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable


# ============================================================
# Enums
# ============================================================


class AnticoagulationGuideline(str, Enum):
    """Referencial utilizado na avaliação."""

    ACC_AHA_2023 = "acc_aha_2023"
    ESC_2024 = "esc_2024"
    INSTITUTIONAL = "institutional"
    UNDETERMINED = "undetermined"


class AtrialArrhythmiaType(str, Enum):
    """Tipo de arritmia atrial."""

    ATRIAL_FIBRILLATION = "atrial_fibrillation"
    ATRIAL_FLUTTER = "atrial_flutter"
    DEVICE_DETECTED_AF = "device_detected_af"
    NONE = "none"
    UNDETERMINED = "undetermined"


class StrokeRiskCategory(str, Enum):
    """Categoria operacional de risco tromboembólico."""

    LOW = "low"
    INTERMEDIATE = "intermediate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNDETERMINED = "undetermined"


class BleedingRiskCategory(str, Enum):
    """Categoria operacional do HAS-BLED."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNDETERMINED = "undetermined"


class AnticoagulationRecommendationStatus(str, Enum):
    """Estado da recomendação de revisão."""

    NOT_INDICATED_BY_SCORE = "not_indicated_by_score"
    CONSIDER = "consider"
    RECOMMENDED_BY_SCORE = "recommended_by_score"
    ALREADY_ANTICOAGULATED = "already_anticoagulated"
    SPECIALIST_REVIEW_REQUIRED = "specialist_review_required"
    NOT_APPLICABLE = "not_applicable"
    UNDETERMINED = "undetermined"


class AnticoagulantClass(str, Enum):
    """Classe de anticoagulante atualmente utilizada."""

    NONE = "none"
    WARFARIN = "warfarin"
    APIXABAN = "apixaban"
    RIVAROXABAN = "rivaroxaban"
    DABIGATRAN = "dabigatran"
    EDOXABAN = "edoxaban"
    PARENTERAL = "parenteral"
    OTHER = "other"
    UNDETERMINED = "undetermined"


# ============================================================
# Entrada
# ============================================================


@dataclass(slots=True)
class AnticoagulationAssessmentInput:
    """Entrada normalizada para avaliação de anticoagulação."""

    age_years: float | None = None

    atrial_arrhythmia: AtrialArrhythmiaType = (
        AtrialArrhythmiaType.UNDETERMINED
    )

    guideline: AnticoagulationGuideline = (
        AnticoagulationGuideline.ESC_2024
    )

    female_sex: bool = False

    congestive_heart_failure: bool = False
    hypertension: bool = False
    diabetes: bool = False

    previous_stroke: bool = False
    previous_tia: bool = False
    previous_systemic_embolism: bool = False

    vascular_disease: bool = False
    previous_myocardial_infarction: bool = False
    peripheral_arterial_disease: bool = False
    aortic_plaque: bool = False

    abnormal_renal_function: bool = False
    abnormal_hepatic_function: bool = False

    egfr_ml_min_1_73m2: float | None = None
    dialysis: bool = False

    cirrhosis: bool = False
    significant_liver_dysfunction: bool = False

    previous_major_bleeding: bool = False
    bleeding_predisposition: bool = False

    labile_inr: bool = False
    time_in_therapeutic_range_percent: float | None = None

    concomitant_antiplatelet: bool = False
    concomitant_nsaid: bool = False

    harmful_alcohol_use: bool = False

    mechanical_heart_valve: bool = False
    moderate_or_severe_mitral_stenosis: bool = False

    active_major_bleeding: bool = False
    severe_thrombocytopenia: bool = False
    platelet_count_per_mm3: float | None = None

    recent_intracranial_hemorrhage: bool = False
    recent_major_surgery: bool = False

    pregnancy: bool = False

    current_anticoagulant: AnticoagulantClass = (
        AnticoagulantClass.NONE
    )

    anticoagulant_medications: list[str] = field(
        default_factory=list,
    )

    interacting_medications: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Configuração e resultados
# ============================================================


@dataclass(slots=True)
class AnticoagulationAssessmentConfig:
    """Configurações operacionais."""

    high_bleeding_risk_has_bled: int = 3

    severe_thrombocytopenia_threshold: float = 50_000.0

    poor_ttr_threshold_percent: float = 65.0

    cha2ds2_va_consider_threshold: int = 1
    cha2ds2_va_recommend_threshold: int = 2

    cha2ds2_vasc_male_consider_threshold: int = 1
    cha2ds2_vasc_male_recommend_threshold: int = 2

    cha2ds2_vasc_female_consider_threshold: int = 2
    cha2ds2_vasc_female_recommend_threshold: int = 3


@dataclass(slots=True)
class AnticoagulationValidation:
    """Resultado de validação."""

    valid: bool

    missing_fields: list[str] = field(
        default_factory=list,
    )

    invalid_fields: list[str] = field(
        default_factory=list,
    )

    warnings: list[str] = field(
        default_factory=list,
    )


@dataclass(slots=True)
class ThromboembolicScoreResult:
    """Resultado dos escores tromboembólicos."""

    cha2ds2_vasc_score: int = 0
    cha2ds2_va_score: int = 0

    category: StrokeRiskCategory = (
        StrokeRiskCategory.UNDETERMINED
    )

    components: dict[str, int] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class HASBLEDResult:
    """Resultado do HAS-BLED."""

    score: int = 0

    category: BleedingRiskCategory = (
        BleedingRiskCategory.UNDETERMINED
    )

    components: dict[str, int] = field(
        default_factory=dict,
    )

    modifiable_risk_factors: list[str] = field(
        default_factory=list,
    )


@dataclass(slots=True)
class AnticoagulationAssessmentResult:
    """Resultado integrado do motor."""

    valid: bool = False

    thromboembolic_score: ThromboembolicScoreResult = field(
        default_factory=ThromboembolicScoreResult,
    )

    has_bled: HASBLEDResult = field(
        default_factory=HASBLEDResult,
    )

    recommendation_status: (
        AnticoagulationRecommendationStatus
    ) = AnticoagulationRecommendationStatus.UNDETERMINED

    anticoagulation_review_required: bool = False
    bleeding_risk_review_required: bool = False
    specialist_review_required: bool = False
    immediate_evaluation_required: bool = False

    absolute_alerts: list[str] = field(
        default_factory=list,
    )

    warnings: list[str] = field(
        default_factory=list,
    )

    recommendations: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Motor principal
# ============================================================


class AnticoagulationAssessmentEngine:
    """Motor de avaliação de anticoagulação na FA/flutter."""

    def __init__(
        self,
        config: AnticoagulationAssessmentConfig | None = None,
    ) -> None:
        self.config = (
            config
            or AnticoagulationAssessmentConfig()
        )

    def assess(
        self,
        data: AnticoagulationAssessmentInput,
    ) -> AnticoagulationAssessmentResult:
        """Executa avaliação integrada."""

        validation = self.validate(data)

        if not validation.valid:
            return AnticoagulationAssessmentResult(
                valid=False,
                warnings=self._unique_strings(
                    validation.warnings
                    + [
                        (
                            "Campos ausentes: "
                            + ", ".join(
                                validation.missing_fields
                            )
                        )
                        if validation.missing_fields
                        else "",
                        (
                            "Campos inválidos: "
                            + ", ".join(
                                validation.invalid_fields
                            )
                        )
                        if validation.invalid_fields
                        else "",
                    ]
                ),
                metadata={
                    "missing_fields": list(
                        validation.missing_fields
                    ),
                    "invalid_fields": list(
                        validation.invalid_fields
                    ),
                },
            )

        thromboembolic = (
            self.calculate_thromboembolic_scores(data)
        )

        has_bled = self.calculate_has_bled(data)

        absolute_alerts = (
            self.detect_critical_conditions(data)
        )

        recommendation_status = (
            self.resolve_recommendation_status(
                data=data,
                scores=thromboembolic,
                critical_conditions=absolute_alerts,
            )
        )

        warnings = list(validation.warnings)
        recommendations: list[str] = []

        if has_bled.score >= 3:
            warnings.append(
                "HAS-BLED elevado. Isso indica necessidade "
                "de revisão dos fatores hemorrágicos e "
                "monitorização mais próxima, não suspensão "
                "automática da anticoagulação."
            )

        if has_bled.modifiable_risk_factors:
            recommendations.append(
                "Corrigir ou reduzir fatores hemorrágicos "
                "modificáveis quando clinicamente possível."
            )

        if data.concomitant_antiplatelet:
            warnings.append(
                "Uso concomitante de antiagregante informado. "
                "Confirmar indicação, duração e risco de "
                "sangramento."
            )

        if data.concomitant_nsaid:
            warnings.append(
                "Uso concomitante de AINE informado. "
                "Reavaliar necessidade devido ao aumento "
                "potencial do risco hemorrágico."
            )

        if data.harmful_alcohol_use:
            warnings.append(
                "Consumo nocivo de álcool informado. "
                "Considerar intervenção e monitorização."
            )

        if data.labile_inr:
            recommendations.append(
                "Revisar adesão, dieta, interações, frequência "
                "de monitorização e tempo em faixa terapêutica."
            )

        if data.abnormal_renal_function or data.dialysis:
            recommendations.append(
                "Confirmar função renal e adequação do "
                "anticoagulante e da dose antes da validação."
            )

        if data.abnormal_hepatic_function:
            recommendations.append(
                "Confirmar gravidade da disfunção hepática "
                "antes da seleção ou manutenção do tratamento."
            )

        if (
            data.mechanical_heart_valve
            or data.moderate_or_severe_mitral_stenosis
        ):
            recommendations.append(
                "Avaliação especializada necessária para "
                "seleção do anticoagulante."
            )

        if data.interacting_medications:
            recommendations.append(
                "Revisar interações farmacocinéticas e "
                "farmacodinâmicas dos medicamentos informados."
            )

        already_anticoagulated = (
            data.current_anticoagulant
            not in {
                AnticoagulantClass.NONE,
                AnticoagulantClass.UNDETERMINED,
            }
        )

        anticoagulation_review_required = (
            recommendation_status
            in {
                AnticoagulationRecommendationStatus.CONSIDER,
                AnticoagulationRecommendationStatus
                .RECOMMENDED_BY_SCORE,
                AnticoagulationRecommendationStatus
                .SPECIALIST_REVIEW_REQUIRED,
            }
            or already_anticoagulated
        )

        immediate_evaluation = any(
            (
                data.active_major_bleeding,
                data.recent_intracranial_hemorrhage,
            )
        )

        return AnticoagulationAssessmentResult(
            valid=True,
            thromboembolic_score=thromboembolic,
            has_bled=has_bled,
            recommendation_status=(
                recommendation_status
            ),
            anticoagulation_review_required=(
                anticoagulation_review_required
            ),
            bleeding_risk_review_required=(
                has_bled.score
                >= self.config.high_bleeding_risk_has_bled
            ),
            specialist_review_required=bool(
                absolute_alerts
                or data.mechanical_heart_valve
                or data.moderate_or_severe_mitral_stenosis
                or data.pregnancy
            ),
            immediate_evaluation_required=(
                immediate_evaluation
            ),
            absolute_alerts=self._unique_strings(
                absolute_alerts
            ),
            warnings=self._unique_strings(warnings),
            recommendations=self._unique_strings(
                recommendations
            ),
            metadata={
                "guideline": data.guideline.value,
                "atrial_arrhythmia": (
                    data.atrial_arrhythmia.value
                ),
                "current_anticoagulant": (
                    data.current_anticoagulant.value
                ),
                "already_anticoagulated": (
                    already_anticoagulated
                ),
                "interacting_medication_count": len(
                    self._unique_strings(
                        data.interacting_medications
                    )
                ),
            },
        )

    # ========================================================
    # Validação
    # ========================================================

    def validate(
        self,
        data: AnticoagulationAssessmentInput,
    ) -> AnticoagulationValidation:
        """Valida aplicabilidade e plausibilidade."""

        missing: list[str] = []
        invalid: list[str] = []
        warnings: list[str] = []

        if data.age_years is None:
            missing.append("age_years")

        elif (
            not self._valid_number(data.age_years)
            or float(data.age_years) < 0
            or float(data.age_years) > 130
        ):
            invalid.append("age_years")

        if (
            data.atrial_arrhythmia
            == AtrialArrhythmiaType.UNDETERMINED
        ):
            missing.append("atrial_arrhythmia")

        if (
            data.egfr_ml_min_1_73m2 is not None
            and (
                not self._valid_number(
                    data.egfr_ml_min_1_73m2
                )
                or float(data.egfr_ml_min_1_73m2) < 0
            )
        ):
            invalid.append("egfr_ml_min_1_73m2")

        if (
            data.platelet_count_per_mm3 is not None
            and (
                not self._valid_number(
                    data.platelet_count_per_mm3
                )
                or float(data.platelet_count_per_mm3) < 0
            )
        ):
            invalid.append("platelet_count_per_mm3")

        if (
            data.time_in_therapeutic_range_percent
            is not None
            and (
                not self._valid_number(
                    data.time_in_therapeutic_range_percent
                )
                or not 0
                <= float(
                    data.time_in_therapeutic_range_percent
                )
                <= 100
            )
        ):
            invalid.append(
                "time_in_therapeutic_range_percent"
            )

        if (
            data.atrial_arrhythmia
            == AtrialArrhythmiaType.NONE
        ):
            warnings.append(
                "Não foi informada fibrilação atrial ou "
                "flutter. Os escores podem não ser aplicáveis."
            )

        return AnticoagulationValidation(
            valid=not missing and not invalid,
            missing_fields=self._unique_strings(missing),
            invalid_fields=self._unique_strings(invalid),
            warnings=self._unique_strings(warnings),
        )

    # ========================================================
    # CHA2DS2-VASc e CHA2DS2-VA
    # ========================================================

    def calculate_thromboembolic_scores(
        self,
        data: AnticoagulationAssessmentInput,
    ) -> ThromboembolicScoreResult:
        """Calcula os dois escores tromboembólicos."""

        assert data.age_years is not None

        age = float(data.age_years)

        heart_failure = int(
            data.congestive_heart_failure
        )

        hypertension = int(data.hypertension)

        age_75_or_more = 2 if age >= 75 else 0
        age_65_to_74 = (
            1 if 65 <= age < 75 else 0
        )

        diabetes = int(data.diabetes)

        previous_thromboembolism = (
            2
            if any(
                (
                    data.previous_stroke,
                    data.previous_tia,
                    data.previous_systemic_embolism,
                )
            )
            else 0
        )

        vascular = int(
            any(
                (
                    data.vascular_disease,
                    data.previous_myocardial_infarction,
                    data.peripheral_arterial_disease,
                    data.aortic_plaque,
                )
            )
        )

        female = int(data.female_sex)

        cha2ds2_va = sum(
            (
                heart_failure,
                hypertension,
                age_75_or_more,
                age_65_to_74,
                diabetes,
                previous_thromboembolism,
                vascular,
            )
        )

        cha2ds2_vasc = cha2ds2_va + female

        return ThromboembolicScoreResult(
            cha2ds2_vasc_score=cha2ds2_vasc,
            cha2ds2_va_score=cha2ds2_va,
            category=self.classify_stroke_risk(
                cha2ds2_va
            ),
            components={
                "congestive_heart_failure": (
                    heart_failure
                ),
                "hypertension": hypertension,
                "age_75_or_more": age_75_or_more,
                "age_65_to_74": age_65_to_74,
                "diabetes": diabetes,
                "stroke_tia_or_systemic_embolism": (
                    previous_thromboembolism
                ),
                "vascular_disease": vascular,
                "female_sex": female,
            },
        )

    @staticmethod
    def classify_stroke_risk(
        cha2ds2_va_score: int,
    ) -> StrokeRiskCategory:
        """Classificação operacional pelo CHA2DS2-VA."""

        if cha2ds2_va_score <= 0:
            return StrokeRiskCategory.LOW

        if cha2ds2_va_score == 1:
            return StrokeRiskCategory.INTERMEDIATE

        if cha2ds2_va_score <= 3:
            return StrokeRiskCategory.HIGH

        return StrokeRiskCategory.VERY_HIGH

    # ========================================================
    # HAS-BLED
    # ========================================================

    def calculate_has_bled(
        self,
        data: AnticoagulationAssessmentInput,
    ) -> HASBLEDResult:
        """Calcula HAS-BLED e fatores modificáveis."""

        assert data.age_years is not None

        hypertension = int(data.hypertension)

        renal = int(
            data.abnormal_renal_function
            or data.dialysis
        )

        hepatic = int(
            data.abnormal_hepatic_function
            or data.cirrhosis
            or data.significant_liver_dysfunction
        )

        stroke = int(
            data.previous_stroke
            or data.previous_tia
        )

        bleeding = int(
            data.previous_major_bleeding
            or data.bleeding_predisposition
        )

        labile_inr = int(
            data.labile_inr
            or (
                data.time_in_therapeutic_range_percent
                is not None
                and data.time_in_therapeutic_range_percent
                < self.config.poor_ttr_threshold_percent
            )
        )

        elderly = int(float(data.age_years) > 65)

        drugs = int(
            data.concomitant_antiplatelet
            or data.concomitant_nsaid
        )

        alcohol = int(data.harmful_alcohol_use)

        score = sum(
            (
                hypertension,
                renal,
                hepatic,
                stroke,
                bleeding,
                labile_inr,
                elderly,
                drugs,
                alcohol,
            )
        )

        modifiable: list[str] = []

        if data.hypertension:
            modifiable.append(
                "Controle inadequado da pressão arterial."
            )

        if labile_inr:
            modifiable.append(
                "INR lábil ou tempo em faixa terapêutica baixo."
            )

        if data.concomitant_antiplatelet:
            modifiable.append(
                "Uso concomitante de antiagregante."
            )

        if data.concomitant_nsaid:
            modifiable.append(
                "Uso concomitante de AINE."
            )

        if data.harmful_alcohol_use:
            modifiable.append(
                "Consumo nocivo de álcool."
            )

        if score <= 1:
            category = BleedingRiskCategory.LOW

        elif score == 2:
            category = BleedingRiskCategory.MODERATE

        else:
            category = BleedingRiskCategory.HIGH

        return HASBLEDResult(
            score=score,
            category=category,
            components={
                "hypertension": hypertension,
                "abnormal_renal_function": renal,
                "abnormal_hepatic_function": hepatic,
                "stroke_history": stroke,
                "bleeding_history": bleeding,
                "labile_inr": labile_inr,
                "age_over_65": elderly,
                "drugs": drugs,
                "alcohol": alcohol,
            },
            modifiable_risk_factors=(
                self._unique_strings(modifiable)
            ),
        )

    # ========================================================
    # Recomendação e alertas
    # ========================================================

    def resolve_recommendation_status(
        self,
        *,
        data: AnticoagulationAssessmentInput,
        scores: ThromboembolicScoreResult,
        critical_conditions: list[str],
    ) -> AnticoagulationRecommendationStatus:
        """Resolve o estado da recomendação."""

        if (
            data.atrial_arrhythmia
            == AtrialArrhythmiaType.NONE
        ):
            return (
                AnticoagulationRecommendationStatus
                .NOT_APPLICABLE
            )

        if critical_conditions:
            return (
                AnticoagulationRecommendationStatus
                .SPECIALIST_REVIEW_REQUIRED
            )

        if (
            data.current_anticoagulant
            not in {
                AnticoagulantClass.NONE,
                AnticoagulantClass.UNDETERMINED,
            }
        ):
            return (
                AnticoagulationRecommendationStatus
                .ALREADY_ANTICOAGULATED
            )

        if data.guideline == AnticoagulationGuideline.ESC_2024:
            if (
                scores.cha2ds2_va_score
                >= self.config
                .cha2ds2_va_recommend_threshold
            ):
                return (
                    AnticoagulationRecommendationStatus
                    .RECOMMENDED_BY_SCORE
                )

            if (
                scores.cha2ds2_va_score
                >= self.config
                .cha2ds2_va_consider_threshold
            ):
                return (
                    AnticoagulationRecommendationStatus
                    .CONSIDER
                )

            return (
                AnticoagulationRecommendationStatus
                .NOT_INDICATED_BY_SCORE
            )

        threshold_consider = (
            self.config
            .cha2ds2_vasc_female_consider_threshold
            if data.female_sex
            else self.config
            .cha2ds2_vasc_male_consider_threshold
        )

        threshold_recommend = (
            self.config
            .cha2ds2_vasc_female_recommend_threshold
            if data.female_sex
            else self.config
            .cha2ds2_vasc_male_recommend_threshold
        )

        if (
            scores.cha2ds2_vasc_score
            >= threshold_recommend
        ):
            return (
                AnticoagulationRecommendationStatus
                .RECOMMENDED_BY_SCORE
            )

        if (
            scores.cha2ds2_vasc_score
            >= threshold_consider
        ):
            return (
                AnticoagulationRecommendationStatus
                .CONSIDER
            )

        return (
            AnticoagulationRecommendationStatus
            .NOT_INDICATED_BY_SCORE
        )

    def detect_critical_conditions(
        self,
        data: AnticoagulationAssessmentInput,
    ) -> list[str]:
        """Detecta situações que impedem decisão automática."""

        alerts: list[str] = []

        if data.active_major_bleeding:
            alerts.append(
                "Sangramento maior ativo informado."
            )

        if data.recent_intracranial_hemorrhage:
            alerts.append(
                "Hemorragia intracraniana recente informada."
            )

        if data.mechanical_heart_valve:
            alerts.append(
                "Prótese valvar mecânica informada."
            )

        if data.moderate_or_severe_mitral_stenosis:
            alerts.append(
                "Estenose mitral moderada ou grave informada."
            )

        if data.pregnancy:
            alerts.append(
                "Gestação informada."
            )

        severe_platelet_reduction = (
            data.severe_thrombocytopenia
            or (
                data.platelet_count_per_mm3 is not None
                and data.platelet_count_per_mm3
                < self.config
                .severe_thrombocytopenia_threshold
            )
        )

        if severe_platelet_reduction:
            alerts.append(
                "Trombocitopenia grave informada."
            )

        return self._unique_strings(alerts)

    # ========================================================
    # Utilidades
    # ========================================================

    @staticmethod
    def _valid_number(
        value: object,
    ) -> bool:
        """Verifica número finito."""

        try:
            return isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _unique_strings(
        values: Iterable[str],
    ) -> list[str]:
        """Remove textos vazios e duplicados."""

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(
                str(value).strip().split()
            )

            if not normalized:
                continue

            key = normalized.casefold()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result