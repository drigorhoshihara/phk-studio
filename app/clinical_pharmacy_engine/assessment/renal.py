"""
PHK Studio
Clinical Pharmacy Engine

Renal Assessment Engine.

Responsabilidades:

- calcular CKD-EPI 2021;
- estimar depuração de creatinina por Cockcroft-Gault;
- classificar categoria de TFG;
- classificar albuminúria;
- identificar possível lesão renal aguda;
- gerar alertas e recomendações farmacêuticas;
- sinalizar necessidade de revisão profissional.

Este módulo fornece suporte à decisão clínica.
Não substitui diagnóstico médico, interpretação laboratorial
ou validação farmacêutica.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

from app.clinical_pharmacy_engine.assessment.models import (
    AssessmentAlert,
    AssessmentDataQuality,
    AssessmentStatus,
    BaseAssessmentResult,
    ClinicalRecommendation,
    ClinicalRiskLevel,
    RecommendationCategory,
    RecommendationPriority,
)


# ============================================================
# Enums
# ============================================================


class BiologicalSex(str, Enum):
    """Sexo biológico usado pelas equações renais."""

    FEMALE = "female"
    MALE = "male"


class RenalEquation(str, Enum):
    """Equações disponíveis no motor renal."""

    CKD_EPI_2021_CREATININE = (
        "ckd_epi_2021_creatinine"
    )
    COCKCROFT_GAULT = "cockcroft_gault"


class GFRCategory(str, Enum):
    """Categorias KDIGO de taxa de filtração glomerular."""

    G1 = "G1"
    G2 = "G2"
    G3A = "G3a"
    G3B = "G3b"
    G4 = "G4"
    G5 = "G5"
    UNDETERMINED = "undetermined"


class AlbuminuriaCategory(str, Enum):
    """Categorias KDIGO de albuminúria."""

    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    UNDETERMINED = "undetermined"


class AKIStage(str, Enum):
    """Estadiamento de possível lesão renal aguda."""

    NONE = "none"
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"
    POSSIBLE = "possible"
    UNDETERMINED = "undetermined"


class RenalRiskCategory(str, Enum):
    """Categoria prognóstica combinada simplificada."""

    LOW = "low"
    MODERATELY_INCREASED = "moderately_increased"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNDETERMINED = "undetermined"


# ============================================================
# Entradas e resultados
# ============================================================


@dataclass(slots=True)
class RenalAssessmentInput:
    """Dados necessários para avaliação renal."""

    age_years: float | None = None
    biological_sex: BiologicalSex | None = None

    serum_creatinine_mg_dl: float | None = None

    weight_kg: float | None = None
    height_cm: float | None = None

    albumin_creatinine_ratio_mg_g: (
        float | None
    ) = None

    previous_creatinine_mg_dl: float | None = None

    creatinine_48h_ago_mg_dl: float | None = None
    baseline_creatinine_7d_mg_dl: float | None = None

    urine_output_ml_kg_h: float | None = None
    urine_output_duration_hours: float | None = None

    dialysis: bool = False
    kidney_transplant: bool = False

    pregnant: bool = False
    amputee: bool = False
    severe_malnutrition: bool = False
    extreme_muscle_mass: bool = False

    medications: list[str] = field(
        default_factory=list,
    )

    clinical_conditions: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class RenalCalculation:
    """Resultado numérico de uma equação renal."""

    equation: RenalEquation
    value: float | None
    unit: str

    valid: bool = True
    warning: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class RenalAssessmentResult(
    BaseAssessmentResult
):
    """Resultado completo da avaliação renal."""

    egfr_ckd_epi_2021: float | None = None
    creatinine_clearance_cg: float | None = None

    gfr_category: GFRCategory = (
        GFRCategory.UNDETERMINED
    )

    albuminuria_category: AlbuminuriaCategory = (
        AlbuminuriaCategory.UNDETERMINED
    )

    renal_risk_category: RenalRiskCategory = (
        RenalRiskCategory.UNDETERMINED
    )

    aki_stage: AKIStage = AKIStage.UNDETERMINED

    calculations: list[RenalCalculation] = field(
        default_factory=list,
    )

    renal_dose_review_required: bool = False
    nephrology_review_suggested: bool = False


# ============================================================
# Motor principal
# ============================================================


class RenalAssessmentEngine:
    """Motor de avaliação da função renal."""

    REFERENCES = [
        (
            "KDIGO 2024 Clinical Practice Guideline "
            "for the Evaluation and Management of CKD"
        ),
        (
            "CKD-EPI 2021 creatinine equation"
        ),
        (
            "KDIGO Clinical Practice Guideline "
            "for Acute Kidney Injury"
        ),
    ]

    def assess(
        self,
        data: RenalAssessmentInput,
    ) -> RenalAssessmentResult:
        """
        Executa a avaliação renal integrada.

        Nenhuma recomendação altera automaticamente
        prescrição ou tratamento.
        """

        quality = self._validate_input(data)

        if quality.invalid_fields:
            return RenalAssessmentResult(
                assessment_type="renal",
                status=AssessmentStatus.INVALID_DATA,
                risk_level=(
                    ClinicalRiskLevel.UNDETERMINED
                ),
                summary=(
                    "Não foi possível realizar a avaliação "
                    "renal devido a dados inválidos."
                ),
                data_quality=quality,
                references=list(self.REFERENCES),
            )

        egfr = self.calculate_ckd_epi_2021(data)

        cockcroft = self.calculate_cockcroft_gault(
            data,
        )

        gfr_category = self.classify_gfr(
            egfr.value,
        )

        albuminuria_category = (
            self.classify_albuminuria(
                data.albumin_creatinine_ratio_mg_g,
            )
        )

        aki_stage = self.assess_aki(data)

        renal_risk = self.classify_combined_risk(
            gfr_category,
            albuminuria_category,
        )

        calculations = [
            egfr,
            cockcroft,
        ]

        alerts = self._build_alerts(
            data=data,
            egfr=egfr.value,
            creatinine_clearance=cockcroft.value,
            gfr_category=gfr_category,
            albuminuria_category=(
                albuminuria_category
            ),
            renal_risk=renal_risk,
            aki_stage=aki_stage,
        )

        recommendations = (
            self._build_recommendations(
                data=data,
                egfr=egfr.value,
                creatinine_clearance=(
                    cockcroft.value
                ),
                gfr_category=gfr_category,
                albuminuria_category=(
                    albuminuria_category
                ),
                renal_risk=renal_risk,
                aki_stage=aki_stage,
            )
        )

        risk_level = self._clinical_risk_level(
            renal_risk=renal_risk,
            aki_stage=aki_stage,
            dialysis=data.dialysis,
        )

        renal_dose_review_required = (
            self._requires_dose_review(
                egfr=egfr.value,
                creatinine_clearance=(
                    cockcroft.value
                ),
                aki_stage=aki_stage,
                dialysis=data.dialysis,
            )
        )

        nephrology_review_suggested = (
            self._suggest_nephrology_review(
                gfr_category=gfr_category,
                albuminuria_category=(
                    albuminuria_category
                ),
                aki_stage=aki_stage,
                dialysis=data.dialysis,
                kidney_transplant=(
                    data.kidney_transplant
                ),
            )
        )

        status = self._assessment_status(
            egfr=egfr.value,
            albuminuria=(
                data.albumin_creatinine_ratio_mg_g
            ),
            quality=quality,
        )

        summary = self._build_summary(
            egfr=egfr.value,
            creatinine_clearance=(
                cockcroft.value
            ),
            gfr_category=gfr_category,
            albuminuria_category=(
                albuminuria_category
            ),
            renal_risk=renal_risk,
            aki_stage=aki_stage,
        )

        return RenalAssessmentResult(
            assessment_type="renal",
            status=status,
            risk_level=risk_level,
            summary=summary,
            alerts=alerts,
            recommendations=recommendations,
            data_quality=quality,
            references=list(self.REFERENCES),
            requires_pharmacist_review=True,
            requires_prescriber_contact=any(
                recommendation.requires_prescriber_contact
                for recommendation in recommendations
            ),
            requires_referral=(
                nephrology_review_suggested
            ),
            requires_emergency_referral=any(
                recommendation.requires_immediate_action
                for recommendation in recommendations
            ),
            egfr_ckd_epi_2021=egfr.value,
            creatinine_clearance_cg=(
                cockcroft.value
            ),
            gfr_category=gfr_category,
            albuminuria_category=(
                albuminuria_category
            ),
            renal_risk_category=renal_risk,
            aki_stage=aki_stage,
            calculations=calculations,
            renal_dose_review_required=(
                renal_dose_review_required
            ),
            nephrology_review_suggested=(
                nephrology_review_suggested
            ),
            metadata={
                "dialysis": data.dialysis,
                "kidney_transplant": (
                    data.kidney_transplant
                ),
                "pregnant": data.pregnant,
            },
        )

    # ========================================================
    # Validação
    # ========================================================

    def _validate_input(
        self,
        data: RenalAssessmentInput,
    ) -> AssessmentDataQuality:
        """Valida dados clínicos e laboratoriais."""

        missing: list[str] = []
        invalid: list[str] = []
        warnings: list[str] = []

        if data.age_years is None:
            missing.append("age_years")
        elif (
            not self._is_valid_number(data.age_years)
            or data.age_years <= 0
            or data.age_years > 130
        ):
            invalid.append("age_years")

        if data.biological_sex is None:
            missing.append("biological_sex")

        if data.serum_creatinine_mg_dl is None:
            missing.append(
                "serum_creatinine_mg_dl"
            )
        elif (
            not self._is_valid_number(
                data.serum_creatinine_mg_dl
            )
            or data.serum_creatinine_mg_dl <= 0
            or data.serum_creatinine_mg_dl > 30
        ):
            invalid.append(
                "serum_creatinine_mg_dl"
            )

        self._validate_optional_positive(
            "weight_kg",
            data.weight_kg,
            invalid,
        )

        self._validate_optional_positive(
            "height_cm",
            data.height_cm,
            invalid,
        )

        self._validate_optional_nonnegative(
            "albumin_creatinine_ratio_mg_g",
            data.albumin_creatinine_ratio_mg_g,
            invalid,
        )

        for field_name, value in (
            (
                "previous_creatinine_mg_dl",
                data.previous_creatinine_mg_dl,
            ),
            (
                "creatinine_48h_ago_mg_dl",
                data.creatinine_48h_ago_mg_dl,
            ),
            (
                "baseline_creatinine_7d_mg_dl",
                data.baseline_creatinine_7d_mg_dl,
            ),
        ):
            self._validate_optional_positive(
                field_name,
                value,
                invalid,
            )

        if data.age_years is not None:
            if data.age_years < 18:
                warnings.append(
                    "CKD-EPI 2021 e Cockcroft-Gault "
                    "não devem ser interpretadas como "
                    "equações pediátricas."
                )

        if data.pregnant:
            warnings.append(
                "Estimativas baseadas em creatinina devem "
                "ser interpretadas com cautela na gestação."
            )

        if data.amputee:
            warnings.append(
                "Amputação pode reduzir a geração de "
                "creatinina e limitar a precisão da estimativa."
            )

        if data.severe_malnutrition:
            warnings.append(
                "Desnutrição grave pode reduzir a geração "
                "de creatinina e superestimar a função renal."
            )

        if data.extreme_muscle_mass:
            warnings.append(
                "Massa muscular extrema pode limitar a "
                "precisão das equações baseadas em creatinina."
            )

        complete = (
            not missing
            and not invalid
        )

        confidence = 1.0

        confidence -= min(
            0.15 * len(missing),
            0.60,
        )

        confidence -= min(
            0.25 * len(invalid),
            0.75,
        )

        confidence -= min(
            0.05 * len(warnings),
            0.25,
        )

        return AssessmentDataQuality(
            complete=complete,
            missing_fields=missing,
            invalid_fields=invalid,
            warnings=warnings,
            confidence=max(
                0.0,
                confidence,
            ),
        )

    # ========================================================
    # Equações
    # ========================================================

    def calculate_ckd_epi_2021(
        self,
        data: RenalAssessmentInput,
    ) -> RenalCalculation:
        """
        Calcula eGFR pela CKD-EPI 2021 creatinina.

        Unidade:
        mL/min/1,73 m²
        """

        age = data.age_years
        creatinine = data.serum_creatinine_mg_dl
        sex = data.biological_sex

        if (
            age is None
            or creatinine is None
            or sex is None
        ):
            return RenalCalculation(
                equation=(
                    RenalEquation.CKD_EPI_2021_CREATININE
                ),
                value=None,
                unit="mL/min/1.73 m²",
                valid=False,
                warning=(
                    "Idade, sexo biológico e creatinina "
                    "são necessários."
                ),
            )

        if (
            age < 18
            or creatinine <= 0
        ):
            return RenalCalculation(
                equation=(
                    RenalEquation.CKD_EPI_2021_CREATININE
                ),
                value=None,
                unit="mL/min/1.73 m²",
                valid=False,
                warning=(
                    "CKD-EPI 2021 deste módulo é destinada "
                    "à avaliação de adultos."
                ),
            )

        if sex == BiologicalSex.FEMALE:
            kappa = 0.7
            alpha = -0.241
            sex_factor = 1.012
        else:
            kappa = 0.9
            alpha = -0.302
            sex_factor = 1.0

        creatinine_ratio = creatinine / kappa

        value = (
            142.0
            * min(creatinine_ratio, 1.0) ** alpha
            * max(creatinine_ratio, 1.0) ** -1.200
            * 0.9938 ** age
            * sex_factor
        )

        return RenalCalculation(
            equation=(
                RenalEquation.CKD_EPI_2021_CREATININE
            ),
            value=round(value, 2),
            unit="mL/min/1.73 m²",
            metadata={
                "age_years": age,
                "serum_creatinine_mg_dl": creatinine,
                "biological_sex": sex.value,
                "race_coefficient_used": False,
            },
        )

    def calculate_cockcroft_gault(
        self,
        data: RenalAssessmentInput,
    ) -> RenalCalculation:
        """
        Calcula depuração de creatinina por Cockcroft-Gault.

        Fórmula:
        ((140 - idade) × peso) / (72 × creatinina)

        Aplica fator 0,85 para sexo feminino.
        """

        age = data.age_years
        weight = data.weight_kg
        creatinine = data.serum_creatinine_mg_dl
        sex = data.biological_sex

        if (
            age is None
            or weight is None
            or creatinine is None
            or sex is None
        ):
            return RenalCalculation(
                equation=(
                    RenalEquation.COCKCROFT_GAULT
                ),
                value=None,
                unit="mL/min",
                valid=False,
                warning=(
                    "Idade, peso, sexo biológico e "
                    "creatinina são necessários."
                ),
            )

        if (
            age < 18
            or age >= 140
            or weight <= 0
            or creatinine <= 0
        ):
            return RenalCalculation(
                equation=(
                    RenalEquation.COCKCROFT_GAULT
                ),
                value=None,
                unit="mL/min",
                valid=False,
                warning=(
                    "Dados fora do intervalo válido para "
                    "o cálculo."
                ),
            )

        value = (
            (140.0 - age)
            * weight
            / (72.0 * creatinine)
        )

        if sex == BiologicalSex.FEMALE:
            value *= 0.85

        warning = None

        if data.height_cm is None:
            warning = (
                "O cálculo utilizou o peso informado sem "
                "avaliação de peso ideal ou ajustado."
            )

        return RenalCalculation(
            equation=(
                RenalEquation.COCKCROFT_GAULT
            ),
            value=round(value, 2),
            unit="mL/min",
            warning=warning,
            metadata={
                "weight_kg_used": weight,
                "weight_strategy": "reported_weight",
            },
        )

    # ========================================================
    # Classificações
    # ========================================================

    @staticmethod
    def classify_gfr(
        egfr: float | None,
    ) -> GFRCategory:
        """Classifica a categoria KDIGO de TFG."""

        if egfr is None:
            return GFRCategory.UNDETERMINED

        if egfr >= 90:
            return GFRCategory.G1

        if egfr >= 60:
            return GFRCategory.G2

        if egfr >= 45:
            return GFRCategory.G3A

        if egfr >= 30:
            return GFRCategory.G3B

        if egfr >= 15:
            return GFRCategory.G4

        return GFRCategory.G5

    @staticmethod
    def classify_albuminuria(
        acr_mg_g: float | None,
    ) -> AlbuminuriaCategory:
        """Classifica a albuminúria pelo ACR."""

        if acr_mg_g is None:
            return (
                AlbuminuriaCategory.UNDETERMINED
            )

        if acr_mg_g < 30:
            return AlbuminuriaCategory.A1

        if acr_mg_g <= 300:
            return AlbuminuriaCategory.A2

        return AlbuminuriaCategory.A3

    @staticmethod
    def classify_combined_risk(
        gfr: GFRCategory,
        albuminuria: AlbuminuriaCategory,
    ) -> RenalRiskCategory:
        """
        Classificação combinada simplificada de risco.

        Não substitui o mapa prognóstico completo KDIGO.
        """

        if (
            gfr == GFRCategory.UNDETERMINED
            or albuminuria
            == AlbuminuriaCategory.UNDETERMINED
        ):
            return RenalRiskCategory.UNDETERMINED

        risk_matrix = {
            GFRCategory.G1: {
                AlbuminuriaCategory.A1: (
                    RenalRiskCategory.LOW
                ),
                AlbuminuriaCategory.A2: (
                    RenalRiskCategory.MODERATELY_INCREASED
                ),
                AlbuminuriaCategory.A3: (
                    RenalRiskCategory.HIGH
                ),
            },
            GFRCategory.G2: {
                AlbuminuriaCategory.A1: (
                    RenalRiskCategory.LOW
                ),
                AlbuminuriaCategory.A2: (
                    RenalRiskCategory.MODERATELY_INCREASED
                ),
                AlbuminuriaCategory.A3: (
                    RenalRiskCategory.HIGH
                ),
            },
            GFRCategory.G3A: {
                AlbuminuriaCategory.A1: (
                    RenalRiskCategory.MODERATELY_INCREASED
                ),
                AlbuminuriaCategory.A2: (
                    RenalRiskCategory.HIGH
                ),
                AlbuminuriaCategory.A3: (
                    RenalRiskCategory.VERY_HIGH
                ),
            },
            GFRCategory.G3B: {
                AlbuminuriaCategory.A1: (
                    RenalRiskCategory.HIGH
                ),
                AlbuminuriaCategory.A2: (
                    RenalRiskCategory.VERY_HIGH
                ),
                AlbuminuriaCategory.A3: (
                    RenalRiskCategory.VERY_HIGH
                ),
            },
            GFRCategory.G4: {
                AlbuminuriaCategory.A1: (
                    RenalRiskCategory.VERY_HIGH
                ),
                AlbuminuriaCategory.A2: (
                    RenalRiskCategory.VERY_HIGH
                ),
                AlbuminuriaCategory.A3: (
                    RenalRiskCategory.VERY_HIGH
                ),
            },
            GFRCategory.G5: {
                AlbuminuriaCategory.A1: (
                    RenalRiskCategory.VERY_HIGH
                ),
                AlbuminuriaCategory.A2: (
                    RenalRiskCategory.VERY_HIGH
                ),
                AlbuminuriaCategory.A3: (
                    RenalRiskCategory.VERY_HIGH
                ),
            },
        }

        return risk_matrix[gfr][albuminuria]

    # ========================================================
    # Lesão renal aguda
    # ========================================================

    def assess_aki(
        self,
        data: RenalAssessmentInput,
    ) -> AKIStage:
        """
        Identifica possível AKI pela variação da creatinina
        e, quando disponível, pelo débito urinário.

        A determinação depende de contexto clínico e série
        temporal adequados.
        """

        current = data.serum_creatinine_mg_dl

        if current is None:
            return AKIStage.UNDETERMINED

        stages: list[AKIStage] = []

        creatinine_48h = (
            data.creatinine_48h_ago_mg_dl
        )

        if creatinine_48h is not None:
            delta = current - creatinine_48h

            if delta >= 0.3:
                stages.append(AKIStage.STAGE_1)

        baseline = (
            data.baseline_creatinine_7d_mg_dl
            or data.previous_creatinine_mg_dl
        )

        if baseline is not None and baseline > 0:
            ratio = current / baseline

            if ratio >= 3.0:
                stages.append(AKIStage.STAGE_3)
            elif ratio >= 2.0:
                stages.append(AKIStage.STAGE_2)
            elif ratio >= 1.5:
                stages.append(AKIStage.STAGE_1)

        urine_stage = self._aki_by_urine_output(data)

        if urine_stage != AKIStage.UNDETERMINED:
            stages.append(urine_stage)

        if not stages:
            if (
                baseline is None
                and creatinine_48h is None
                and data.urine_output_ml_kg_h is None
            ):
                return AKIStage.UNDETERMINED

            return AKIStage.NONE

        priority = {
            AKIStage.NONE: 0,
            AKIStage.POSSIBLE: 1,
            AKIStage.STAGE_1: 2,
            AKIStage.STAGE_2: 3,
            AKIStage.STAGE_3: 4,
            AKIStage.UNDETERMINED: -1,
        }

        return max(
            stages,
            key=lambda stage: priority[stage],
        )

    @staticmethod
    def _aki_by_urine_output(
        data: RenalAssessmentInput,
    ) -> AKIStage:
        """Classifica possível AKI pelo débito urinário."""

        output = data.urine_output_ml_kg_h
        duration = data.urine_output_duration_hours

        if output is None or duration is None:
            return AKIStage.UNDETERMINED

        if output < 0:
            return AKIStage.UNDETERMINED

        if output < 0.3 and duration >= 24:
            return AKIStage.STAGE_3

        if output < 0.5 and duration >= 12:
            return AKIStage.STAGE_2

        if output < 0.5 and duration >= 6:
            return AKIStage.STAGE_1

        return AKIStage.NONE

    # ========================================================
    # Alertas
    # ========================================================

    def _build_alerts(
        self,
        *,
        data: RenalAssessmentInput,
        egfr: float | None,
        creatinine_clearance: float | None,
        gfr_category: GFRCategory,
        albuminuria_category: AlbuminuriaCategory,
        renal_risk: RenalRiskCategory,
        aki_stage: AKIStage,
    ) -> list[AssessmentAlert]:
        """Gera alertas estruturados."""

        alerts: list[AssessmentAlert] = []

        if aki_stage in {
            AKIStage.STAGE_1,
            AKIStage.STAGE_2,
            AKIStage.STAGE_3,
            AKIStage.POSSIBLE,
        }:
            risk = (
                ClinicalRiskLevel.CRITICAL
                if aki_stage == AKIStage.STAGE_3
                else ClinicalRiskLevel.HIGH
            )

            alerts.append(
                AssessmentAlert(
                    code="RENAL_POSSIBLE_AKI",
                    title=(
                        "Possível lesão renal aguda"
                    ),
                    description=(
                        "Foi identificada variação de "
                        "creatinina ou débito urinário "
                        "compatível com possível lesão "
                        "renal aguda. Confirmar cronologia, "
                        "estado volêmico, causas reversíveis "
                        "e contexto clínico."
                    ),
                    risk_level=risk,
                    requires_immediate_action=(
                        aki_stage == AKIStage.STAGE_3
                    ),
                    evidence=[
                        f"AKI stage: {aki_stage.value}",
                    ],
                )
            )

        if gfr_category in {
            GFRCategory.G4,
            GFRCategory.G5,
        }:
            alerts.append(
                AssessmentAlert(
                    code="RENAL_SEVERE_REDUCTION",
                    title=(
                        "Redução importante da função renal"
                    ),
                    description=(
                        "A estimativa de filtração encontra-se "
                        "em categoria associada a maior risco "
                        "clínico e farmacoterapêutico."
                    ),
                    risk_level=(
                        ClinicalRiskLevel.HIGH
                        if gfr_category == GFRCategory.G4
                        else ClinicalRiskLevel.CRITICAL
                    ),
                    evidence=[
                        f"eGFR: {egfr}",
                        (
                            "GFR category: "
                            f"{gfr_category.value}"
                        ),
                    ],
                )
            )

        if (
            albuminuria_category
            == AlbuminuriaCategory.A3
        ):
            alerts.append(
                AssessmentAlert(
                    code="RENAL_SEVERE_ALBUMINURIA",
                    title=(
                        "Albuminúria acentuadamente aumentada"
                    ),
                    description=(
                        "O valor informado de relação "
                        "albumina/creatinina corresponde à "
                        "categoria A3."
                    ),
                    risk_level=ClinicalRiskLevel.HIGH,
                    evidence=[
                        (
                            "ACR: "
                            f"{data.albumin_creatinine_ratio_mg_g}"
                            " mg/g"
                        ),
                    ],
                )
            )

        if renal_risk == RenalRiskCategory.VERY_HIGH:
            alerts.append(
                AssessmentAlert(
                    code="RENAL_VERY_HIGH_RISK",
                    title=(
                        "Risco renal combinado muito alto"
                    ),
                    description=(
                        "A combinação da categoria de TFG "
                        "com a albuminúria indica risco renal "
                        "e cardiovascular aumentado."
                    ),
                    risk_level=ClinicalRiskLevel.HIGH,
                )
            )

        if (
            egfr is not None
            and creatinine_clearance is not None
            and abs(egfr - creatinine_clearance) >= 20
        ):
            alerts.append(
                AssessmentAlert(
                    code="RENAL_ESTIMATE_DISCORDANCE",
                    title=(
                        "Discordância entre estimativas renais"
                    ),
                    description=(
                        "Existe diferença relevante entre "
                        "CKD-EPI e Cockcroft-Gault. Revisar "
                        "peso utilizado, composição corporal, "
                        "estabilidade da creatinina e método "
                        "adotado pela referência do medicamento."
                    ),
                    risk_level=ClinicalRiskLevel.MODERATE,
                    evidence=[
                        f"CKD-EPI: {egfr}",
                        (
                            "Cockcroft-Gault: "
                            f"{creatinine_clearance}"
                        ),
                    ],
                )
            )

        if data.dialysis:
            alerts.append(
                AssessmentAlert(
                    code="RENAL_DIALYSIS",
                    title="Paciente em diálise",
                    description=(
                        "A farmacoterapia deve considerar "
                        "modalidade dialítica, frequência, "
                        "depuração extracorpórea e momento "
                        "de administração."
                    ),
                    risk_level=ClinicalRiskLevel.HIGH,
                )
            )

        return alerts

    # ========================================================
    # Recomendações
    # ========================================================

    def _build_recommendations(
        self,
        *,
        data: RenalAssessmentInput,
        egfr: float | None,
        creatinine_clearance: float | None,
        gfr_category: GFRCategory,
        albuminuria_category: AlbuminuriaCategory,
        renal_risk: RenalRiskCategory,
        aki_stage: AKIStage,
    ) -> list[ClinicalRecommendation]:
        """Gera recomendações farmacêuticas."""

        recommendations: list[
            ClinicalRecommendation
        ] = []

        if self._requires_dose_review(
            egfr=egfr,
            creatinine_clearance=(
                creatinine_clearance
            ),
            aki_stage=aki_stage,
            dialysis=data.dialysis,
        ):
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Revisar medicamentos dependentes "
                        "da função renal"
                    ),
                    description=(
                        "Revisar dose, intervalo, indicação, "
                        "contraindicações e necessidade de "
                        "monitorização dos medicamentos com "
                        "eliminação renal relevante."
                    ),
                    category=(
                        RecommendationCategory.DOSE_REVIEW
                    ),
                    priority=(
                        RecommendationPriority.URGENT
                        if aki_stage
                        in {
                            AKIStage.STAGE_2,
                            AKIStage.STAGE_3,
                        }
                        else RecommendationPriority.PRIORITY
                    ),
                    rationale=(
                        "A redução ou instabilidade da função "
                        "renal pode alterar exposição, eficácia "
                        "e toxicidade farmacológica."
                    ),
                    related_medications=list(
                        data.medications,
                    ),
                    requires_prescriber_contact=(
                        aki_stage
                        in {
                            AKIStage.STAGE_2,
                            AKIStage.STAGE_3,
                        }
                    ),
                )
            )

        if aki_stage in {
            AKIStage.STAGE_1,
            AKIStage.STAGE_2,
            AKIStage.STAGE_3,
        }:
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Avaliação clínica imediata da "
                        "possível lesão renal aguda"
                    ),
                    description=(
                        "Confirmar tendência da creatinina, "
                        "débito urinário, estado volêmico, "
                        "exposição a nefrotóxicos e possíveis "
                        "causas obstrutivas ou hemodinâmicas."
                    ),
                    category=(
                        RecommendationCategory.LABORATORY_REVIEW
                    ),
                    priority=(
                        RecommendationPriority.IMMEDIATE
                        if aki_stage == AKIStage.STAGE_3
                        else RecommendationPriority.URGENT
                    ),
                    monitoring_parameters=[
                        "creatinina sérica",
                        "ureia",
                        "potássio",
                        "bicarbonato",
                        "débito urinário",
                        "estado volêmico",
                    ],
                    requires_prescriber_contact=True,
                    requires_immediate_action=(
                        aki_stage == AKIStage.STAGE_3
                    ),
                )
            )

        if self._suggest_nephrology_review(
            gfr_category=gfr_category,
            albuminuria_category=(
                albuminuria_category
            ),
            aki_stage=aki_stage,
            dialysis=data.dialysis,
            kidney_transplant=data.kidney_transplant,
        ):
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Considerar avaliação nefrológica"
                    ),
                    description=(
                        "Avaliar necessidade e urgência de "
                        "encaminhamento conforme persistência "
                        "das alterações, velocidade de "
                        "progressão, albuminúria, complicações "
                        "e contexto clínico."
                    ),
                    category=(
                        RecommendationCategory.REFERRAL
                    ),
                    priority=(
                        RecommendationPriority.URGENT
                        if aki_stage
                        in {
                            AKIStage.STAGE_2,
                            AKIStage.STAGE_3,
                        }
                        else RecommendationPriority.PRIORITY
                    ),
                )
            )

        if (
            data.albumin_creatinine_ratio_mg_g
            is None
        ):
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Obter avaliação de albuminúria"
                    ),
                    description=(
                        "Considerar relação albumina/"
                        "creatinina urinária para completar "
                        "a classificação de risco renal."
                    ),
                    category=(
                        RecommendationCategory.LABORATORY_REVIEW
                    ),
                    priority=(
                        RecommendationPriority.ROUTINE
                    ),
                    monitoring_parameters=[
                        (
                            "relação albumina/creatinina "
                            "urinária"
                        ),
                    ],
                )
            )

        return recommendations

    # ========================================================
    # Regras auxiliares
    # ========================================================

    @staticmethod
    def _requires_dose_review(
        *,
        egfr: float | None,
        creatinine_clearance: float | None,
        aki_stage: AKIStage,
        dialysis: bool,
    ) -> bool:
        """Determina necessidade de revisão posológica."""

        if dialysis:
            return True

        if aki_stage in {
            AKIStage.STAGE_1,
            AKIStage.STAGE_2,
            AKIStage.STAGE_3,
            AKIStage.POSSIBLE,
        }:
            return True

        if (
            egfr is not None
            and egfr < 60
        ):
            return True

        if (
            creatinine_clearance is not None
            and creatinine_clearance < 60
        ):
            return True

        return False

    @staticmethod
    def _suggest_nephrology_review(
        *,
        gfr_category: GFRCategory,
        albuminuria_category: AlbuminuriaCategory,
        aki_stage: AKIStage,
        dialysis: bool,
        kidney_transplant: bool,
    ) -> bool:
        """Sinaliza possível necessidade de nefrologia."""

        if dialysis or kidney_transplant:
            return True

        if gfr_category in {
            GFRCategory.G4,
            GFRCategory.G5,
        }:
            return True

        if (
            albuminuria_category
            == AlbuminuriaCategory.A3
        ):
            return True

        return aki_stage in {
            AKIStage.STAGE_2,
            AKIStage.STAGE_3,
        }

    @staticmethod
    def _clinical_risk_level(
        *,
        renal_risk: RenalRiskCategory,
        aki_stage: AKIStage,
        dialysis: bool,
    ) -> ClinicalRiskLevel:
        """Converte achados em nível de risco clínico."""

        if (
            dialysis
            or aki_stage == AKIStage.STAGE_3
        ):
            return ClinicalRiskLevel.CRITICAL

        if aki_stage in {
            AKIStage.STAGE_1,
            AKIStage.STAGE_2,
        }:
            return ClinicalRiskLevel.HIGH

        mapping = {
            RenalRiskCategory.LOW: (
                ClinicalRiskLevel.LOW
            ),
            RenalRiskCategory.MODERATELY_INCREASED: (
                ClinicalRiskLevel.MODERATE
            ),
            RenalRiskCategory.HIGH: (
                ClinicalRiskLevel.HIGH
            ),
            RenalRiskCategory.VERY_HIGH: (
                ClinicalRiskLevel.HIGH
            ),
            RenalRiskCategory.UNDETERMINED: (
                ClinicalRiskLevel.UNDETERMINED
            ),
        }

        return mapping[renal_risk]

    @staticmethod
    def _assessment_status(
        *,
        egfr: float | None,
        albuminuria: float | None,
        quality: AssessmentDataQuality,
    ) -> AssessmentStatus:
        """Determina o estado final da avaliação."""

        if quality.invalid_fields:
            return AssessmentStatus.INVALID_DATA

        if egfr is None:
            return AssessmentStatus.INSUFFICIENT_DATA

        if albuminuria is None or quality.missing_fields:
            return AssessmentStatus.PARTIAL

        return AssessmentStatus.COMPLETED

    @staticmethod
    def _build_summary(
        *,
        egfr: float | None,
        creatinine_clearance: float | None,
        gfr_category: GFRCategory,
        albuminuria_category: AlbuminuriaCategory,
        renal_risk: RenalRiskCategory,
        aki_stage: AKIStage,
    ) -> str:
        """Cria resumo textual da avaliação."""

        parts: list[str] = []

        if egfr is not None:
            parts.append(
                f"CKD-EPI 2021: {egfr:.2f} "
                "mL/min/1,73 m²"
            )

        parts.append(
            f"categoria de TFG: {gfr_category.value}"
        )

        if creatinine_clearance is not None:
            parts.append(
                "Cockcroft-Gault: "
                f"{creatinine_clearance:.2f} mL/min"
            )

        parts.append(
            "albuminúria: "
            f"{albuminuria_category.value}"
        )

        parts.append(
            f"risco combinado: {renal_risk.value}"
        )

        parts.append(
            f"avaliação de AKI: {aki_stage.value}"
        )

        return "; ".join(parts) + "."

    @staticmethod
    def _is_valid_number(
        value: float,
    ) -> bool:
        """Verifica se o valor é numérico e finito."""

        try:
            return isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def _validate_optional_positive(
        self,
        field_name: str,
        value: float | None,
        invalid: list[str],
    ) -> None:
        """Valida campo opcional estritamente positivo."""

        if value is None:
            return

        if (
            not self._is_valid_number(value)
            or value <= 0
        ):
            invalid.append(field_name)

    def _validate_optional_nonnegative(
        self,
        field_name: str,
        value: float | None,
        invalid: list[str],
    ) -> None:
        """Valida campo opcional não negativo."""

        if value is None:
            return

        if (
            not self._is_valid_number(value)
            or value < 0
        ):
            invalid.append(field_name)