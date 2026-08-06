"""
PHK Studio
Clinical Pharmacy Engine

Cardiovascular Assessment Models.

Modelos compartilhados pelos componentes cardiovasculares:

- risco cardiovascular global;
- hipertensão arterial;
- dislipidemia;
- insuficiência cardíaca;
- síndrome coronariana aguda;
- anticoagulação;
- risco hemorrágico;
- intervalo QT;
- recomendações farmacêuticas.

Este módulo contém estruturas de dados, não algoritmos
definitivos de diagnóstico ou tratamento.

Todo resultado gerado pelo sistema exige revisão clínica
e farmacêutica antes de qualquer intervenção.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.clinical_pharmacy_engine.assessment.models import (
    BaseAssessmentResult,
)


# ============================================================
# Enums gerais
# ============================================================


class CardiovascularSex(str, Enum):
    """Sexo biológico utilizado por equações cardiovasculares."""

    FEMALE = "female"
    MALE = "male"
    UNDETERMINED = "undetermined"


class SmokingStatus(str, Enum):
    """Situação relacionada ao tabagismo."""

    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"
    UNKNOWN = "unknown"


class DiabetesStatus(str, Enum):
    """Situação clínica relacionada ao diabetes."""

    NONE = "none"
    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    OTHER = "other"
    UNKNOWN = "unknown"


class CardiovascularRiskCategory(str, Enum):
    """Categoria geral de risco cardiovascular."""

    LOW = "low"
    BORDERLINE = "borderline"
    MODERATE = "moderate"
    INTERMEDIATE = "intermediate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"
    UNDETERMINED = "undetermined"


class RiskEquationType(str, Enum):
    """Equações ou famílias de estimadores cardiovasculares."""

    ASCVD_PCE = "ascvd_pooled_cohort_equations"
    PREVENT = "prevent"
    FRAMINGHAM = "framingham"
    SCORE2 = "score2"
    SCORE2_OP = "score2_op"
    SCORE2_LAC = "score2_lac"
    REYNOLDS = "reynolds"
    QRISK3 = "qrisk3"
    OTHER = "other"


class PreventionContext(str, Enum):
    """Contexto preventivo do paciente."""

    PRIMARY = "primary_prevention"
    SECONDARY = "secondary_prevention"
    UNDETERMINED = "undetermined"


# ============================================================
# Pressão arterial
# ============================================================


class BloodPressureContext(str, Enum):
    """Contexto em que a pressão foi obtida."""

    OFFICE = "office"
    HOME = "home"
    AMBULATORY_DAYTIME = "ambulatory_daytime"
    AMBULATORY_NIGHTTIME = "ambulatory_nighttime"
    AMBULATORY_24H = "ambulatory_24h"
    EMERGENCY = "emergency"
    INPATIENT = "inpatient"
    UNKNOWN = "unknown"


class BloodPressureClassification(str, Enum):
    """Classificação genérica de pressão arterial."""

    OPTIMAL = "optimal"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH_NORMAL = "high_normal"
    HYPERTENSION_STAGE_1 = "hypertension_stage_1"
    HYPERTENSION_STAGE_2 = "hypertension_stage_2"
    HYPERTENSION_STAGE_3 = "hypertension_stage_3"
    ISOLATED_SYSTOLIC_HYPERTENSION = (
        "isolated_systolic_hypertension"
    )
    HYPERTENSIVE_CRISIS = "hypertensive_crisis"
    HYPOTENSION = "hypotension"
    UNDETERMINED = "undetermined"


class HypertensionGuideline(str, Enum):
    """Referencial de classificação pressórica."""

    AHA_ACC = "aha_acc"
    ESC_ESH = "esc_esh"
    SBC = "sbc"
    WHO = "who"
    CUSTOM = "custom"


class HypertensionPhenotype(str, Enum):
    """Fenótipos clínicos relacionados à pressão arterial."""

    NONE = "none"
    SUSTAINED = "sustained"
    WHITE_COAT = "white_coat"
    MASKED = "masked"
    RESISTANT = "resistant"
    APPARENT_RESISTANT = "apparent_resistant"
    REFRACTORY = "refractory"
    ISOLATED_SYSTOLIC = "isolated_systolic"
    ISOLATED_DIASTOLIC = "isolated_diastolic"
    UNDETERMINED = "undetermined"


class HypertensiveEventType(str, Enum):
    """Sinalização de elevação pressórica aguda."""

    NONE = "none"
    SEVERE_ASYMPTOMATIC = "severe_asymptomatic"
    POSSIBLE_URGENCY = "possible_urgency"
    POSSIBLE_EMERGENCY = "possible_emergency"
    UNDETERMINED = "undetermined"


@dataclass(slots=True)
class BloodPressureMeasurement:
    """Registro individual de pressão arterial."""

    systolic_mm_hg: float
    diastolic_mm_hg: float

    heart_rate_bpm: float | None = None

    context: BloodPressureContext = (
        BloodPressureContext.OFFICE
    )

    measurement_date: str | None = None
    arm: str | None = None
    position: str | None = None
    device_type: str | None = None

    rested_before_measurement: bool | None = None
    validated_device: bool | None = None

    symptoms: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    @property
    def pulse_pressure(self) -> float:
        """Calcula pressão de pulso."""

        return round(
            self.systolic_mm_hg
            - self.diastolic_mm_hg,
            2,
        )

    @property
    def mean_arterial_pressure(self) -> float:
        """Calcula pressão arterial média aproximada."""

        value = (
            self.diastolic_mm_hg
            + (
                self.systolic_mm_hg
                - self.diastolic_mm_hg
            )
            / 3.0
        )

        return round(value, 2)


@dataclass(slots=True)
class HypertensionAssessmentResult:
    """Resultado da avaliação de pressão arterial."""

    classification: BloodPressureClassification = (
        BloodPressureClassification.UNDETERMINED
    )

    guideline: HypertensionGuideline = (
        HypertensionGuideline.CUSTOM
    )

    phenotype: HypertensionPhenotype = (
        HypertensionPhenotype.UNDETERMINED
    )

    acute_event_type: HypertensiveEventType = (
        HypertensiveEventType.UNDETERMINED
    )

    average_systolic_mm_hg: float | None = None
    average_diastolic_mm_hg: float | None = None

    target_systolic_mm_hg: float | None = None
    target_diastolic_mm_hg: float | None = None

    valid: bool = False

    measurements_used: int = 0

    possible_target_organ_damage: bool = False
    requires_confirmation: bool = True
    requires_immediate_evaluation: bool = False

    warnings: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Perfil lipídico
# ============================================================


class LipidUnit(str, Enum):
    """Unidade dos parâmetros lipídicos."""

    MG_DL = "mg/dL"
    MMOL_L = "mmol/L"


class StatinIntensity(str, Enum):
    """Intensidade terapêutica da estatina."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    MAXIMALLY_TOLERATED = "maximally_tolerated"
    UNDETERMINED = "undetermined"


class LipidTargetStatus(str, Enum):
    """Situação em relação ao alvo lipídico."""

    AT_TARGET = "at_target"
    ABOVE_TARGET = "above_target"
    FAR_ABOVE_TARGET = "far_above_target"
    BELOW_SAFETY_THRESHOLD = "below_safety_threshold"
    UNDETERMINED = "undetermined"


@dataclass(slots=True)
class LipidProfile:
    """Perfil lipídico utilizado na avaliação."""

    total_cholesterol: float | None = None
    ldl_cholesterol: float | None = None
    hdl_cholesterol: float | None = None
    triglycerides: float | None = None

    non_hdl_cholesterol: float | None = None
    apolipoprotein_b: float | None = None
    lipoprotein_a: float | None = None

    unit: LipidUnit = LipidUnit.MG_DL

    fasting: bool | None = None
    collected_at: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class DyslipidemiaAssessmentResult:
    """Resultado da avaliação de dislipidemia."""

    risk_category: CardiovascularRiskCategory = (
        CardiovascularRiskCategory.UNDETERMINED
    )

    ldl_target: float | None = None
    non_hdl_target: float | None = None
    apolipoprotein_b_target: float | None = None

    current_ldl: float | None = None
    current_non_hdl: float | None = None
    current_apolipoprotein_b: float | None = None

    ldl_status: LipidTargetStatus = (
        LipidTargetStatus.UNDETERMINED
    )

    suggested_statin_intensity: StatinIntensity = (
        StatinIntensity.UNDETERMINED
    )

    familial_hypercholesterolemia_suspected: bool = False
    severe_hypertriglyceridemia: bool = False

    valid: bool = False

    warnings: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Insuficiência cardíaca
# ============================================================


class NYHAClass(str, Enum):
    """Classes funcionais NYHA."""

    I = "I"
    II = "II"
    III = "III"
    IV = "IV"
    UNDETERMINED = "undetermined"


class HeartFailurePhenotype(str, Enum):
    """Fenótipo definido pela fração de ejeção."""

    HFrEF = "heart_failure_reduced_ef"
    HFmrEF = "heart_failure_mildly_reduced_ef"
    HFpEF = "heart_failure_preserved_ef"
    HFimpEF = "heart_failure_improved_ef"
    RIGHT_SIDED = "right_sided_heart_failure"
    UNDETERMINED = "undetermined"


class CongestionStatus(str, Enum):
    """Estado clínico de congestão."""

    NONE = "none"
    POSSIBLE = "possible"
    PRESENT = "present"
    SEVERE = "severe"
    UNDETERMINED = "undetermined"


class PerfusionStatus(str, Enum):
    """Estado clínico simplificado de perfusão."""

    WARM = "warm"
    COLD = "cold"
    UNDETERMINED = "undetermined"


@dataclass(slots=True)
class EchocardiogramData:
    """Dados essenciais de ecocardiograma."""

    left_ventricular_ejection_fraction_percent: (
        float | None
    ) = None

    previous_ejection_fraction_percent: (
        float | None
    ) = None

    left_atrial_volume_index_ml_m2: float | None = None
    e_over_e_prime: float | None = None
    tricuspid_regurgitation_velocity_m_s: (
        float | None
    ) = None

    left_ventricular_hypertrophy: bool | None = None
    right_ventricular_dysfunction: bool | None = None
    significant_valvular_disease: bool | None = None

    report_date: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class HeartFailureAssessmentResult:
    """Resultado da avaliação de insuficiência cardíaca."""

    phenotype: HeartFailurePhenotype = (
        HeartFailurePhenotype.UNDETERMINED
    )

    nyha_class: NYHAClass = NYHAClass.UNDETERMINED

    congestion_status: CongestionStatus = (
        CongestionStatus.UNDETERMINED
    )

    perfusion_status: PerfusionStatus = (
        PerfusionStatus.UNDETERMINED
    )

    ejection_fraction_percent: float | None = None

    acute_decompensation_suspected: bool = False
    cardiogenic_shock_suspected: bool = False

    guideline_directed_therapy_review_required: bool = False

    valid: bool = False

    warnings: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Síndrome coronariana
# ============================================================


class ChestPainType(str, Enum):
    """Caracterização inicial da dor torácica."""

    NON_ANGINAL = "non_anginal"
    ATYPICAL_ANGINA = "atypical_angina"
    TYPICAL_ANGINA = "typical_angina"
    POSSIBLE_ACS = "possible_acute_coronary_syndrome"
    UNDETERMINED = "undetermined"


class ACSRiskCategory(str, Enum):
    """Categoria de risco para síndrome coronariana."""

    LOW = "low"
    INTERMEDIATE = "intermediate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNDETERMINED = "undetermined"


@dataclass(slots=True)
class AcuteCoronaryAssessmentResult:
    """Resultado integrado da avaliação coronariana."""

    chest_pain_type: ChestPainType = (
        ChestPainType.UNDETERMINED
    )

    heart_score: int | None = None
    heart_risk: ACSRiskCategory = (
        ACSRiskCategory.UNDETERMINED
    )

    timi_score: int | None = None
    timi_risk: ACSRiskCategory = (
        ACSRiskCategory.UNDETERMINED
    )

    grace_score: int | None = None
    grace_risk: ACSRiskCategory = (
        ACSRiskCategory.UNDETERMINED
    )

    acute_coronary_syndrome_suspected: bool = False
    stemi_suspected: bool = False
    immediate_evaluation_required: bool = False

    valid: bool = False

    warnings: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Anticoagulação
# ============================================================


class AtrialFibrillationType(str, Enum):
    """Tipo clínico de fibrilação atrial."""

    NONE = "none"
    FIRST_DIAGNOSED = "first_diagnosed"
    PAROXYSMAL = "paroxysmal"
    PERSISTENT = "persistent"
    LONG_STANDING_PERSISTENT = (
        "long_standing_persistent"
    )
    PERMANENT = "permanent"
    UNDETERMINED = "undetermined"


class AnticoagulantType(str, Enum):
    """Classes de anticoagulantes relevantes."""

    NONE = "none"
    WARFARIN = "warfarin"
    APIXABAN = "apixaban"
    DABIGATRAN = "dabigatran"
    EDOXABAN = "edoxaban"
    RIVAROXABAN = "rivaroxaban"
    UNFRACTIONATED_HEPARIN = (
        "unfractionated_heparin"
    )
    LOW_MOLECULAR_WEIGHT_HEPARIN = (
        "low_molecular_weight_heparin"
    )
    FONDAPARINUX = "fondaparinux"
    OTHER = "other"


class ThromboembolicRiskCategory(str, Enum):
    """Risco tromboembólico estimado."""

    LOW = "low"
    INTERMEDIATE = "intermediate"
    HIGH = "high"
    UNDETERMINED = "undetermined"


class BleedingRiskCategory(str, Enum):
    """Risco hemorrágico estimado."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNDETERMINED = "undetermined"


@dataclass(slots=True)
class AnticoagulationAssessmentResult:
    """Resultado da avaliação de anticoagulação."""

    atrial_fibrillation_type: AtrialFibrillationType = (
        AtrialFibrillationType.UNDETERMINED
    )

    cha2ds2_vasc_score: int | None = None

    thromboembolic_risk: ThromboembolicRiskCategory = (
        ThromboembolicRiskCategory.UNDETERMINED
    )

    has_bled_score: int | None = None

    bleeding_risk: BleedingRiskCategory = (
        BleedingRiskCategory.UNDETERMINED
    )

    current_anticoagulant: AnticoagulantType = (
        AnticoagulantType.NONE
    )

    anticoagulation_review_required: bool = False
    dose_review_required: bool = False
    interaction_review_required: bool = False

    contraindication_suspected: bool = False
    active_bleeding_alert: bool = False

    valid: bool = False

    warnings: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Intervalo QT
# ============================================================


class QTCorrectionFormula(str, Enum):
    """Fórmulas disponíveis para correção do QT."""

    BAZETT = "bazett"
    FRIDERICIA = "fridericia"
    FRAMINGHAM = "framingham"
    HODGES = "hodges"


class QTClassification(str, Enum):
    """Classificação clínica simplificada do QTc."""

    NORMAL = "normal"
    BORDERLINE = "borderline"
    PROLONGED = "prolonged"
    MARKEDLY_PROLONGED = "markedly_prolonged"
    EXTREME = "extreme"
    UNDETERMINED = "undetermined"


class TorsadesRiskCategory(str, Enum):
    """Categoria de risco para torsades de pointes."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNDETERMINED = "undetermined"


@dataclass(slots=True)
class ECGData:
    """Dados eletrocardiográficos básicos."""

    heart_rate_bpm: float | None = None

    qt_interval_ms: float | None = None
    rr_interval_ms: float | None = None
    qrs_duration_ms: float | None = None

    corrected_qt_ms: float | None = None

    rhythm: str | None = None

    atrial_fibrillation_present: bool = False
    ventricular_arrhythmia_present: bool = False
    bundle_branch_block_present: bool = False
    paced_rhythm: bool = False

    report_date: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class QTAssessmentResult:
    """Resultado da avaliação do intervalo QT."""

    raw_qt_ms: float | None = None
    heart_rate_bpm: float | None = None

    qtc_bazett_ms: float | None = None
    qtc_fridericia_ms: float | None = None
    qtc_framingham_ms: float | None = None
    qtc_hodges_ms: float | None = None

    preferred_qtc_ms: float | None = None

    preferred_formula: QTCorrectionFormula | None = None

    classification: QTClassification = (
        QTClassification.UNDETERMINED
    )

    torsades_risk: TorsadesRiskCategory = (
        TorsadesRiskCategory.UNDETERMINED
    )

    qt_prolonging_medications: list[str] = field(
        default_factory=list,
    )

    electrolyte_risk_present: bool = False
    immediate_review_required: bool = False

    valid: bool = False

    warnings: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Resultado de equações de risco
# ============================================================


@dataclass(slots=True)
class CardiovascularRiskEstimate:
    """Resultado padronizado de uma equação de risco."""

    equation: RiskEquationType

    risk_percent_10_years: float | None = None
    risk_percent_30_years: float | None = None

    risk_category: CardiovascularRiskCategory = (
        CardiovascularRiskCategory.UNDETERMINED
    )

    endpoint: str = ""
    population: str = ""

    valid: bool = False

    missing_fields: list[str] = field(
        default_factory=list,
    )

    warnings: list[str] = field(
        default_factory=list,
    )

    calculation_version: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Entrada integrada
# ============================================================


@dataclass(slots=True)
class CardiovascularAssessmentInput:
    """Dados clínicos utilizados pelo motor cardiovascular."""

    age_years: float | None = None

    biological_sex: CardiovascularSex = (
        CardiovascularSex.UNDETERMINED
    )

    ethnicity_or_population_group: str | None = None
    country_or_region: str | None = None

    smoking_status: SmokingStatus = (
        SmokingStatus.UNKNOWN
    )

    diabetes_status: DiabetesStatus = (
        DiabetesStatus.UNKNOWN
    )

    prevention_context: PreventionContext = (
        PreventionContext.UNDETERMINED
    )

    blood_pressure_measurements: list[
        BloodPressureMeasurement
    ] = field(
        default_factory=list,
    )

    treated_hypertension: bool = False

    lipid_profile: LipidProfile = field(
        default_factory=LipidProfile,
    )

    body_mass_index: float | None = None
    waist_circumference_cm: float | None = None

    egfr_ml_min_1_73m2: float | None = None
    creatinine_clearance_ml_min: float | None = None

    family_history_premature_cvd: bool = False

    established_ascvd: bool = False
    prior_myocardial_infarction: bool = False
    prior_stroke_or_tia: bool = False
    peripheral_arterial_disease: bool = False

    chronic_kidney_disease: bool = False
    heart_failure: bool = False
    atrial_fibrillation: bool = False

    atrial_fibrillation_type: AtrialFibrillationType = (
        AtrialFibrillationType.UNDETERMINED
    )

    vascular_disease: bool = False
    hypertension_history: bool = False

    previous_major_bleeding: bool = False
    active_bleeding: bool = False

    labile_inr: bool = False
    alcohol_use_risk: bool = False

    liver_disease: bool = False
    renal_disease: bool = False

    echocardiogram: EchocardiogramData = field(
        default_factory=EchocardiogramData,
    )

    ecg: ECGData = field(
        default_factory=ECGData,
    )

    nyha_class: NYHAClass = NYHAClass.UNDETERMINED

    chest_pain_present: bool = False
    dyspnea_present: bool = False
    syncope_present: bool = False
    palpitations_present: bool = False

    edema_present: bool = False
    orthopnea_present: bool = False
    pulmonary_rales_present: bool = False
    jugular_venous_distension_present: bool = False

    troponin_value: float | None = None
    troponin_upper_reference_limit: float | None = None

    bnp_pg_ml: float | None = None
    nt_probnp_pg_ml: float | None = None

    potassium_mmol_l: float | None = None
    magnesium_mg_dl: float | None = None
    calcium_mg_dl: float | None = None

    current_anticoagulant: AnticoagulantType = (
        AnticoagulantType.NONE
    )

    medications: list[str] = field(
        default_factory=list,
    )

    qt_prolonging_medications: list[str] = field(
        default_factory=list,
    )

    antiplatelet_medications: list[str] = field(
        default_factory=list,
    )

    diagnoses: list[str] = field(
        default_factory=list,
    )

    symptoms: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Resultado integrado
# ============================================================


@dataclass(slots=True)
class CardiovascularAssessmentResult(
    BaseAssessmentResult
):
    """Resultado completo do motor cardiovascular."""

    risk_estimates: list[
        CardiovascularRiskEstimate
    ] = field(
        default_factory=list,
    )

    overall_cardiovascular_risk: (
        CardiovascularRiskCategory
    ) = CardiovascularRiskCategory.UNDETERMINED

    hypertension: HypertensionAssessmentResult = field(
        default_factory=HypertensionAssessmentResult,
    )

    dyslipidemia: DyslipidemiaAssessmentResult = field(
        default_factory=DyslipidemiaAssessmentResult,
    )

    heart_failure: HeartFailureAssessmentResult = field(
        default_factory=HeartFailureAssessmentResult,
    )

    acute_coronary: AcuteCoronaryAssessmentResult = field(
        default_factory=AcuteCoronaryAssessmentResult,
    )

    anticoagulation: AnticoagulationAssessmentResult = field(
        default_factory=AnticoagulationAssessmentResult,
    )

    qt_assessment: QTAssessmentResult = field(
        default_factory=QTAssessmentResult,
    )

    secondary_prevention: bool = False

    medication_review_required: bool = False
    urgent_medical_evaluation_required: bool = False
    emergency_referral_required: bool = False
    cardiology_review_suggested: bool = False

    calculated_values: dict[str, float] = field(
        default_factory=dict,
    )