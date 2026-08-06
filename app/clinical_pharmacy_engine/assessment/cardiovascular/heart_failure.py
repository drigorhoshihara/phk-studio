"""
PHK Studio
Clinical Pharmacy Engine

Heart Failure Assessment Engine.

Responsabilidades:

- classificar insuficiência cardíaca pela fração de ejeção;
- reconhecer HFrEF, HFmrEF, HFpEF e HFimpEF;
- avaliar sinais e sintomas de congestão;
- registrar classe funcional NYHA;
- identificar possível descompensação aguda;
- avaliar perfusão e perfil hemodinâmico;
- analisar peptídeos natriuréticos;
- identificar barreiras à otimização farmacoterapêutica;
- revisar presença dos pilares terapêuticos da HFrEF;
- gerar alertas estruturados e auditáveis.

O módulo não:

- confirma diagnóstico isoladamente;
- inicia ou suspende medicamentos;
- seleciona doses automaticamente;
- substitui avaliação clínica, ecocardiográfica ou laboratorial;
- utiliza fração de ejeção isolada para confirmar HFpEF/HFmrEF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable


# ============================================================
# Enums
# ============================================================


class HeartFailurePhenotype(str, Enum):
    """Fenótipo pela fração de ejeção."""

    HFrEF = "hfref"
    HFmrEF = "hfmref"
    HFpEF = "hfpef"
    HFimpEF = "hfimpEF"
    PRESYMPTOMATIC_LV_DYSFUNCTION = (
        "presymptomatic_lv_dysfunction"
    )
    NO_REDUCED_EJECTION_FRACTION = (
        "no_reduced_ejection_fraction"
    )
    UNDETERMINED = "undetermined"


class NYHAClass(str, Enum):
    """Classe funcional NYHA informada."""

    I = "I"
    II = "II"
    III = "III"
    IV = "IV"
    UNDETERMINED = "undetermined"


class HeartFailureStage(str, Enum):
    """Estágio estrutural e clínico."""

    AT_RISK = "at_risk"
    PRE_HEART_FAILURE = "pre_heart_failure"
    SYMPTOMATIC_HEART_FAILURE = (
        "symptomatic_heart_failure"
    )
    ADVANCED_HEART_FAILURE = (
        "advanced_heart_failure"
    )
    UNDETERMINED = "undetermined"


class CongestionCategory(str, Enum):
    """Carga clínica de congestão."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNDETERMINED = "undetermined"


class PerfusionCategory(str, Enum):
    """Estado clínico de perfusão."""

    ADEQUATE = "adequate"
    POSSIBLY_REDUCED = "possibly_reduced"
    REDUCED = "reduced"
    UNDETERMINED = "undetermined"


class HemodynamicProfile(str, Enum):
    """Perfil simplificado quente/frio e seco/congesto."""

    WARM_DRY = "warm_dry"
    WARM_WET = "warm_wet"
    COLD_DRY = "cold_dry"
    COLD_WET = "cold_wet"
    UNDETERMINED = "undetermined"


class NatriureticPeptideStatus(str, Enum):
    """Interpretação operacional dos peptídeos."""

    NOT_AVAILABLE = "not_available"
    BELOW_CONFIGURED_THRESHOLD = (
        "below_configured_threshold"
    )
    ELEVATED = "elevated"
    MARKEDLY_ELEVATED = "markedly_elevated"
    REQUIRES_CONTEXTUAL_INTERPRETATION = (
        "requires_contextual_interpretation"
    )


class HeartFailureUrgency(str, Enum):
    """Prioridade clínica identificada."""

    ROUTINE = "routine"
    PRIORITY = "priority"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    UNDETERMINED = "undetermined"


class MedicationPillarStatus(str, Enum):
    """Estado de cada pilar farmacoterapêutico."""

    PRESENT = "present"
    ABSENT = "absent"
    CONTRAINDICATION_REPORTED = (
        "contraindication_reported"
    )
    INTOLERANCE_REPORTED = "intolerance_reported"
    NOT_ASSESSED = "not_assessed"
    NOT_APPLICABLE = "not_applicable"


# ============================================================
# Entrada
# ============================================================


@dataclass(slots=True)
class HeartFailureAssessmentInput:
    """Entrada normalizada da avaliação."""

    age_years: float | None = None

    current_lvef_percent: float | None = None
    previous_lvef_percent: float | None = None

    heart_failure_diagnosis_established: bool = False
    structural_heart_disease_present: bool = False
    increased_filling_pressure_evidence: bool = False

    nyha_class: NYHAClass = NYHAClass.UNDETERMINED

    dyspnea: bool = False
    exertional_dyspnea: bool = False
    orthopnea: bool = False
    paroxysmal_nocturnal_dyspnea: bool = False
    fatigue: bool = False
    reduced_exercise_tolerance: bool = False

    peripheral_edema: bool = False
    pulmonary_rales: bool = False
    elevated_jugular_venous_pressure: bool = False
    ascites: bool = False
    pulmonary_edema: bool = False
    rapid_weight_gain: bool = False

    weight_gain_kg: float | None = None
    weight_gain_period_days: float | None = None

    systolic_blood_pressure_mm_hg: float | None = None
    diastolic_blood_pressure_mm_hg: float | None = None
    heart_rate_bpm: float | None = None
    oxygen_saturation_percent: float | None = None

    cool_extremities: bool = False
    altered_mental_status: bool = False
    oliguria: bool = False
    dizziness_or_presyncope: bool = False
    syncope: bool = False

    chest_pain_suspected_ischemic: bool = False
    sustained_ventricular_arrhythmia: bool = False

    bnp_pg_ml: float | None = None
    nt_pro_bnp_pg_ml: float | None = None

    creatinine_mg_dl: float | None = None
    egfr_ml_min_1_73m2: float | None = None

    potassium_mmol_l: float | None = None
    sodium_mmol_l: float | None = None

    acute_kidney_injury_suspected: bool = False
    worsening_renal_function: bool = False

    active_hyperkalemia: bool = False
    symptomatic_hypotension: bool = False
    symptomatic_bradycardia: bool = False

    ace_inhibitor_present: bool = False
    arb_present: bool = False
    arni_present: bool = False

    evidence_based_beta_blocker_present: bool = False
    mineralocorticoid_receptor_antagonist_present: bool = False
    sglt2_inhibitor_present: bool = False

    loop_diuretic_present: bool = False
    hydralazine_nitrate_present: bool = False
    ivabradine_present: bool = False
    digoxin_present: bool = False

    renin_angiotensin_system_intolerance: bool = False
    beta_blocker_intolerance: bool = False
    mra_intolerance: bool = False
    sglt2_inhibitor_intolerance: bool = False

    renin_angiotensin_system_contraindication: bool = False
    beta_blocker_contraindication: bool = False
    mra_contraindication: bool = False
    sglt2_inhibitor_contraindication: bool = False

    recent_hospitalization_for_heart_failure: bool = False
    recurrent_hospitalizations: bool = False

    persistent_symptoms_despite_therapy: bool = False
    inotrope_dependence: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Configuração
# ============================================================


@dataclass(slots=True)
class HeartFailureAssessmentConfig:
    """Limites operacionais configuráveis."""

    hfref_max_lvef_percent: float = 40.0
    hfmref_max_lvef_percent: float = 49.0

    hypotension_systolic_threshold_mm_hg: float = 90.0
    severe_hypotension_systolic_mm_hg: float = 80.0

    bradycardia_threshold_bpm: float = 50.0
    severe_bradycardia_threshold_bpm: float = 40.0

    tachycardia_threshold_bpm: float = 100.0
    severe_tachycardia_threshold_bpm: float = 130.0

    hypoxemia_threshold_percent: float = 92.0
    severe_hypoxemia_threshold_percent: float = 88.0

    hyperkalemia_threshold_mmol_l: float = 5.5
    severe_hyperkalemia_threshold_mmol_l: float = 6.0

    hyponatremia_threshold_mmol_l: float = 135.0
    severe_hyponatremia_threshold_mmol_l: float = 125.0

    severe_renal_impairment_egfr: float = 30.0

    rapid_weight_gain_kg: float = 2.0
    rapid_weight_gain_days: float = 3.0

    outpatient_bnp_threshold_pg_ml: float = 35.0
    outpatient_nt_pro_bnp_threshold_pg_ml: float = 125.0

    marked_bnp_multiplier: float = 5.0

    advanced_hf_nyha_classes: tuple[NYHAClass, ...] = (
        NYHAClass.III,
        NYHAClass.IV,
    )


@dataclass(slots=True)
class HeartFailureValidation:
    """Resultado da validação."""

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


# ============================================================
# Resultados
# ============================================================


@dataclass(slots=True)
class HeartFailurePillarReview:
    """Revisão dos pilares terapêuticos da HFrEF."""

    renin_angiotensin_system: MedicationPillarStatus = (
        MedicationPillarStatus.NOT_ASSESSED
    )

    evidence_based_beta_blocker: MedicationPillarStatus = (
        MedicationPillarStatus.NOT_ASSESSED
    )

    mineralocorticoid_receptor_antagonist: (
        MedicationPillarStatus
    ) = MedicationPillarStatus.NOT_ASSESSED

    sglt2_inhibitor: MedicationPillarStatus = (
        MedicationPillarStatus.NOT_ASSESSED
    )

    present_count: int = 0
    assessable_count: int = 0

    missing_pillars: list[str] = field(
        default_factory=list,
    )

    barriers: list[str] = field(
        default_factory=list,
    )


@dataclass(slots=True)
class HeartFailureAssessmentResult:
    """Resultado integrado."""

    valid: bool = False

    phenotype: HeartFailurePhenotype = (
        HeartFailurePhenotype.UNDETERMINED
    )

    stage: HeartFailureStage = (
        HeartFailureStage.UNDETERMINED
    )

    nyha_class: NYHAClass = NYHAClass.UNDETERMINED

    congestion: CongestionCategory = (
        CongestionCategory.UNDETERMINED
    )

    perfusion: PerfusionCategory = (
        PerfusionCategory.UNDETERMINED
    )

    hemodynamic_profile: HemodynamicProfile = (
        HemodynamicProfile.UNDETERMINED
    )

    natriuretic_peptide_status: (
        NatriureticPeptideStatus
    ) = NatriureticPeptideStatus.NOT_AVAILABLE

    urgency: HeartFailureUrgency = (
        HeartFailureUrgency.UNDETERMINED
    )

    possible_acute_decompensation: bool = False
    advanced_heart_failure_signal: bool = False

    pillar_review: HeartFailurePillarReview = field(
        default_factory=HeartFailurePillarReview,
    )

    immediate_evaluation_required: bool = False
    medication_review_required: bool = False
    specialist_review_required: bool = False

    alerts: list[str] = field(
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


class HeartFailureAssessmentEngine:
    """Motor estruturado de insuficiência cardíaca."""

    def __init__(
        self,
        config: HeartFailureAssessmentConfig | None = None,
    ) -> None:
        self.config = (
            config
            or HeartFailureAssessmentConfig()
        )

    def assess(
        self,
        data: HeartFailureAssessmentInput,
    ) -> HeartFailureAssessmentResult:
        """Executa a avaliação completa."""

        validation = self.validate(data)

        if not validation.valid:
            return HeartFailureAssessmentResult(
                valid=False,
                nyha_class=data.nyha_class,
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

        phenotype = self.classify_phenotype(data)
        stage = self.classify_stage(data)

        congestion_score = self.calculate_congestion_score(
            data
        )

        congestion = self.classify_congestion(
            congestion_score
        )

        perfusion_score = self.calculate_hypoperfusion_score(
            data
        )

        perfusion = self.classify_perfusion(
            perfusion_score
        )

        hemodynamic_profile = (
            self.resolve_hemodynamic_profile(
                congestion=congestion,
                perfusion=perfusion,
            )
        )

        peptide_status = (
            self.classify_natriuretic_peptides(data)
        )

        alerts = self.detect_critical_alerts(data)

        possible_decompensation = (
            self.detect_possible_acute_decompensation(
                data=data,
                congestion=congestion,
                perfusion=perfusion,
            )
        )

        advanced_signal = (
            self.detect_advanced_heart_failure(data)
        )

        pillar_review = self.review_pillars(
            data=data,
            phenotype=phenotype,
        )

        urgency = self.resolve_urgency(
            data=data,
            alerts=alerts,
            possible_decompensation=(
                possible_decompensation
            ),
            congestion=congestion,
            perfusion=perfusion,
            advanced_signal=advanced_signal,
        )

        warnings = list(validation.warnings)
        recommendations: list[str] = []

        if (
            phenotype
            in {
                HeartFailurePhenotype.HFmrEF,
                HeartFailurePhenotype.HFpEF,
            }
            and not data.increased_filling_pressure_evidence
        ):
            warnings.append(
                "A fração de ejeção isolada não confirma "
                "HFmrEF ou HFpEF. É necessária evidência "
                "clínica ou objetiva adicional de aumento "
                "das pressões de enchimento."
            )

        if possible_decompensation:
            warnings.append(
                "Conjunto de achados compatível com possível "
                "descompensação de insuficiência cardíaca."
            )

        if congestion in {
            CongestionCategory.MODERATE,
            CongestionCategory.SEVERE,
        }:
            recommendations.append(
                "Revisar estado volêmico, adesão, ingestão "
                "de sódio e líquidos, diuréticos e causas "
                "precipitantes."
            )

        if perfusion == PerfusionCategory.REDUCED:
            recommendations.append(
                "Avaliar imediatamente sinais de baixo débito "
                "e comprometimento de órgãos-alvo."
            )

        if data.active_hyperkalemia:
            recommendations.append(
                "Confirmar potássio, eletrocardiograma, função "
                "renal e medicamentos associados."
            )

        if (
            data.worsening_renal_function
            or data.acute_kidney_injury_suspected
        ):
            recommendations.append(
                "Reavaliar perfusão, congestão, função renal, "
                "diuréticos e medicamentos hemodinamicamente "
                "ativos."
            )

        if pillar_review.missing_pillars:
            recommendations.append(
                "Revisar elegibilidade, tolerabilidade e "
                "barreiras aos pilares farmacoterapêuticos "
                "ausentes."
            )

        if advanced_signal:
            recommendations.append(
                "Considerar avaliação por equipe especializada "
                "em insuficiência cardíaca avançada."
            )

        medication_review_required = any(
            (
                bool(pillar_review.missing_pillars),
                bool(pillar_review.barriers),
                data.active_hyperkalemia,
                data.symptomatic_hypotension,
                data.symptomatic_bradycardia,
                data.worsening_renal_function,
                data.recent_hospitalization_for_heart_failure,
                possible_decompensation,
            )
        )

        specialist_review_required = any(
            (
                advanced_signal,
                phenotype
                == HeartFailurePhenotype.HFimpEF,
                data.recurrent_hospitalizations,
                data.sustained_ventricular_arrhythmia,
                data.inotrope_dependence,
            )
        )

        immediate_evaluation_required = (
            urgency
            in {
                HeartFailureUrgency.URGENT,
                HeartFailureUrgency.EMERGENCY,
            }
        )

        return HeartFailureAssessmentResult(
            valid=True,
            phenotype=phenotype,
            stage=stage,
            nyha_class=data.nyha_class,
            congestion=congestion,
            perfusion=perfusion,
            hemodynamic_profile=hemodynamic_profile,
            natriuretic_peptide_status=peptide_status,
            urgency=urgency,
            possible_acute_decompensation=(
                possible_decompensation
            ),
            advanced_heart_failure_signal=(
                advanced_signal
            ),
            pillar_review=pillar_review,
            immediate_evaluation_required=(
                immediate_evaluation_required
            ),
            medication_review_required=(
                medication_review_required
            ),
            specialist_review_required=(
                specialist_review_required
            ),
            alerts=self._unique_strings(alerts),
            warnings=self._unique_strings(warnings),
            recommendations=self._unique_strings(
                recommendations
            ),
            metadata={
                "current_lvef_percent": (
                    data.current_lvef_percent
                ),
                "previous_lvef_percent": (
                    data.previous_lvef_percent
                ),
                "congestion_score": congestion_score,
                "hypoperfusion_score": perfusion_score,
                "four_pillars_present": (
                    pillar_review.present_count
                ),
                "four_pillars_assessable": (
                    pillar_review.assessable_count
                ),
                "heart_failure_diagnosis_established": (
                    data.heart_failure_diagnosis_established
                ),
                "increased_filling_pressure_evidence": (
                    data.increased_filling_pressure_evidence
                ),
            },
        )

    # ========================================================
    # Validação
    # ========================================================

    def validate(
        self,
        data: HeartFailureAssessmentInput,
    ) -> HeartFailureValidation:
        """Valida plausibilidade dos dados."""

        missing: list[str] = []
        invalid: list[str] = []
        warnings: list[str] = []

        if (
            data.current_lvef_percent is None
            and not data.heart_failure_diagnosis_established
            and not data.structural_heart_disease_present
        ):
            warnings.append(
                "Fração de ejeção, diagnóstico estabelecido "
                "e doença estrutural não foram informados."
            )

        self._validate_optional_range(
            "age_years",
            data.age_years,
            0,
            130,
            invalid,
        )

        self._validate_optional_range(
            "current_lvef_percent",
            data.current_lvef_percent,
            0,
            100,
            invalid,
        )

        self._validate_optional_range(
            "previous_lvef_percent",
            data.previous_lvef_percent,
            0,
            100,
            invalid,
        )

        self._validate_optional_range(
            "systolic_blood_pressure_mm_hg",
            data.systolic_blood_pressure_mm_hg,
            30,
            300,
            invalid,
        )

        self._validate_optional_range(
            "diastolic_blood_pressure_mm_hg",
            data.diastolic_blood_pressure_mm_hg,
            10,
            200,
            invalid,
        )

        self._validate_optional_range(
            "heart_rate_bpm",
            data.heart_rate_bpm,
            20,
            250,
            invalid,
        )

        self._validate_optional_range(
            "oxygen_saturation_percent",
            data.oxygen_saturation_percent,
            20,
            100,
            invalid,
        )

        for field_name, value in {
            "bnp_pg_ml": data.bnp_pg_ml,
            "nt_pro_bnp_pg_ml": data.nt_pro_bnp_pg_ml,
            "creatinine_mg_dl": data.creatinine_mg_dl,
            "egfr_ml_min_1_73m2": (
                data.egfr_ml_min_1_73m2
            ),
            "potassium_mmol_l": data.potassium_mmol_l,
            "sodium_mmol_l": data.sodium_mmol_l,
            "weight_gain_kg": data.weight_gain_kg,
            "weight_gain_period_days": (
                data.weight_gain_period_days
            ),
        }.items():
            if value is None:
                continue

            if (
                not self._valid_number(value)
                or float(value) < 0
            ):
                invalid.append(field_name)

        return HeartFailureValidation(
            valid=not missing and not invalid,
            missing_fields=self._unique_strings(missing),
            invalid_fields=self._unique_strings(invalid),
            warnings=self._unique_strings(warnings),
        )

    # ========================================================
    # Fenótipo e estágio
    # ========================================================

    def classify_phenotype(
        self,
        data: HeartFailureAssessmentInput,
    ) -> HeartFailurePhenotype:
        """Classifica pelo histórico da FEVE."""

        current = data.current_lvef_percent
        previous = data.previous_lvef_percent

        if current is None:
            return HeartFailurePhenotype.UNDETERMINED

        if (
            previous is not None
            and previous <= 40
            and current > 40
        ):
            return HeartFailurePhenotype.HFimpEF

        if current <= self.config.hfref_max_lvef_percent:
            if (
                data.heart_failure_diagnosis_established
                or self._has_heart_failure_symptoms(data)
            ):
                return HeartFailurePhenotype.HFrEF

            return (
                HeartFailurePhenotype
                .PRESYMPTOMATIC_LV_DYSFUNCTION
            )

        if (
            current
            <= self.config.hfmref_max_lvef_percent
        ):
            return HeartFailurePhenotype.HFmrEF

        if (
            data.heart_failure_diagnosis_established
            or self._has_heart_failure_symptoms(data)
            or data.increased_filling_pressure_evidence
        ):
            return HeartFailurePhenotype.HFpEF

        return (
            HeartFailurePhenotype
            .NO_REDUCED_EJECTION_FRACTION
        )

    def classify_stage(
        self,
        data: HeartFailureAssessmentInput,
    ) -> HeartFailureStage:
        """Classifica estágio clínico simplificado."""

        if self.detect_advanced_heart_failure(data):
            return HeartFailureStage.ADVANCED_HEART_FAILURE

        if (
            data.heart_failure_diagnosis_established
            or self._has_heart_failure_symptoms(data)
        ):
            return (
                HeartFailureStage
                .SYMPTOMATIC_HEART_FAILURE
            )

        if (
            data.structural_heart_disease_present
            or (
                data.current_lvef_percent is not None
                and data.current_lvef_percent <= 40
            )
            or data.increased_filling_pressure_evidence
        ):
            return HeartFailureStage.PRE_HEART_FAILURE

        return HeartFailureStage.AT_RISK

    # ========================================================
    # Congestão e perfusão
    # ========================================================

    @staticmethod
    def calculate_congestion_score(
        data: HeartFailureAssessmentInput,
    ) -> int:
        """Conta sinais clínicos de congestão."""

        score = 0

        score += int(data.peripheral_edema)
        score += int(data.pulmonary_rales)
        score += int(
            data.elevated_jugular_venous_pressure
        )
        score += int(data.ascites)
        score += int(data.orthopnea)
        score += int(
            data.paroxysmal_nocturnal_dyspnea
        )
        score += int(data.rapid_weight_gain)

        if data.pulmonary_edema:
            score += 3

        return score

    @staticmethod
    def classify_congestion(
        score: int,
    ) -> CongestionCategory:
        """Converte escore em categoria."""

        if score <= 0:
            return CongestionCategory.NONE

        if score <= 2:
            return CongestionCategory.MILD

        if score <= 5:
            return CongestionCategory.MODERATE

        return CongestionCategory.SEVERE

    def calculate_hypoperfusion_score(
        self,
        data: HeartFailureAssessmentInput,
    ) -> int:
        """Conta sinais sugestivos de hipoperfusão."""

        score = 0

        score += int(data.cool_extremities)
        score += int(data.oliguria)
        score += int(data.altered_mental_status)
        score += int(data.dizziness_or_presyncope)

        if data.syncope:
            score += 2

        if (
            data.systolic_blood_pressure_mm_hg
            is not None
            and data.systolic_blood_pressure_mm_hg
            < self.config
            .hypotension_systolic_threshold_mm_hg
        ):
            score += 1

        if (
            data.systolic_blood_pressure_mm_hg
            is not None
            and data.systolic_blood_pressure_mm_hg
            < self.config
            .severe_hypotension_systolic_mm_hg
        ):
            score += 2

        return score

    @staticmethod
    def classify_perfusion(
        score: int,
    ) -> PerfusionCategory:
        """Classifica perfusão."""

        if score <= 0:
            return PerfusionCategory.ADEQUATE

        if score <= 2:
            return PerfusionCategory.POSSIBLY_REDUCED

        return PerfusionCategory.REDUCED

    @staticmethod
    def resolve_hemodynamic_profile(
        *,
        congestion: CongestionCategory,
        perfusion: PerfusionCategory,
    ) -> HemodynamicProfile:
        """Resolve perfil clínico simplificado."""

        wet = congestion in {
            CongestionCategory.MODERATE,
            CongestionCategory.SEVERE,
        }

        cold = perfusion == PerfusionCategory.REDUCED

        if not wet and not cold:
            return HemodynamicProfile.WARM_DRY

        if wet and not cold:
            return HemodynamicProfile.WARM_WET

        if not wet and cold:
            return HemodynamicProfile.COLD_DRY

        return HemodynamicProfile.COLD_WET

    # ========================================================
    # Peptídeos natriuréticos
    # ========================================================

    def classify_natriuretic_peptides(
        self,
        data: HeartFailureAssessmentInput,
    ) -> NatriureticPeptideStatus:
        """Classificação operacional e configurável."""

        values_available = any(
            (
                data.bnp_pg_ml is not None,
                data.nt_pro_bnp_pg_ml is not None,
            )
        )

        if not values_available:
            return NatriureticPeptideStatus.NOT_AVAILABLE

        elevated = False
        markedly_elevated = False

        if data.bnp_pg_ml is not None:
            elevated = (
                elevated
                or data.bnp_pg_ml
                >= self.config.outpatient_bnp_threshold_pg_ml
            )

            markedly_elevated = (
                markedly_elevated
                or data.bnp_pg_ml
                >= (
                    self.config.outpatient_bnp_threshold_pg_ml
                    * self.config.marked_bnp_multiplier
                )
            )

        if data.nt_pro_bnp_pg_ml is not None:
            elevated = (
                elevated
                or data.nt_pro_bnp_pg_ml
                >= (
                    self.config
                    .outpatient_nt_pro_bnp_threshold_pg_ml
                )
            )

            markedly_elevated = (
                markedly_elevated
                or data.nt_pro_bnp_pg_ml
                >= (
                    self.config
                    .outpatient_nt_pro_bnp_threshold_pg_ml
                    * self.config.marked_bnp_multiplier
                )
            )

        if markedly_elevated:
            return (
                NatriureticPeptideStatus.MARKEDLY_ELEVATED
            )

        if elevated:
            return NatriureticPeptideStatus.ELEVATED

        return (
            NatriureticPeptideStatus
            .BELOW_CONFIGURED_THRESHOLD
        )

    # ========================================================
    # Quatro pilares
    # ========================================================

    def review_pillars(
        self,
        *,
        data: HeartFailureAssessmentInput,
        phenotype: HeartFailurePhenotype,
    ) -> HeartFailurePillarReview:
        """Revisa presença e barreiras dos pilares."""

        if phenotype not in {
            HeartFailurePhenotype.HFrEF,
            HeartFailurePhenotype.HFimpEF,
        }:
            return HeartFailurePillarReview(
                renin_angiotensin_system=(
                    MedicationPillarStatus.NOT_APPLICABLE
                ),
                evidence_based_beta_blocker=(
                    MedicationPillarStatus.NOT_APPLICABLE
                ),
                mineralocorticoid_receptor_antagonist=(
                    MedicationPillarStatus.NOT_APPLICABLE
                ),
                sglt2_inhibitor=(
                    MedicationPillarStatus.NOT_APPLICABLE
                ),
            )

        ras_present = any(
            (
                data.ace_inhibitor_present,
                data.arb_present,
                data.arni_present,
            )
        )

        ras = self._pillar_status(
            present=ras_present,
            intolerance=(
                data.renin_angiotensin_system_intolerance
            ),
            contraindication=(
                data
                .renin_angiotensin_system_contraindication
            ),
        )

        beta_blocker = self._pillar_status(
            present=(
                data.evidence_based_beta_blocker_present
            ),
            intolerance=data.beta_blocker_intolerance,
            contraindication=(
                data.beta_blocker_contraindication
            ),
        )

        mra = self._pillar_status(
            present=(
                data
                .mineralocorticoid_receptor_antagonist_present
            ),
            intolerance=data.mra_intolerance,
            contraindication=data.mra_contraindication,
        )

        sglt2 = self._pillar_status(
            present=data.sglt2_inhibitor_present,
            intolerance=(
                data.sglt2_inhibitor_intolerance
            ),
            contraindication=(
                data.sglt2_inhibitor_contraindication
            ),
        )

        statuses = {
            "sistema renina-angiotensina": ras,
            "betabloqueador baseado em evidência": (
                beta_blocker
            ),
            "antagonista do receptor mineralocorticoide": (
                mra
            ),
            "inibidor de SGLT2": sglt2,
        }

        missing = [
            name
            for name, status in statuses.items()
            if status == MedicationPillarStatus.ABSENT
        ]

        barriers = [
            name
            for name, status in statuses.items()
            if status
            in {
                MedicationPillarStatus
                .CONTRAINDICATION_REPORTED,
                MedicationPillarStatus
                .INTOLERANCE_REPORTED,
            }
        ]

        present_count = sum(
            status == MedicationPillarStatus.PRESENT
            for status in statuses.values()
        )

        return HeartFailurePillarReview(
            renin_angiotensin_system=ras,
            evidence_based_beta_blocker=beta_blocker,
            mineralocorticoid_receptor_antagonist=mra,
            sglt2_inhibitor=sglt2,
            present_count=present_count,
            assessable_count=4,
            missing_pillars=missing,
            barriers=barriers,
        )

    @staticmethod
    def _pillar_status(
        *,
        present: bool,
        intolerance: bool,
        contraindication: bool,
    ) -> MedicationPillarStatus:
        """Resolve estado de um pilar."""

        if present:
            return MedicationPillarStatus.PRESENT

        if contraindication:
            return (
                MedicationPillarStatus
                .CONTRAINDICATION_REPORTED
            )

        if intolerance:
            return (
                MedicationPillarStatus.INTOLERANCE_REPORTED
            )

        return MedicationPillarStatus.ABSENT

    # ========================================================
    # Alertas e urgência
    # ========================================================

    def detect_critical_alerts(
        self,
        data: HeartFailureAssessmentInput,
    ) -> list[str]:
        """Detecta condições críticas."""

        alerts: list[str] = []

        if data.pulmonary_edema:
            alerts.append(
                "Edema pulmonar informado."
            )

        if data.altered_mental_status:
            alerts.append(
                "Alteração do estado mental informada."
            )

        if data.syncope:
            alerts.append(
                "Síncope informada."
            )

        if data.chest_pain_suspected_ischemic:
            alerts.append(
                "Dor torácica possivelmente isquêmica."
            )

        if data.sustained_ventricular_arrhythmia:
            alerts.append(
                "Arritmia ventricular sustentada informada."
            )

        if (
            data.oxygen_saturation_percent is not None
            and data.oxygen_saturation_percent
            < self.config.severe_hypoxemia_threshold_percent
        ):
            alerts.append(
                "Hipoxemia grave informada."
            )

        if (
            data.systolic_blood_pressure_mm_hg is not None
            and data.systolic_blood_pressure_mm_hg
            < self.config
            .severe_hypotension_systolic_mm_hg
        ):
            alerts.append(
                "Hipotensão arterial grave informada."
            )

        if (
            data.potassium_mmol_l is not None
            and data.potassium_mmol_l
            >= self.config
            .severe_hyperkalemia_threshold_mmol_l
        ):
            alerts.append(
                "Hipercalemia grave informada."
            )

        return self._unique_strings(alerts)

    def detect_possible_acute_decompensation(
        self,
        *,
        data: HeartFailureAssessmentInput,
        congestion: CongestionCategory,
        perfusion: PerfusionCategory,
    ) -> bool:
        """Detecta conjunto compatível com descompensação."""

        return any(
            (
                data.pulmonary_edema,
                congestion == CongestionCategory.SEVERE,
                perfusion == PerfusionCategory.REDUCED,
                data.rapid_weight_gain
                and congestion
                in {
                    CongestionCategory.MODERATE,
                    CongestionCategory.SEVERE,
                },
                data.recent_hospitalization_for_heart_failure
                and self._has_heart_failure_symptoms(data),
            )
        )

    def detect_advanced_heart_failure(
        self,
        data: HeartFailureAssessmentInput,
    ) -> bool:
        """Sinaliza possível insuficiência avançada."""

        return any(
            (
                data.inotrope_dependence,
                data.recurrent_hospitalizations,
                (
                    data.nyha_class
                    in self.config.advanced_hf_nyha_classes
                    and data.persistent_symptoms_despite_therapy
                ),
            )
        )

    def resolve_urgency(
        self,
        *,
        data: HeartFailureAssessmentInput,
        alerts: list[str],
        possible_decompensation: bool,
        congestion: CongestionCategory,
        perfusion: PerfusionCategory,
        advanced_signal: bool,
    ) -> HeartFailureUrgency:
        """Resolve prioridade clínica."""

        if alerts:
            return HeartFailureUrgency.EMERGENCY

        if (
            possible_decompensation
            or perfusion == PerfusionCategory.REDUCED
            or congestion == CongestionCategory.SEVERE
        ):
            return HeartFailureUrgency.URGENT

        if (
            advanced_signal
            or congestion == CongestionCategory.MODERATE
            or data.recent_hospitalization_for_heart_failure
        ):
            return HeartFailureUrgency.PRIORITY

        return HeartFailureUrgency.ROUTINE

    # ========================================================
    # Utilidades
    # ========================================================

    @staticmethod
    def _has_heart_failure_symptoms(
        data: HeartFailureAssessmentInput,
    ) -> bool:
        """Verifica sintomas ou sinais compatíveis."""

        return any(
            (
                data.dyspnea,
                data.exertional_dyspnea,
                data.orthopnea,
                data.paroxysmal_nocturnal_dyspnea,
                data.fatigue,
                data.reduced_exercise_tolerance,
                data.peripheral_edema,
                data.pulmonary_rales,
                data.elevated_jugular_venous_pressure,
                data.ascites,
                data.pulmonary_edema,
            )
        )

    @staticmethod
    def _validate_optional_range(
        field_name: str,
        value: float | None,
        minimum: float,
        maximum: float,
        invalid: list[str],
    ) -> None:
        """Valida campo numérico opcional."""

        if value is None:
            return

        if (
            not HeartFailureAssessmentEngine
            ._valid_number(value)
            or not minimum <= float(value) <= maximum
        ):
            invalid.append(field_name)

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