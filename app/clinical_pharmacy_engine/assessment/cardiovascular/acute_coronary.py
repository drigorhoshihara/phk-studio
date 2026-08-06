"""
PHK Studio
Clinical Pharmacy Engine

Acute Coronary Syndrome Assessment Engine.

Responsabilidades:

- estruturar a triagem de suspeita de síndrome coronariana aguda;
- avaliar sintomas compatíveis com isquemia miocárdica;
- interpretar dados estruturados de ECG;
- interpretar troponina de forma relativa ao limite superior;
- reconhecer lesão miocárdica aguda;
- distinguir cenários compatíveis com STEMI e NSTE-ACS;
- identificar instabilidade hemodinâmica ou elétrica;
- calcular um HEART Score operacional;
- gerar alertas, recomendações e metadados auditáveis;
- sinalizar necessidade de avaliação emergencial.

O módulo não:

- confirma infarto isoladamente;
- interpreta imagens brutas de ECG;
- substitui avaliação médica emergencial;
- recomenda trombólise automaticamente;
- seleciona estratégia invasiva automaticamente;
- prescreve antiagregantes ou anticoagulantes;
- interpreta troponina sem considerar o ensaio utilizado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable


# ============================================================
# Enums
# ============================================================


class AcuteCoronarySyndromeType(str, Enum):
    """Classificação clínica operacional."""

    STEMI_COMPATIBLE = "stemi_compatible"
    NSTEMI_COMPATIBLE = "nstemi_compatible"
    UNSTABLE_ANGINA_COMPATIBLE = (
        "unstable_angina_compatible"
    )
    ACUTE_MYOCARDIAL_INJURY = (
        "acute_myocardial_injury"
    )
    CHRONIC_MYOCARDIAL_INJURY = (
        "chronic_myocardial_injury"
    )
    ISCHEMIA_UNDER_INVESTIGATION = (
        "ischemia_under_investigation"
    )
    NON_ISCHEMIC_PRESENTATION = (
        "non_ischemic_presentation"
    )
    UNDETERMINED = "undetermined"


class AcuteCoronaryUrgency(str, Enum):
    """Prioridade clínica."""

    ROUTINE = "routine"
    PRIORITY = "priority"
    URGENT = "urgent"
    EMERGENCY = "emergency"
    UNDETERMINED = "undetermined"


class ChestPainPattern(str, Enum):
    """Padrão estruturado da dor."""

    HIGHLY_SUSPICIOUS = "highly_suspicious"
    MODERATELY_SUSPICIOUS = "moderately_suspicious"
    SLIGHTLY_SUSPICIOUS = "slightly_suspicious"
    NON_SUSPICIOUS = "non_suspicious"
    ABSENT = "absent"
    UNDETERMINED = "undetermined"


class ECGIschemiaPattern(str, Enum):
    """Padrão estruturado do ECG."""

    PERSISTENT_ST_ELEVATION = (
        "persistent_st_elevation"
    )
    TRANSIENT_ST_ELEVATION = (
        "transient_st_elevation"
    )
    ST_DEPRESSION = "st_depression"
    T_WAVE_INVERSION = "t_wave_inversion"
    NEW_LEFT_BUNDLE_BRANCH_BLOCK = (
        "new_left_bundle_branch_block"
    )
    POSTERIOR_INFARCTION_PATTERN = (
        "posterior_infarction_pattern"
    )
    RIGHT_VENTRICULAR_INFARCTION_PATTERN = (
        "right_ventricular_infarction_pattern"
    )
    NONSPECIFIC_CHANGES = "nonspecific_changes"
    NORMAL = "normal"
    NOT_AVAILABLE = "not_available"
    UNDETERMINED = "undetermined"


class TroponinStatus(str, Enum):
    """Interpretação operacional da troponina."""

    NOT_AVAILABLE = "not_available"
    BELOW_UPPER_REFERENCE_LIMIT = (
        "below_upper_reference_limit"
    )
    ABOVE_UPPER_REFERENCE_LIMIT = (
        "above_upper_reference_limit"
    )
    RISING_OR_FALLING = "rising_or_falling"
    STABLE_ELEVATION = "stable_elevation"
    UNINTERPRETABLE = "uninterpretable"


class MyocardialInjuryType(str, Enum):
    """Tipo de lesão miocárdica."""

    NONE = "none"
    ACUTE = "acute"
    CHRONIC = "chronic"
    UNDETERMINED = "undetermined"


class HEARTRiskCategory(str, Enum):
    """Categoria do HEART Score."""

    LOW = "low"
    INTERMEDIATE = "intermediate"
    HIGH = "high"
    UNDETERMINED = "undetermined"


class HemodynamicStatus(str, Enum):
    """Estado hemodinâmico simplificado."""

    STABLE = "stable"
    POSSIBLY_UNSTABLE = "possibly_unstable"
    UNSTABLE = "unstable"
    SHOCK = "shock"
    UNDETERMINED = "undetermined"


# ============================================================
# Entrada
# ============================================================


@dataclass(slots=True)
class AcuteCoronaryAssessmentInput:
    """Entrada normalizada para avaliação."""

    age_years: float | None = None

    chest_pain_present: bool = False

    chest_pain_pattern: ChestPainPattern = (
        ChestPainPattern.UNDETERMINED
    )

    chest_pain_duration_minutes: float | None = None
    persistent_chest_pain: bool = False
    recurrent_chest_pain: bool = False

    retrosternal_pain: bool = False
    pressure_or_tightness: bool = False
    exertional_trigger: bool = False
    relief_with_rest: bool = False

    radiation_to_arm_or_jaw: bool = False
    diaphoresis: bool = False
    nausea_or_vomiting: bool = False
    dyspnea: bool = False
    syncope_or_presyncope: bool = False

    atypical_presentation_possible: bool = False

    ecg_pattern: ECGIschemiaPattern = (
        ECGIschemiaPattern.NOT_AVAILABLE
    )

    st_elevation_mm: float | None = None
    st_depression_mm: float | None = None

    dynamic_ecg_changes: bool = False
    reciprocal_changes: bool = False

    ecg_performed_within_10_minutes: bool | None = None

    troponin_value: float | None = None
    troponin_upper_reference_limit: float | None = None

    previous_troponin_value: float | None = None

    troponin_measurement_interval_hours: float | None = None

    known_coronary_artery_disease: bool = False
    previous_myocardial_infarction: bool = False
    previous_pci_or_cabg: bool = False

    hypertension: bool = False
    diabetes: bool = False
    dyslipidemia: bool = False
    current_smoker: bool = False
    family_history_premature_cad: bool = False
    obesity: bool = False

    systolic_blood_pressure_mm_hg: float | None = None
    diastolic_blood_pressure_mm_hg: float | None = None
    heart_rate_bpm: float | None = None
    oxygen_saturation_percent: float | None = None

    altered_mental_status: bool = False
    cool_extremities: bool = False
    oliguria: bool = False

    pulmonary_edema: bool = False
    acute_heart_failure: bool = False

    sustained_ventricular_arrhythmia: bool = False
    cardiac_arrest: bool = False
    mechanical_complication_suspected: bool = False

    active_major_bleeding: bool = False
    recent_intracranial_hemorrhage: bool = False

    severe_thrombocytopenia: bool = False
    platelet_count_per_mm3: float | None = None

    current_anticoagulation: bool = False
    current_antiplatelet_therapy: bool = False

    creatinine_mg_dl: float | None = None
    egfr_ml_min_1_73m2: float | None = None

    hemoglobin_g_dl: float | None = None

    alternative_cause_of_troponin_elevation: bool = False

    suspected_alternative_diagnoses: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Configuração
# ============================================================


@dataclass(slots=True)
class AcuteCoronaryAssessmentConfig:
    """Configurações operacionais."""

    hypotension_threshold_mm_hg: float = 90.0
    severe_hypotension_threshold_mm_hg: float = 80.0

    tachycardia_threshold_bpm: float = 100.0
    severe_tachycardia_threshold_bpm: float = 130.0

    bradycardia_threshold_bpm: float = 50.0
    severe_bradycardia_threshold_bpm: float = 40.0

    hypoxemia_threshold_percent: float = 90.0
    severe_hypoxemia_threshold_percent: float = 85.0

    severe_thrombocytopenia_threshold: float = 50_000.0

    significant_st_depression_mm: float = 0.5
    significant_st_elevation_mm: float = 1.0

    relative_troponin_change_percent: float = 20.0

    heart_low_risk_max_score: int = 3
    heart_intermediate_risk_max_score: int = 6

    prolonged_pain_minutes: float = 20.0


@dataclass(slots=True)
class AcuteCoronaryValidation:
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
class HEARTScoreResult:
    """Resultado do HEART Score."""

    score: int = 0

    category: HEARTRiskCategory = (
        HEARTRiskCategory.UNDETERMINED
    )

    components: dict[str, int] = field(
        default_factory=dict,
    )

    valid: bool = False

    missing_components: list[str] = field(
        default_factory=list,
    )


@dataclass(slots=True)
class AcuteCoronaryAssessmentResult:
    """Resultado integrado."""

    valid: bool = False

    syndrome_type: AcuteCoronarySyndromeType = (
        AcuteCoronarySyndromeType.UNDETERMINED
    )

    urgency: AcuteCoronaryUrgency = (
        AcuteCoronaryUrgency.UNDETERMINED
    )

    hemodynamic_status: HemodynamicStatus = (
        HemodynamicStatus.UNDETERMINED
    )

    troponin_status: TroponinStatus = (
        TroponinStatus.NOT_AVAILABLE
    )

    myocardial_injury_type: MyocardialInjuryType = (
        MyocardialInjuryType.UNDETERMINED
    )

    heart_score: HEARTScoreResult = field(
        default_factory=HEARTScoreResult,
    )

    ischemic_symptom_signal: bool = False
    ischemic_ecg_signal: bool = False
    myocardial_injury_present: bool = False

    possible_stemi: bool = False
    possible_nstemi: bool = False
    possible_unstable_angina: bool = False

    immediate_evaluation_required: bool = False
    cardiology_review_required: bool = False
    medication_review_required: bool = False
    bleeding_risk_review_required: bool = False

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


class AcuteCoronaryAssessmentEngine:
    """Motor estruturado de síndrome coronariana aguda."""

    def __init__(
        self,
        config: AcuteCoronaryAssessmentConfig | None = None,
    ) -> None:
        self.config = (
            config
            or AcuteCoronaryAssessmentConfig()
        )

    def assess(
        self,
        data: AcuteCoronaryAssessmentInput,
    ) -> AcuteCoronaryAssessmentResult:
        """Executa a avaliação completa."""

        validation = self.validate(data)

        if not validation.valid:
            return AcuteCoronaryAssessmentResult(
                valid=False,
                warnings=self._unique_strings(
                    validation.warnings
                    + [
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
                    "invalid_fields": list(
                        validation.invalid_fields
                    ),
                },
            )

        ischemic_symptom_signal = (
            self.detect_ischemic_symptom_signal(data)
        )

        ischemic_ecg_signal = (
            self.detect_ischemic_ecg_signal(data)
        )

        troponin_status = self.classify_troponin(data)

        injury_type = self.classify_myocardial_injury(
            troponin_status
        )

        myocardial_injury_present = (
            injury_type
            in {
                MyocardialInjuryType.ACUTE,
                MyocardialInjuryType.CHRONIC,
            }
        )

        hemodynamic_status = (
            self.classify_hemodynamic_status(data)
        )

        critical_alerts = self.detect_critical_alerts(
            data=data,
            hemodynamic_status=hemodynamic_status,
        )

        possible_stemi = self.detect_possible_stemi(
            data=data,
            ischemic_symptom_signal=(
                ischemic_symptom_signal
            ),
        )

        possible_nstemi = self.detect_possible_nstemi(
            ischemic_symptom_signal=(
                ischemic_symptom_signal
            ),
            ischemic_ecg_signal=(
                ischemic_ecg_signal
            ),
            injury_type=injury_type,
            possible_stemi=possible_stemi,
            alternative_cause=(
                data
                .alternative_cause_of_troponin_elevation
            ),
        )

        possible_unstable_angina = (
            self.detect_possible_unstable_angina(
                ischemic_symptom_signal=(
                    ischemic_symptom_signal
                ),
                ischemic_ecg_signal=(
                    ischemic_ecg_signal
                ),
                troponin_status=troponin_status,
                possible_stemi=possible_stemi,
                possible_nstemi=possible_nstemi,
            )
        )

        syndrome_type = self.resolve_syndrome_type(
            data=data,
            ischemic_symptom_signal=(
                ischemic_symptom_signal
            ),
            ischemic_ecg_signal=(
                ischemic_ecg_signal
            ),
            injury_type=injury_type,
            possible_stemi=possible_stemi,
            possible_nstemi=possible_nstemi,
            possible_unstable_angina=(
                possible_unstable_angina
            ),
        )

        heart_score = self.calculate_heart_score(
            data
        )

        urgency = self.resolve_urgency(
            data=data,
            syndrome_type=syndrome_type,
            hemodynamic_status=(
                hemodynamic_status
            ),
            critical_alerts=critical_alerts,
            heart_score=heart_score,
        )

        warnings = list(validation.warnings)
        recommendations: list[str] = []

        if (
            data.ecg_pattern
            == ECGIschemiaPattern.NOT_AVAILABLE
        ):
            warnings.append(
                "ECG não informado. A avaliação de síndrome "
                "coronariana aguda permanece incompleta."
            )

            recommendations.append(
                "Obter e interpretar ECG de 12 derivações "
                "sem atraso."
            )

        if (
            data.ecg_performed_within_10_minutes is False
        ):
            warnings.append(
                "O ECG não foi realizado dentro dos "
                "primeiros dez minutos informados."
            )

        if (
            data.troponin_value is None
            or data.troponin_upper_reference_limit is None
        ):
            warnings.append(
                "Troponina ou limite superior de referência "
                "não informado."
            )

            recommendations.append(
                "Solicitar troponina cardíaca com método e "
                "limite de referência documentados."
            )

        if (
            troponin_status
            == TroponinStatus.ABOVE_UPPER_REFERENCE_LIMIT
        ):
            warnings.append(
                "Troponina acima do limite superior indica "
                "lesão miocárdica, mas não define isoladamente "
                "infarto do miocárdio."
            )

        if injury_type == MyocardialInjuryType.ACUTE:
            warnings.append(
                "Foi identificada variação de troponina "
                "compatível com lesão miocárdica aguda."
            )

        if (
            data.alternative_cause_of_troponin_elevation
            and myocardial_injury_present
        ):
            warnings.append(
                "Foi informada possível causa alternativa "
                "para a elevação da troponina."
            )

        if possible_stemi:
            recommendations.append(
                "Ativar avaliação emergencial para possível "
                "oclusão coronariana aguda."
            )

        elif possible_nstemi:
            recommendations.append(
                "Realizar avaliação emergencial ou urgente "
                "para possível NSTEMI."
            )

        elif possible_unstable_angina:
            recommendations.append(
                "Manter investigação urgente de isquemia "
                "miocárdica apesar de troponina não elevada."
            )

        if data.persistent_chest_pain:
            recommendations.append(
                "Dor persistente exige reavaliação clínica "
                "e eletrocardiográfica imediata."
            )

        if data.dynamic_ecg_changes:
            recommendations.append(
                "Alterações dinâmicas do ECG exigem avaliação "
                "cardiológica prioritária."
            )

        if data.active_major_bleeding:
            recommendations.append(
                "Revisar imediatamente risco hemorrágico "
                "antes de terapia antitrombótica."
            )

        if (
            data.current_anticoagulation
            or data.current_antiplatelet_therapy
        ):
            recommendations.append(
                "Confirmar medicamentos antitrombóticos, "
                "última dose, indicação e adesão."
            )

        if data.egfr_ml_min_1_73m2 is not None:
            recommendations.append(
                "Considerar função renal na seleção e dose "
                "de medicamentos e contraste."
            )

        immediate_evaluation_required = (
            urgency
            in {
                AcuteCoronaryUrgency.URGENT,
                AcuteCoronaryUrgency.EMERGENCY,
            }
        )

        cardiology_review_required = any(
            (
                possible_stemi,
                possible_nstemi,
                possible_unstable_angina,
                ischemic_ecg_signal,
                critical_alerts,
            )
        )

        bleeding_review_required = any(
            (
                data.active_major_bleeding,
                data.recent_intracranial_hemorrhage,
                data.severe_thrombocytopenia,
                (
                    data.platelet_count_per_mm3 is not None
                    and data.platelet_count_per_mm3
                    < self.config
                    .severe_thrombocytopenia_threshold
                ),
                data.current_anticoagulation,
            )
        )

        medication_review_required = any(
            (
                data.current_anticoagulation,
                data.current_antiplatelet_therapy,
                bleeding_review_required,
                myocardial_injury_present,
                cardiology_review_required,
            )
        )

        troponin_ratio = self.troponin_ratio(data)
        relative_change = self.relative_troponin_change(data)

        return AcuteCoronaryAssessmentResult(
            valid=True,
            syndrome_type=syndrome_type,
            urgency=urgency,
            hemodynamic_status=hemodynamic_status,
            troponin_status=troponin_status,
            myocardial_injury_type=injury_type,
            heart_score=heart_score,
            ischemic_symptom_signal=(
                ischemic_symptom_signal
            ),
            ischemic_ecg_signal=(
                ischemic_ecg_signal
            ),
            myocardial_injury_present=(
                myocardial_injury_present
            ),
            possible_stemi=possible_stemi,
            possible_nstemi=possible_nstemi,
            possible_unstable_angina=(
                possible_unstable_angina
            ),
            immediate_evaluation_required=(
                immediate_evaluation_required
            ),
            cardiology_review_required=(
                cardiology_review_required
            ),
            medication_review_required=(
                medication_review_required
            ),
            bleeding_risk_review_required=(
                bleeding_review_required
            ),
            alerts=self._unique_strings(
                critical_alerts
            ),
            warnings=self._unique_strings(
                warnings
            ),
            recommendations=self._unique_strings(
                recommendations
            ),
            metadata={
                "troponin_ratio_to_url": (
                    round(troponin_ratio, 4)
                    if troponin_ratio is not None
                    else None
                ),
                "relative_troponin_change_percent": (
                    round(relative_change, 2)
                    if relative_change is not None
                    else None
                ),
                "ecg_pattern": data.ecg_pattern.value,
                "chest_pain_pattern": (
                    data.chest_pain_pattern.value
                ),
                "suspected_alternative_diagnoses": (
                    self._unique_strings(
                        data.suspected_alternative_diagnoses
                    )
                ),
            },
        )

    # ========================================================
    # Validação
    # ========================================================

    def validate(
        self,
        data: AcuteCoronaryAssessmentInput,
    ) -> AcuteCoronaryValidation:
        """Valida plausibilidade dos dados."""

        invalid: list[str] = []
        warnings: list[str] = []

        self._validate_optional_range(
            field_name="age_years",
            value=data.age_years,
            minimum=0,
            maximum=130,
            invalid=invalid,
        )

        self._validate_optional_range(
            field_name="chest_pain_duration_minutes",
            value=data.chest_pain_duration_minutes,
            minimum=0,
            maximum=10_000,
            invalid=invalid,
        )

        self._validate_optional_range(
            field_name="systolic_blood_pressure_mm_hg",
            value=data.systolic_blood_pressure_mm_hg,
            minimum=20,
            maximum=350,
            invalid=invalid,
        )

        self._validate_optional_range(
            field_name="diastolic_blood_pressure_mm_hg",
            value=data.diastolic_blood_pressure_mm_hg,
            minimum=10,
            maximum=250,
            invalid=invalid,
        )

        self._validate_optional_range(
            field_name="heart_rate_bpm",
            value=data.heart_rate_bpm,
            minimum=10,
            maximum=300,
            invalid=invalid,
        )

        self._validate_optional_range(
            field_name="oxygen_saturation_percent",
            value=data.oxygen_saturation_percent,
            minimum=20,
            maximum=100,
            invalid=invalid,
        )

        for field_name, value in {
            "st_elevation_mm": data.st_elevation_mm,
            "st_depression_mm": data.st_depression_mm,
            "troponin_value": data.troponin_value,
            "troponin_upper_reference_limit": (
                data.troponin_upper_reference_limit
            ),
            "previous_troponin_value": (
                data.previous_troponin_value
            ),
            "troponin_measurement_interval_hours": (
                data.troponin_measurement_interval_hours
            ),
            "platelet_count_per_mm3": (
                data.platelet_count_per_mm3
            ),
            "creatinine_mg_dl": data.creatinine_mg_dl,
            "egfr_ml_min_1_73m2": (
                data.egfr_ml_min_1_73m2
            ),
            "hemoglobin_g_dl": data.hemoglobin_g_dl,
        }.items():
            if value is None:
                continue

            if (
                not self._valid_number(value)
                or float(value) < 0
            ):
                invalid.append(field_name)

        if (
            data.troponin_value is not None
            and data.troponin_upper_reference_limit
            is not None
            and data.troponin_upper_reference_limit <= 0
        ):
            invalid.append(
                "troponin_upper_reference_limit"
            )

        if (
            data.chest_pain_present
            and data.chest_pain_pattern
            == ChestPainPattern.UNDETERMINED
        ):
            warnings.append(
                "Dor torácica presente, mas padrão clínico "
                "não classificado."
            )

        return AcuteCoronaryValidation(
            valid=not invalid,
            missing_fields=[],
            invalid_fields=self._unique_strings(
                invalid
            ),
            warnings=self._unique_strings(
                warnings
            ),
        )

    # ========================================================
    # Sintomas e ECG
    # ========================================================

    def detect_ischemic_symptom_signal(
        self,
        data: AcuteCoronaryAssessmentInput,
    ) -> bool:
        """Detecta sintomas compatíveis com isquemia."""

        if (
            data.chest_pain_pattern
            in {
                ChestPainPattern.HIGHLY_SUSPICIOUS,
                ChestPainPattern.MODERATELY_SUSPICIOUS,
            }
        ):
            return True

        typical_features = sum(
            (
                int(data.retrosternal_pain),
                int(data.pressure_or_tightness),
                int(data.exertional_trigger),
                int(data.radiation_to_arm_or_jaw),
                int(data.diaphoresis),
                int(data.nausea_or_vomiting),
            )
        )

        if data.chest_pain_present and typical_features >= 2:
            return True

        return any(
            (
                data.persistent_chest_pain,
                data.recurrent_chest_pain
                and typical_features >= 1,
                data.dyspnea
                and data.atypical_presentation_possible,
                data.syncope_or_presyncope
                and data.atypical_presentation_possible,
            )
        )

    def detect_ischemic_ecg_signal(
        self,
        data: AcuteCoronaryAssessmentInput,
    ) -> bool:
        """Detecta padrões isquêmicos informados."""

        if (
            data.ecg_pattern
            in {
                ECGIschemiaPattern
                .PERSISTENT_ST_ELEVATION,
                ECGIschemiaPattern
                .TRANSIENT_ST_ELEVATION,
                ECGIschemiaPattern.ST_DEPRESSION,
                ECGIschemiaPattern.T_WAVE_INVERSION,
                ECGIschemiaPattern
                .NEW_LEFT_BUNDLE_BRANCH_BLOCK,
                ECGIschemiaPattern
                .POSTERIOR_INFARCTION_PATTERN,
                ECGIschemiaPattern
                .RIGHT_VENTRICULAR_INFARCTION_PATTERN,
            }
        ):
            return True

        if data.dynamic_ecg_changes:
            return True

        if (
            data.st_depression_mm is not None
            and data.st_depression_mm
            >= self.config.significant_st_depression_mm
        ):
            return True

        if (
            data.st_elevation_mm is not None
            and data.st_elevation_mm
            >= self.config.significant_st_elevation_mm
        ):
            return True

        return False

    # ========================================================
    # Troponina
    # ========================================================

    def classify_troponin(
        self,
        data: AcuteCoronaryAssessmentInput,
    ) -> TroponinStatus:
        """Classifica troponina em relação ao ensaio."""

        ratio = self.troponin_ratio(data)

        if ratio is None:
            return TroponinStatus.NOT_AVAILABLE

        if ratio <= 1.0:
            return (
                TroponinStatus
                .BELOW_UPPER_REFERENCE_LIMIT
            )

        change = self.relative_troponin_change(data)

        if change is None:
            return (
                TroponinStatus
                .ABOVE_UPPER_REFERENCE_LIMIT
            )

        if (
            abs(change)
            >= self.config
            .relative_troponin_change_percent
        ):
            return TroponinStatus.RISING_OR_FALLING

        return TroponinStatus.STABLE_ELEVATION

    @staticmethod
    def classify_myocardial_injury(
        troponin_status: TroponinStatus,
    ) -> MyocardialInjuryType:
        """Classifica lesão miocárdica."""

        if (
            troponin_status
            == TroponinStatus.RISING_OR_FALLING
        ):
            return MyocardialInjuryType.ACUTE

        if (
            troponin_status
            in {
                TroponinStatus
                .ABOVE_UPPER_REFERENCE_LIMIT,
                TroponinStatus.STABLE_ELEVATION,
            }
        ):
            return MyocardialInjuryType.CHRONIC

        if (
            troponin_status
            == TroponinStatus
            .BELOW_UPPER_REFERENCE_LIMIT
        ):
            return MyocardialInjuryType.NONE

        return MyocardialInjuryType.UNDETERMINED

    @staticmethod
    def troponin_ratio(
        data: AcuteCoronaryAssessmentInput,
    ) -> float | None:
        """Calcula troponina dividida pelo limite superior."""

        if (
            data.troponin_value is None
            or data.troponin_upper_reference_limit is None
            or data.troponin_upper_reference_limit <= 0
        ):
            return None

        return (
            float(data.troponin_value)
            / float(data.troponin_upper_reference_limit)
        )

    @staticmethod
    def relative_troponin_change(
        data: AcuteCoronaryAssessmentInput,
    ) -> float | None:
        """Calcula variação relativa entre duas medidas."""

        if (
            data.previous_troponin_value is None
            or data.troponin_value is None
        ):
            return None

        previous = float(
            data.previous_troponin_value
        )

        current = float(data.troponin_value)

        if previous == 0:
            if current > 0:
                return 100.0

            return 0.0

        return (
            (current - previous)
            / abs(previous)
            * 100.0
        )

    # ========================================================
    # Classificação sindrômica
    # ========================================================

    def detect_possible_stemi(
        self,
        *,
        data: AcuteCoronaryAssessmentInput,
        ischemic_symptom_signal: bool,
    ) -> bool:
        """Detecta cenário compatível com STEMI."""

        stemi_pattern = (
            data.ecg_pattern
            in {
                ECGIschemiaPattern
                .PERSISTENT_ST_ELEVATION,
                ECGIschemiaPattern
                .POSTERIOR_INFARCTION_PATTERN,
            }
        )

        measured_elevation = bool(
            data.st_elevation_mm is not None
            and data.st_elevation_mm
            >= self.config.significant_st_elevation_mm
        )

        return bool(
            ischemic_symptom_signal
            and (
                stemi_pattern
                or measured_elevation
            )
        )

    @staticmethod
    def detect_possible_nstemi(
        *,
        ischemic_symptom_signal: bool,
        ischemic_ecg_signal: bool,
        injury_type: MyocardialInjuryType,
        possible_stemi: bool,
        alternative_cause: bool,
    ) -> bool:
        """Detecta cenário compatível com NSTEMI."""

        if possible_stemi:
            return False

        return bool(
            injury_type == MyocardialInjuryType.ACUTE
            and (
                ischemic_symptom_signal
                or ischemic_ecg_signal
            )
            and not alternative_cause
        )

    @staticmethod
    def detect_possible_unstable_angina(
        *,
        ischemic_symptom_signal: bool,
        ischemic_ecg_signal: bool,
        troponin_status: TroponinStatus,
        possible_stemi: bool,
        possible_nstemi: bool,
    ) -> bool:
        """Detecta cenário compatível com angina instável."""

        if possible_stemi or possible_nstemi:
            return False

        return bool(
            ischemic_symptom_signal
            and (
                ischemic_ecg_signal
                or troponin_status
                in {
                    TroponinStatus
                    .BELOW_UPPER_REFERENCE_LIMIT,
                    TroponinStatus.NOT_AVAILABLE,
                }
            )
        )

    @staticmethod
    def resolve_syndrome_type(
        *,
        data: AcuteCoronaryAssessmentInput,
        ischemic_symptom_signal: bool,
        ischemic_ecg_signal: bool,
        injury_type: MyocardialInjuryType,
        possible_stemi: bool,
        possible_nstemi: bool,
        possible_unstable_angina: bool,
    ) -> AcuteCoronarySyndromeType:
        """Resolve classificação operacional."""

        if possible_stemi:
            return (
                AcuteCoronarySyndromeType
                .STEMI_COMPATIBLE
            )

        if possible_nstemi:
            return (
                AcuteCoronarySyndromeType
                .NSTEMI_COMPATIBLE
            )

        if possible_unstable_angina:
            return (
                AcuteCoronarySyndromeType
                .UNSTABLE_ANGINA_COMPATIBLE
            )

        if injury_type == MyocardialInjuryType.ACUTE:
            return (
                AcuteCoronarySyndromeType
                .ACUTE_MYOCARDIAL_INJURY
            )

        if injury_type == MyocardialInjuryType.CHRONIC:
            return (
                AcuteCoronarySyndromeType
                .CHRONIC_MYOCARDIAL_INJURY
            )

        if ischemic_symptom_signal or ischemic_ecg_signal:
            return (
                AcuteCoronarySyndromeType
                .ISCHEMIA_UNDER_INVESTIGATION
            )

        if (
            not data.chest_pain_present
            and not ischemic_ecg_signal
        ):
            return (
                AcuteCoronarySyndromeType
                .NON_ISCHEMIC_PRESENTATION
            )

        return AcuteCoronarySyndromeType.UNDETERMINED

    # ========================================================
    # HEART Score
    # ========================================================

    def calculate_heart_score(
        self,
        data: AcuteCoronaryAssessmentInput,
    ) -> HEARTScoreResult:
        """Calcula HEART Score operacional."""

        missing: list[str] = []

        history_score = self._heart_history_score(
            data.chest_pain_pattern
        )

        if history_score is None:
            missing.append("history")

        ecg_score = self._heart_ecg_score(
            data.ecg_pattern
        )

        if ecg_score is None:
            missing.append("ecg")

        age_score = self._heart_age_score(
            data.age_years
        )

        if age_score is None:
            missing.append("age")

        risk_score = self._heart_risk_factor_score(
            data
        )

        troponin_score = self._heart_troponin_score(
            data
        )

        if troponin_score is None:
            missing.append("troponin")

        if missing:
            return HEARTScoreResult(
                score=0,
                category=HEARTRiskCategory.UNDETERMINED,
                components={},
                valid=False,
                missing_components=self._unique_strings(
                    missing
                ),
            )

        assert history_score is not None
        assert ecg_score is not None
        assert age_score is not None
        assert troponin_score is not None

        score = sum(
            (
                history_score,
                ecg_score,
                age_score,
                risk_score,
                troponin_score,
            )
        )

        if score <= self.config.heart_low_risk_max_score:
            category = HEARTRiskCategory.LOW

        elif (
            score
            <= self.config
            .heart_intermediate_risk_max_score
        ):
            category = HEARTRiskCategory.INTERMEDIATE

        else:
            category = HEARTRiskCategory.HIGH

        return HEARTScoreResult(
            score=score,
            category=category,
            components={
                "history": history_score,
                "ecg": ecg_score,
                "age": age_score,
                "risk_factors": risk_score,
                "troponin": troponin_score,
            },
            valid=True,
            missing_components=[],
        )

    @staticmethod
    def _heart_history_score(
        pattern: ChestPainPattern,
    ) -> int | None:
        """Pontuação do histórico."""

        mapping = {
            ChestPainPattern.NON_SUSPICIOUS: 0,
            ChestPainPattern.SLIGHTLY_SUSPICIOUS: 0,
            ChestPainPattern.MODERATELY_SUSPICIOUS: 1,
            ChestPainPattern.HIGHLY_SUSPICIOUS: 2,
            ChestPainPattern.ABSENT: 0,
        }

        return mapping.get(pattern)

    @staticmethod
    def _heart_ecg_score(
        pattern: ECGIschemiaPattern,
    ) -> int | None:
        """Pontuação do ECG."""

        if pattern == ECGIschemiaPattern.NOT_AVAILABLE:
            return None

        if pattern == ECGIschemiaPattern.NORMAL:
            return 0

        if (
            pattern
            in {
                ECGIschemiaPattern.NONSPECIFIC_CHANGES,
                ECGIschemiaPattern.T_WAVE_INVERSION,
            }
        ):
            return 1

        return 2

    @staticmethod
    def _heart_age_score(
        age_years: float | None,
    ) -> int | None:
        """Pontuação da idade."""

        if age_years is None:
            return None

        if age_years < 45:
            return 0

        if age_years < 65:
            return 1

        return 2

    @staticmethod
    def _heart_risk_factor_score(
        data: AcuteCoronaryAssessmentInput,
    ) -> int:
        """Pontuação dos fatores de risco."""

        count = sum(
            (
                int(data.hypertension),
                int(data.diabetes),
                int(data.dyslipidemia),
                int(data.current_smoker),
                int(data.family_history_premature_cad),
                int(data.obesity),
            )
        )

        known_atherosclerotic_disease = any(
            (
                data.known_coronary_artery_disease,
                data.previous_myocardial_infarction,
                data.previous_pci_or_cabg,
            )
        )

        if known_atherosclerotic_disease or count >= 3:
            return 2

        if count >= 1:
            return 1

        return 0

    def _heart_troponin_score(
        self,
        data: AcuteCoronaryAssessmentInput,
    ) -> int | None:
        """Pontuação da troponina."""

        ratio = self.troponin_ratio(data)

        if ratio is None:
            return None

        if ratio <= 1.0:
            return 0

        if ratio <= 3.0:
            return 1

        return 2

    # ========================================================
    # Hemodinâmica, alertas e urgência
    # ========================================================

    def classify_hemodynamic_status(
        self,
        data: AcuteCoronaryAssessmentInput,
    ) -> HemodynamicStatus:
        """Classifica estabilidade hemodinâmica."""

        shock_signals = sum(
            (
                int(data.altered_mental_status),
                int(data.cool_extremities),
                int(data.oliguria),
            )
        )

        severe_hypotension = bool(
            data.systolic_blood_pressure_mm_hg is not None
            and data.systolic_blood_pressure_mm_hg
            < self.config
            .severe_hypotension_threshold_mm_hg
        )

        hypotension = bool(
            data.systolic_blood_pressure_mm_hg is not None
            and data.systolic_blood_pressure_mm_hg
            < self.config.hypotension_threshold_mm_hg
        )

        if severe_hypotension and shock_signals >= 1:
            return HemodynamicStatus.SHOCK

        if severe_hypotension or shock_signals >= 2:
            return HemodynamicStatus.UNSTABLE

        if hypotension or shock_signals == 1:
            return (
                HemodynamicStatus.POSSIBLY_UNSTABLE
            )

        return HemodynamicStatus.STABLE

    def detect_critical_alerts(
        self,
        *,
        data: AcuteCoronaryAssessmentInput,
        hemodynamic_status: HemodynamicStatus,
    ) -> list[str]:
        """Detecta sinais críticos."""

        alerts: list[str] = []

        if data.cardiac_arrest:
            alerts.append(
                "Parada cardiorrespiratória informada."
            )

        if data.sustained_ventricular_arrhythmia:
            alerts.append(
                "Arritmia ventricular sustentada informada."
            )

        if (
            hemodynamic_status
            in {
                HemodynamicStatus.UNSTABLE,
                HemodynamicStatus.SHOCK,
            }
        ):
            alerts.append(
                "Instabilidade hemodinâmica informada."
            )

        if data.pulmonary_edema:
            alerts.append(
                "Edema pulmonar agudo informado."
            )

        if data.mechanical_complication_suspected:
            alerts.append(
                "Suspeita de complicação mecânica aguda."
            )

        if (
            data.oxygen_saturation_percent is not None
            and data.oxygen_saturation_percent
            < self.config
            .severe_hypoxemia_threshold_percent
        ):
            alerts.append(
                "Hipoxemia grave informada."
            )

        if data.persistent_chest_pain:
            alerts.append(
                "Dor torácica persistente informada."
            )

        return self._unique_strings(alerts)

    def resolve_urgency(
        self,
        *,
        data: AcuteCoronaryAssessmentInput,
        syndrome_type: AcuteCoronarySyndromeType,
        hemodynamic_status: HemodynamicStatus,
        critical_alerts: list[str],
        heart_score: HEARTScoreResult,
    ) -> AcuteCoronaryUrgency:
        """Resolve prioridade clínica."""

        if (
            critical_alerts
            or syndrome_type
            == AcuteCoronarySyndromeType
            .STEMI_COMPATIBLE
            or hemodynamic_status
            in {
                HemodynamicStatus.UNSTABLE,
                HemodynamicStatus.SHOCK,
            }
        ):
            return AcuteCoronaryUrgency.EMERGENCY

        if (
            syndrome_type
            in {
                AcuteCoronarySyndromeType
                .NSTEMI_COMPATIBLE,
                AcuteCoronarySyndromeType
                .UNSTABLE_ANGINA_COMPATIBLE,
                AcuteCoronarySyndromeType
                .ACUTE_MYOCARDIAL_INJURY,
            }
            or data.dynamic_ecg_changes
            or (
                heart_score.valid
                and heart_score.category
                == HEARTRiskCategory.HIGH
            )
        ):
            return AcuteCoronaryUrgency.URGENT

        if (
            syndrome_type
            == AcuteCoronarySyndromeType
            .ISCHEMIA_UNDER_INVESTIGATION
            or (
                heart_score.valid
                and heart_score.category
                == HEARTRiskCategory.INTERMEDIATE
            )
        ):
            return AcuteCoronaryUrgency.PRIORITY

        return AcuteCoronaryUrgency.ROUTINE

    # ========================================================
    # Utilidades
    # ========================================================

    @staticmethod
    def _validate_optional_range(
        *,
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
            not AcuteCoronaryAssessmentEngine
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