"""
PHK Studio
Clinical Pharmacy Engine

Modelo clínico unificado do paciente.

Este módulo concentra os dados utilizados pelos motores
renal, hepático, cardiovascular e pelos futuros domínios
clínicos.

As estruturas aqui definidas representam dados clínicos,
não decisões terapêuticas automáticas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4


# ============================================================
# Enumerações gerais
# ============================================================


class BiologicalSex(str, Enum):
    """Sexo biológico utilizado em cálculos clínicos."""

    FEMALE = "female"
    MALE = "male"
    INTERSEX = "intersex"
    UNDETERMINED = "undetermined"


class SmokingStatus(str, Enum):
    """Situação relacionada ao tabagismo."""

    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"
    PASSIVE_EXPOSURE = "passive_exposure"
    UNDETERMINED = "undetermined"


class AlcoholUseStatus(str, Enum):
    """Classificação geral de consumo de álcool."""

    NONE = "none"
    LOW_RISK = "low_risk"
    MODERATE_RISK = "moderate_risk"
    HIGH_RISK = "high_risk"
    DEPENDENCE_SUSPECTED = "dependence_suspected"
    UNDETERMINED = "undetermined"


class PregnancyStatus(str, Enum):
    """Situação gestacional."""

    NOT_APPLICABLE = "not_applicable"
    NOT_PREGNANT = "not_pregnant"
    PREGNANT = "pregnant"
    POSSIBLY_PREGNANT = "possibly_pregnant"
    POSTPARTUM = "postpartum"
    UNDETERMINED = "undetermined"


class MedicationRoute(str, Enum):
    """Principais vias de administração."""

    ORAL = "oral"
    INTRAVENOUS = "intravenous"
    INTRAMUSCULAR = "intramuscular"
    SUBCUTANEOUS = "subcutaneous"
    INHALATION = "inhalation"
    TOPICAL = "topical"
    TRANSDERMAL = "transdermal"
    RECTAL = "rectal"
    VAGINAL = "vaginal"
    OPHTHALMIC = "ophthalmic"
    OTIC = "otic"
    NASAL = "nasal"
    OTHER = "other"
    UNDETERMINED = "undetermined"


class MedicationStatus(str, Enum):
    """Estado do medicamento na farmacoterapia."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISCONTINUED = "discontinued"
    COMPLETED = "completed"
    PROPOSED = "proposed"
    UNDETERMINED = "undetermined"


class AllergySeverity(str, Enum):
    """Gravidade registrada de uma alergia."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"
    UNDETERMINED = "undetermined"


class DiagnosisStatus(str, Enum):
    """Estado de um diagnóstico ou problema clínico."""

    ACTIVE = "active"
    CONTROLLED = "controlled"
    RESOLVED = "resolved"
    SUSPECTED = "suspected"
    HISTORY = "history"
    UNDETERMINED = "undetermined"


# ============================================================
# Demografia
# ============================================================


@dataclass(slots=True)
class PatientDemographics:
    """Dados demográficos e antropométricos."""

    date_of_birth: date | None = None
    age_years: float | None = None

    biological_sex: BiologicalSex = (
        BiologicalSex.UNDETERMINED
    )

    gender_identity: str | None = None

    ethnicity_or_population_group: str | None = None

    weight_kg: float | None = None
    height_cm: float | None = None
    body_mass_index: float | None = None

    body_surface_area_m2: float | None = None

    pregnancy_status: PregnancyStatus = (
        PregnancyStatus.UNDETERMINED
    )

    gestational_age_weeks: float | None = None

    breastfeeding: bool = False

    def resolved_age_years(
        self,
        reference_date: date | None = None,
    ) -> float | None:
        """Retorna idade informada ou calculada."""

        if self.age_years is not None:
            return self.age_years

        if self.date_of_birth is None:
            return None

        reference = reference_date or date.today()

        days = (
            reference - self.date_of_birth
        ).days

        return round(days / 365.2425, 2)

    def resolved_bmi(self) -> float | None:
        """Retorna IMC informado ou calculado."""

        if self.body_mass_index is not None:
            return self.body_mass_index

        if (
            self.weight_kg is None
            or self.height_cm is None
            or self.height_cm <= 0
        ):
            return None

        height_m = self.height_cm / 100

        return round(
            self.weight_kg / (height_m**2),
            2,
        )


# ============================================================
# Medicamentos
# ============================================================


@dataclass(slots=True)
class ClinicalMedication:
    """Medicamento estruturado da farmacoterapia."""

    id: UUID = field(default_factory=uuid4)

    name: str = ""
    active_ingredient: str | None = None
    brand_name: str | None = None

    dose_value: float | None = None
    dose_unit: str | None = None

    pharmaceutical_form: str | None = None

    route: MedicationRoute = (
        MedicationRoute.UNDETERMINED
    )

    frequency: str | None = None
    administration_times: list[str] = field(
        default_factory=list
    )

    indication: str | None = None

    start_date: date | None = None
    end_date: date | None = None

    status: MedicationStatus = (
        MedicationStatus.ACTIVE
    )

    as_needed: bool = False

    prescriber: str | None = None

    renal_adjustment_required: bool = False
    hepatic_adjustment_required: bool = False

    therapeutic_drug_monitoring_required: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def normalized_name(self) -> str:
        """Nome normalizado para comparação."""

        source = (
            self.active_ingredient
            or self.name
            or self.brand_name
            or ""
        )

        return " ".join(
            source.strip().casefold().split()
        )


# ============================================================
# Diagnósticos e alergias
# ============================================================


@dataclass(slots=True)
class ClinicalDiagnosis:
    """Diagnóstico, condição ou problema clínico."""

    id: UUID = field(default_factory=uuid4)

    name: str = ""

    code: str | None = None
    coding_system: str | None = None

    status: DiagnosisStatus = (
        DiagnosisStatus.ACTIVE
    )

    onset_date: date | None = None
    resolution_date: date | None = None

    severity: str | None = None

    notes: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ClinicalAllergy:
    """Alergia ou hipersensibilidade registrada."""

    id: UUID = field(default_factory=uuid4)

    substance: str = ""
    reaction: str | None = None

    severity: AllergySeverity = (
        AllergySeverity.UNDETERMINED
    )

    confirmed: bool = False
    life_threatening: bool = False

    onset_date: date | None = None

    notes: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Sinais vitais
# ============================================================


@dataclass(slots=True)
class VitalSigns:
    """Conjunto de sinais vitais em um instante."""

    measured_at: datetime = field(
        default_factory=datetime.utcnow
    )

    systolic_blood_pressure_mm_hg: float | None = None
    diastolic_blood_pressure_mm_hg: float | None = None

    heart_rate_bpm: float | None = None
    respiratory_rate_per_min: float | None = None

    oxygen_saturation_percent: float | None = None
    temperature_celsius: float | None = None

    weight_kg: float | None = None

    pain_score_0_10: float | None = None

    context: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def mean_arterial_pressure_mm_hg(
        self,
    ) -> float | None:
        """Calcula pressão arterial média."""

        systolic = self.systolic_blood_pressure_mm_hg
        diastolic = self.diastolic_blood_pressure_mm_hg

        if systolic is None or diastolic is None:
            return None

        return round(
            (
                systolic
                + (2 * diastolic)
            )
            / 3,
            2,
        )

    @property
    def pulse_pressure_mm_hg(
        self,
    ) -> float | None:
        """Calcula pressão de pulso."""

        systolic = self.systolic_blood_pressure_mm_hg
        diastolic = self.diastolic_blood_pressure_mm_hg

        if systolic is None or diastolic is None:
            return None

        return round(
            systolic - diastolic,
            2,
        )


# ============================================================
# Exames laboratoriais
# ============================================================


@dataclass(slots=True)
class LaboratoryResult:
    """Resultado laboratorial individual."""

    name: str = ""
    value: float | str | bool | None = None
    unit: str | None = None

    reference_min: float | None = None
    reference_max: float | None = None

    collected_at: datetime | None = None
    resulted_at: datetime | None = None

    abnormal: bool | None = None
    critical: bool = False

    code: str | None = None
    coding_system: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ClinicalLaboratoryPanel:
    """Painel consolidado dos exames mais utilizados."""

    creatinine_mg_dl: float | None = None
    urea_mg_dl: float | None = None
    egfr_ml_min_1_73m2: float | None = None

    sodium_mmol_l: float | None = None
    potassium_mmol_l: float | None = None
    magnesium_mg_dl: float | None = None
    calcium_mg_dl: float | None = None

    ast_u_l: float | None = None
    alt_u_l: float | None = None
    alkaline_phosphatase_u_l: float | None = None
    gamma_gt_u_l: float | None = None

    total_bilirubin_mg_dl: float | None = None
    direct_bilirubin_mg_dl: float | None = None

    albumin_g_dl: float | None = None
    inr: float | None = None

    hemoglobin_g_dl: float | None = None
    hematocrit_percent: float | None = None
    leukocytes_per_mm3: float | None = None
    neutrophils_per_mm3: float | None = None
    platelets_per_mm3: float | None = None

    glucose_mg_dl: float | None = None
    hba1c_percent: float | None = None

    total_cholesterol_mg_dl: float | None = None
    ldl_cholesterol_mg_dl: float | None = None
    hdl_cholesterol_mg_dl: float | None = None
    triglycerides_mg_dl: float | None = None

    troponin_value: float | None = None
    troponin_upper_reference_limit: float | None = None

    bnp_pg_ml: float | None = None
    nt_pro_bnp_pg_ml: float | None = None

    crp_mg_l: float | None = None
    procalcitonin_ng_ml: float | None = None

    lactate_mmol_l: float | None = None

    additional_results: list[
        LaboratoryResult
    ] = field(default_factory=list)

    collected_at: datetime | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# ECG e imagem
# ============================================================


@dataclass(slots=True)
class ClinicalECG:
    """Dados estruturados do eletrocardiograma."""

    performed_at: datetime | None = None

    rhythm: str | None = None

    heart_rate_bpm: float | None = None

    pr_interval_ms: float | None = None
    qrs_duration_ms: float | None = None

    qt_interval_ms: float | None = None
    corrected_qt_ms: float | None = None

    st_elevation_mm: float | None = None
    st_depression_mm: float | None = None

    dynamic_changes: bool = False
    ischemic_changes: bool = False

    atrial_fibrillation: bool = False
    ventricular_arrhythmia_present: bool = False

    interpretation: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ClinicalEchocardiogram:
    """Dados estruturados do ecocardiograma."""

    performed_at: datetime | None = None

    left_ventricular_ejection_fraction_percent: (
        float | None
    ) = None

    previous_ejection_fraction_percent: (
        float | None
    ) = None

    left_ventricular_hypertrophy: bool = False
    right_ventricular_dysfunction: bool = False

    significant_valvular_disease: bool = False

    left_atrial_volume_index_ml_m2: float | None = None
    e_over_e_prime: float | None = None

    tricuspid_regurgitation_velocity_m_s: (
        float | None
    ) = None

    interpretation: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Contexto clínico
# ============================================================


@dataclass(slots=True)
class ClinicalSymptoms:
    """Sintomas e sinais clínicos estruturados."""

    chest_pain: bool = False
    dyspnea: bool = False
    orthopnea: bool = False
    edema: bool = False

    fatigue: bool = False
    syncope: bool = False
    dizziness: bool = False

    nausea_or_vomiting: bool = False

    fever: bool = False
    cough: bool = False

    confusion: bool = False
    reduced_urine_output: bool = False

    bleeding: bool = False

    additional_symptoms: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class ClinicalLifestyle:
    """Hábitos e fatores comportamentais."""

    smoking_status: SmokingStatus = (
        SmokingStatus.UNDETERMINED
    )

    pack_years: float | None = None

    alcohol_use_status: AlcoholUseStatus = (
        AlcoholUseStatus.UNDETERMINED
    )

    physical_activity_level: str | None = None

    diet_pattern: str | None = None

    medication_adherence_percent: float | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================
# Paciente clínico unificado
# ============================================================


@dataclass(slots=True)
class ClinicalPatient:
    """
    Representação clínica unificada do paciente.

    Este objeto será traduzido posteriormente para os inputs
    específicos dos motores renal, hepático e cardiovascular.
    """

    id: UUID = field(default_factory=uuid4)

    external_id: str | None = None

    name: str | None = None

    demographics: PatientDemographics = field(
        default_factory=PatientDemographics
    )

    diagnoses: list[ClinicalDiagnosis] = field(
        default_factory=list
    )

    medications: list[ClinicalMedication] = field(
        default_factory=list
    )

    allergies: list[ClinicalAllergy] = field(
        default_factory=list
    )

    vital_signs: list[VitalSigns] = field(
        default_factory=list
    )

    laboratory: ClinicalLaboratoryPanel = field(
        default_factory=ClinicalLaboratoryPanel
    )

    ecg: ClinicalECG = field(
        default_factory=ClinicalECG
    )

    echocardiogram: ClinicalEchocardiogram = field(
        default_factory=ClinicalEchocardiogram
    )

    symptoms: ClinicalSymptoms = field(
        default_factory=ClinicalSymptoms
    )

    lifestyle: ClinicalLifestyle = field(
        default_factory=ClinicalLifestyle
    )

    clinical_notes: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    updated_at: datetime = field(
        default_factory=datetime.utcnow
    )

    @property
    def age_years(self) -> float | None:
        """Idade clínica resolvida."""

        return self.demographics.resolved_age_years()

    @property
    def active_medications(
        self,
    ) -> list[ClinicalMedication]:
        """Medicamentos atualmente ativos."""

        return [
            medication
            for medication in self.medications
            if medication.status
            == MedicationStatus.ACTIVE
        ]

    @property
    def active_diagnoses(
        self,
    ) -> list[ClinicalDiagnosis]:
        """Diagnósticos ativos ou suspeitos."""

        return [
            diagnosis
            for diagnosis in self.diagnoses
            if diagnosis.status
            in {
                DiagnosisStatus.ACTIVE,
                DiagnosisStatus.CONTROLLED,
                DiagnosisStatus.SUSPECTED,
            }
        ]

    @property
    def latest_vital_signs(
        self,
    ) -> VitalSigns | None:
        """Sinais vitais mais recentes."""

        if not self.vital_signs:
            return None

        return max(
            self.vital_signs,
            key=lambda item: item.measured_at,
        )

    def has_diagnosis(
        self,
        *terms: str,
    ) -> bool:
        """Verifica diagnóstico por nome ou código."""

        normalized_terms = {
            term.strip().casefold()
            for term in terms
            if term.strip()
        }

        for diagnosis in self.active_diagnoses:
            values = {
                diagnosis.name.strip().casefold(),
                (
                    diagnosis.code.strip().casefold()
                    if diagnosis.code
                    else ""
                ),
            }

            if any(
                term in value
                for term in normalized_terms
                for value in values
            ):
                return True

        return False

    def uses_medication(
        self,
        *terms: str,
    ) -> bool:
        """Verifica medicamento ativo por nome."""

        normalized_terms = {
            term.strip().casefold()
            for term in terms
            if term.strip()
        }

        return any(
            term in medication.normalized_name
            for medication in self.active_medications
            for term in normalized_terms
        )

    def medication_names(self) -> list[str]:
        """Lista os nomes dos medicamentos ativos."""

        return [
            (
                medication.active_ingredient
                or medication.name
                or medication.brand_name
                or ""
            )
            for medication in self.active_medications
        ]

    def touch(self) -> None:
        """Atualiza o instante da última modificação."""

        self.updated_at = datetime.utcnow()