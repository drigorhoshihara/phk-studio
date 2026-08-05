"""
PHK Studio
Clinical Pharmacy Engine

Modelos centrais utilizados por todo o módulo clínico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, List, Optional
from uuid import uuid4


# =========================================================
# ENUMS
# =========================================================


class BiologicalSex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    INTERSEX = "intersex"
    NOT_INFORMED = "not_informed"


class SeverityLevel(str, Enum):
    INFORMATIONAL = "informational"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class UrgencyLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    ORANGE = "orange"
    RED = "red"


class InteractionType(str, Enum):
    PHARMACOKINETIC = "pharmacokinetic"
    PHARMACODYNAMIC = "pharmacodynamic"
    MEDICATION_FOOD = "medication_food"
    MEDICATION_ALCOHOL = "medication_alcohol"
    MEDICATION_HERBAL = "medication_herbal"
    MEDICATION_SUPPLEMENT = "medication_supplement"
    MEDICATION_LABORATORY = "medication_laboratory"


class AdverseReactionType(str, Enum):
    TYPE_A = "type_a"
    TYPE_B = "type_b"
    TYPE_C = "type_c"
    TYPE_D = "type_d"
    TYPE_E = "type_e"
    TYPE_F = "type_f"
    UNCLASSIFIED = "unclassified"


class ConsultationType(str, Enum):
    FIRST_VISIT = "first_visit"
    FOLLOW_UP = "follow_up"
    PRESCRIPTION_REVIEW = "prescription_review"
    MEDICATION_RECONCILIATION = "medication_reconciliation"
    PHARMACOVIGILANCE = "pharmacovigilance"
    TOXICOLOGY = "toxicology"
    TELEPHARMACY = "telepharmacy"
    OTHER = "other"


class ConsultationStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    PENDING_REVIEW = "pending_review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MedicationStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"
    UNKNOWN = "unknown"


class DrugTherapyProblemType(str, Enum):
    UNTREATED_CONDITION = "untreated_condition"
    MEDICATION_WITHOUT_INDICATION = "medication_without_indication"
    INAPPROPRIATE_MEDICATION = "inappropriate_medication"
    LOW_DOSE = "low_dose"
    HIGH_DOSE = "high_dose"
    INAPPROPRIATE_FREQUENCY = "inappropriate_frequency"
    INAPPROPRIATE_DURATION = "inappropriate_duration"
    THERAPEUTIC_DUPLICATION = "therapeutic_duplication"
    DRUG_INTERACTION = "drug_interaction"
    FOOD_INTERACTION = "food_interaction"
    CONTRAINDICATION = "contraindication"
    ADVERSE_DRUG_REACTION = "adverse_drug_reaction"
    NON_ADHERENCE = "non_adherence"
    INCORRECT_ADMINISTRATION = "incorrect_administration"
    ACCESS_PROBLEM = "access_problem"
    THERAPEUTIC_FAILURE = "therapeutic_failure"
    MONITORING_NEEDED = "monitoring_needed"
    RENAL_ADJUSTMENT = "renal_adjustment"
    HEPATIC_ADJUSTMENT = "hepatic_adjustment"
    OTHER = "other"


class InterventionStatus(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    PARTIALLY_ACCEPTED = "partially_accepted"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    CANCELLED = "cancelled"


class ReferralDestination(str, Enum):
    PHARMACIST_FOLLOW_UP = "pharmacist_follow_up"
    PRIMARY_CARE = "primary_care"
    MEDICAL_CLINIC = "medical_clinic"
    SPECIALIST = "specialist"
    URGENT_CARE = "urgent_care"
    EMERGENCY_DEPARTMENT = "emergency_department"
    HOSPITAL = "hospital"
    MOBILE_EMERGENCY_SERVICE = "mobile_emergency_service"
    TOXICOLOGY_CENTER = "toxicology_center"
    OTHER = "other"


# =========================================================
# PACIENTE
# =========================================================


@dataclass
class AllergyRecord:
    substance: str
    reaction: Optional[str] = None
    severity: SeverityLevel = SeverityLevel.MODERATE
    confirmed: bool = False
    active: bool = True
    notes: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class ClinicalCondition:
    name: str
    icd10_code: Optional[str] = None
    snomed_code: Optional[str] = None
    diagnosis_date: Optional[date] = None
    active: bool = True
    controlled: Optional[bool] = None
    notes: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid4()))


@dataclass
class PatientProfile:
    full_name: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    preferred_name: Optional[str] = None
    birth_date: Optional[date] = None

    biological_sex: BiologicalSex = (
        BiologicalSex.NOT_INFORMED
    )

    national_identifier: Optional[str] = None
    health_system_identifier: Optional[str] = None

    phone: Optional[str] = None
    email: Optional[str] = None

    city: Optional[str] = None
    state: Optional[str] = None

    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

    allergies: List[AllergyRecord] = field(
        default_factory=list,
    )

    conditions: List[ClinicalCondition] = field(
        default_factory=list,
    )

    pregnancy: bool = False
    breastfeeding: bool = False

    smoker: bool = False
    alcohol_use: bool = False

    renal_function: Optional[float] = None
    hepatic_function: Optional[str] = None

    consent_for_care: bool = False
    consent_for_data_processing: bool = False

    notes: Optional[str] = None

    created_at: datetime = field(
        default_factory=datetime.utcnow,
    )

    updated_at: datetime = field(
        default_factory=datetime.utcnow,
    )

    @property
    def age(self) -> Optional[int]:
        if self.birth_date is None:
            return None

        today = date.today()

        return (
            today.year
            - self.birth_date.year
            - (
                (today.month, today.day)
                < (
                    self.birth_date.month,
                    self.birth_date.day,
                )
            )
        )

    @property
    def body_mass_index(self) -> Optional[float]:
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


# =========================================================
# MEDICAMENTOS
# =========================================================


@dataclass
class MedicationRecord:
    name: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    active_ingredient: Optional[str] = None

    concentration: Optional[str] = None
    dosage_form: Optional[str] = None

    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None

    indication: Optional[str] = None

    atc_code: Optional[str] = None
    rxnorm_code: Optional[str] = None

    start_date: Optional[date] = None
    end_date: Optional[date] = None

    status: MedicationStatus = MedicationStatus.ACTIVE

    prescriber_name: Optional[str] = None
    prescriber_registration: Optional[str] = None

    self_medication: bool = False
    adherence_confirmed: Optional[bool] = None

    notes: Optional[str] = None


@dataclass
class PrescriptionItem:
    medication_name: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    active_ingredient: Optional[str] = None
    concentration: Optional[str] = None
    dosage_form: Optional[str] = None

    dose: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None

    indication: Optional[str] = None
    instructions: Optional[str] = None


@dataclass
class PrescriptionRecord:
    patient_id: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    issue_date: Optional[date] = None
    expiration_date: Optional[date] = None

    prescriber_name: Optional[str] = None
    prescriber_registration: Optional[str] = None
    prescriber_specialty: Optional[str] = None

    institution: Optional[str] = None

    items: List[PrescriptionItem] = field(
        default_factory=list,
    )

    source_file: Optional[str] = None
    extracted_automatically: bool = False

    pharmacist_validated: bool = False
    validation_notes: Optional[str] = None

    created_at: datetime = field(
        default_factory=datetime.utcnow,
    )


# =========================================================
# AVALIAÇÃO CLÍNICA
# =========================================================


@dataclass
class VitalSigns:
    systolic_blood_pressure: Optional[int] = None
    diastolic_blood_pressure: Optional[int] = None

    heart_rate: Optional[int] = None
    respiratory_rate: Optional[int] = None

    oxygen_saturation: Optional[float] = None
    temperature_celsius: Optional[float] = None

    capillary_glucose_mg_dl: Optional[float] = None

    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

    pain_score: Optional[int] = None

    measured_at: datetime = field(
        default_factory=datetime.utcnow,
    )

    notes: Optional[str] = None


@dataclass
class LaboratoryResult:
    test_name: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    value: Optional[float | str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None

    loinc_code: Optional[str] = None

    collected_at: Optional[datetime] = None
    reported_at: Optional[datetime] = None

    abnormal: Optional[bool] = None
    critical: Optional[bool] = None

    notes: Optional[str] = None


@dataclass
class SymptomRecord:
    name: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    description: Optional[str] = None
    onset: Optional[datetime] = None
    duration: Optional[str] = None

    intensity: Optional[str] = None
    progression: Optional[str] = None

    associated_symptoms: List[str] = field(
        default_factory=list,
    )

    red_flag: bool = False


# =========================================================
# INTERAÇÕES E RAM
# =========================================================


@dataclass
class DrugInteraction:
    medication_a: str
    medication_b: str

    interaction_type: InteractionType
    severity: SeverityLevel

    mechanism: str
    clinical_effect: str
    recommendation: str

    evidence_level: str = "not_assessed"
    confidence: float = 0.0

    monitoring_parameters: List[str] = field(
        default_factory=list,
    )

    requires_pharmacist_review: bool = True

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )


@dataclass
class AdverseDrugReaction:
    medication: str
    reaction: str

    reaction_type: AdverseReactionType
    severity: SeverityLevel

    causality: Optional[str] = None
    seriousness: Optional[str] = None
    preventability: Optional[str] = None

    onset: Optional[datetime] = None
    recovered: Optional[bool] = None

    notification_recommended: bool = False
    pharmacist_validated: bool = False

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )


# =========================================================
# PROBLEMAS FARMACOTERAPÊUTICOS
# =========================================================


@dataclass
class DrugTherapyProblem:
    patient_id: str
    title: str
    description: str

    problem_type: DrugTherapyProblemType

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    consultation_id: Optional[str] = None

    involved_medications: List[str] = field(
        default_factory=list,
    )

    related_condition: Optional[str] = None

    severity: SeverityLevel = SeverityLevel.MODERATE
    urgency: UrgencyLevel = UrgencyLevel.GREEN

    recommendation: Optional[str] = None

    detected_automatically: bool = False
    requires_pharmacist_review: bool = True

    confirmed_by_pharmacist: bool = False
    pharmacist_notes: Optional[str] = None

    resolved: bool = False
    resolved_at: Optional[datetime] = None


# =========================================================
# INTERVENÇÃO FARMACÊUTICA
# =========================================================


@dataclass
class PharmaceuticalIntervention:
    patient_id: str
    description: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    consultation_id: Optional[str] = None
    drug_therapy_problem_id: Optional[str] = None

    intervention_type: str = "other"
    rationale: Optional[str] = None

    target_professional: Optional[str] = None

    status: InterventionStatus = (
        InterventionStatus.PROPOSED
    )

    accepted_by: Optional[str] = None
    acceptance_reason: Optional[str] = None
    rejection_reason: Optional[str] = None

    implemented_at: Optional[datetime] = None
    outcome: Optional[str] = None

    requires_follow_up: bool = False
    follow_up_date: Optional[date] = None


# =========================================================
# ENCAMINHAMENTO
# =========================================================


@dataclass
class HealthcareFacility:
    name: str

    facility_type: Optional[str] = None
    specialty: Optional[str] = None

    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None

    phone: Optional[str] = None
    opening_hours: Optional[str] = None

    distance_km: Optional[float] = None

    emergency_service: bool = False
    public_service: Optional[bool] = None


@dataclass
class ClinicalReferral:
    patient_id: str
    reason: str
    clinical_summary: str

    destination: ReferralDestination

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    consultation_id: Optional[str] = None
    specialty: Optional[str] = None

    urgency: UrgencyLevel = UrgencyLevel.GREEN
    severity: SeverityLevel = SeverityLevel.MODERATE

    red_flags: List[str] = field(
        default_factory=list,
    )

    selected_facility: Optional[
        HealthcareFacility
    ] = None

    suggested_facilities: List[
        HealthcareFacility
    ] = field(
        default_factory=list,
    )

    immediate_actions: List[str] = field(
        default_factory=list,
    )

    pharmacist_validated: bool = False
    validated_at: Optional[datetime] = None

    patient_accepted: Optional[bool] = None
    patient_refusal_reason: Optional[str] = None

    completed: bool = False
    outcome: Optional[str] = None


# =========================================================
# PLANO DE CUIDADO
# =========================================================


@dataclass
class TherapeuticGoal:
    description: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    target_value: Optional[str] = None
    deadline: Optional[date] = None

    monitoring_parameters: List[str] = field(
        default_factory=list,
    )

    achieved: bool = False
    achieved_at: Optional[datetime] = None


@dataclass
class CarePlan:
    patient_id: str

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    consultation_id: Optional[str] = None

    goals: List[TherapeuticGoal] = field(
        default_factory=list,
    )

    problem_ids: List[str] = field(
        default_factory=list,
    )

    intervention_ids: List[str] = field(
        default_factory=list,
    )

    patient_guidance: List[str] = field(
        default_factory=list,
    )

    non_pharmacological_guidance: List[str] = field(
        default_factory=list,
    )

    clinical_monitoring: List[str] = field(
        default_factory=list,
    )

    laboratory_monitoring: List[str] = field(
        default_factory=list,
    )

    warning_signs: List[str] = field(
        default_factory=list,
    )

    next_review_date: Optional[date] = None

    pharmacist_validated: bool = False
    validated_at: Optional[datetime] = None


# =========================================================
# CONSULTA FARMACÊUTICA
# =========================================================


@dataclass
class PharmaceuticalConsultation:
    patient_id: str
    consultation_type: ConsultationType

    id: str = field(
        default_factory=lambda: str(uuid4()),
    )

    status: ConsultationStatus = (
        ConsultationStatus.DRAFT
    )

    started_at: datetime = field(
        default_factory=datetime.utcnow,
    )

    completed_at: Optional[datetime] = None

    chief_complaint: Optional[str] = None
    history_of_present_illness: Optional[str] = None

    subjective: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    plan: Optional[str] = None

    symptoms: List[SymptomRecord] = field(
        default_factory=list,
    )

    vital_signs: List[VitalSigns] = field(
        default_factory=list,
    )

    laboratory_results: List[
        LaboratoryResult
    ] = field(
        default_factory=list,
    )

    medications: List[MedicationRecord] = field(
        default_factory=list,
    )

    clinical_findings: List[str] = field(
        default_factory=list,
    )

    red_flags: List[str] = field(
        default_factory=list,
    )

    drug_therapy_problems: List[
        DrugTherapyProblem
    ] = field(
        default_factory=list,
    )

    interventions: List[
        PharmaceuticalIntervention
    ] = field(
        default_factory=list,
    )

    referrals: List[ClinicalReferral] = field(
        default_factory=list,
    )

    care_plan: Optional[CarePlan] = None

    pharmacist_name: Optional[str] = None
    pharmacist_registration: Optional[str] = None

    reviewed_by_pharmacist: bool = False
    reviewed_at: Optional[datetime] = None

    clinical_notes: Optional[str] = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# =========================================================
# RESULTADOS CONSOLIDADOS
# =========================================================


@dataclass
class ClinicalDecisionSupportResult:
    patient_id: str

    consultation_id: Optional[str] = None

    findings: List[str] = field(
        default_factory=list,
    )

    interactions: List[DrugInteraction] = field(
        default_factory=list,
    )

    adverse_reactions: List[
        AdverseDrugReaction
    ] = field(
        default_factory=list,
    )

    therapy_problems: List[
        DrugTherapyProblem
    ] = field(
        default_factory=list,
    )

    interventions: List[
        PharmaceuticalIntervention
    ] = field(
        default_factory=list,
    )

    referrals: List[ClinicalReferral] = field(
        default_factory=list,
    )

    overall_risk: SeverityLevel = (
        SeverityLevel.INFORMATIONAL
    )

    urgency: UrgencyLevel = UrgencyLevel.GREEN

    confidence: float = 0.0

    protocol_versions: dict[str, str] = field(
        default_factory=dict,
    )

    requires_pharmacist_review: bool = True

    generated_at: datetime = field(
        default_factory=datetime.utcnow,
    )

    disclaimer: str = (
        "Resultado automatizado de apoio à decisão clínica. "
        "Exige revisão e validação de farmacêutico habilitado."
    )


@dataclass
class PrescriptionReviewResult:
    prescription_id: str
    patient_id: str

    problems: List[DrugTherapyProblem] = field(
        default_factory=list,
    )

    proposed_interventions: List[
        PharmaceuticalIntervention
    ] = field(
        default_factory=list,
    )

    interactions: List[DrugInteraction] = field(
        default_factory=list,
    )

    adverse_reactions: List[
        AdverseDrugReaction
    ] = field(
        default_factory=list,
    )

    interaction_count: int = 0
    contraindication_count: int = 0
    high_risk_count: int = 0

    overall_risk: SeverityLevel = (
        SeverityLevel.INFORMATIONAL
    )

    requires_urgent_review: bool = False
    requires_medical_contact: bool = False
    requires_referral: bool = False

    pharmacist_validated: bool = False
    pharmacist_notes: Optional[str] = None

    analyzed_at: datetime = field(
        default_factory=datetime.utcnow,
    )