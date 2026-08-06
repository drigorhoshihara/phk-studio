"""
PHK Studio
Clinical Pharmacy Engine

Adaptador compatível entre:

ClinicalPatient
    ↓
CardiovascularPatientAdapter
    ↓
CardiovascularAssessmentInput
    ↓
CardiovascularAssessmentEngine

Esta implementação utiliza introspecção das dataclasses e
resolução defensiva de enums para reduzir incompatibilidades
entre versões dos modelos cardiovasculares.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, TypeVar

from app.clinical_pharmacy_engine.assessment.core.patient import (
    AlcoholUseStatus,
    BiologicalSex,
    ClinicalPatient,
    SmokingStatus as CoreSmokingStatus,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.models import (
    AnticoagulantType,
    AtrialFibrillationType,
    BloodPressureContext,
    BloodPressureMeasurement,
    CardiovascularAssessmentInput,
    CardiovascularSex,
    DiabetesStatus,
    ECGData,
    EchocardiogramData,
    LipidProfile,
    NYHAClass,
    PreventionContext,
    SmokingStatus,
)


T = TypeVar("T")


@dataclass(slots=True)
class CardiovascularAdapterConfig:
    """
    Configuração do adaptador cardiovascular.

    Valores padrão de enum permanecem opcionais para impedir
    referências a membros que não existam no modelo instalado.
    """

    default_prevention_context: PreventionContext | None = None

    default_blood_pressure_context: (
        BloodPressureContext | None
    ) = None

    include_all_vital_signs: bool = True

    infer_diagnoses_from_names: bool = True
    infer_medications_from_names: bool = True

    country_code: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


class CardiovascularPatientAdapter:
    """
    Traduz ClinicalPatient para CardiovascularAssessmentInput.

    O adaptador organiza dados, mas não substitui julgamento
    clínico nem validação farmacêutica.
    """

    def __init__(
        self,
        config: CardiovascularAdapterConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else CardiovascularAdapterConfig()
        )

    # ========================================================
    # API pública
    # ========================================================

    def adapt(
        self,
        patient: ClinicalPatient,
    ) -> CardiovascularAssessmentInput:
        """Converte um ClinicalPatient para o modelo cardiovascular."""

        diagnosis_names = [
            diagnosis.name
            for diagnosis in patient.active_diagnoses
            if diagnosis.name
        ]

        medication_names = patient.medication_names()

        normalized_diagnoses = [
            self._normalize_text(value)
            for value in diagnosis_names
        ]

        normalized_medications = [
            self._normalize_text(value)
            for value in medication_names
        ]

        established_ascvd = self._has_any_diagnosis(
            normalized_diagnoses,
            {
                "doença cardiovascular aterosclerótica",
                "doenca cardiovascular aterosclerotica",
                "doença arterial coronariana",
                "doenca arterial coronariana",
                "aterosclerose",
                "ascvd",
                "dac",
            },
        )

        prior_myocardial_infarction = (
            self._has_any_diagnosis(
                normalized_diagnoses,
                {
                    "infarto agudo do miocárdio",
                    "infarto agudo do miocardio",
                    "infarto do miocárdio",
                    "infarto do miocardio",
                    "myocardial infarction",
                    "iam",
                },
            )
        )

        prior_stroke_or_tia = self._has_any_diagnosis(
            normalized_diagnoses,
            {
                "acidente vascular cerebral",
                "ataque isquêmico transitório",
                "ataque isquemico transitorio",
                "stroke",
                "avc",
                "ait",
                "tia",
            },
        )

        peripheral_arterial_disease = (
            self._has_any_diagnosis(
                normalized_diagnoses,
                {
                    "doença arterial periférica",
                    "doenca arterial periferica",
                    "peripheral arterial disease",
                    "dap",
                },
            )
        )

        hypertension_history = (
            self._has_any_diagnosis(
                normalized_diagnoses,
                {
                    "hipertensão",
                    "hipertensao",
                    "hipertensão arterial",
                    "hipertensao arterial",
                    "hypertension",
                    "has",
                },
            )
        )

        treated_hypertension = (
            hypertension_history
            and self._uses_any_medication(
                normalized_medications,
                self._antihypertensive_terms(),
            )
        )

        chronic_kidney_disease = (
            self._has_any_diagnosis(
                normalized_diagnoses,
                {
                    "doença renal crônica",
                    "doenca renal cronica",
                    "chronic kidney disease",
                    "drc",
                },
            )
            or (
                patient.laboratory.egfr_ml_min_1_73m2
                is not None
                and patient.laboratory.egfr_ml_min_1_73m2 < 60
            )
        )

        renal_disease = (
            chronic_kidney_disease
            or self._has_any_diagnosis(
                normalized_diagnoses,
                {
                    "insuficiência renal",
                    "insuficiencia renal",
                    "lesão renal",
                    "lesao renal",
                    "doença renal",
                    "doenca renal",
                },
            )
        )

        liver_disease = self._has_any_diagnosis(
            normalized_diagnoses,
            {
                "cirrose",
                "hepatopatia",
                "insuficiência hepática",
                "insuficiencia hepatica",
                "doença hepática",
                "doenca hepatica",
            },
        )

        heart_failure = self._has_any_diagnosis(
            normalized_diagnoses,
            {
                "insuficiência cardíaca",
                "insuficiencia cardiaca",
                "heart failure",
                "icfer",
                "icfep",
                "icfemr",
            },
        )

        atrial_fibrillation = (
            patient.ecg.atrial_fibrillation
            or self._has_any_diagnosis(
                normalized_diagnoses,
                {
                    "fibrilação atrial",
                    "fibrilacao atrial",
                    "atrial fibrillation",
                },
            )
        )

        active_bleeding = (
            patient.symptoms.bleeding
            or self._has_any_diagnosis(
                normalized_diagnoses,
                {
                    "sangramento ativo",
                    "hemorragia ativa",
                    "active bleeding",
                },
            )
        )

        alcohol_use_risk = (
            patient.lifestyle.alcohol_use_status
            in {
                AlcoholUseStatus.HIGH_RISK,
                AlcoholUseStatus.DEPENDENCE_SUSPECTED,
            }
        )

        antiplatelet_medications = [
            original
            for original, normalized in zip(
                medication_names,
                normalized_medications,
                strict=False,
            )
            if self._contains_any(
                normalized,
                {
                    "ácido acetilsalicílico",
                    "acido acetilsalicilico",
                    "aspirina",
                    "clopidogrel",
                    "prasugrel",
                    "ticagrelor",
                    "ticlopidina",
                    "aas",
                },
            )
        ]

        vascular_disease = any(
            (
                established_ascvd,
                prior_myocardial_infarction,
                prior_stroke_or_tia,
                peripheral_arterial_disease,
            )
        )

        metadata = self._build_metadata(
            patient=patient,
            normalized_diagnoses=normalized_diagnoses,
        )

        input_values: dict[str, Any] = {
            "age_years": patient.age_years,
            "biological_sex": self._resolve_cardiovascular_sex(
                patient
            ),
            "ethnicity_or_population_group": (
                patient.demographics.ethnicity_or_population_group
            ),
            "smoking_status": self._resolve_smoking_status(
                patient
            ),
            "pack_years": patient.lifestyle.pack_years,
            "diabetes_status": self._resolve_diabetes_status(
                normalized_diagnoses
            ),
            "body_mass_index": (
                patient.demographics.resolved_bmi()
            ),
            "prevention_context": (
                self._resolve_prevention_context(
                    established_ascvd=established_ascvd,
                    prior_myocardial_infarction=(
                        prior_myocardial_infarction
                    ),
                    prior_stroke_or_tia=prior_stroke_or_tia,
                    peripheral_arterial_disease=(
                        peripheral_arterial_disease
                    ),
                )
            ),
            "established_ascvd": established_ascvd,
            "prior_myocardial_infarction": (
                prior_myocardial_infarction
            ),
            "prior_stroke_or_tia": prior_stroke_or_tia,
            "peripheral_arterial_disease": (
                peripheral_arterial_disease
            ),
            "vascular_disease": vascular_disease,
            "family_history_premature_cvd": bool(
                patient.metadata.get(
                    "family_history_premature_cvd",
                    False,
                )
            ),
            "blood_pressure_measurements": (
                self._build_blood_pressure_measurements(
                    patient
                )
            ),
            "treated_hypertension": treated_hypertension,
            "hypertension_history": hypertension_history,
            "lipid_profile": self._build_lipid_profile(
                patient
            ),
            "egfr_ml_min_1_73m2": (
                patient.laboratory.egfr_ml_min_1_73m2
            ),
            "creatinine_mg_dl": (
                patient.laboratory.creatinine_mg_dl
            ),
            "potassium_mmol_l": (
                patient.laboratory.potassium_mmol_l
            ),
            "chronic_kidney_disease": (
                chronic_kidney_disease
            ),
            "renal_disease": renal_disease,
            "liver_disease": liver_disease,
            "atrial_fibrillation": atrial_fibrillation,
            "atrial_fibrillation_type": (
                self._resolve_atrial_fibrillation_type(
                    atrial_fibrillation
                )
            ),
            "current_anticoagulant": (
                self._resolve_anticoagulant(
                    normalized_medications
                )
            ),
            "antiplatelet_medications": (
                antiplatelet_medications
            ),
            "previous_major_bleeding": bool(
                patient.metadata.get(
                    "previous_major_bleeding",
                    False,
                )
            ),
            "active_bleeding": active_bleeding,
            "labile_inr": bool(
                patient.metadata.get(
                    "labile_inr",
                    False,
                )
            ),
            "alcohol_use_risk": alcohol_use_risk,
            "echocardiogram": self._build_echocardiogram(
                patient
            ),
            "ecg": self._build_ecg(patient),
            "nyha_class": self._resolve_nyha_class(
                patient
            ),
            "heart_failure": heart_failure,
            "chest_pain_present": (
                patient.symptoms.chest_pain
            ),
            "dyspnea_present": patient.symptoms.dyspnea,
            "syncope_present": patient.symptoms.syncope,
            "palpitations_present": bool(
                patient.metadata.get(
                    "palpitations_present",
                    False,
                )
            ),
            "edema_present": patient.symptoms.edema,
            "orthopnea_present": (
                patient.symptoms.orthopnea
            ),
            "pulmonary_rales_present": bool(
                patient.metadata.get(
                    "pulmonary_rales_present",
                    False,
                )
            ),
            "jugular_venous_distension_present": bool(
                patient.metadata.get(
                    "jugular_venous_distension_present",
                    False,
                )
            ),
            "bnp_pg_ml": patient.laboratory.bnp_pg_ml,
            "nt_probnp_pg_ml": (
                patient.laboratory.nt_pro_bnp_pg_ml
            ),
            "troponin_value": (
                patient.laboratory.troponin_value
            ),
            "troponin_upper_reference_limit": (
                patient.laboratory
                .troponin_upper_reference_limit
            ),
            "medications": medication_names,
            "symptoms": self._build_symptom_names(
                patient
            ),
            "metadata": metadata,
        }

        return self._construct_dataclass(
            CardiovascularAssessmentInput,
            input_values,
        )

    def adapt_and_assess(
        self,
        patient: ClinicalPatient,
        *,
        engine: Any | None = None,
    ) -> Any:
        """Adapta o paciente e executa o motor cardiovascular."""

        if engine is None:
            from app.clinical_pharmacy_engine.assessment.cardiovascular import (
                CardiovascularAssessmentEngine,
            )

            engine = CardiovascularAssessmentEngine()

        cardiovascular_input = self.adapt(patient)

        return engine.assess(cardiovascular_input)

    # ========================================================
    # Construção das estruturas auxiliares
    # ========================================================

    def _build_blood_pressure_measurements(
        self,
        patient: ClinicalPatient,
    ) -> list[BloodPressureMeasurement]:
        result: list[BloodPressureMeasurement] = []

        vital_signs = list(patient.vital_signs)

        if (
            not self.config.include_all_vital_signs
            and patient.latest_vital_signs is not None
        ):
            vital_signs = [patient.latest_vital_signs]

        for vital in vital_signs:
            systolic = (
                vital.systolic_blood_pressure_mm_hg
            )

            diastolic = (
                vital.diastolic_blood_pressure_mm_hg
            )

            if systolic is None or diastolic is None:
                continue

            values = {
                "systolic_mm_hg": systolic,
                "diastolic_mm_hg": diastolic,
                "heart_rate_bpm": vital.heart_rate_bpm,
                "context": self._resolve_bp_context(
                    vital.context
                ),
                "measured_at": self._datetime_to_string(
                    vital.measured_at
                ),
                "metadata": dict(vital.metadata),
            }

            result.append(
                self._construct_dataclass(
                    BloodPressureMeasurement,
                    values,
                )
            )

        return result

    def _build_lipid_profile(
        self,
        patient: ClinicalPatient,
    ) -> LipidProfile:
        laboratory = patient.laboratory

        values = {
            "total_cholesterol": (
                laboratory.total_cholesterol_mg_dl
            ),
            "ldl_cholesterol": (
                laboratory.ldl_cholesterol_mg_dl
            ),
            "hdl_cholesterol": (
                laboratory.hdl_cholesterol_mg_dl
            ),
            "triglycerides": (
                laboratory.triglycerides_mg_dl
            ),
            "non_hdl_cholesterol": (
                self._calculate_non_hdl(patient)
            ),
            "apolipoprotein_b": (
                laboratory.metadata.get(
                    "apolipoprotein_b"
                )
            ),
            "lipoprotein_a": (
                laboratory.metadata.get(
                    "lipoprotein_a"
                )
            ),
            "fasting": laboratory.metadata.get(
                "fasting"
            ),
            "collected_at": self._datetime_to_string(
                laboratory.collected_at
            ),
            "metadata": dict(laboratory.metadata),
        }

        return self._construct_dataclass(
            LipidProfile,
            values,
        )

    def _build_ecg(
        self,
        patient: ClinicalPatient,
    ) -> ECGData:
        ecg = patient.ecg

        values = {
            "rhythm": ecg.rhythm,
            "heart_rate_bpm": ecg.heart_rate_bpm,
            "pr_interval_ms": ecg.pr_interval_ms,
            "qrs_duration_ms": ecg.qrs_duration_ms,
            "qt_interval_ms": ecg.qt_interval_ms,
            "corrected_qt_ms": ecg.corrected_qt_ms,
            "st_elevation_mm": ecg.st_elevation_mm,
            "st_depression_mm": ecg.st_depression_mm,
            "dynamic_changes": ecg.dynamic_changes,
            "ischemic_changes": ecg.ischemic_changes,
            "atrial_fibrillation": (
                ecg.atrial_fibrillation
            ),
            "ventricular_arrhythmia_present": (
                ecg.ventricular_arrhythmia_present
            ),
            "report_date": self._datetime_to_string(
                ecg.performed_at
            ),
            "metadata": {
                **dict(ecg.metadata),
                "interpretation": ecg.interpretation,
            },
        }

        return self._construct_dataclass(
            ECGData,
            values,
        )

    def _build_echocardiogram(
        self,
        patient: ClinicalPatient,
    ) -> EchocardiogramData:
        echo = patient.echocardiogram

        values = {
            "left_ventricular_ejection_fraction_percent": (
                echo
                .left_ventricular_ejection_fraction_percent
            ),
            "previous_ejection_fraction_percent": (
                echo.previous_ejection_fraction_percent
            ),
            "left_atrial_volume_index_ml_m2": (
                echo.left_atrial_volume_index_ml_m2
            ),
            "e_over_e_prime": echo.e_over_e_prime,
            "tricuspid_regurgitation_velocity_m_s": (
                echo
                .tricuspid_regurgitation_velocity_m_s
            ),
            "left_ventricular_hypertrophy": (
                echo.left_ventricular_hypertrophy
            ),
            "right_ventricular_dysfunction": (
                echo.right_ventricular_dysfunction
            ),
            "significant_valvular_disease": (
                echo.significant_valvular_disease
            ),
            "report_date": self._datetime_to_string(
                echo.performed_at
            ),
            "metadata": {
                **dict(echo.metadata),
                "interpretation": echo.interpretation,
            },
        }

        return self._construct_dataclass(
            EchocardiogramData,
            values,
        )

    # ========================================================
    # Resolução de enums
    # ========================================================

    def _resolve_bp_context(
        self,
        value: str | None,
    ) -> BloodPressureContext:
        normalized = self._normalize_text(value or "")

        aliases: dict[str, tuple[str, ...]] = {
            "office": (
                "OFFICE",
                "office",
                "CLINIC",
                "clinic",
            ),
            "consultorio": (
                "OFFICE",
                "office",
                "CLINIC",
                "clinic",
            ),
            "consultório": (
                "OFFICE",
                "office",
                "CLINIC",
                "clinic",
            ),
            "home": (
                "HOME",
                "home",
            ),
            "domiciliar": (
                "HOME",
                "home",
            ),
            "ambulatory": (
                "AMBULATORY",
                "ambulatory",
                "ABPM",
                "abpm",
            ),
            "ambulatorial": (
                "AMBULATORY",
                "ambulatory",
            ),
            "hospital": (
                "HOSPITAL",
                "hospital",
                "INPATIENT",
                "inpatient",
            ),
            "hospitalar": (
                "HOSPITAL",
                "hospital",
                "INPATIENT",
                "inpatient",
            ),
        }

        candidates = aliases.get(
            normalized,
            (
                normalized,
                "OFFICE",
                "office",
            ),
        )

        return self._resolve_enum(
            BloodPressureContext,
            *candidates,
            default=(
                self.config
                .default_blood_pressure_context
            ),
        )

    def _resolve_prevention_context(
        self,
        *,
        established_ascvd: bool,
        prior_myocardial_infarction: bool,
        prior_stroke_or_tia: bool,
        peripheral_arterial_disease: bool,
    ) -> PreventionContext:
        secondary = any(
            (
                established_ascvd,
                prior_myocardial_infarction,
                prior_stroke_or_tia,
                peripheral_arterial_disease,
            )
        )

        if secondary:
            return self._resolve_enum(
                PreventionContext,
                "SECONDARY",
                "secondary",
                "SECONDARY_PREVENTION",
                "secondary_prevention",
            )

        if self.config.default_prevention_context is not None:
            return self.config.default_prevention_context

        return self._resolve_enum(
            PreventionContext,
            "PRIMARY",
            "primary",
            "PRIMARY_PREVENTION",
            "primary_prevention",
        )

    def _resolve_cardiovascular_sex(
        self,
        patient: ClinicalPatient,
    ) -> CardiovascularSex:
        biological_sex = (
            patient.demographics.biological_sex
        )

        if biological_sex == BiologicalSex.MALE:
            candidates = ("MALE", "male", "M")

        elif biological_sex == BiologicalSex.FEMALE:
            candidates = ("FEMALE", "female", "F")

        else:
            candidates = (
                "UNDETERMINED",
                "undetermined",
                "UNKNOWN",
                "unknown",
                "OTHER",
                "other",
            )

        return self._resolve_enum(
            CardiovascularSex,
            *candidates,
        )

    def _resolve_smoking_status(
        self,
        patient: ClinicalPatient,
    ) -> SmokingStatus:
        mapping = {
            CoreSmokingStatus.NEVER: (
                "NEVER",
                "never",
                "NON_SMOKER",
                "non_smoker",
            ),
            CoreSmokingStatus.FORMER: (
                "FORMER",
                "former",
                "EX_SMOKER",
                "ex_smoker",
            ),
            CoreSmokingStatus.CURRENT: (
                "CURRENT",
                "current",
                "SMOKER",
                "smoker",
            ),
            CoreSmokingStatus.PASSIVE_EXPOSURE: (
                "PASSIVE_EXPOSURE",
                "passive_exposure",
                "PASSIVE",
                "passive",
            ),
        }

        candidates = mapping.get(
            patient.lifestyle.smoking_status,
            (
                "UNDETERMINED",
                "undetermined",
                "UNKNOWN",
                "unknown",
            ),
        )

        return self._resolve_enum(
            SmokingStatus,
            *candidates,
        )

    def _resolve_diabetes_status(
        self,
        diagnoses: list[str],
    ) -> DiabetesStatus:
        if self._has_any_diagnosis(
            diagnoses,
            {
                "diabetes mellitus tipo 1",
                "diabetes tipo 1",
                "type 1 diabetes",
                "dm1",
            },
        ):
            candidates = (
                "TYPE_1",
                "type_1",
                "TYPE1",
                "type1",
            )

        elif self._has_any_diagnosis(
            diagnoses,
            {
                "diabetes mellitus tipo 2",
                "diabetes tipo 2",
                "type 2 diabetes",
                "dm2",
            },
        ):
            candidates = (
                "TYPE_2",
                "type_2",
                "TYPE2",
                "type2",
            )

        elif self._has_any_diagnosis(
            diagnoses,
            {
                "diabetes",
                "diabetes mellitus",
            },
        ):
            candidates = (
                "OTHER",
                "other",
                "PRESENT",
                "present",
            )

        else:
            candidates = (
                "NONE",
                "none",
                "ABSENT",
                "absent",
                "NO_DIABETES",
                "no_diabetes",
            )

        return self._resolve_enum(
            DiabetesStatus,
            *candidates,
        )

    def _resolve_atrial_fibrillation_type(
        self,
        present: bool,
    ) -> AtrialFibrillationType:
        if present:
            candidates = (
                "UNDETERMINED",
                "undetermined",
                "UNKNOWN",
                "unknown",
                "PRESENT",
                "present",
            )
        else:
            candidates = (
                "NONE",
                "none",
                "ABSENT",
                "absent",
                "NO_AF",
                "no_af",
            )

        return self._resolve_enum(
            AtrialFibrillationType,
            *candidates,
        )

    def _resolve_nyha_class(
        self,
        patient: ClinicalPatient,
    ) -> NYHAClass:
        value = patient.metadata.get("nyha_class")

        if isinstance(value, NYHAClass):
            return value

        normalized = (
            str(value).strip().upper()
            if value is not None
            else ""
        )

        candidates = {
            "I": ("I", "1"),
            "1": ("I", "1"),
            "II": ("II", "2"),
            "2": ("II", "2"),
            "III": ("III", "3"),
            "3": ("III", "3"),
            "IV": ("IV", "4"),
            "4": ("IV", "4"),
        }.get(
            normalized,
            (
                "UNDETERMINED",
                "undetermined",
                "UNKNOWN",
                "unknown",
            ),
        )

        return self._resolve_enum(
            NYHAClass,
            *candidates,
        )

    def _resolve_anticoagulant(
        self,
        medications: list[str],
    ) -> AnticoagulantType:
        mappings: tuple[
            tuple[set[str], tuple[str, ...]],
            ...,
        ] = (
            (
                {"warfarin", "varfarina", "marevan"},
                ("WARFARIN", "warfarin"),
            ),
            (
                {"apixaban", "apixabana", "eliquis"},
                ("APIXABAN", "apixaban"),
            ),
            (
                {
                    "rivaroxaban",
                    "rivaroxabana",
                    "xarelto",
                },
                ("RIVAROXABAN", "rivaroxaban"),
            ),
            (
                {
                    "dabigatran",
                    "dabigatrana",
                    "pradaxa",
                },
                ("DABIGATRAN", "dabigatran"),
            ),
            (
                {"edoxaban", "edoxabana"},
                ("EDOXABAN", "edoxaban"),
            ),
            (
                {"enoxaparin", "enoxaparina"},
                (
                    "LOW_MOLECULAR_WEIGHT_HEPARIN",
                    "low_molecular_weight_heparin",
                ),
            ),
            (
                {"heparina", "heparin"},
                (
                    "UNFRACTIONATED_HEPARIN",
                    "unfractionated_heparin",
                ),
            ),
            (
                {"fondaparinux"},
                ("FONDAPARINUX", "fondaparinux"),
            ),
        )

        for medication_terms, enum_candidates in mappings:
            if self._uses_any_medication(
                medications,
                medication_terms,
            ):
                return self._resolve_enum(
                    AnticoagulantType,
                    *enum_candidates,
                )

        return self._resolve_enum(
            AnticoagulantType,
            "NONE",
            "none",
            "NO_ANTICOAGULANT",
            "no_anticoagulant",
            "OTHER",
            "other",
        )

    # ========================================================
    # Metadados e utilidades clínicas
    # ========================================================

    def _build_metadata(
        self,
        *,
        patient: ClinicalPatient,
        normalized_diagnoses: list[str],
    ) -> dict[str, Any]:
        latest_vitals = patient.latest_vital_signs

        metadata: dict[str, Any] = {
            **dict(patient.metadata),
            **dict(self.config.metadata),
            "clinical_patient_id": str(patient.id),
            "clinical_patient_external_id": (
                patient.external_id
            ),
            "adapter": (
                "CardiovascularPatientAdapter"
            ),
            "adapted_at": datetime.utcnow().isoformat(),
            "possible_target_organ_damage": any(
                (
                    patient.symptoms.confusion,
                    patient.symptoms.chest_pain,
                    patient.symptoms.dyspnea,
                    patient.symptoms.reduced_urine_output,
                )
            ),
            "oxygen_saturation_percent": (
                latest_vitals.oxygen_saturation_percent
                if latest_vitals is not None
                else None
            ),
            "fatigue": patient.symptoms.fatigue,
            "altered_mental_status": (
                patient.symptoms.confusion
            ),
            "oliguria": (
                patient.symptoms.reduced_urine_output
            ),
            "dizziness_or_presyncope": (
                patient.symptoms.dizziness
            ),
            "ascites": self._has_any_diagnosis(
                normalized_diagnoses,
                {"ascite", "ascites"},
            ),
        }

        country_code = (
            self.config.country_code
            or patient.metadata.get("country_code")
        )

        if country_code:
            metadata["country_code"] = (
                str(country_code).upper()
            )

        return metadata

    def _build_symptom_names(
        self,
        patient: ClinicalPatient,
    ) -> list[str]:
        symptom_map = {
            "dor torácica": patient.symptoms.chest_pain,
            "dispneia": patient.symptoms.dyspnea,
            "ortopneia": patient.symptoms.orthopnea,
            "edema": patient.symptoms.edema,
            "fadiga": patient.symptoms.fatigue,
            "síncope": patient.symptoms.syncope,
            "tontura": patient.symptoms.dizziness,
            "náusea ou vômito": (
                patient.symptoms.nausea_or_vomiting
            ),
            "febre": patient.symptoms.fever,
            "confusão": patient.symptoms.confusion,
            "oligúria": (
                patient.symptoms.reduced_urine_output
            ),
            "sangramento": patient.symptoms.bleeding,
        }

        values = [
            name
            for name, present in symptom_map.items()
            if present
        ]

        values.extend(
            patient.symptoms.additional_symptoms
        )

        return self._unique_strings(values)

    @staticmethod
    def _calculate_non_hdl(
        patient: ClinicalPatient,
    ) -> float | None:
        total = (
            patient.laboratory.total_cholesterol_mg_dl
        )

        hdl = (
            patient.laboratory.hdl_cholesterol_mg_dl
        )

        if total is None or hdl is None:
            return None

        return round(total - hdl, 2)

    # ========================================================
    # Introspecção e compatibilidade
    # ========================================================

    @staticmethod
    def _construct_dataclass(
        cls: type[T],
        values: dict[str, Any],
    ) -> T:
        """
        Instancia uma dataclass usando somente campos existentes.

        Campos desconhecidos são descartados. Campos obrigatórios
        ausentes são preenchidos defensivamente quando possível.
        """

        if not is_dataclass(cls):
            raise TypeError(
                f"{cls.__name__} não é uma dataclass."
            )

        class_fields = {
            item.name: item
            for item in fields(cls)
        }

        compatible_values = {
            name: value
            for name, value in values.items()
            if name in class_fields
        }

        for name, item in class_fields.items():
            if name in compatible_values:
                continue

            has_default = item.default is not MISSING

            has_factory = (
                item.default_factory is not MISSING
            )

            if has_default or has_factory:
                continue

            compatible_values[name] = (
                CardiovascularPatientAdapter
                ._safe_missing_value(item.type)
            )

        return cls(**compatible_values)

    @staticmethod
    def _safe_missing_value(
        annotation: Any,
    ) -> Any:
        """
        Produz valor neutro para campos obrigatórios ausentes.

        Esta função existe apenas como proteção de compatibilidade.
        """

        text = str(annotation).casefold()

        if "bool" in text:
            return False

        if "list" in text:
            return []

        if "dict" in text:
            return {}

        if "set" in text:
            return set()

        if "tuple" in text:
            return ()

        return None

    @staticmethod
    def _resolve_enum(
        enum_class: type[T],
        *candidates: str,
        default: T | None = None,
    ) -> T:
        """
        Resolve enum por nome ou valor, ignorando maiúsculas.

        Quando nenhum candidato for localizado, utiliza o default
        configurado ou o primeiro membro disponível.
        """

        if default is not None:
            if isinstance(default, enum_class):
                fallback = default
            else:
                fallback = None
        else:
            fallback = None

        normalized_candidates = {
            str(candidate).strip().casefold()
            for candidate in candidates
            if candidate is not None
            and str(candidate).strip()
        }

        if issubclass(enum_class, Enum):
            for member_name, member in (
                enum_class.__members__.items()
            ):
                normalized_name = (
                    str(member_name)
                    .strip()
                    .casefold()
                )

                normalized_value = (
                    str(member.value)
                    .strip()
                    .casefold()
                )

                if (
                    normalized_name
                    in normalized_candidates
                    or normalized_value
                    in normalized_candidates
                ):
                    return member

        if fallback is not None:
            return fallback

        members = list(enum_class)

        if members:
            return members[0]

        raise ValueError(
            f"O enum {enum_class.__name__} não possui membros."
        )

    # ========================================================
    # Comparação de textos
    # ========================================================

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        return " ".join(
            str(value).strip().casefold().split()
        )

    @classmethod
    def _contains_any(
        cls,
        value: str,
        terms: set[str],
    ) -> bool:
        normalized_value = cls._normalize_text(
            value
        )

        return any(
            cls._normalize_text(term)
            in normalized_value
            for term in terms
        )

    @classmethod
    def _has_any_diagnosis(
        cls,
        diagnoses: list[str],
        terms: set[str],
    ) -> bool:
        return any(
            cls._contains_any(
                diagnosis,
                terms,
            )
            for diagnosis in diagnoses
        )

    @classmethod
    def _uses_any_medication(
        cls,
        medications: list[str],
        terms: set[str],
    ) -> bool:
        return any(
            cls._contains_any(
                medication,
                terms,
            )
            for medication in medications
        )

    @staticmethod
    def _antihypertensive_terms() -> set[str]:
        return {
            "losartana",
            "valsartana",
            "candesartana",
            "telmisartana",
            "enalapril",
            "captopril",
            "ramipril",
            "lisinopril",
            "perindopril",
            "amlodipino",
            "anlodipino",
            "nifedipino",
            "hidroclorotiazida",
            "clortalidona",
            "indapamida",
            "carvedilol",
            "bisoprolol",
            "metoprolol",
            "atenolol",
            "propranolol",
            "espironolactona",
            "eplerenona",
            "sacubitril",
        }

    @classmethod
    def _unique_strings(
        cls,
        values: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = cls._normalize_text(value)

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)
            result.append(str(value).strip())

        return result

    @staticmethod
    def _datetime_to_string(
        value: datetime | date | None,
    ) -> str | None:
        if value is None:
            return None

        return value.isoformat()