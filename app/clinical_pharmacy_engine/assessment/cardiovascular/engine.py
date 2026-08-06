"""
PHK Studio
Clinical Pharmacy Engine

Cardiovascular Assessment Orchestrator.

Orquestra os motores cardiovasculares especializados:

- hipertensão;
- ASCVD Pooled Cohort Equations;
- Framingham;
- SCORE2;
- dislipidemia;
- intervalo QT;
- anticoagulação;
- insuficiência cardíaca;
- síndrome coronariana aguda.

Responsabilidades:

- transformar CardiovascularAssessmentInput nas entradas
  específicas de cada motor;
- executar cada módulo de forma isolada;
- impedir que a falha de um módulo interrompa toda a análise;
- consolidar riscos, alertas e recomendações;
- adaptar resultados especializados aos modelos públicos;
- registrar rastreabilidade e estado de execução;
- produzir CardiovascularAssessmentResult.

O motor fornece suporte à decisão clínica.
Nenhuma recomendação altera automaticamente a prescrição.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

from app.clinical_pharmacy_engine.assessment.models import (
    AssessmentAlert,
    AssessmentDataQuality,
    AssessmentStatus,
    ClinicalRecommendation,
    ClinicalRiskLevel,
    RecommendationCategory,
    RecommendationPriority,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.models import (
    ACSRiskCategory,
    AnticoagulantType,
    AnticoagulationAssessmentResult,
    AtrialFibrillationType,
    BloodPressureContext,
    CardiovascularAssessmentInput,
    CardiovascularAssessmentResult,
    CardiovascularRiskCategory,
    CardiovascularRiskEstimate,
    CardiovascularSex,
    ChestPainType,
    CongestionStatus,
    DiabetesStatus,
    DyslipidemiaAssessmentResult,
    HeartFailureAssessmentResult,
    HeartFailurePhenotype,
    HypertensionAssessmentResult,
    NYHAClass,
    PerfusionStatus,
    PreventionContext,
    QTAssessmentResult,
    SmokingStatus,
    ThromboembolicRiskCategory,
    BleedingRiskCategory,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.hypertension import (
    HypertensionAssessmentEngine,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.ascvd import (
    ASCVDPooledCohortEngine,
    PCEPopulationGroup,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.framingham import (
    FraminghamRiskEngine,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.score2 import (
    SCORE2Region,
    SCORE2RiskEngine,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.dyslipidemia import (
    DyslipidemiaAssessmentEngine,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.qt_interval import (
    QTIntervalAssessmentEngine,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.anticoagulation import (
    AnticoagulantClass,
    AnticoagulationAssessmentEngine,
    AnticoagulationAssessmentInput,
    AnticoagulationGuideline,
    AtrialArrhythmiaType,
    BleedingRiskCategory as InternalBleedingRiskCategory,
    StrokeRiskCategory,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.heart_failure import (
    CongestionCategory,
    HeartFailureAssessmentEngine,
    HeartFailureAssessmentInput,
    HeartFailurePhenotype as InternalHeartFailurePhenotype,
    HemodynamicProfile,
    NYHAClass as InternalNYHAClass,
    PerfusionCategory,
)

from app.clinical_pharmacy_engine.assessment.cardiovascular.acute_coronary import (
    AcuteCoronaryAssessmentEngine,
    AcuteCoronaryAssessmentInput,
    AcuteCoronarySyndromeType,
    ChestPainPattern,
    ECGIschemiaPattern,
    HEARTRiskCategory,
)


T = TypeVar("T")


# ============================================================
# Estado de execução
# ============================================================


class CardiovascularModuleState(str, Enum):
    """Estado de execução de um módulo cardiovascular."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(slots=True)
class CardiovascularModuleExecution:
    """Registro auditável da execução de um módulo."""

    module_name: str
    state: CardiovascularModuleState

    message: str = ""

    error_type: str | None = None
    error_message: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class CardiovascularEngineConfig:
    """Configurações do orquestrador."""

    enable_hypertension: bool = True
    enable_ascvd: bool = True
    enable_framingham: bool = True
    enable_score2: bool = True
    enable_dyslipidemia: bool = True
    enable_qt: bool = True
    enable_anticoagulation: bool = True
    enable_heart_failure: bool = True
    enable_acute_coronary: bool = True

    pce_population_group: PCEPopulationGroup = (
        PCEPopulationGroup.OTHER
    )

    score2_region: SCORE2Region = (
        SCORE2Region.UNDETERMINED
    )

    anticoagulation_guideline: AnticoagulationGuideline = (
        AnticoagulationGuideline.ESC_2024
    )

    continue_on_module_error: bool = True

    include_invalid_risk_estimates: bool = False

    references: list[str] = field(
        default_factory=lambda: [
            "PHK Studio Cardiovascular Assessment Engine",
            "ACC/AHA cardiovascular risk framework",
            "ESC cardiovascular prevention framework",
            "Heart Failure guideline framework",
            "Acute Coronary Syndrome guideline framework",
        ]
    )


# ============================================================
# Orquestrador
# ============================================================


class CardiovascularAssessmentEngine:
    """Orquestrador central do domínio cardiovascular."""

    def __init__(
        self,
        config: CardiovascularEngineConfig | None = None,
        *,
        hypertension_engine: (
            HypertensionAssessmentEngine | None
        ) = None,
        ascvd_engine: (
            ASCVDPooledCohortEngine | None
        ) = None,
        framingham_engine: (
            FraminghamRiskEngine | None
        ) = None,
        score2_engine: SCORE2RiskEngine | None = None,
        dyslipidemia_engine: (
            DyslipidemiaAssessmentEngine | None
        ) = None,
        qt_engine: (
            QTIntervalAssessmentEngine | None
        ) = None,
        anticoagulation_engine: (
            AnticoagulationAssessmentEngine | None
        ) = None,
        heart_failure_engine: (
            HeartFailureAssessmentEngine | None
        ) = None,
        acute_coronary_engine: (
            AcuteCoronaryAssessmentEngine | None
        ) = None,
    ) -> None:
        self.config = (
            config
            or CardiovascularEngineConfig()
        )

        self.hypertension_engine = (
            hypertension_engine
            or HypertensionAssessmentEngine()
        )

        self.ascvd_engine = (
            ascvd_engine
            or ASCVDPooledCohortEngine()
        )

        self.framingham_engine = (
            framingham_engine
            or FraminghamRiskEngine()
        )

        self.score2_engine = (
            score2_engine
            or SCORE2RiskEngine()
        )

        self.dyslipidemia_engine = (
            dyslipidemia_engine
            or DyslipidemiaAssessmentEngine()
        )

        self.qt_engine = (
            qt_engine
            or QTIntervalAssessmentEngine()
        )

        self.anticoagulation_engine = (
            anticoagulation_engine
            or AnticoagulationAssessmentEngine()
        )

        self.heart_failure_engine = (
            heart_failure_engine
            or HeartFailureAssessmentEngine()
        )

        self.acute_coronary_engine = (
            acute_coronary_engine
            or AcuteCoronaryAssessmentEngine()
        )

    def assess(
        self,
        data: CardiovascularAssessmentInput,
    ) -> CardiovascularAssessmentResult:
        """
        Executa a avaliação cardiovascular integrada.

        Cada módulo é executado de forma independente.
        Uma falha isolada não invalida automaticamente
        toda a avaliação.
        """

        executions: list[
            CardiovascularModuleExecution
        ] = []

        module_errors: list[str] = []

        risk_estimates: list[
            CardiovascularRiskEstimate
        ] = []

        hypertension_result = (
            HypertensionAssessmentResult()
        )

        dyslipidemia_result = (
            DyslipidemiaAssessmentResult()
        )

        qt_result = QTAssessmentResult()

        anticoagulation_result = (
            AnticoagulationAssessmentResult()
        )

        heart_failure_result = (
            HeartFailureAssessmentResult()
        )

        acute_coronary_result = (
            self._empty_acute_coronary_result()
        )

        systolic, diastolic, heart_rate = (
            self._average_vitals(data)
        )

        diabetes = self._has_diabetes(data)
        current_smoker = (
            data.smoking_status
            == SmokingStatus.CURRENT
        )

        secondary_prevention = self._is_secondary_prevention(
            data
        )

        # ----------------------------------------------------
        # Hipertensão
        # ----------------------------------------------------

        if self.config.enable_hypertension:
            if data.blood_pressure_measurements:
                hypertension_result = self._execute_module(
                    module_name="hypertension",
                    executions=executions,
                    errors=module_errors,
                    default=hypertension_result,
                    function=lambda: (
                        self.hypertension_engine.assess(
                            data.blood_pressure_measurements,
                            treated_hypertension=(
                                data.treated_hypertension
                            ),
                            hypertension_history=(
                                data.hypertension_history
                            ),
                            symptoms=data.symptoms,
                            possible_target_organ_damage=bool(
                                data.metadata.get(
                                    "possible_target_organ_damage",
                                    False,
                                )
                            ),
                            office_measurements=(
                                self._measurements_by_context(
                                    data,
                                    BloodPressureContext.OFFICE,
                                )
                            ),
                            home_measurements=(
                                self._measurements_by_context(
                                    data,
                                    BloodPressureContext.HOME,
                                )
                            ),
                        )
                    ),
                )
            else:
                self._record_skipped(
                    executions,
                    "hypertension",
                    "Nenhuma medição de pressão arterial.",
                )

        # ----------------------------------------------------
        # ASCVD PCE
        # ----------------------------------------------------

        if self.config.enable_ascvd:
            ascvd = self._execute_module(
                module_name="ascvd",
                executions=executions,
                errors=module_errors,
                default=None,
                function=lambda: (
                    self.ascvd_engine.assess_integrated_input(
                        data,
                        population_group=(
                            self._resolve_pce_population(data)
                        ),
                    )
                ),
            )

            if (
                ascvd is not None
                and (
                    ascvd.valid
                    or self.config
                    .include_invalid_risk_estimates
                )
            ):
                risk_estimates.append(ascvd)

        # ----------------------------------------------------
        # Framingham
        # ----------------------------------------------------

        if self.config.enable_framingham:
            framingham = self._execute_module(
                module_name="framingham",
                executions=executions,
                errors=module_errors,
                default=None,
                function=lambda: (
                    self.framingham_engine
                    .assess_integrated_input(data)
                ),
            )

            if (
                framingham is not None
                and (
                    framingham.valid
                    or self.config
                    .include_invalid_risk_estimates
                )
            ):
                risk_estimates.append(framingham)

        # ----------------------------------------------------
        # SCORE2
        # ----------------------------------------------------

        if self.config.enable_score2:
            score2_region = self._resolve_score2_region(
                data
            )

            score2 = self._execute_module(
                module_name="score2",
                executions=executions,
                errors=module_errors,
                default=None,
                function=lambda: (
                    self.score2_engine.assess_integrated_input(
                        data,
                        region=score2_region,
                        country_code=(
                            self._country_code(data)
                        ),
                    )
                ),
            )

            if score2 is not None:
                if score2.valid:
                    risk_estimates.append(score2)

                elif (
                    score2.metadata.get(
                        "calculation_blocked"
                    )
                ):
                    self._replace_module_state(
                        executions,
                        "score2",
                        CardiovascularModuleState.BLOCKED,
                        (
                            "SCORE2 bloqueado por ausência "
                            "de tabela validada ou aplicabilidade."
                        ),
                    )

                elif (
                    self.config
                    .include_invalid_risk_estimates
                ):
                    risk_estimates.append(score2)

        overall_risk = self._resolve_overall_risk(
            data=data,
            risk_estimates=risk_estimates,
        )

        # ----------------------------------------------------
        # Dislipidemia
        # ----------------------------------------------------

        if self.config.enable_dyslipidemia:
            dyslipidemia_engine_result = (
                self._execute_module(
                    module_name="dyslipidemia",
                    executions=executions,
                    errors=module_errors,
                    default=None,
                    function=lambda: (
                        self.dyslipidemia_engine
                        .assess_integrated_input(
                            data,
                            risk_estimates=risk_estimates,
                            statin_intolerance=bool(
                                data.metadata.get(
                                    "statin_intolerance",
                                    False,
                                )
                            ),
                            current_lipid_lowering_therapy=bool(
                                data.metadata.get(
                                    "current_lipid_lowering_therapy",
                                    False,
                                )
                            ),
                        )
                    ),
                )
            )

            if dyslipidemia_engine_result is not None:
                dyslipidemia_result = (
                    dyslipidemia_engine_result.assessment
                )

                if not (
                    dyslipidemia_engine_result
                    .target_profile_applied
                ):
                    self._replace_module_state(
                        executions,
                        "dyslipidemia",
                        CardiovascularModuleState.PARTIAL,
                        (
                            "Avaliação laboratorial concluída, "
                            "sem protocolo validado de metas."
                        ),
                    )

        # ----------------------------------------------------
        # QT
        # ----------------------------------------------------

        if self.config.enable_qt:
            if (
                data.ecg.qt_interval_ms is not None
                or data.ecg.corrected_qt_ms is not None
            ):
                qt_result = self._execute_module(
                    module_name="qt_interval",
                    executions=executions,
                    errors=module_errors,
                    default=qt_result,
                    function=lambda: (
                        self.qt_engine
                        .assess_integrated_input(
                            data,
                            congenital_long_qt_syndrome=bool(
                                data.metadata.get(
                                    "congenital_long_qt_syndrome",
                                    False,
                                )
                            ),
                            previous_torsades_de_pointes=bool(
                                data.metadata.get(
                                    "previous_torsades_de_pointes",
                                    False,
                                )
                            ),
                            previous_syncope_suspected_arrhythmic=bool(
                                data.metadata.get(
                                    "previous_arrhythmic_syncope",
                                    False,
                                )
                            ),
                        )
                    ),
                )
            else:
                self._record_skipped(
                    executions,
                    "qt_interval",
                    "QT ou QTc não informado.",
                )

        # ----------------------------------------------------
        # Anticoagulação
        # ----------------------------------------------------

        if self.config.enable_anticoagulation:
            if (
                data.atrial_fibrillation
                or data.atrial_fibrillation_type
                not in {
                    AtrialFibrillationType.NONE,
                    AtrialFibrillationType.UNDETERMINED,
                }
            ):
                internal_anticoagulation = (
                    self._execute_module(
                        module_name="anticoagulation",
                        executions=executions,
                        errors=module_errors,
                        default=None,
                        function=lambda: (
                            self.anticoagulation_engine.assess(
                                self._build_anticoagulation_input(
                                    data,
                                    diabetes=diabetes,
                                )
                            )
                        ),
                    )
                )

                if internal_anticoagulation is not None:
                    anticoagulation_result = (
                        self._adapt_anticoagulation_result(
                            internal_anticoagulation,
                            data,
                        )
                    )
            else:
                self._record_skipped(
                    executions,
                    "anticoagulation",
                    (
                        "Fibrilação atrial ou flutter "
                        "não informados."
                    ),
                )

        # ----------------------------------------------------
        # Insuficiência cardíaca
        # ----------------------------------------------------

        if self.config.enable_heart_failure:
            if self._should_run_heart_failure(data):
                internal_heart_failure = (
                    self._execute_module(
                        module_name="heart_failure",
                        executions=executions,
                        errors=module_errors,
                        default=None,
                        function=lambda: (
                            self.heart_failure_engine.assess(
                                self._build_heart_failure_input(
                                    data,
                                    systolic=systolic,
                                    diastolic=diastolic,
                                    heart_rate=heart_rate,
                                )
                            )
                        ),
                    )
                )

                if internal_heart_failure is not None:
                    heart_failure_result = (
                        self._adapt_heart_failure_result(
                            internal_heart_failure
                        )
                    )
            else:
                self._record_skipped(
                    executions,
                    "heart_failure",
                    (
                        "Sem dados clínicos ou estruturais "
                        "suficientes para avaliação."
                    ),
                )

        # ----------------------------------------------------
        # Síndrome coronariana aguda
        # ----------------------------------------------------

        if self.config.enable_acute_coronary:
            if self._should_run_acute_coronary(data):
                internal_acute_coronary = (
                    self._execute_module(
                        module_name="acute_coronary",
                        executions=executions,
                        errors=module_errors,
                        default=None,
                        function=lambda: (
                            self.acute_coronary_engine.assess(
                                self._build_acute_coronary_input(
                                    data,
                                    systolic=systolic,
                                    diastolic=diastolic,
                                    heart_rate=heart_rate,
                                    diabetes=diabetes,
                                    current_smoker=(
                                        current_smoker
                                    ),
                                )
                            )
                        ),
                    )
                )

                if internal_acute_coronary is not None:
                    acute_coronary_result = (
                        self._adapt_acute_coronary_result(
                            internal_acute_coronary
                        )
                    )
            else:
                self._record_skipped(
                    executions,
                    "acute_coronary",
                    (
                        "Sem dor torácica, troponina ou "
                        "sinal clínico de síndrome coronariana."
                    ),
                )

        # ----------------------------------------------------
        # Consolidação
        # ----------------------------------------------------

        alerts = self._build_alerts(
            hypertension=hypertension_result,
            dyslipidemia=dyslipidemia_result,
            qt=qt_result,
            anticoagulation=anticoagulation_result,
            heart_failure=heart_failure_result,
            acute_coronary=acute_coronary_result,
        )

        recommendations = self._build_recommendations(
            hypertension=hypertension_result,
            dyslipidemia=dyslipidemia_result,
            qt=qt_result,
            anticoagulation=anticoagulation_result,
            heart_failure=heart_failure_result,
            acute_coronary=acute_coronary_result,
        )

        urgent_medical_evaluation = any(
            (
                hypertension_result
                .requires_immediate_evaluation,
                qt_result.immediate_review_required,
                heart_failure_result
                .acute_decompensation_suspected,
                acute_coronary_result
                .immediate_evaluation_required,
                anticoagulation_result
                .active_bleeding_alert,
            )
        )

        emergency_referral = any(
            (
                acute_coronary_result.stemi_suspected,
                acute_coronary_result
                .immediate_evaluation_required
                and bool(
                    acute_coronary_result.metadata.get(
                        "emergency",
                        False,
                    )
                ),
                qt_result.classification.value
                == "extreme",
                anticoagulation_result
                .active_bleeding_alert,
            )
        )

        medication_review_required = any(
            (
                dyslipidemia_result
                .familial_hypercholesterolemia_suspected,
                dyslipidemia_result
                .severe_hypertriglyceridemia,
                qt_result.immediate_review_required,
                anticoagulation_result
                .anticoagulation_review_required,
                anticoagulation_result
                .dose_review_required,
                heart_failure_result
                .guideline_directed_therapy_review_required,
                acute_coronary_result
                .acute_coronary_syndrome_suspected,
            )
        )

        cardiology_review = any(
            (
                secondary_prevention,
                heart_failure_result.valid,
                acute_coronary_result
                .acute_coronary_syndrome_suspected,
                anticoagulation_result
                .anticoagulation_review_required,
                qt_result.immediate_review_required,
                overall_risk
                in {
                    CardiovascularRiskCategory.HIGH,
                    CardiovascularRiskCategory.VERY_HIGH,
                    CardiovascularRiskCategory.EXTREME,
                },
            )
        )

        risk_level = self._resolve_clinical_risk_level(
            overall_risk=overall_risk,
            urgent=urgent_medical_evaluation,
            emergency=emergency_referral,
        )

        quality = self._build_data_quality(
            data=data,
            executions=executions,
            errors=module_errors,
        )

        status = self._resolve_status(
            executions=executions,
            errors=module_errors,
        )

        calculated_values = self._build_calculated_values(
            systolic=systolic,
            diastolic=diastolic,
            risk_estimates=risk_estimates,
            hypertension=hypertension_result,
            qt=qt_result,
            anticoagulation=anticoagulation_result,
            heart_failure=heart_failure_result,
            acute_coronary=acute_coronary_result,
        )

        summary = self._build_summary(
            overall_risk=overall_risk,
            hypertension=hypertension_result,
            dyslipidemia=dyslipidemia_result,
            qt=qt_result,
            anticoagulation=anticoagulation_result,
            heart_failure=heart_failure_result,
            acute_coronary=acute_coronary_result,
        )

        return CardiovascularAssessmentResult(
            assessment_type="cardiovascular",
            status=status,
            risk_level=risk_level,
            summary=summary,
            alerts=alerts,
            recommendations=recommendations,
            data_quality=quality,
            references=list(
                self.config.references
            ),
            requires_pharmacist_review=True,
            requires_prescriber_contact=any(
                recommendation.requires_prescriber_contact
                for recommendation in recommendations
            ),
            requires_referral=cardiology_review,
            requires_emergency_referral=(
                emergency_referral
            ),
            risk_estimates=risk_estimates,
            overall_cardiovascular_risk=(
                overall_risk
            ),
            hypertension=hypertension_result,
            dyslipidemia=dyslipidemia_result,
            heart_failure=heart_failure_result,
            acute_coronary=acute_coronary_result,
            anticoagulation=anticoagulation_result,
            qt_assessment=qt_result,
            secondary_prevention=secondary_prevention,
            medication_review_required=(
                medication_review_required
            ),
            urgent_medical_evaluation_required=(
                urgent_medical_evaluation
            ),
            emergency_referral_required=(
                emergency_referral
            ),
            cardiology_review_suggested=(
                cardiology_review
            ),
            calculated_values=calculated_values,
            metadata={
                "module_executions": [
                    {
                        "module_name": item.module_name,
                        "state": item.state.value,
                        "message": item.message,
                        "error_type": item.error_type,
                        "error_message": item.error_message,
                        "metadata": item.metadata,
                    }
                    for item in executions
                ],
                "module_errors": list(
                    module_errors
                ),
                "completed_modules": sum(
                    item.state
                    == CardiovascularModuleState.COMPLETED
                    for item in executions
                ),
                "partial_modules": sum(
                    item.state
                    == CardiovascularModuleState.PARTIAL
                    for item in executions
                ),
                "blocked_modules": sum(
                    item.state
                    == CardiovascularModuleState.BLOCKED
                    for item in executions
                ),
                "failed_modules": sum(
                    item.state
                    == CardiovascularModuleState.FAILED
                    for item in executions
                ),
                "skipped_modules": sum(
                    item.state
                    == CardiovascularModuleState.SKIPPED
                    for item in executions
                ),
            },
        )

    # ========================================================
    # Execução isolada
    # ========================================================

    def _execute_module(
        self,
        *,
        module_name: str,
        executions: list[
            CardiovascularModuleExecution
        ],
        errors: list[str],
        default: T,
        function: Callable[[], T],
    ) -> T:
        """Executa módulo com contenção de falhas."""

        try:
            result = function()

            executions.append(
                CardiovascularModuleExecution(
                    module_name=module_name,
                    state=(
                        CardiovascularModuleState.COMPLETED
                    ),
                    message="Execução concluída.",
                )
            )

            return result

        except Exception as exc:
            error_message = (
                f"{module_name}: "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append(error_message)

            executions.append(
                CardiovascularModuleExecution(
                    module_name=module_name,
                    state=(
                        CardiovascularModuleState.FAILED
                    ),
                    message=(
                        "Falha isolada durante a execução."
                    ),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

            if not self.config.continue_on_module_error:
                raise

            return default

    @staticmethod
    def _record_skipped(
        executions: list[
            CardiovascularModuleExecution
        ],
        module_name: str,
        message: str,
    ) -> None:
        executions.append(
            CardiovascularModuleExecution(
                module_name=module_name,
                state=CardiovascularModuleState.SKIPPED,
                message=message,
            )
        )

    @staticmethod
    def _replace_module_state(
        executions: list[
            CardiovascularModuleExecution
        ],
        module_name: str,
        state: CardiovascularModuleState,
        message: str,
    ) -> None:
        for execution in reversed(executions):
            if execution.module_name == module_name:
                execution.state = state
                execution.message = message
                return

    # ========================================================
    # Conversores de entrada
    # ========================================================

    def _build_anticoagulation_input(
        self,
        data: CardiovascularAssessmentInput,
        *,
        diabetes: bool,
    ) -> AnticoagulationAssessmentInput:
        return AnticoagulationAssessmentInput(
            age_years=data.age_years,
            atrial_arrhythmia=(
                self._map_atrial_arrhythmia(data)
            ),
            guideline=(
                self.config.anticoagulation_guideline
            ),
            female_sex=(
                data.biological_sex
                == CardiovascularSex.FEMALE
            ),
            congestive_heart_failure=(
                data.heart_failure
            ),
            hypertension=(
                data.hypertension_history
                or data.treated_hypertension
            ),
            diabetes=diabetes,
            previous_stroke=(
                data.prior_stroke_or_tia
            ),
            previous_tia=(
                data.prior_stroke_or_tia
            ),
            vascular_disease=(
                data.vascular_disease
                or data.established_ascvd
            ),
            previous_myocardial_infarction=(
                data.prior_myocardial_infarction
            ),
            peripheral_arterial_disease=(
                data.peripheral_arterial_disease
            ),
            abnormal_renal_function=(
                data.renal_disease
                or data.chronic_kidney_disease
            ),
            abnormal_hepatic_function=(
                data.liver_disease
            ),
            egfr_ml_min_1_73m2=(
                data.egfr_ml_min_1_73m2
            ),
            previous_major_bleeding=(
                data.previous_major_bleeding
            ),
            bleeding_predisposition=(
                data.active_bleeding
            ),
            labile_inr=data.labile_inr,
            concomitant_antiplatelet=bool(
                data.antiplatelet_medications
            ),
            concomitant_nsaid=self._contains_any(
                data.medications,
                {
                    "ibuprofen",
                    "diclofenac",
                    "naproxen",
                    "nimesulide",
                    "nimesulida",
                    "ketoprofen",
                    "cetoprofeno",
                    "meloxicam",
                    "celecoxib",
                    "aine",
                    "nsaid",
                },
            ),
            harmful_alcohol_use=(
                data.alcohol_use_risk
            ),
            active_major_bleeding=(
                data.active_bleeding
            ),
            current_anticoagulant=(
                self._map_anticoagulant(
                    data.current_anticoagulant
                )
            ),
            anticoagulant_medications=[
                medication
                for medication in data.medications
                if self._is_anticoagulant_name(
                    medication
                )
            ],
            interacting_medications=list(
                data.metadata.get(
                    "anticoagulant_interacting_medications",
                    [],
                )
            ),
            metadata={
                **dict(data.metadata),
                "source": (
                    "CardiovascularAssessmentInput"
                ),
            },
        )

    def _build_heart_failure_input(
        self,
        data: CardiovascularAssessmentInput,
        *,
        systolic: float | None,
        diastolic: float | None,
        heart_rate: float | None,
    ) -> HeartFailureAssessmentInput:
        medications = data.medications

        return HeartFailureAssessmentInput(
            age_years=data.age_years,
            current_lvef_percent=(
                data.echocardiogram
                .left_ventricular_ejection_fraction_percent
            ),
            previous_lvef_percent=(
                data.echocardiogram
                .previous_ejection_fraction_percent
            ),
            heart_failure_diagnosis_established=(
                data.heart_failure
            ),
            structural_heart_disease_present=any(
                (
                    data.heart_failure,
                    bool(
                        data.echocardiogram
                        .left_ventricular_hypertrophy
                    ),
                    bool(
                        data.echocardiogram
                        .right_ventricular_dysfunction
                    ),
                    bool(
                        data.echocardiogram
                        .significant_valvular_disease
                    ),
                )
            ),
            increased_filling_pressure_evidence=any(
                (
                    data.echocardiogram
                    .e_over_e_prime is not None,
                    data.echocardiogram
                    .left_atrial_volume_index_ml_m2
                    is not None,
                    data.echocardiogram
                    .tricuspid_regurgitation_velocity_m_s
                    is not None,
                )
            ),
            nyha_class=self._map_nyha_class(
                data.nyha_class
            ),
            dyspnea=data.dyspnea_present,
            exertional_dyspnea=bool(
                data.metadata.get(
                    "exertional_dyspnea",
                    data.dyspnea_present,
                )
            ),
            orthopnea=data.orthopnea_present,
            paroxysmal_nocturnal_dyspnea=bool(
                data.metadata.get(
                    "paroxysmal_nocturnal_dyspnea",
                    False,
                )
            ),
            fatigue=bool(
                data.metadata.get(
                    "fatigue",
                    False,
                )
            ),
            reduced_exercise_tolerance=bool(
                data.metadata.get(
                    "reduced_exercise_tolerance",
                    False,
                )
            ),
            peripheral_edema=data.edema_present,
            pulmonary_rales=(
                data.pulmonary_rales_present
            ),
            elevated_jugular_venous_pressure=(
                data
                .jugular_venous_distension_present
            ),
            ascites=bool(
                data.metadata.get(
                    "ascites",
                    False,
                )
            ),
            pulmonary_edema=bool(
                data.metadata.get(
                    "pulmonary_edema",
                    False,
                )
            ),
            rapid_weight_gain=bool(
                data.metadata.get(
                    "rapid_weight_gain",
                    False,
                )
            ),
            weight_gain_kg=data.metadata.get(
                "weight_gain_kg"
            ),
            weight_gain_period_days=data.metadata.get(
                "weight_gain_period_days"
            ),
            systolic_blood_pressure_mm_hg=systolic,
            diastolic_blood_pressure_mm_hg=diastolic,
            heart_rate_bpm=heart_rate,
            oxygen_saturation_percent=data.metadata.get(
                "oxygen_saturation_percent"
            ),
            cool_extremities=bool(
                data.metadata.get(
                    "cool_extremities",
                    False,
                )
            ),
            altered_mental_status=bool(
                data.metadata.get(
                    "altered_mental_status",
                    False,
                )
            ),
            oliguria=bool(
                data.metadata.get(
                    "oliguria",
                    False,
                )
            ),
            dizziness_or_presyncope=bool(
                data.metadata.get(
                    "dizziness_or_presyncope",
                    False,
                )
            ),
            syncope=data.syncope_present,
            chest_pain_suspected_ischemic=(
                data.chest_pain_present
            ),
            sustained_ventricular_arrhythmia=(
                data.ecg.ventricular_arrhythmia_present
            ),
            bnp_pg_ml=data.bnp_pg_ml,
            nt_pro_bnp_pg_ml=data.nt_probnp_pg_ml,
            egfr_ml_min_1_73m2=(
                data.egfr_ml_min_1_73m2
            ),
            potassium_mmol_l=(
                data.potassium_mmol_l
            ),
            sodium_mmol_l=data.metadata.get(
                "sodium_mmol_l"
            ),
            acute_kidney_injury_suspected=bool(
                data.metadata.get(
                    "acute_kidney_injury_suspected",
                    False,
                )
            ),
            worsening_renal_function=bool(
                data.metadata.get(
                    "worsening_renal_function",
                    False,
                )
            ),
            active_hyperkalemia=bool(
                data.potassium_mmol_l is not None
                and data.potassium_mmol_l >= 5.5
            ),
            symptomatic_hypotension=bool(
                data.metadata.get(
                    "symptomatic_hypotension",
                    False,
                )
            ),
            symptomatic_bradycardia=bool(
                data.metadata.get(
                    "symptomatic_bradycardia",
                    False,
                )
            ),
            ace_inhibitor_present=self._contains_any(
                medications,
                {
                    "enalapril",
                    "captopril",
                    "ramipril",
                    "lisinopril",
                    "perindopril",
                },
            ),
            arb_present=self._contains_any(
                medications,
                {
                    "losartan",
                    "losartana",
                    "valsartan",
                    "valsartana",
                    "candesartan",
                    "candesartana",
                    "telmisartan",
                    "telmisartana",
                },
            ),
            arni_present=self._contains_any(
                medications,
                {
                    "sacubitril",
                    "entresto",
                },
            ),
            evidence_based_beta_blocker_present=(
                self._contains_any(
                    medications,
                    {
                        "carvedilol",
                        "bisoprolol",
                        "metoprolol succinate",
                        "succinato de metoprolol",
                    },
                )
            ),
            mineralocorticoid_receptor_antagonist_present=(
                self._contains_any(
                    medications,
                    {
                        "spironolactone",
                        "espironolactona",
                        "eplerenone",
                        "eplerenona",
                    },
                )
            ),
            sglt2_inhibitor_present=self._contains_any(
                medications,
                {
                    "dapagliflozin",
                    "dapagliflozina",
                    "empagliflozin",
                    "empagliflozina",
                },
            ),
            loop_diuretic_present=self._contains_any(
                medications,
                {
                    "furosemide",
                    "furosemida",
                    "bumetanide",
                    "bumetanida",
                    "torsemide",
                    "torsemida",
                },
            ),
            persistent_symptoms_despite_therapy=bool(
                data.metadata.get(
                    "persistent_symptoms_despite_therapy",
                    False,
                )
            ),
            recent_hospitalization_for_heart_failure=bool(
                data.metadata.get(
                    "recent_hf_hospitalization",
                    False,
                )
            ),
            recurrent_hospitalizations=bool(
                data.metadata.get(
                    "recurrent_hf_hospitalizations",
                    False,
                )
            ),
            inotrope_dependence=bool(
                data.metadata.get(
                    "inotrope_dependence",
                    False,
                )
            ),
            metadata={
                **dict(data.metadata),
                "source": (
                    "CardiovascularAssessmentInput"
                ),
            },
        )

    def _build_acute_coronary_input(
        self,
        data: CardiovascularAssessmentInput,
        *,
        systolic: float | None,
        diastolic: float | None,
        heart_rate: float | None,
        diabetes: bool,
        current_smoker: bool,
    ) -> AcuteCoronaryAssessmentInput:
        return AcuteCoronaryAssessmentInput(
            age_years=data.age_years,
            chest_pain_present=(
                data.chest_pain_present
            ),
            chest_pain_pattern=(
                self._resolve_chest_pain_pattern(data)
            ),
            chest_pain_duration_minutes=(
                data.metadata.get(
                    "chest_pain_duration_minutes"
                )
            ),
            persistent_chest_pain=bool(
                data.metadata.get(
                    "persistent_chest_pain",
                    False,
                )
            ),
            recurrent_chest_pain=bool(
                data.metadata.get(
                    "recurrent_chest_pain",
                    False,
                )
            ),
            retrosternal_pain=bool(
                data.metadata.get(
                    "retrosternal_pain",
                    False,
                )
            ),
            pressure_or_tightness=bool(
                data.metadata.get(
                    "pressure_or_tightness",
                    False,
                )
            ),
            exertional_trigger=bool(
                data.metadata.get(
                    "exertional_trigger",
                    False,
                )
            ),
            relief_with_rest=bool(
                data.metadata.get(
                    "relief_with_rest",
                    False,
                )
            ),
            radiation_to_arm_or_jaw=bool(
                data.metadata.get(
                    "radiation_to_arm_or_jaw",
                    False,
                )
            ),
            diaphoresis=bool(
                data.metadata.get(
                    "diaphoresis",
                    False,
                )
            ),
            nausea_or_vomiting=bool(
                data.metadata.get(
                    "nausea_or_vomiting",
                    False,
                )
            ),
            dyspnea=data.dyspnea_present,
            syncope_or_presyncope=(
                data.syncope_present
            ),
            atypical_presentation_possible=bool(
                data.metadata.get(
                    "atypical_acs_presentation",
                    False,
                )
            ),
            ecg_pattern=self._resolve_ecg_pattern(
                data
            ),
            st_elevation_mm=data.metadata.get(
                "st_elevation_mm"
            ),
            st_depression_mm=data.metadata.get(
                "st_depression_mm"
            ),
            dynamic_ecg_changes=bool(
                data.metadata.get(
                    "dynamic_ecg_changes",
                    False,
                )
            ),
            reciprocal_changes=bool(
                data.metadata.get(
                    "reciprocal_changes",
                    False,
                )
            ),
            ecg_performed_within_10_minutes=(
                data.metadata.get(
                    "ecg_performed_within_10_minutes"
                )
            ),
            troponin_value=data.troponin_value,
            troponin_upper_reference_limit=(
                data.troponin_upper_reference_limit
            ),
            previous_troponin_value=(
                data.metadata.get(
                    "previous_troponin_value"
                )
            ),
            troponin_measurement_interval_hours=(
                data.metadata.get(
                    "troponin_measurement_interval_hours"
                )
            ),
            known_coronary_artery_disease=(
                data.established_ascvd
            ),
            previous_myocardial_infarction=(
                data.prior_myocardial_infarction
            ),
            previous_pci_or_cabg=bool(
                data.metadata.get(
                    "previous_pci_or_cabg",
                    False,
                )
            ),
            hypertension=(
                data.hypertension_history
                or data.treated_hypertension
            ),
            diabetes=diabetes,
            dyslipidemia=bool(
                data.lipid_profile.ldl_cholesterol
                is not None
                and data.lipid_profile.ldl_cholesterol
                >= 130
            ),
            current_smoker=current_smoker,
            family_history_premature_cad=(
                data.family_history_premature_cvd
            ),
            obesity=bool(
                data.body_mass_index is not None
                and data.body_mass_index >= 30
            ),
            systolic_blood_pressure_mm_hg=systolic,
            diastolic_blood_pressure_mm_hg=diastolic,
            heart_rate_bpm=heart_rate,
            oxygen_saturation_percent=(
                data.metadata.get(
                    "oxygen_saturation_percent"
                )
            ),
            altered_mental_status=bool(
                data.metadata.get(
                    "altered_mental_status",
                    False,
                )
            ),
            cool_extremities=bool(
                data.metadata.get(
                    "cool_extremities",
                    False,
                )
            ),
            oliguria=bool(
                data.metadata.get(
                    "oliguria",
                    False,
                )
            ),
            pulmonary_edema=bool(
                data.metadata.get(
                    "pulmonary_edema",
                    False,
                )
            ),
            acute_heart_failure=bool(
                data.metadata.get(
                    "acute_heart_failure",
                    False,
                )
            ),
            sustained_ventricular_arrhythmia=(
                data.ecg.ventricular_arrhythmia_present
            ),
            cardiac_arrest=bool(
                data.metadata.get(
                    "cardiac_arrest",
                    False,
                )
            ),
            mechanical_complication_suspected=bool(
                data.metadata.get(
                    "mechanical_complication_suspected",
                    False,
                )
            ),
            active_major_bleeding=(
                data.active_bleeding
            ),
            current_anticoagulation=(
                data.current_anticoagulant
                != AnticoagulantType.NONE
            ),
            current_antiplatelet_therapy=bool(
                data.antiplatelet_medications
            ),
            egfr_ml_min_1_73m2=(
                data.egfr_ml_min_1_73m2
            ),
            alternative_cause_of_troponin_elevation=bool(
                data.metadata.get(
                    "alternative_troponin_cause",
                    False,
                )
            ),
            suspected_alternative_diagnoses=list(
                data.metadata.get(
                    "suspected_alternative_diagnoses",
                    [],
                )
            ),
            metadata={
                **dict(data.metadata),
                "source": (
                    "CardiovascularAssessmentInput"
                ),
            },
        )

    # ========================================================
    # Adaptadores de resultados
    # ========================================================

    @staticmethod
    def _adapt_anticoagulation_result(
        result: Any,
        data: CardiovascularAssessmentInput,
    ) -> AnticoagulationAssessmentResult:
        return AnticoagulationAssessmentResult(
            atrial_fibrillation_type=(
                data.atrial_fibrillation_type
            ),
            cha2ds2_vasc_score=(
                result.thromboembolic_score
                .cha2ds2_vasc_score
            ),
            thromboembolic_risk=(
                CardiovascularAssessmentEngine
                ._map_stroke_risk(
                    result.thromboembolic_score.category
                )
            ),
            has_bled_score=result.has_bled.score,
            bleeding_risk=(
                CardiovascularAssessmentEngine
                ._map_bleeding_risk(
                    result.has_bled.category
                )
            ),
            current_anticoagulant=(
                data.current_anticoagulant
            ),
            anticoagulation_review_required=(
                result.anticoagulation_review_required
            ),
            dose_review_required=bool(
                data.renal_disease
                or data.chronic_kidney_disease
                or data.liver_disease
            ),
            interaction_review_required=bool(
                result.recommendations
            ),
            contraindication_suspected=bool(
                result.absolute_alerts
            ),
            active_bleeding_alert=(
                data.active_bleeding
            ),
            valid=result.valid,
            warnings=list(result.warnings),
            metadata={
                **dict(result.metadata),
                "cha2ds2_va_score": (
                    result.thromboembolic_score
                    .cha2ds2_va_score
                ),
                "recommendation_status": (
                    result.recommendation_status.value
                ),
                "modifiable_risk_factors": list(
                    result.has_bled
                    .modifiable_risk_factors
                ),
                "absolute_alerts": list(
                    result.absolute_alerts
                ),
                "recommendations": list(
                    result.recommendations
                ),
            },
        )

    @staticmethod
    def _adapt_heart_failure_result(
        result: Any,
    ) -> HeartFailureAssessmentResult:
        return HeartFailureAssessmentResult(
            phenotype=(
                CardiovascularAssessmentEngine
                ._map_heart_failure_phenotype(
                    result.phenotype
                )
            ),
            nyha_class=(
                CardiovascularAssessmentEngine
                ._map_public_nyha(
                    result.nyha_class
                )
            ),
            congestion_status=(
                CardiovascularAssessmentEngine
                ._map_congestion(
                    result.congestion
                )
            ),
            perfusion_status=(
                CardiovascularAssessmentEngine
                ._map_perfusion(
                    result.perfusion
                )
            ),
            ejection_fraction_percent=(
                result.metadata.get(
                    "current_lvef_percent"
                )
            ),
            acute_decompensation_suspected=(
                result.possible_acute_decompensation
            ),
            cardiogenic_shock_suspected=(
                result.hemodynamic_profile
                == HemodynamicProfile.COLD_WET
                and result.immediate_evaluation_required
            ),
            guideline_directed_therapy_review_required=(
                result.medication_review_required
            ),
            valid=result.valid,
            warnings=list(result.warnings),
            metadata={
                **dict(result.metadata),
                "stage": result.stage.value,
                "hemodynamic_profile": (
                    result.hemodynamic_profile.value
                ),
                "urgency": result.urgency.value,
                "advanced_heart_failure_signal": (
                    result.advanced_heart_failure_signal
                ),
                "alerts": list(result.alerts),
                "recommendations": list(
                    result.recommendations
                ),
                "missing_pillars": list(
                    result.pillar_review
                    .missing_pillars
                ),
                "pillar_barriers": list(
                    result.pillar_review.barriers
                ),
            },
        )

    @staticmethod
    def _adapt_acute_coronary_result(
        result: Any,
    ) -> Any:
        public = (
            CardiovascularAssessmentEngine
            ._empty_acute_coronary_result()
        )

        public.chest_pain_type = (
            CardiovascularAssessmentEngine
            ._map_chest_pain_type(
                result.syndrome_type
            )
        )

        public.heart_score = (
            result.heart_score.score
            if result.heart_score.valid
            else None
        )

        public.heart_risk = (
            CardiovascularAssessmentEngine
            ._map_heart_risk(
                result.heart_score.category
            )
        )

        public.acute_coronary_syndrome_suspected = any(
            (
                result.possible_stemi,
                result.possible_nstemi,
                result.possible_unstable_angina,
            )
        )

        public.stemi_suspected = (
            result.possible_stemi
        )

        public.immediate_evaluation_required = (
            result.immediate_evaluation_required
        )

        public.valid = result.valid
        public.warnings = list(result.warnings)

        public.metadata = {
            **dict(result.metadata),
            "syndrome_type": (
                result.syndrome_type.value
            ),
            "urgency": result.urgency.value,
            "hemodynamic_status": (
                result.hemodynamic_status.value
            ),
            "troponin_status": (
                result.troponin_status.value
            ),
            "myocardial_injury_type": (
                result.myocardial_injury_type.value
            ),
            "possible_nstemi": (
                result.possible_nstemi
            ),
            "possible_unstable_angina": (
                result.possible_unstable_angina
            ),
            "alerts": list(result.alerts),
            "recommendations": list(
                result.recommendations
            ),
            "emergency": (
                result.urgency.value == "emergency"
            ),
        }

        return public

    @staticmethod
    def _empty_acute_coronary_result() -> Any:
        from app.clinical_pharmacy_engine.assessment.cardiovascular.models import (
            AcuteCoronaryAssessmentResult,
        )

        return AcuteCoronaryAssessmentResult()

    # ========================================================
    # Alertas e recomendações
    # ========================================================

    def _build_alerts(
        self,
        *,
        hypertension: HypertensionAssessmentResult,
        dyslipidemia: DyslipidemiaAssessmentResult,
        qt: QTAssessmentResult,
        anticoagulation: AnticoagulationAssessmentResult,
        heart_failure: HeartFailureAssessmentResult,
        acute_coronary: Any,
    ) -> list[AssessmentAlert]:
        alerts: list[AssessmentAlert] = []

        if hypertension.requires_immediate_evaluation:
            alerts.append(
                AssessmentAlert(
                    code="CV_HYPERTENSIVE_EMERGENCY",
                    title=(
                        "Possível emergência hipertensiva"
                    ),
                    description=(
                        "Pressão arterial severamente elevada "
                        "associada a sintomas ou possível dano "
                        "agudo de órgão-alvo."
                    ),
                    risk_level=ClinicalRiskLevel.CRITICAL,
                    requires_immediate_action=True,
                )
            )

        if qt.immediate_review_required:
            alerts.append(
                AssessmentAlert(
                    code="CV_QT_RISK",
                    title=(
                        "Risco relevante relacionado ao QT"
                    ),
                    description=(
                        "QTc ou fatores associados indicam "
                        "necessidade de revisão imediata."
                    ),
                    risk_level=(
                        ClinicalRiskLevel.CRITICAL
                        if qt.classification.value == "extreme"
                        else ClinicalRiskLevel.HIGH
                    ),
                    related_medications=list(
                        qt.qt_prolonging_medications
                    ),
                    requires_immediate_action=True,
                )
            )

        if heart_failure.acute_decompensation_suspected:
            alerts.append(
                AssessmentAlert(
                    code="CV_HF_DECOMPENSATION",
                    title=(
                        "Possível descompensação de "
                        "insuficiência cardíaca"
                    ),
                    description=(
                        "Foram identificados sinais ou sintomas "
                        "compatíveis com possível descompensação."
                    ),
                    risk_level=ClinicalRiskLevel.HIGH,
                )
            )

        if acute_coronary.stemi_suspected:
            alerts.append(
                AssessmentAlert(
                    code="CV_POSSIBLE_STEMI",
                    title="Possível STEMI",
                    description=(
                        "O conjunto clínico e eletrocardiográfico "
                        "é compatível com possível STEMI."
                    ),
                    risk_level=ClinicalRiskLevel.CRITICAL,
                    requires_immediate_action=True,
                )
            )

        elif (
            acute_coronary
            .acute_coronary_syndrome_suspected
        ):
            alerts.append(
                AssessmentAlert(
                    code="CV_POSSIBLE_ACS",
                    title=(
                        "Possível síndrome coronariana aguda"
                    ),
                    description=(
                        "Foram identificados dados compatíveis "
                        "com possível síndrome coronariana."
                    ),
                    risk_level=ClinicalRiskLevel.HIGH,
                )
            )

        if anticoagulation.active_bleeding_alert:
            alerts.append(
                AssessmentAlert(
                    code="CV_ACTIVE_BLEEDING",
                    title=(
                        "Sangramento ativo em contexto "
                        "de anticoagulação"
                    ),
                    description=(
                        "Foi informado sangramento ativo "
                        "durante avaliação antitrombótica."
                    ),
                    risk_level=ClinicalRiskLevel.CRITICAL,
                    requires_immediate_action=True,
                )
            )

        if dyslipidemia.severe_hypertriglyceridemia:
            alerts.append(
                AssessmentAlert(
                    code="CV_SEVERE_HYPERTRIGLYCERIDEMIA",
                    title=(
                        "Hipertrigliceridemia grave"
                    ),
                    description=(
                        "Triglicerídeos em faixa que exige "
                        "avaliação clínica prioritária."
                    ),
                    risk_level=ClinicalRiskLevel.HIGH,
                )
            )

        return alerts

    def _build_recommendations(
        self,
        *,
        hypertension: HypertensionAssessmentResult,
        dyslipidemia: DyslipidemiaAssessmentResult,
        qt: QTAssessmentResult,
        anticoagulation: AnticoagulationAssessmentResult,
        heart_failure: HeartFailureAssessmentResult,
        acute_coronary: Any,
    ) -> list[ClinicalRecommendation]:
        recommendations: list[
            ClinicalRecommendation
        ] = []

        if (
            hypertension.valid
            and hypertension.classification.value
            not in {
                "normal",
                "optimal",
                "undetermined",
            }
        ):
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Revisar controle da pressão arterial"
                    ),
                    description=(
                        "Confirmar medições, adesão, metas, "
                        "tolerabilidade e farmacoterapia."
                    ),
                    category=(
                        RecommendationCategory.MONITORING
                    ),
                    priority=(
                        RecommendationPriority.PRIORITY
                    ),
                    monitoring_parameters=[
                        "pressão arterial",
                        "frequência cardíaca",
                    ],
                )
            )

        if (
            dyslipidemia.valid
            and (
                dyslipidemia
                .familial_hypercholesterolemia_suspected
                or dyslipidemia.ldl_status.value
                in {
                    "above_target",
                    "far_above_target",
                }
            )
        ):
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Revisar tratamento hipolipemiante"
                    ),
                    description=(
                        "Avaliar categoria de risco, metas "
                        "versionadas, adesão, tolerabilidade "
                        "e causas secundárias."
                    ),
                    category=(
                        RecommendationCategory.DOSE_REVIEW
                    ),
                    priority=(
                        RecommendationPriority.PRIORITY
                    ),
                )
            )

        if qt.immediate_review_required:
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Revisar fatores de prolongamento do QT"
                    ),
                    description=(
                        "Revisar medicamentos, eletrólitos, "
                        "frequência cardíaca, ECG e fatores "
                        "clínicos associados."
                    ),
                    category=(
                        RecommendationCategory
                        .LABORATORY_REVIEW
                    ),
                    priority=(
                        RecommendationPriority.IMMEDIATE
                    ),
                    related_medications=list(
                        qt.qt_prolonging_medications
                    ),
                    monitoring_parameters=[
                        "ECG",
                        "potássio",
                        "magnésio",
                        "cálcio",
                    ],
                    requires_prescriber_contact=True,
                    requires_immediate_action=True,
                )
            )

        if (
            anticoagulation
            .anticoagulation_review_required
        ):
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Revisar anticoagulação"
                    ),
                    description=(
                        "Revisar risco tromboembólico, risco "
                        "hemorrágico, função renal, função "
                        "hepática, dose e interações."
                    ),
                    category=(
                        RecommendationCategory.DOSE_REVIEW
                    ),
                    priority=(
                        RecommendationPriority.PRIORITY
                    ),
                    requires_prescriber_contact=True,
                )
            )

        if (
            heart_failure
            .guideline_directed_therapy_review_required
        ):
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Revisar farmacoterapia da "
                        "insuficiência cardíaca"
                    ),
                    description=(
                        "Avaliar pilares terapêuticos, barreiras, "
                        "congestão, função renal, eletrólitos e "
                        "tolerabilidade hemodinâmica."
                    ),
                    category=(
                        RecommendationCategory.DOSE_REVIEW
                    ),
                    priority=(
                        RecommendationPriority.URGENT
                        if heart_failure
                        .acute_decompensation_suspected
                        else RecommendationPriority.PRIORITY
                    ),
                    requires_prescriber_contact=(
                        heart_failure
                        .acute_decompensation_suspected
                    ),
                )
            )

        if (
            acute_coronary
            .acute_coronary_syndrome_suspected
        ):
            recommendations.append(
                ClinicalRecommendation(
                    title=(
                        "Avaliação imediata da possível "
                        "síndrome coronariana aguda"
                    ),
                    description=(
                        "Confirmar ECG, troponina seriada, "
                        "estabilidade hemodinâmica, risco "
                        "hemorrágico e estratégia clínica."
                    ),
                    category=(
                        RecommendationCategory
                        .EMERGENCY_REFERRAL
                    ),
                    priority=(
                        RecommendationPriority.IMMEDIATE
                    ),
                    requires_prescriber_contact=True,
                    requires_immediate_action=True,
                )
            )

        return recommendations

    # ========================================================
    # Consolidação
    # ========================================================

    @staticmethod
    def _resolve_overall_risk(
        *,
        data: CardiovascularAssessmentInput,
        risk_estimates: list[
            CardiovascularRiskEstimate
        ],
    ) -> CardiovascularRiskCategory:
        if (
            data.established_ascvd
            or data.prior_myocardial_infarction
            or data.prior_stroke_or_tia
            or data.peripheral_arterial_disease
        ):
            return CardiovascularRiskCategory.VERY_HIGH

        valid_categories = [
            estimate.risk_category
            for estimate in risk_estimates
            if estimate.valid
        ]

        if not valid_categories:
            return (
                CardiovascularRiskCategory.UNDETERMINED
            )

        return max(
            valid_categories,
            key=CardiovascularAssessmentEngine
            ._risk_category_rank,
        )

    @staticmethod
    def _resolve_clinical_risk_level(
        *,
        overall_risk: CardiovascularRiskCategory,
        urgent: bool,
        emergency: bool,
    ) -> ClinicalRiskLevel:
        if emergency:
            return ClinicalRiskLevel.CRITICAL

        if urgent:
            return ClinicalRiskLevel.HIGH

        mapping = {
            CardiovascularRiskCategory.LOW: (
                ClinicalRiskLevel.LOW
            ),
            CardiovascularRiskCategory.BORDERLINE: (
                ClinicalRiskLevel.MODERATE
            ),
            CardiovascularRiskCategory.MODERATE: (
                ClinicalRiskLevel.MODERATE
            ),
            CardiovascularRiskCategory.INTERMEDIATE: (
                ClinicalRiskLevel.MODERATE
            ),
            CardiovascularRiskCategory.HIGH: (
                ClinicalRiskLevel.HIGH
            ),
            CardiovascularRiskCategory.VERY_HIGH: (
                ClinicalRiskLevel.HIGH
            ),
            CardiovascularRiskCategory.EXTREME: (
                ClinicalRiskLevel.CRITICAL
            ),
            CardiovascularRiskCategory.UNDETERMINED: (
                ClinicalRiskLevel.UNDETERMINED
            ),
        }

        return mapping[overall_risk]

    @staticmethod
    def _resolve_status(
        *,
        executions: list[
            CardiovascularModuleExecution
        ],
        errors: list[str],
    ) -> AssessmentStatus:
        completed = sum(
            item.state
            in {
                CardiovascularModuleState.COMPLETED,
                CardiovascularModuleState.PARTIAL,
            }
            for item in executions
        )

        if completed == 0:
            return AssessmentStatus.INSUFFICIENT_DATA

        if errors:
            return AssessmentStatus.PARTIAL

        if any(
            item.state
            in {
                CardiovascularModuleState.PARTIAL,
                CardiovascularModuleState.BLOCKED,
            }
            for item in executions
        ):
            return AssessmentStatus.PARTIAL

        return AssessmentStatus.COMPLETED

    @staticmethod
    def _build_data_quality(
        *,
        data: CardiovascularAssessmentInput,
        executions: list[
            CardiovascularModuleExecution
        ],
        errors: list[str],
    ) -> AssessmentDataQuality:
        missing: list[str] = []
        warnings: list[str] = []

        if data.age_years is None:
            missing.append("age_years")

        if (
            data.biological_sex
            == CardiovascularSex.UNDETERMINED
        ):
            missing.append("biological_sex")

        if not data.blood_pressure_measurements:
            warnings.append(
                "Pressão arterial não informada."
            )

        if (
            data.lipid_profile.total_cholesterol
            is None
            and data.lipid_profile.ldl_cholesterol
            is None
        ):
            warnings.append(
                "Perfil lipídico incompleto."
            )

        failed_count = sum(
            item.state
            == CardiovascularModuleState.FAILED
            for item in executions
        )

        confidence = 1.0

        confidence -= min(
            len(missing) * 0.15,
            0.45,
        )

        confidence -= min(
            len(warnings) * 0.05,
            0.20,
        )

        confidence -= min(
            failed_count * 0.15,
            0.45,
        )

        return AssessmentDataQuality(
            complete=(
                not missing
                and not errors
                and failed_count == 0
            ),
            missing_fields=missing,
            invalid_fields=[],
            warnings=warnings + list(errors),
            confidence=max(
                0.0,
                confidence,
            ),
        )

    @staticmethod
    def _build_calculated_values(
        *,
        systolic: float | None,
        diastolic: float | None,
        risk_estimates: list[
            CardiovascularRiskEstimate
        ],
        hypertension: HypertensionAssessmentResult,
        qt: QTAssessmentResult,
        anticoagulation: AnticoagulationAssessmentResult,
        heart_failure: HeartFailureAssessmentResult,
        acute_coronary: Any,
    ) -> dict[str, float]:
        values: dict[str, float] = {}

        if systolic is not None:
            values["average_systolic_mm_hg"] = (
                round(systolic, 2)
            )

        if diastolic is not None:
            values["average_diastolic_mm_hg"] = (
                round(diastolic, 2)
            )

        for estimate in risk_estimates:
            if estimate.risk_percent_10_years is not None:
                values[
                    (
                        f"{estimate.equation.value}"
                        "_risk_percent_10_years"
                    )
                ] = estimate.risk_percent_10_years

        if qt.preferred_qtc_ms is not None:
            values["preferred_qtc_ms"] = (
                qt.preferred_qtc_ms
            )

        if (
            anticoagulation.cha2ds2_vasc_score
            is not None
        ):
            values["cha2ds2_vasc_score"] = float(
                anticoagulation.cha2ds2_vasc_score
            )

        if anticoagulation.has_bled_score is not None:
            values["has_bled_score"] = float(
                anticoagulation.has_bled_score
            )

        if (
            heart_failure.ejection_fraction_percent
            is not None
        ):
            values["lvef_percent"] = float(
                heart_failure.ejection_fraction_percent
            )

        if acute_coronary.heart_score is not None:
            values["heart_score"] = float(
                acute_coronary.heart_score
            )

        return values

    @staticmethod
    def _build_summary(
        *,
        overall_risk: CardiovascularRiskCategory,
        hypertension: HypertensionAssessmentResult,
        dyslipidemia: DyslipidemiaAssessmentResult,
        qt: QTAssessmentResult,
        anticoagulation: AnticoagulationAssessmentResult,
        heart_failure: HeartFailureAssessmentResult,
        acute_coronary: Any,
    ) -> str:
        parts = [
            (
                "risco cardiovascular global: "
                f"{overall_risk.value}"
            )
        ]

        if hypertension.valid:
            parts.append(
                "pressão arterial: "
                f"{hypertension.classification.value}"
            )

        if dyslipidemia.valid:
            parts.append(
                "risco lipídico: "
                f"{dyslipidemia.risk_category.value}"
            )

        if qt.valid:
            parts.append(
                f"QT: {qt.classification.value}"
            )

        if anticoagulation.valid:
            parts.append(
                "risco tromboembólico: "
                f"{anticoagulation.thromboembolic_risk.value}"
            )

        if heart_failure.valid:
            parts.append(
                "insuficiência cardíaca: "
                f"{heart_failure.phenotype.value}"
            )

        if (
            acute_coronary
            .acute_coronary_syndrome_suspected
        ):
            parts.append(
                "possível síndrome coronariana aguda"
            )

        return "; ".join(parts) + "."

    # ========================================================
    # Mapeamentos
    # ========================================================

    def _resolve_pce_population(
        self,
        data: CardiovascularAssessmentInput,
    ) -> PCEPopulationGroup:
        value = (
            data.ethnicity_or_population_group
            or ""
        ).strip().casefold()

        if value in {
            "white",
            "branco",
            "branca",
        }:
            return PCEPopulationGroup.WHITE

        if value in {
            "black",
            "negro",
            "negra",
            "preto",
            "preta",
        }:
            return PCEPopulationGroup.BLACK

        if (
            self.config.pce_population_group
            != PCEPopulationGroup.UNDETERMINED
        ):
            return self.config.pce_population_group

        return PCEPopulationGroup.OTHER

    def _resolve_score2_region(
        self,
        data: CardiovascularAssessmentInput,
    ) -> SCORE2Region:
        configured = self.config.score2_region

        if configured != SCORE2Region.UNDETERMINED:
            return configured

        value = str(
            data.metadata.get(
                "score2_region",
                "",
            )
        ).casefold()

        mapping = {
            "low": SCORE2Region.LOW,
            "moderate": SCORE2Region.MODERATE,
            "high": SCORE2Region.HIGH,
            "very_high": SCORE2Region.VERY_HIGH,
        }

        return mapping.get(
            value,
            SCORE2Region.UNDETERMINED,
        )

    @staticmethod
    def _map_atrial_arrhythmia(
        data: CardiovascularAssessmentInput,
    ) -> AtrialArrhythmiaType:
        if data.atrial_fibrillation:
            return (
                AtrialArrhythmiaType
                .ATRIAL_FIBRILLATION
            )

        if (
            data.atrial_fibrillation_type
            == AtrialFibrillationType.NONE
        ):
            return AtrialArrhythmiaType.NONE

        if (
            data.atrial_fibrillation_type
            == AtrialFibrillationType.UNDETERMINED
        ):
            return (
                AtrialArrhythmiaType.UNDETERMINED
            )

        return (
            AtrialArrhythmiaType
            .ATRIAL_FIBRILLATION
        )

    @staticmethod
    def _map_anticoagulant(
        value: AnticoagulantType,
    ) -> AnticoagulantClass:
        mapping = {
            AnticoagulantType.NONE: (
                AnticoagulantClass.NONE
            ),
            AnticoagulantType.WARFARIN: (
                AnticoagulantClass.WARFARIN
            ),
            AnticoagulantType.APIXABAN: (
                AnticoagulantClass.APIXABAN
            ),
            AnticoagulantType.RIVAROXABAN: (
                AnticoagulantClass.RIVAROXABAN
            ),
            AnticoagulantType.DABIGATRAN: (
                AnticoagulantClass.DABIGATRAN
            ),
            AnticoagulantType.EDOXABAN: (
                AnticoagulantClass.EDOXABAN
            ),
            AnticoagulantType.UNFRACTIONATED_HEPARIN: (
                AnticoagulantClass.PARENTERAL
            ),
            AnticoagulantType.LOW_MOLECULAR_WEIGHT_HEPARIN: (
                AnticoagulantClass.PARENTERAL
            ),
            AnticoagulantType.FONDAPARINUX: (
                AnticoagulantClass.PARENTERAL
            ),
            AnticoagulantType.OTHER: (
                AnticoagulantClass.OTHER
            ),
        }

        return mapping.get(
            value,
            AnticoagulantClass.UNDETERMINED,
        )

    @staticmethod
    def _map_nyha_class(
        value: NYHAClass,
    ) -> InternalNYHAClass:
        try:
            return InternalNYHAClass(value.value)
        except ValueError:
            return InternalNYHAClass.UNDETERMINED

    @staticmethod
    def _map_public_nyha(
        value: InternalNYHAClass,
    ) -> NYHAClass:
        try:
            return NYHAClass(value.value)
        except ValueError:
            return NYHAClass.UNDETERMINED

    @staticmethod
    def _map_heart_failure_phenotype(
        value: InternalHeartFailurePhenotype,
    ) -> HeartFailurePhenotype:
        mapping = {
            InternalHeartFailurePhenotype.HFrEF: (
                HeartFailurePhenotype.HFrEF
            ),
            InternalHeartFailurePhenotype.HFmrEF: (
                HeartFailurePhenotype.HFmrEF
            ),
            InternalHeartFailurePhenotype.HFpEF: (
                HeartFailurePhenotype.HFpEF
            ),
            InternalHeartFailurePhenotype.HFimpEF: (
                HeartFailurePhenotype.HFimpEF
            ),
        }

        return mapping.get(
            value,
            HeartFailurePhenotype.UNDETERMINED,
        )

    @staticmethod
    def _map_congestion(
        value: CongestionCategory,
    ) -> CongestionStatus:
        mapping = {
            CongestionCategory.NONE: (
                CongestionStatus.NONE
            ),
            CongestionCategory.MILD: (
                CongestionStatus.POSSIBLE
            ),
            CongestionCategory.MODERATE: (
                CongestionStatus.PRESENT
            ),
            CongestionCategory.SEVERE: (
                CongestionStatus.SEVERE
            ),
        }

        return mapping.get(
            value,
            CongestionStatus.UNDETERMINED,
        )

    @staticmethod
    def _map_perfusion(
        value: PerfusionCategory,
    ) -> PerfusionStatus:
        if value == PerfusionCategory.REDUCED:
            return PerfusionStatus.COLD

        if value in {
            PerfusionCategory.ADEQUATE,
            PerfusionCategory.POSSIBLY_REDUCED,
        }:
            return PerfusionStatus.WARM

        return PerfusionStatus.UNDETERMINED

    @staticmethod
    def _map_stroke_risk(
        value: StrokeRiskCategory,
    ) -> ThromboembolicRiskCategory:
        mapping = {
            StrokeRiskCategory.LOW: (
                ThromboembolicRiskCategory.LOW
            ),
            StrokeRiskCategory.INTERMEDIATE: (
                ThromboembolicRiskCategory.INTERMEDIATE
            ),
            StrokeRiskCategory.HIGH: (
                ThromboembolicRiskCategory.HIGH
            ),
            StrokeRiskCategory.VERY_HIGH: (
                ThromboembolicRiskCategory.HIGH
            ),
        }

        return mapping.get(
            value,
            ThromboembolicRiskCategory.UNDETERMINED,
        )

    @staticmethod
    def _map_bleeding_risk(
        value: InternalBleedingRiskCategory,
    ) -> BleedingRiskCategory:
        mapping = {
            InternalBleedingRiskCategory.LOW: (
                BleedingRiskCategory.LOW
            ),
            InternalBleedingRiskCategory.MODERATE: (
                BleedingRiskCategory.MODERATE
            ),
            InternalBleedingRiskCategory.HIGH: (
                BleedingRiskCategory.HIGH
            ),
        }

        return mapping.get(
            value,
            BleedingRiskCategory.UNDETERMINED,
        )

    @staticmethod
    def _map_heart_risk(
        value: HEARTRiskCategory,
    ) -> ACSRiskCategory:
        mapping = {
            HEARTRiskCategory.LOW: (
                ACSRiskCategory.LOW
            ),
            HEARTRiskCategory.INTERMEDIATE: (
                ACSRiskCategory.INTERMEDIATE
            ),
            HEARTRiskCategory.HIGH: (
                ACSRiskCategory.HIGH
            ),
        }

        return mapping.get(
            value,
            ACSRiskCategory.UNDETERMINED,
        )

    @staticmethod
    def _map_chest_pain_type(
        syndrome: AcuteCoronarySyndromeType,
    ) -> ChestPainType:
        if syndrome in {
            AcuteCoronarySyndromeType
            .STEMI_COMPATIBLE,
            AcuteCoronarySyndromeType
            .NSTEMI_COMPATIBLE,
            AcuteCoronarySyndromeType
            .UNSTABLE_ANGINA_COMPATIBLE,
        }:
            return ChestPainType.POSSIBLE_ACS

        return ChestPainType.UNDETERMINED

    # ========================================================
    # Utilidades
    # ========================================================

    @staticmethod
    def _average_vitals(
        data: CardiovascularAssessmentInput,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
    ]:
        measurements = (
            data.blood_pressure_measurements
        )

        if not measurements:
            return None, None, None

        systolic_values = [
            item.systolic_mm_hg
            for item in measurements
            if item.systolic_mm_hg > 0
        ]

        diastolic_values = [
            item.diastolic_mm_hg
            for item in measurements
            if item.diastolic_mm_hg > 0
        ]

        heart_rate_values = [
            item.heart_rate_bpm
            for item in measurements
            if (
                item.heart_rate_bpm is not None
                and item.heart_rate_bpm > 0
            )
        ]

        systolic = (
            sum(systolic_values)
            / len(systolic_values)
            if systolic_values
            else None
        )

        diastolic = (
            sum(diastolic_values)
            / len(diastolic_values)
            if diastolic_values
            else None
        )

        heart_rate = (
            sum(heart_rate_values)
            / len(heart_rate_values)
            if heart_rate_values
            else None
        )

        return systolic, diastolic, heart_rate

    @staticmethod
    def _measurements_by_context(
        data: CardiovascularAssessmentInput,
        context: BloodPressureContext,
    ) -> list[Any]:
        return [
            measurement
            for measurement
            in data.blood_pressure_measurements
            if measurement.context == context
        ]

    @staticmethod
    def _has_diabetes(
        data: CardiovascularAssessmentInput,
    ) -> bool:
        return data.diabetes_status in {
            DiabetesStatus.TYPE_1,
            DiabetesStatus.TYPE_2,
            DiabetesStatus.OTHER,
        }

    @staticmethod
    def _is_secondary_prevention(
        data: CardiovascularAssessmentInput,
    ) -> bool:
        return any(
            (
                data.prevention_context
                == PreventionContext.SECONDARY,
                data.established_ascvd,
                data.prior_myocardial_infarction,
                data.prior_stroke_or_tia,
                data.peripheral_arterial_disease,
            )
        )

    @staticmethod
    def _should_run_heart_failure(
        data: CardiovascularAssessmentInput,
    ) -> bool:
        return any(
            (
                data.heart_failure,
                data.echocardiogram
                .left_ventricular_ejection_fraction_percent
                is not None,
                data.dyspnea_present,
                data.edema_present,
                data.orthopnea_present,
                data.pulmonary_rales_present,
                data
                .jugular_venous_distension_present,
                data.bnp_pg_ml is not None,
                data.nt_probnp_pg_ml is not None,
            )
        )

    @staticmethod
    def _should_run_acute_coronary(
        data: CardiovascularAssessmentInput,
    ) -> bool:
        return any(
            (
                data.chest_pain_present,
                data.troponin_value is not None,
                bool(
                    data.metadata.get(
                        "dynamic_ecg_changes",
                        False,
                    )
                ),
                bool(
                    data.metadata.get(
                        "st_elevation_mm"
                    )
                ),
                bool(
                    data.metadata.get(
                        "st_depression_mm"
                    )
                ),
            )
        )

    @staticmethod
    def _resolve_chest_pain_pattern(
        data: CardiovascularAssessmentInput,
    ) -> ChestPainPattern:
        value = str(
            data.metadata.get(
                "chest_pain_pattern",
                "",
            )
        ).casefold()

        mapping = {
            "highly_suspicious": (
                ChestPainPattern.HIGHLY_SUSPICIOUS
            ),
            "moderately_suspicious": (
                ChestPainPattern.MODERATELY_SUSPICIOUS
            ),
            "slightly_suspicious": (
                ChestPainPattern.SLIGHTLY_SUSPICIOUS
            ),
            "non_suspicious": (
                ChestPainPattern.NON_SUSPICIOUS
            ),
            "absent": ChestPainPattern.ABSENT,
        }

        if value in mapping:
            return mapping[value]

        if data.chest_pain_present:
            return (
                ChestPainPattern.MODERATELY_SUSPICIOUS
            )

        return ChestPainPattern.ABSENT

    @staticmethod
    def _resolve_ecg_pattern(
        data: CardiovascularAssessmentInput,
    ) -> ECGIschemiaPattern:
        value = str(
            data.metadata.get(
                "ecg_ischemia_pattern",
                "",
            )
        ).casefold()

        mapping = {
            item.value: item
            for item in ECGIschemiaPattern
        }

        if value in mapping:
            return mapping[value]

        if (
            data.metadata.get(
                "st_elevation_mm"
            )
        ):
            return (
                ECGIschemiaPattern
                .PERSISTENT_ST_ELEVATION
            )

        if (
            data.metadata.get(
                "st_depression_mm"
            )
        ):
            return ECGIschemiaPattern.ST_DEPRESSION

        if data.ecg.report_date is not None:
            return ECGIschemiaPattern.NORMAL

        return ECGIschemiaPattern.NOT_AVAILABLE

    @staticmethod
    def _country_code(
        data: CardiovascularAssessmentInput,
    ) -> str | None:
        value = data.metadata.get(
            "country_code"
        )

        if value:
            return str(value).upper()

        return None

    @staticmethod
    def _contains_any(
        medications: list[str],
        terms: set[str],
    ) -> bool:
        normalized_medications = [
            medication.casefold()
            for medication in medications
        ]

        return any(
            term.casefold() in medication
            for medication in normalized_medications
            for term in terms
        )

    @staticmethod
    def _is_anticoagulant_name(
        medication: str,
    ) -> bool:
        terms = {
            "warfarin",
            "varfarina",
            "apixaban",
            "apixabana",
            "rivaroxaban",
            "rivaroxabana",
            "dabigatran",
            "dabigatrana",
            "edoxaban",
            "edoxabana",
            "heparin",
            "heparina",
            "enoxaparin",
            "enoxaparina",
            "fondaparinux",
        }

        normalized = medication.casefold()

        return any(
            term in normalized
            for term in terms
        )

    @staticmethod
    def _risk_category_rank(
        category: CardiovascularRiskCategory,
    ) -> int:
        ranking = {
            CardiovascularRiskCategory.UNDETERMINED: 0,
            CardiovascularRiskCategory.LOW: 1,
            CardiovascularRiskCategory.BORDERLINE: 2,
            CardiovascularRiskCategory.MODERATE: 3,
            CardiovascularRiskCategory.INTERMEDIATE: 4,
            CardiovascularRiskCategory.HIGH: 5,
            CardiovascularRiskCategory.VERY_HIGH: 6,
            CardiovascularRiskCategory.EXTREME: 7,
        }

        return ranking.get(category, 0)