"""
PHK Studio
Clinical Pharmacy Engine

Dyslipidemia Assessment Engine.

Responsabilidades:

- validar o perfil lipídico;
- calcular colesterol não HDL;
- estimar LDL-C quando apropriado;
- identificar hipercolesterolemia grave;
- sinalizar possível hipercolesterolemia familiar;
- classificar hipertrigliceridemia;
- combinar risco cardiovascular e contexto preventivo;
- aplicar metas apenas por protocolo validado e versionado;
- sugerir intensidade terapêutica para revisão profissional;
- gerar avisos e metadados auditáveis.

O módulo não altera prescrições automaticamente.

Metas e recomendações farmacoterapêuticas dependem de:

- população;
- contexto preventivo;
- diretriz adotada;
- comorbidades;
- tolerabilidade;
- função renal e hepática;
- interações medicamentosas;
- decisão clínica compartilhada.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Sequence

from app.clinical_pharmacy_engine.assessment.cardiovascular.models import (
    CardiovascularAssessmentInput,
    CardiovascularRiskCategory,
    CardiovascularRiskEstimate,
    DyslipidemiaAssessmentResult,
    LipidProfile,
    LipidTargetStatus,
    LipidUnit,
    PreventionContext,
    StatinIntensity,
)


# ============================================================
# Enums específicos
# ============================================================


class DyslipidemiaGuideline(str, Enum):
    """Referencial clínico utilizado pelo protocolo."""

    SBC_2025 = "sbc_2025"
    ESC_EAS_2025 = "esc_eas_2025"
    AHA_ACC_2026 = "aha_acc_2026"
    INSTITUTIONAL = "institutional"
    TEST_ONLY = "test_only"
    UNDETERMINED = "undetermined"


class TriglycerideCategory(str, Enum):
    """Classificação operacional dos triglicerídeos."""

    NORMAL = "normal"
    BORDERLINE_HIGH = "borderline_high"
    HIGH = "high"
    SEVERE = "severe"
    VERY_SEVERE = "very_severe"
    UNDETERMINED = "undetermined"


class LDLCalculationMethod(str, Enum):
    """Método utilizado para obtenção do LDL-C."""

    DIRECT = "direct"
    FRIEDEWALD = "friedewald"
    PROVIDED = "provided"
    UNAVAILABLE = "unavailable"


class FamilialHypercholesterolemiaSignal(str, Enum):
    """Força do sinal de possível hipercolesterolemia familiar."""

    NONE = "none"
    POSSIBLE = "possible"
    STRONG = "strong"
    UNDETERMINED = "undetermined"


class LipidAssessmentStatus(str, Enum):
    """Estado operacional da avaliação."""

    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    INVALID = "invalid"


# ============================================================
# Entrada normalizada
# ============================================================


@dataclass(slots=True)
class DyslipidemiaInput:
    """Entrada normalizada do motor de dislipidemia."""

    lipid_profile: LipidProfile = field(
        default_factory=LipidProfile,
    )

    age_years: float | None = None

    prevention_context: PreventionContext = (
        PreventionContext.UNDETERMINED
    )

    established_ascvd: bool = False
    prior_myocardial_infarction: bool = False
    prior_stroke_or_tia: bool = False
    peripheral_arterial_disease: bool = False

    diabetes: bool = False
    chronic_kidney_disease: bool = False

    egfr_ml_min_1_73m2: float | None = None

    family_history_premature_cvd: bool = False
    known_familial_hypercholesterolemia: bool = False

    current_lipid_lowering_therapy: bool = False
    current_statin_intensity: StatinIntensity = (
        StatinIntensity.UNDETERMINED
    )

    statin_intolerance: bool = False

    risk_estimates: list[
        CardiovascularRiskEstimate
    ] = field(
        default_factory=list,
    )

    external_risk_category: (
        CardiovascularRiskCategory | None
    ) = None

    secondary_causes_suspected: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Protocolos de metas
# ============================================================


@dataclass(frozen=True, slots=True)
class LipidTarget:
    """Metas para uma categoria de risco."""

    risk_category: CardiovascularRiskCategory

    ldl_target_mg_dl: float | None = None
    non_hdl_target_mg_dl: float | None = None
    apolipoprotein_b_target_mg_dl: float | None = None

    minimum_ldl_reduction_percent: float | None = None

    suggested_statin_intensity: StatinIntensity = (
        StatinIntensity.UNDETERMINED
    )


@dataclass(slots=True)
class DyslipidemiaTargetProfile:
    """
    Protocolo versionado de metas lipídicas.

    O perfil deve ser validado antes do uso clínico.
    """

    guideline: DyslipidemiaGuideline = (
        DyslipidemiaGuideline.UNDETERMINED
    )

    version: str = ""
    source_name: str = ""
    source_reference: str = ""

    targets: dict[
        CardiovascularRiskCategory,
        LipidTarget,
    ] = field(
        default_factory=dict,
    )

    validated: bool = False

    approved_by: str | None = None
    approved_at: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def is_usable(self) -> bool:
        """Confirma se o protocolo pode orientar metas."""

        return bool(
            self.validated
            and self.version.strip()
            and self.source_name.strip()
            and self.targets
        )

    def target_for(
        self,
        risk_category: CardiovascularRiskCategory,
    ) -> LipidTarget | None:
        """Obtém a meta da categoria solicitada."""

        return self.targets.get(risk_category)


# ============================================================
# Configuração
# ============================================================


@dataclass(slots=True)
class DyslipidemiaConfig:
    """Configurações operacionais do motor."""

    allow_friedewald_calculation: bool = True

    friedewald_max_triglycerides_mg_dl: float = 400.0

    severe_hypercholesterolemia_ldl_mg_dl: float = 190.0

    possible_familial_hypercholesterolemia_ldl_mg_dl: (
        float
    ) = 190.0

    strong_familial_hypercholesterolemia_ldl_mg_dl: (
        float
    ) = 250.0

    severe_hypertriglyceridemia_mg_dl: float = 500.0

    very_severe_hypertriglyceridemia_mg_dl: float = 1000.0

    low_hdl_male_mg_dl: float = 40.0
    low_hdl_female_mg_dl: float = 50.0

    non_hdl_difference_from_ldl: float = 30.0

    block_targets_without_validated_profile: bool = True


@dataclass(slots=True)
class DyslipidemiaValidation:
    """Resultado da validação dos dados."""

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
# Resultado interno ampliado
# ============================================================


@dataclass(slots=True)
class DyslipidemiaEngineResult:
    """Resultado ampliado usado internamente pelo PHK Studio."""

    assessment: DyslipidemiaAssessmentResult

    status: LipidAssessmentStatus = (
        LipidAssessmentStatus.PARTIAL
    )

    calculated_ldl_mg_dl: float | None = None
    calculated_non_hdl_mg_dl: float | None = None

    ldl_calculation_method: LDLCalculationMethod = (
        LDLCalculationMethod.UNAVAILABLE
    )

    triglyceride_category: TriglycerideCategory = (
        TriglycerideCategory.UNDETERMINED
    )

    familial_hypercholesterolemia_signal: (
        FamilialHypercholesterolemiaSignal
    ) = FamilialHypercholesterolemiaSignal.UNDETERMINED

    severe_hypercholesterolemia: bool = False
    low_hdl_present: bool = False

    target_profile_applied: bool = False
    target_profile_version: str | None = None

    medication_review_required: bool = False

    warnings: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


# ============================================================
# Motor principal
# ============================================================


class DyslipidemiaAssessmentEngine:
    """Motor estruturado de avaliação de dislipidemia."""

    def __init__(
        self,
        target_profile: (
            DyslipidemiaTargetProfile | None
        ) = None,
        config: DyslipidemiaConfig | None = None,
    ) -> None:
        self.target_profile = (
            target_profile
            or DyslipidemiaTargetProfile()
        )

        self.config = (
            config
            or DyslipidemiaConfig()
        )

    def assess(
        self,
        data: DyslipidemiaInput,
    ) -> DyslipidemiaEngineResult:
        """Executa avaliação lipídica integrada."""

        validation = self.validate(data)

        if not validation.valid:
            assessment = DyslipidemiaAssessmentResult(
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
                        else ""
                    ]
                    + [
                        (
                            "Campos ausentes: "
                            + ", ".join(
                                validation.missing_fields
                            )
                        )
                        if validation.missing_fields
                        else ""
                    ]
                ),
                metadata={
                    "invalid_fields": list(
                        validation.invalid_fields
                    ),
                    "missing_fields": list(
                        validation.missing_fields
                    ),
                },
            )

            return DyslipidemiaEngineResult(
                assessment=assessment,
                status=LipidAssessmentStatus.INVALID,
                warnings=list(assessment.warnings),
                metadata={
                    "validation_failed": True,
                },
            )

        warnings = list(validation.warnings)

        ldl, ldl_method = self.resolve_ldl(
            data.lipid_profile
        )

        non_hdl = self.resolve_non_hdl(
            data.lipid_profile
        )

        triglyceride_category = (
            self.classify_triglycerides(
                data.lipid_profile.triglycerides
            )
        )

        risk_category = self.resolve_risk_category(
            data
        )

        severe_hypercholesterolemia = bool(
            ldl is not None
            and ldl
            >= (
                self.config
                .severe_hypercholesterolemia_ldl_mg_dl
            )
        )

        familial_signal = (
            self.evaluate_familial_hypercholesterolemia(
                ldl_mg_dl=ldl,
                known_familial_hypercholesterolemia=(
                    data
                    .known_familial_hypercholesterolemia
                ),
                family_history_premature_cvd=(
                    data.family_history_premature_cvd
                ),
            )
        )

        severe_triglyceridemia = (
            triglyceride_category
            in {
                TriglycerideCategory.SEVERE,
                TriglycerideCategory.VERY_SEVERE,
            }
        )

        low_hdl = self._low_hdl_present(
            profile=data.lipid_profile,
            metadata=data.metadata,
        )

        target = None
        target_profile_applied = False

        if self.target_profile.is_usable():
            target = self.target_profile.target_for(
                risk_category
            )

            if target is None:
                warnings.append(
                    "O protocolo carregado não possui meta "
                    "para a categoria de risco identificada."
                )
            else:
                target_profile_applied = True

        elif (
            self.config
            .block_targets_without_validated_profile
        ):
            warnings.append(
                "Nenhum protocolo de metas lipídicas "
                "validado e versionado foi carregado. "
                "O sistema realizou a avaliação laboratorial, "
                "mas não atribuiu metas terapêuticas."
            )

        ldl_status = self._target_status(
            current_value=ldl,
            target_value=(
                target.ldl_target_mg_dl
                if target
                else None
            ),
        )

        suggested_intensity = (
            target.suggested_statin_intensity
            if target
            else StatinIntensity.UNDETERMINED
        )

        medication_review_required = any(
            (
                severe_hypercholesterolemia,
                severe_triglyceridemia,
                ldl_status
                in {
                    LipidTargetStatus.ABOVE_TARGET,
                    LipidTargetStatus.FAR_ABOVE_TARGET,
                },
                data.established_ascvd
                and not data.current_lipid_lowering_therapy,
                data.known_familial_hypercholesterolemia,
                data.statin_intolerance,
            )
        )

        if severe_hypercholesterolemia:
            warnings.append(
                "LDL-C em faixa de hipercolesterolemia "
                "grave. Avaliar causas secundárias, "
                "hipercolesterolemia familiar e necessidade "
                "de terapia intensiva."
            )

        if (
            familial_signal
            == FamilialHypercholesterolemiaSignal.STRONG
        ):
            warnings.append(
                "Há sinal forte de possível "
                "hipercolesterolemia familiar. "
                "Recomenda-se avaliação diagnóstica "
                "estruturada e rastreamento familiar."
            )

        elif (
            familial_signal
            == FamilialHypercholesterolemiaSignal.POSSIBLE
        ):
            warnings.append(
                "Há sinal de possível hipercolesterolemia "
                "familiar. Confirmar história familiar, "
                "exame físico e valores prévios sem terapia."
            )

        if severe_triglyceridemia:
            warnings.append(
                "Triglicerídeos em faixa grave. Avaliar "
                "risco de pancreatite, causas secundárias "
                "e necessidade de abordagem prioritária."
            )

        if (
            triglyceride_category
            == TriglycerideCategory.VERY_SEVERE
        ):
            warnings.append(
                "Triglicerídeos em faixa muito grave. "
                "Recomenda-se avaliação clínica urgente."
            )

        if data.secondary_causes_suspected:
            warnings.append(
                "Existem sinais de possível causa secundária "
                "de dislipidemia."
            )

        if data.statin_intolerance:
            warnings.append(
                "Intolerância à estatina informada. "
                "Confirmar causalidade, dose, molécula, "
                "reexposição e alternativas terapêuticas."
            )

        assessment = DyslipidemiaAssessmentResult(
            risk_category=risk_category,
            ldl_target=(
                target.ldl_target_mg_dl
                if target
                else None
            ),
            non_hdl_target=(
                target.non_hdl_target_mg_dl
                if target
                else None
            ),
            apolipoprotein_b_target=(
                target.apolipoprotein_b_target_mg_dl
                if target
                else None
            ),
            current_ldl=ldl,
            current_non_hdl=non_hdl,
            current_apolipoprotein_b=(
                data.lipid_profile.apolipoprotein_b
            ),
            ldl_status=ldl_status,
            suggested_statin_intensity=(
                suggested_intensity
            ),
            familial_hypercholesterolemia_suspected=(
                familial_signal
                in {
                    FamilialHypercholesterolemiaSignal
                    .POSSIBLE,
                    FamilialHypercholesterolemiaSignal
                    .STRONG,
                }
            ),
            severe_hypertriglyceridemia=(
                severe_triglyceridemia
            ),
            valid=True,
            warnings=self._unique_strings(warnings),
            metadata={
                "status": (
                    LipidAssessmentStatus.COMPLETED.value
                    if target_profile_applied
                    else LipidAssessmentStatus.PARTIAL.value
                ),
                "ldl_calculation_method": (
                    ldl_method.value
                ),
                "triglyceride_category": (
                    triglyceride_category.value
                ),
                "familial_hypercholesterolemia_signal": (
                    familial_signal.value
                ),
                "severe_hypercholesterolemia": (
                    severe_hypercholesterolemia
                ),
                "low_hdl_present": low_hdl,
                "target_profile_applied": (
                    target_profile_applied
                ),
                "target_profile_guideline": (
                    self.target_profile.guideline.value
                ),
                "target_profile_version": (
                    self.target_profile.version
                ),
                "medication_review_required": (
                    medication_review_required
                ),
                "minimum_ldl_reduction_percent": (
                    target.minimum_ldl_reduction_percent
                    if target
                    else None
                ),
            },
        )

        return DyslipidemiaEngineResult(
            assessment=assessment,
            status=(
                LipidAssessmentStatus.COMPLETED
                if target_profile_applied
                else LipidAssessmentStatus.PARTIAL
            ),
            calculated_ldl_mg_dl=ldl,
            calculated_non_hdl_mg_dl=non_hdl,
            ldl_calculation_method=ldl_method,
            triglyceride_category=(
                triglyceride_category
            ),
            familial_hypercholesterolemia_signal=(
                familial_signal
            ),
            severe_hypercholesterolemia=(
                severe_hypercholesterolemia
            ),
            low_hdl_present=low_hdl,
            target_profile_applied=(
                target_profile_applied
            ),
            target_profile_version=(
                self.target_profile.version or None
            ),
            medication_review_required=(
                medication_review_required
            ),
            warnings=list(assessment.warnings),
            metadata=dict(assessment.metadata),
        )

    def assess_integrated_input(
        self,
        data: CardiovascularAssessmentInput,
        *,
        risk_estimates: (
            Sequence[CardiovascularRiskEstimate] | None
        ) = None,
        statin_intolerance: bool = False,
        current_lipid_lowering_therapy: bool = False,
        current_statin_intensity: StatinIntensity = (
            StatinIntensity.UNDETERMINED
        ),
    ) -> DyslipidemiaEngineResult:
        """Converte a entrada cardiovascular integrada."""

        diabetes = (
            data.diabetes_status.value
            in {
                "type_1",
                "type_2",
                "other",
            }
        )

        established_ascvd = any(
            (
                data.established_ascvd,
                data.prior_myocardial_infarction,
                data.prior_stroke_or_tia,
                data.peripheral_arterial_disease,
            )
        )

        return self.assess(
            DyslipidemiaInput(
                lipid_profile=data.lipid_profile,
                age_years=data.age_years,
                prevention_context=(
                    data.prevention_context
                ),
                established_ascvd=established_ascvd,
                prior_myocardial_infarction=(
                    data.prior_myocardial_infarction
                ),
                prior_stroke_or_tia=(
                    data.prior_stroke_or_tia
                ),
                peripheral_arterial_disease=(
                    data.peripheral_arterial_disease
                ),
                diabetes=diabetes,
                chronic_kidney_disease=(
                    data.chronic_kidney_disease
                ),
                egfr_ml_min_1_73m2=(
                    data.egfr_ml_min_1_73m2
                ),
                family_history_premature_cvd=(
                    data.family_history_premature_cvd
                ),
                current_lipid_lowering_therapy=(
                    current_lipid_lowering_therapy
                ),
                current_statin_intensity=(
                    current_statin_intensity
                ),
                statin_intolerance=statin_intolerance,
                risk_estimates=list(
                    risk_estimates or []
                ),
                metadata={
                    **dict(data.metadata),
                    "biological_sex": (
                        data.biological_sex.value
                    ),
                    "source": (
                        "CardiovascularAssessmentInput"
                    ),
                },
            )
        )

    # ========================================================
    # Validação
    # ========================================================

    def validate(
        self,
        data: DyslipidemiaInput,
    ) -> DyslipidemiaValidation:
        """Valida o perfil lipídico informado."""

        missing: list[str] = []
        invalid: list[str] = []
        warnings: list[str] = []

        profile = data.lipid_profile

        numeric_fields = {
            "total_cholesterol": (
                profile.total_cholesterol
            ),
            "ldl_cholesterol": (
                profile.ldl_cholesterol
            ),
            "hdl_cholesterol": (
                profile.hdl_cholesterol
            ),
            "triglycerides": profile.triglycerides,
            "non_hdl_cholesterol": (
                profile.non_hdl_cholesterol
            ),
            "apolipoprotein_b": (
                profile.apolipoprotein_b
            ),
            "lipoprotein_a": profile.lipoprotein_a,
        }

        available = False

        for field_name, value in numeric_fields.items():
            if value is None:
                continue

            available = True

            if (
                not self._valid_number(value)
                or float(value) < 0
            ):
                invalid.append(field_name)

        if not available:
            missing.append("lipid_profile")

        if profile.unit != LipidUnit.MG_DL:
            invalid.append("lipid_profile.unit")

            warnings.append(
                "O motor atual espera perfil lipídico em "
                "mg/dL. Converter explicitamente antes da "
                "avaliação."
            )

        if (
            profile.total_cholesterol is not None
            and profile.hdl_cholesterol is not None
            and self._valid_number(
                profile.total_cholesterol
            )
            and self._valid_number(
                profile.hdl_cholesterol
            )
            and float(profile.hdl_cholesterol)
            > float(profile.total_cholesterol)
        ):
            invalid.append(
                "hdl_cholesterol"
            )

            warnings.append(
                "HDL-C não pode ser maior que o "
                "colesterol total."
            )

        if (
            profile.ldl_cholesterol is None
            and not (
                self.config
                .allow_friedewald_calculation
                and profile.total_cholesterol is not None
                and profile.hdl_cholesterol is not None
                and profile.triglycerides is not None
            )
        ):
            warnings.append(
                "LDL-C direto não informado e dados "
                "insuficientes para estimativa."
            )

        if (
            data.prevention_context
            == PreventionContext.UNDETERMINED
        ):
            warnings.append(
                "Contexto preventivo não determinado. "
                "A estratificação de risco poderá ficar "
                "incompleta."
            )

        return DyslipidemiaValidation(
            valid=not missing and not invalid,
            missing_fields=self._unique_strings(
                missing
            ),
            invalid_fields=self._unique_strings(
                invalid
            ),
            warnings=self._unique_strings(
                warnings
            ),
        )

    # ========================================================
    # LDL e não HDL
    # ========================================================

    def resolve_ldl(
        self,
        profile: LipidProfile,
    ) -> tuple[
        float | None,
        LDLCalculationMethod,
    ]:
        """Obtém LDL direto ou calcula por Friedewald."""

        if self._valid_nonnegative(
            profile.ldl_cholesterol
        ):
            return (
                round(
                    float(profile.ldl_cholesterol),
                    2,
                ),
                LDLCalculationMethod.PROVIDED,
            )

        if not self.config.allow_friedewald_calculation:
            return None, LDLCalculationMethod.UNAVAILABLE

        required_values = (
            profile.total_cholesterol,
            profile.hdl_cholesterol,
            profile.triglycerides,
        )

        if not all(
            self._valid_nonnegative(value)
            for value in required_values
        ):
            return None, LDLCalculationMethod.UNAVAILABLE

        assert profile.total_cholesterol is not None
        assert profile.hdl_cholesterol is not None
        assert profile.triglycerides is not None

        triglycerides = float(
            profile.triglycerides
        )

        if (
            triglycerides
            >= self.config
            .friedewald_max_triglycerides_mg_dl
        ):
            return None, LDLCalculationMethod.UNAVAILABLE

        value = (
            float(profile.total_cholesterol)
            - float(profile.hdl_cholesterol)
            - triglycerides / 5.0
        )

        if value < 0:
            return None, LDLCalculationMethod.UNAVAILABLE

        return (
            round(value, 2),
            LDLCalculationMethod.FRIEDEWALD,
        )

    def resolve_non_hdl(
        self,
        profile: LipidProfile,
    ) -> float | None:
        """Obtém ou calcula colesterol não HDL."""

        if self._valid_nonnegative(
            profile.non_hdl_cholesterol
        ):
            return round(
                float(profile.non_hdl_cholesterol),
                2,
            )

        if not self._valid_nonnegative(
            profile.total_cholesterol
        ):
            return None

        if not self._valid_nonnegative(
            profile.hdl_cholesterol
        ):
            return None

        assert profile.total_cholesterol is not None
        assert profile.hdl_cholesterol is not None

        value = (
            float(profile.total_cholesterol)
            - float(profile.hdl_cholesterol)
        )

        if value < 0:
            return None

        return round(value, 2)

    # ========================================================
    # Estratificação
    # ========================================================

    def resolve_risk_category(
        self,
        data: DyslipidemiaInput,
    ) -> CardiovascularRiskCategory:
        """Resolve a categoria geral de risco."""

        if data.external_risk_category is not None:
            return data.external_risk_category

        if (
            data.established_ascvd
            or data.prior_myocardial_infarction
            or data.prior_stroke_or_tia
            or data.peripheral_arterial_disease
        ):
            return CardiovascularRiskCategory.VERY_HIGH

        if data.known_familial_hypercholesterolemia:
            if (
                data.established_ascvd
                or data.diabetes
                or data.chronic_kidney_disease
            ):
                return (
                    CardiovascularRiskCategory.VERY_HIGH
                )

            return CardiovascularRiskCategory.HIGH

        if (
            data.chronic_kidney_disease
            and data.egfr_ml_min_1_73m2 is not None
        ):
            if data.egfr_ml_min_1_73m2 < 30:
                return (
                    CardiovascularRiskCategory.VERY_HIGH
                )

            if data.egfr_ml_min_1_73m2 < 60:
                return CardiovascularRiskCategory.HIGH

        valid_estimates = [
            estimate
            for estimate in data.risk_estimates
            if estimate.valid
        ]

        if valid_estimates:
            return max(
                (
                    estimate.risk_category
                    for estimate in valid_estimates
                ),
                key=self._risk_rank,
            )

        if data.diabetes:
            return CardiovascularRiskCategory.HIGH

        if (
            data.prevention_context
            == PreventionContext.SECONDARY
        ):
            return CardiovascularRiskCategory.VERY_HIGH

        return CardiovascularRiskCategory.UNDETERMINED

    @staticmethod
    def classify_triglycerides(
        triglycerides_mg_dl: float | None,
    ) -> TriglycerideCategory:
        """Classifica triglicerídeos operacionalmente."""

        if (
            triglycerides_mg_dl is None
            or not DyslipidemiaAssessmentEngine
            ._valid_number(triglycerides_mg_dl)
            or float(triglycerides_mg_dl) < 0
        ):
            return TriglycerideCategory.UNDETERMINED

        value = float(triglycerides_mg_dl)

        if value < 150:
            return TriglycerideCategory.NORMAL

        if value < 200:
            return (
                TriglycerideCategory.BORDERLINE_HIGH
            )

        if value < 500:
            return TriglycerideCategory.HIGH

        if value < 1000:
            return TriglycerideCategory.SEVERE

        return TriglycerideCategory.VERY_SEVERE

    def evaluate_familial_hypercholesterolemia(
        self,
        *,
        ldl_mg_dl: float | None,
        known_familial_hypercholesterolemia: bool,
        family_history_premature_cvd: bool,
    ) -> FamilialHypercholesterolemiaSignal:
        """Gera sinal preliminar de possível HF."""

        if known_familial_hypercholesterolemia:
            return FamilialHypercholesterolemiaSignal.STRONG

        if ldl_mg_dl is None:
            return (
                FamilialHypercholesterolemiaSignal
                .UNDETERMINED
            )

        if (
            ldl_mg_dl
            >= self.config
            .strong_familial_hypercholesterolemia_ldl_mg_dl
        ):
            return FamilialHypercholesterolemiaSignal.STRONG

        if (
            ldl_mg_dl
            >= self.config
            .possible_familial_hypercholesterolemia_ldl_mg_dl
        ):
            if family_history_premature_cvd:
                return (
                    FamilialHypercholesterolemiaSignal
                    .STRONG
                )

            return (
                FamilialHypercholesterolemiaSignal
                .POSSIBLE
            )

        return FamilialHypercholesterolemiaSignal.NONE

    # ========================================================
    # Metas
    # ========================================================

    @staticmethod
    def _target_status(
        *,
        current_value: float | None,
        target_value: float | None,
    ) -> LipidTargetStatus:
        """Compara valor atual e meta."""

        if (
            current_value is None
            or target_value is None
            or target_value <= 0
        ):
            return LipidTargetStatus.UNDETERMINED

        if current_value <= target_value:
            return LipidTargetStatus.AT_TARGET

        if current_value >= target_value * 1.5:
            return LipidTargetStatus.FAR_ABOVE_TARGET

        return LipidTargetStatus.ABOVE_TARGET

    # ========================================================
    # Utilidades
    # ========================================================

    def _low_hdl_present(
        self,
        *,
        profile: LipidProfile,
        metadata: dict[str, Any],
    ) -> bool:
        """Sinaliza HDL baixo conforme sexo informado."""

        if not self._valid_nonnegative(
            profile.hdl_cholesterol
        ):
            return False

        hdl = float(profile.hdl_cholesterol)

        biological_sex = str(
            metadata.get(
                "biological_sex",
                "",
            )
        ).casefold()

        if biological_sex == "male":
            return hdl < self.config.low_hdl_male_mg_dl

        if biological_sex == "female":
            return hdl < self.config.low_hdl_female_mg_dl

        return hdl < self.config.low_hdl_male_mg_dl

    @staticmethod
    def _risk_rank(
        category: CardiovascularRiskCategory,
    ) -> int:
        """Converte categoria em prioridade ordinal."""

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

    @staticmethod
    def _valid_number(
        value: object,
    ) -> bool:
        """Verifica valor numérico finito."""

        try:
            return isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _valid_nonnegative(
        value: object,
    ) -> bool:
        """Verifica valor numérico não negativo."""

        if not DyslipidemiaAssessmentEngine._valid_number(
            value
        ):
            return False

        return float(value) >= 0

    @staticmethod
    def _unique_strings(
        values: Iterable[str],
    ) -> list[str]:
        """Remove textos vazios e repetidos."""

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