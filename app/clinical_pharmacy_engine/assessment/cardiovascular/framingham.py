"""
PHK Studio
Clinical Pharmacy Engine

Framingham General Cardiovascular Risk Engine.

Implementa o modelo lipídico de D'Agostino et al. (2008)
para estimativa de risco cardiovascular global em dez anos.

Desfechos incluídos no modelo original:

- morte coronariana;
- infarto do miocárdio;
- insuficiência coronariana;
- angina;
- AVC isquêmico;
- AVC hemorrágico;
- ataque isquêmico transitório;
- doença arterial periférica;
- insuficiência cardíaca.

População original:

- idade entre 30 e 74 anos;
- ausência de doença cardiovascular no início;
- avaliação em prevenção primária.

O resultado fornece suporte à decisão e não substitui
avaliação clínica, validação farmacêutica ou diretrizes
contemporâneas específicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Any

from app.clinical_pharmacy_engine.assessment.cardiovascular.models import (
    CardiovascularAssessmentInput,
    CardiovascularRiskCategory,
    CardiovascularRiskEstimate,
    CardiovascularSex,
    DiabetesStatus,
    PreventionContext,
    RiskEquationType,
    SmokingStatus,
)


# ============================================================
# Modelos internos
# ============================================================


@dataclass(frozen=True, slots=True)
class FraminghamCoefficients:
    """Coeficientes do modelo lipídico de Framingham."""

    baseline_survival: float
    mean_coefficient_sum: float

    ln_age: float
    ln_total_cholesterol: float
    ln_hdl_cholesterol: float

    ln_systolic_bp_treated: float
    ln_systolic_bp_untreated: float

    current_smoker: float
    diabetes: float


@dataclass(slots=True)
class FraminghamInput:
    """Entrada normalizada do modelo de Framingham."""

    age_years: float | None = None

    biological_sex: CardiovascularSex = (
        CardiovascularSex.UNDETERMINED
    )

    total_cholesterol_mg_dl: float | None = None
    hdl_cholesterol_mg_dl: float | None = None

    systolic_blood_pressure_mm_hg: float | None = None

    treated_hypertension: bool = False
    current_smoker: bool = False
    diabetes: bool = False

    prevention_context: PreventionContext = (
        PreventionContext.PRIMARY
    )

    established_cardiovascular_disease: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class FraminghamConfig:
    """Configuração e limites operacionais."""

    minimum_age_years: float = 30.0
    maximum_age_years: float = 74.0

    minimum_total_cholesterol_mg_dl: float = 100.0
    maximum_total_cholesterol_mg_dl: float = 405.0

    minimum_hdl_mg_dl: float = 10.0
    maximum_hdl_mg_dl: float = 150.0

    minimum_systolic_bp_mm_hg: float = 80.0
    maximum_systolic_bp_mm_hg: float = 250.0

    equation_version: str = "Framingham-General-CVD-2008"


@dataclass(slots=True)
class FraminghamValidation:
    """Resultado da validação da entrada."""

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
# Coeficientes oficiais do modelo lipídico
# ============================================================


FRAMINGHAM_COEFFICIENTS: dict[
    CardiovascularSex,
    FraminghamCoefficients,
] = {
    CardiovascularSex.MALE: FraminghamCoefficients(
        baseline_survival=0.88936,
        mean_coefficient_sum=23.9802,
        ln_age=3.06117,
        ln_total_cholesterol=1.12370,
        ln_hdl_cholesterol=-0.93263,
        ln_systolic_bp_treated=1.99881,
        ln_systolic_bp_untreated=1.93303,
        current_smoker=0.65451,
        diabetes=0.57367,
    ),
    CardiovascularSex.FEMALE: FraminghamCoefficients(
        baseline_survival=0.95012,
        mean_coefficient_sum=26.1931,
        ln_age=2.32888,
        ln_total_cholesterol=1.20904,
        ln_hdl_cholesterol=-0.70833,
        ln_systolic_bp_treated=2.82263,
        ln_systolic_bp_untreated=2.76157,
        current_smoker=0.52873,
        diabetes=0.69154,
    ),
}


# ============================================================
# Motor principal
# ============================================================


class FraminghamRiskEngine:
    """Motor do risco cardiovascular global de Framingham."""

    def __init__(
        self,
        config: FraminghamConfig | None = None,
    ) -> None:
        self.config = config or FraminghamConfig()

    def assess(
        self,
        data: FraminghamInput,
    ) -> CardiovascularRiskEstimate:
        """
        Calcula o risco cardiovascular global em dez anos.
        """

        validation = self.validate(data)

        if not validation.valid:
            return CardiovascularRiskEstimate(
                equation=RiskEquationType.FRAMINGHAM,
                risk_category=(
                    CardiovascularRiskCategory.UNDETERMINED
                ),
                endpoint=(
                    "Doença cardiovascular global em "
                    "dez anos"
                ),
                population=(
                    "Framingham primary prevention"
                ),
                valid=False,
                missing_fields=list(
                    validation.missing_fields
                ),
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
                ),
                calculation_version=(
                    self.config.equation_version
                ),
                metadata={
                    "invalid_fields": list(
                        validation.invalid_fields
                    ),
                    "model_type": (
                        "general_cardiovascular_disease"
                    ),
                },
            )

        coefficients = FRAMINGHAM_COEFFICIENTS[
            data.biological_sex
        ]

        coefficient_sum = self._coefficient_sum(
            data=data,
            coefficients=coefficients,
        )

        exponent = exp(
            coefficient_sum
            - coefficients.mean_coefficient_sum
        )

        risk_probability = (
            1.0
            - coefficients.baseline_survival
            ** exponent
        )

        risk_percent = self._bound(
            risk_probability * 100.0,
            minimum=0.0,
            maximum=100.0,
        )

        warnings = list(validation.warnings)

        warnings.append(
            "O Framingham General CVD estima um conjunto "
            "amplo de eventos cardiovasculares e não deve "
            "ser comparado diretamente ao risco ASCVD PCE."
        )

        warnings.append(
            "Modelo derivado da coorte histórica de "
            "Framingham. Avaliar calibração e aplicabilidade "
            "na população atendida."
        )

        return CardiovascularRiskEstimate(
            equation=RiskEquationType.FRAMINGHAM,
            risk_percent_10_years=round(
                risk_percent,
                2,
            ),
            risk_category=self.classify_risk(
                risk_percent,
            ),
            endpoint=(
                "Doença cardiovascular global em dez anos"
            ),
            population=(
                "Framingham primary prevention"
            ),
            valid=True,
            missing_fields=[],
            warnings=self._unique_strings(warnings),
            calculation_version=(
                self.config.equation_version
            ),
            metadata={
                "coefficient_sum": round(
                    coefficient_sum,
                    8,
                ),
                "baseline_survival": (
                    coefficients.baseline_survival
                ),
                "mean_coefficient_sum": (
                    coefficients.mean_coefficient_sum
                ),
                "treated_hypertension": (
                    data.treated_hypertension
                ),
                "current_smoker": (
                    data.current_smoker
                ),
                "diabetes": data.diabetes,
                "model_type": (
                    "general_cardiovascular_disease"
                ),
                "lipid_based_model": True,
            },
        )

    def assess_integrated_input(
        self,
        data: CardiovascularAssessmentInput,
    ) -> CardiovascularRiskEstimate:
        """
        Converte a entrada cardiovascular integrada para
        o formato utilizado pela equação de Framingham.
        """

        systolic_pressure = (
            self._average_systolic_pressure(data)
        )

        return self.assess(
            FraminghamInput(
                age_years=data.age_years,
                biological_sex=data.biological_sex,
                total_cholesterol_mg_dl=(
                    data.lipid_profile.total_cholesterol
                ),
                hdl_cholesterol_mg_dl=(
                    data.lipid_profile.hdl_cholesterol
                ),
                systolic_blood_pressure_mm_hg=(
                    systolic_pressure
                ),
                treated_hypertension=(
                    data.treated_hypertension
                ),
                current_smoker=(
                    data.smoking_status
                    == SmokingStatus.CURRENT
                ),
                diabetes=(
                    data.diabetes_status
                    in {
                        DiabetesStatus.TYPE_1,
                        DiabetesStatus.TYPE_2,
                        DiabetesStatus.OTHER,
                    }
                ),
                prevention_context=(
                    data.prevention_context
                ),
                established_cardiovascular_disease=(
                    data.established_ascvd
                    or data.prior_myocardial_infarction
                    or data.prior_stroke_or_tia
                    or data.peripheral_arterial_disease
                    or data.heart_failure
                ),
                metadata={
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
        data: FraminghamInput,
    ) -> FraminghamValidation:
        """Valida aplicabilidade e intervalos."""

        missing: list[str] = []
        invalid: list[str] = []
        warnings: list[str] = []

        if data.age_years is None:
            missing.append("age_years")
        elif not self._number_in_range(
            data.age_years,
            self.config.minimum_age_years,
            self.config.maximum_age_years,
        ):
            invalid.append("age_years")

        if (
            data.biological_sex
            not in {
                CardiovascularSex.MALE,
                CardiovascularSex.FEMALE,
            }
        ):
            missing.append("biological_sex")

        self._validate_numeric_field(
            field_name="total_cholesterol_mg_dl",
            value=data.total_cholesterol_mg_dl,
            minimum=(
                self.config
                .minimum_total_cholesterol_mg_dl
            ),
            maximum=(
                self.config
                .maximum_total_cholesterol_mg_dl
            ),
            missing=missing,
            invalid=invalid,
        )

        self._validate_numeric_field(
            field_name="hdl_cholesterol_mg_dl",
            value=data.hdl_cholesterol_mg_dl,
            minimum=self.config.minimum_hdl_mg_dl,
            maximum=self.config.maximum_hdl_mg_dl,
            missing=missing,
            invalid=invalid,
        )

        self._validate_numeric_field(
            field_name=(
                "systolic_blood_pressure_mm_hg"
            ),
            value=(
                data.systolic_blood_pressure_mm_hg
            ),
            minimum=(
                self.config.minimum_systolic_bp_mm_hg
            ),
            maximum=(
                self.config.maximum_systolic_bp_mm_hg
            ),
            missing=missing,
            invalid=invalid,
        )

        if (
            data.prevention_context
            == PreventionContext.SECONDARY
            or data.established_cardiovascular_disease
        ):
            invalid.append("prevention_context")

            warnings.append(
                "O modelo de Framingham General CVD foi "
                "desenvolvido para pessoas sem doença "
                "cardiovascular estabelecida."
            )

        if (
            data.prevention_context
            == PreventionContext.UNDETERMINED
        ):
            warnings.append(
                "Contexto preventivo não determinado. "
                "Confirmar ausência de doença cardiovascular."
            )

        return FraminghamValidation(
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
    # Cálculo
    # ========================================================

    @staticmethod
    def _coefficient_sum(
        *,
        data: FraminghamInput,
        coefficients: FraminghamCoefficients,
    ) -> float:
        """Calcula a soma linear do modelo."""

        assert data.age_years is not None
        assert data.total_cholesterol_mg_dl is not None
        assert data.hdl_cholesterol_mg_dl is not None
        assert (
            data.systolic_blood_pressure_mm_hg
            is not None
        )

        ln_age = log(data.age_years)

        ln_total_cholesterol = log(
            data.total_cholesterol_mg_dl
        )

        ln_hdl = log(
            data.hdl_cholesterol_mg_dl
        )

        ln_systolic = log(
            data.systolic_blood_pressure_mm_hg
        )

        systolic_coefficient = (
            coefficients.ln_systolic_bp_treated
            if data.treated_hypertension
            else coefficients.ln_systolic_bp_untreated
        )

        value = 0.0

        value += coefficients.ln_age * ln_age

        value += (
            coefficients.ln_total_cholesterol
            * ln_total_cholesterol
        )

        value += (
            coefficients.ln_hdl_cholesterol
            * ln_hdl
        )

        value += (
            systolic_coefficient
            * ln_systolic
        )

        value += (
            coefficients.current_smoker
            * (1.0 if data.current_smoker else 0.0)
        )

        value += (
            coefficients.diabetes
            * (1.0 if data.diabetes else 0.0)
        )

        return value

    # ========================================================
    # Classificação
    # ========================================================

    @staticmethod
    def classify_risk(
        risk_percent: float | None,
    ) -> CardiovascularRiskCategory:
        """
        Classificação operacional interna.

        Menor que 10%:
            baixo

        10% a menor que 20%:
            moderado

        20% a menor que 30%:
            alto

        30% ou mais:
            muito alto

        Essas faixas servem para padronização interna e não
        substituem classificações de diretrizes locais.
        """

        if risk_percent is None:
            return (
                CardiovascularRiskCategory.UNDETERMINED
            )

        if risk_percent < 10.0:
            return CardiovascularRiskCategory.LOW

        if risk_percent < 20.0:
            return (
                CardiovascularRiskCategory.MODERATE
            )

        if risk_percent < 30.0:
            return CardiovascularRiskCategory.HIGH

        return CardiovascularRiskCategory.VERY_HIGH

    # ========================================================
    # Utilidades
    # ========================================================

    @staticmethod
    def _average_systolic_pressure(
        data: CardiovascularAssessmentInput,
    ) -> float | None:
        """Calcula a PAS média das medições válidas."""

        values = [
            measurement.systolic_mm_hg
            for measurement
            in data.blood_pressure_measurements
            if (
                FraminghamRiskEngine._valid_number(
                    measurement.systolic_mm_hg
                )
                and 0
                < measurement.systolic_mm_hg
                <= 300
            )
        ]

        if not values:
            return None

        return sum(values) / len(values)

    @staticmethod
    def _validate_numeric_field(
        *,
        field_name: str,
        value: float | None,
        minimum: float,
        maximum: float,
        missing: list[str],
        invalid: list[str],
    ) -> None:
        """Valida campo numérico obrigatório."""

        if value is None:
            missing.append(field_name)
            return

        if not FraminghamRiskEngine._number_in_range(
            value,
            minimum,
            maximum,
        ):
            invalid.append(field_name)

    @staticmethod
    def _number_in_range(
        value: object,
        minimum: float,
        maximum: float,
    ) -> bool:
        """Verifica número finito dentro do intervalo."""

        if not FraminghamRiskEngine._valid_number(
            value
        ):
            return False

        numeric = float(value)

        return minimum <= numeric <= maximum

    @staticmethod
    def _valid_number(
        value: object,
    ) -> bool:
        """Verifica se o valor é numérico e finito."""

        try:
            return isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _bound(
        value: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        """Restringe o valor ao intervalo informado."""

        return max(
            minimum,
            min(value, maximum),
        )

    @staticmethod
    def _unique_strings(
        values: list[str],
    ) -> list[str]:
        """Remove duplicações preservando a ordem."""

        result: list[str] = []
        seen: set[str] = set()

        for value in values:
            normalized = " ".join(
                value.strip().split()
            )

            if not normalized:
                continue

            key = normalized.casefold()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result