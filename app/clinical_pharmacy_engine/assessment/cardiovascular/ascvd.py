"""
PHK Studio
Clinical Pharmacy Engine

Legacy ASCVD Pooled Cohort Equations Engine.

Implementa as Pooled Cohort Equations de 2013 para
estimativa do risco de primeiro evento ASCVD em 10 anos.

Entradas principais:

- idade;
- sexo biológico;
- grupo populacional calibrado;
- colesterol total;
- HDL;
- pressão arterial sistólica;
- tratamento anti-hipertensivo;
- tabagismo atual;
- diabetes.

Importante:

- esta é uma implementação legada e versionada;
- não representa automaticamente a calculadora preferencial
  de diretrizes mais recentes;
- não deve ser utilizada em prevenção secundária;
- não substitui avaliação clínica;
- resultados em populações diferentes das coortes originais
  devem ser interpretados com cautela.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
# Enums e configurações
# ============================================================


class PCEPopulationGroup(str, Enum):
    """
    Grupos de calibração originais do PCE.

    Os nomes representam as coortes utilizadas na equação,
    não categorias biológicas universais.
    """

    WHITE = "white"
    BLACK = "black"
    OTHER = "other"
    UNDETERMINED = "undetermined"


@dataclass(frozen=True, slots=True)
class PCECoefficients:
    """Coeficientes de uma equação PCE específica."""

    baseline_survival: float
    mean_coefficient_sum: float

    ln_age: float = 0.0
    ln_age_squared: float = 0.0

    ln_total_cholesterol: float = 0.0
    ln_age_x_ln_total_cholesterol: float = 0.0

    ln_hdl: float = 0.0
    ln_age_x_ln_hdl: float = 0.0

    ln_treated_sbp: float = 0.0
    ln_age_x_ln_treated_sbp: float = 0.0

    ln_untreated_sbp: float = 0.0
    ln_age_x_ln_untreated_sbp: float = 0.0

    current_smoker: float = 0.0
    ln_age_x_current_smoker: float = 0.0

    diabetes: float = 0.0


@dataclass(slots=True)
class ASCVDPCEInput:
    """Entrada normalizada para as Pooled Cohort Equations."""

    age_years: float | None = None

    biological_sex: CardiovascularSex = (
        CardiovascularSex.UNDETERMINED
    )

    population_group: PCEPopulationGroup = (
        PCEPopulationGroup.UNDETERMINED
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

    established_ascvd: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class ASCVDPCEConfig:
    """Configurações do motor PCE."""

    minimum_age_years: float = 40.0
    maximum_age_years: float = 79.0

    minimum_total_cholesterol_mg_dl: float = 130.0
    maximum_total_cholesterol_mg_dl: float = 320.0

    minimum_hdl_mg_dl: float = 20.0
    maximum_hdl_mg_dl: float = 100.0

    minimum_systolic_bp_mm_hg: float = 90.0
    maximum_systolic_bp_mm_hg: float = 200.0

    equation_version: str = "PCE-2013"

    allow_other_population_using_white_equation: bool = True


@dataclass(slots=True)
class ASCVDPCEValidation:
    """Resultado da validação de uma entrada PCE."""

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
# Coeficientes
# ============================================================


PCE_COEFFICIENTS: dict[
    tuple[CardiovascularSex, PCEPopulationGroup],
    PCECoefficients,
] = {
    (
        CardiovascularSex.MALE,
        PCEPopulationGroup.WHITE,
    ): PCECoefficients(
        baseline_survival=0.9144,
        mean_coefficient_sum=61.18,
        ln_age=12.344,
        ln_total_cholesterol=11.853,
        ln_age_x_ln_total_cholesterol=-2.664,
        ln_hdl=-7.990,
        ln_age_x_ln_hdl=1.769,
        ln_treated_sbp=1.797,
        ln_untreated_sbp=1.764,
        current_smoker=7.837,
        ln_age_x_current_smoker=-1.795,
        diabetes=0.658,
    ),
    (
        CardiovascularSex.MALE,
        PCEPopulationGroup.BLACK,
    ): PCECoefficients(
        baseline_survival=0.8954,
        mean_coefficient_sum=19.54,
        ln_age=2.469,
        ln_total_cholesterol=0.302,
        ln_hdl=-0.307,
        ln_treated_sbp=1.916,
        ln_untreated_sbp=1.809,
        current_smoker=0.549,
        diabetes=0.645,
    ),
    (
        CardiovascularSex.FEMALE,
        PCEPopulationGroup.WHITE,
    ): PCECoefficients(
        baseline_survival=0.9665,
        mean_coefficient_sum=-29.18,
        ln_age=-29.799,
        ln_age_squared=4.884,
        ln_total_cholesterol=13.540,
        ln_age_x_ln_total_cholesterol=-3.114,
        ln_hdl=-13.578,
        ln_age_x_ln_hdl=3.149,
        ln_treated_sbp=2.019,
        ln_untreated_sbp=1.957,
        current_smoker=7.574,
        ln_age_x_current_smoker=-1.665,
        diabetes=0.661,
    ),
    (
        CardiovascularSex.FEMALE,
        PCEPopulationGroup.BLACK,
    ): PCECoefficients(
        baseline_survival=0.9533,
        mean_coefficient_sum=86.61,
        ln_age=17.114,
        ln_total_cholesterol=0.940,
        ln_hdl=-18.920,
        ln_age_x_ln_hdl=4.475,
        ln_treated_sbp=29.291,
        ln_age_x_ln_treated_sbp=-6.432,
        ln_untreated_sbp=27.820,
        ln_age_x_ln_untreated_sbp=-6.087,
        current_smoker=0.691,
        diabetes=0.874,
    ),
}


# ============================================================
# Motor principal
# ============================================================


class ASCVDPooledCohortEngine:
    """Calculadora legada das Pooled Cohort Equations."""

    def __init__(
        self,
        config: ASCVDPCEConfig | None = None,
    ) -> None:
        self.config = config or ASCVDPCEConfig()

    def assess(
        self,
        data: ASCVDPCEInput,
    ) -> CardiovascularRiskEstimate:
        """
        Calcula o risco ASCVD em dez anos.

        Retorna um resultado inválido quando a entrada não
        satisfaz os critérios mínimos da equação.
        """

        validation = self.validate(data)

        if not validation.valid:
            return CardiovascularRiskEstimate(
                equation=RiskEquationType.ASCVD_PCE,
                risk_category=(
                    CardiovascularRiskCategory.UNDETERMINED
                ),
                endpoint=(
                    "Primeiro evento ASCVD fatal ou não fatal "
                    "em dez anos"
                ),
                population=(
                    data.population_group.value
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
                    "legacy_equation": True,
                },
            )

        effective_population = (
            self._effective_population_group(data)
        )

        coefficients = PCE_COEFFICIENTS[
            (
                data.biological_sex,
                effective_population,
            )
        ]

        coefficient_sum = (
            self._coefficient_sum(
                data=data,
                coefficients=coefficients,
            )
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

        if (
            data.population_group
            == PCEPopulationGroup.OTHER
        ):
            warnings.append(
                "A equação para população branca foi "
                "utilizada como aproximação operacional. "
                "O PCE não possui calibração original "
                "específica para este grupo."
            )

        warnings.append(
            "Resultado calculado pelas Pooled Cohort "
            "Equations de 2013. Registrar a versão e evitar "
            "interpretação como estimador cardiovascular "
            "contemporâneo universal."
        )

        return CardiovascularRiskEstimate(
            equation=RiskEquationType.ASCVD_PCE,
            risk_percent_10_years=round(
                risk_percent,
                2,
            ),
            risk_category=self.classify_risk(
                risk_percent,
            ),
            endpoint=(
                "Primeiro evento ASCVD fatal ou não fatal "
                "em dez anos"
            ),
            population=effective_population.value,
            valid=True,
            missing_fields=[],
            warnings=self._unique_strings(warnings),
            calculation_version=(
                self.config.equation_version
            ),
            metadata={
                "legacy_equation": True,
                "requested_population_group": (
                    data.population_group.value
                ),
                "effective_population_group": (
                    effective_population.value
                ),
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
            },
        )

    def assess_integrated_input(
        self,
        data: CardiovascularAssessmentInput,
        *,
        population_group: PCEPopulationGroup,
    ) -> CardiovascularRiskEstimate:
        """
        Converte a entrada cardiovascular integrada
        para o formato do PCE.
        """

        average_systolic = (
            self._average_systolic_pressure(data)
        )

        return self.assess(
            ASCVDPCEInput(
                age_years=data.age_years,
                biological_sex=data.biological_sex,
                population_group=population_group,
                total_cholesterol_mg_dl=(
                    data.lipid_profile.total_cholesterol
                ),
                hdl_cholesterol_mg_dl=(
                    data.lipid_profile.hdl_cholesterol
                ),
                systolic_blood_pressure_mm_hg=(
                    average_systolic
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
                established_ascvd=(
                    data.established_ascvd
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
        data: ASCVDPCEInput,
    ) -> ASCVDPCEValidation:
        """Valida a aplicabilidade e os intervalos do PCE."""

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
                CardiovascularSex.FEMALE,
                CardiovascularSex.MALE,
            }
        ):
            missing.append("biological_sex")

        if (
            data.population_group
            == PCEPopulationGroup.UNDETERMINED
        ):
            missing.append("population_group")

        if (
            data.population_group
            == PCEPopulationGroup.OTHER
            and not (
                self.config
                .allow_other_population_using_white_equation
            )
        ):
            invalid.append("population_group")

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
            or data.established_ascvd
        ):
            invalid.append("prevention_context")

            warnings.append(
                "O PCE não deve ser utilizado para estimar "
                "primeiro evento em paciente com ASCVD "
                "clínica estabelecida."
            )

        if (
            data.prevention_context
            == PreventionContext.UNDETERMINED
        ):
            warnings.append(
                "Contexto preventivo não determinado. "
                "Confirmar ausência de ASCVD clínica."
            )

        if data.population_group == PCEPopulationGroup.OTHER:
            warnings.append(
                "População fora das categorias de "
                "calibração originais do PCE."
            )

        return ASCVDPCEValidation(
            valid=not missing and not invalid,
            missing_fields=self._unique_strings(missing),
            invalid_fields=self._unique_strings(invalid),
            warnings=self._unique_strings(warnings),
        )

    # ========================================================
    # Cálculo
    # ========================================================

    @staticmethod
    def _coefficient_sum(
        *,
        data: ASCVDPCEInput,
        coefficients: PCECoefficients,
    ) -> float:
        """Calcula a soma linear dos coeficientes."""

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

        ln_systolic_bp = log(
            data.systolic_blood_pressure_mm_hg
        )

        current_smoker = (
            1.0 if data.current_smoker else 0.0
        )

        diabetes = 1.0 if data.diabetes else 0.0

        value = 0.0

        value += coefficients.ln_age * ln_age

        value += (
            coefficients.ln_age_squared
            * ln_age
            * ln_age
        )

        value += (
            coefficients.ln_total_cholesterol
            * ln_total_cholesterol
        )

        value += (
            coefficients.ln_age_x_ln_total_cholesterol
            * ln_age
            * ln_total_cholesterol
        )

        value += coefficients.ln_hdl * ln_hdl

        value += (
            coefficients.ln_age_x_ln_hdl
            * ln_age
            * ln_hdl
        )

        if data.treated_hypertension:
            value += (
                coefficients.ln_treated_sbp
                * ln_systolic_bp
            )

            value += (
                coefficients.ln_age_x_ln_treated_sbp
                * ln_age
                * ln_systolic_bp
            )
        else:
            value += (
                coefficients.ln_untreated_sbp
                * ln_systolic_bp
            )

            value += (
                coefficients.ln_age_x_ln_untreated_sbp
                * ln_age
                * ln_systolic_bp
            )

        value += (
            coefficients.current_smoker
            * current_smoker
        )

        value += (
            coefficients.ln_age_x_current_smoker
            * ln_age
            * current_smoker
        )

        value += coefficients.diabetes * diabetes

        return value

    # ========================================================
    # Classificação
    # ========================================================

    @staticmethod
    def classify_risk(
        risk_percent: float | None,
    ) -> CardiovascularRiskCategory:
        """
        Classifica o risco PCE segundo faixas históricas.

        <5%: baixo
        5 a <7,5%: limítrofe
        7,5 a <20%: intermediário
        >=20%: alto
        """

        if risk_percent is None:
            return (
                CardiovascularRiskCategory.UNDETERMINED
            )

        if risk_percent < 5.0:
            return CardiovascularRiskCategory.LOW

        if risk_percent < 7.5:
            return CardiovascularRiskCategory.BORDERLINE

        if risk_percent < 20.0:
            return (
                CardiovascularRiskCategory.INTERMEDIATE
            )

        return CardiovascularRiskCategory.HIGH

    # ========================================================
    # Conversores e utilidades
    # ========================================================

    @staticmethod
    def _effective_population_group(
        data: ASCVDPCEInput,
    ) -> PCEPopulationGroup:
        """
        Resolve a equação utilizada.

        O grupo OTHER utiliza a equação WHITE apenas como
        aproximação explicitamente sinalizada.
        """

        if data.population_group == PCEPopulationGroup.OTHER:
            return PCEPopulationGroup.WHITE

        return data.population_group

    @staticmethod
    def _average_systolic_pressure(
        data: CardiovascularAssessmentInput,
    ) -> float | None:
        """Calcula a PAS média das medições disponíveis."""

        valid_values = [
            measurement.systolic_mm_hg
            for measurement
            in data.blood_pressure_measurements
            if (
                ASCVDPooledCohortEngine._valid_number(
                    measurement.systolic_mm_hg
                )
                and 0
                < measurement.systolic_mm_hg
                <= 300
            )
        ]

        if not valid_values:
            return None

        return sum(valid_values) / len(valid_values)

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
        """Valida um campo numérico obrigatório."""

        if value is None:
            missing.append(field_name)
            return

        if not ASCVDPooledCohortEngine._number_in_range(
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
        """Verifica número finito dentro de um intervalo."""

        if not ASCVDPooledCohortEngine._valid_number(
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
        """Limita um valor ao intervalo solicitado."""

        return max(
            minimum,
            min(value, maximum),
        )

    @staticmethod
    def _unique_strings(
        values: list[str],
    ) -> list[str]:
        """Remove repetições e textos vazios."""

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