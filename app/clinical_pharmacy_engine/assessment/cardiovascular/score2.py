"""
PHK Studio
Clinical Pharmacy Engine

SCORE2 Risk Assessment Engine.

Estrutura segura para estimativa do risco cardiovascular
SCORE2 em dez anos.

O SCORE2 estima eventos cardiovasculares fatais e não fatais
em pessoas aparentemente saudáveis, geralmente entre
40 e 69 anos, sem doença cardiovascular estabelecida e
sem diabetes.

Este módulo não contém coeficientes ou tabelas clínicas
inventadas. Os valores devem ser carregados de uma tabela
validada e explicitamente versionada.

Responsabilidades:

- validar aplicabilidade clínica;
- normalizar fatores de risco;
- calcular colesterol não HDL;
- selecionar região europeia de risco;
- localizar célula correspondente em tabela validada;
- classificar o risco conforme idade;
- registrar versão, origem e limitações;
- integrar com CardiovascularAssessmentInput.

O resultado exige revisão clínica profissional.
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
    CardiovascularSex,
    DiabetesStatus,
    PreventionContext,
    RiskEquationType,
    SmokingStatus,
)


# ============================================================
# Enums
# ============================================================


class SCORE2Region(str, Enum):
    """
    Regiões europeias de calibração do SCORE2.

    A classificação regional deve ser definida por tabela
    oficial versionada ou protocolo institucional.
    """

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNDETERMINED = "undetermined"


class SCORE2ModelType(str, Enum):
    """Família de algoritmo SCORE2."""

    SCORE2 = "score2"
    SCORE2_OP = "score2_op"
    SCORE2_DIABETES = "score2_diabetes"
    SCORE2_LAC = "score2_lac"
    SCORE2_ASIA_PACIFIC = "score2_asia_pacific"
    UNDETERMINED = "undetermined"


class SCORE2LookupMode(str, Enum):
    """Estratégia de consulta da tabela."""

    EXACT = "exact"
    NEAREST = "nearest"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"


# ============================================================
# Entrada e configuração
# ============================================================


@dataclass(slots=True)
class SCORE2Input:
    """Entrada normalizada para o SCORE2."""

    age_years: float | None = None

    biological_sex: CardiovascularSex = (
        CardiovascularSex.UNDETERMINED
    )

    region: SCORE2Region = SCORE2Region.UNDETERMINED

    current_smoker: bool = False

    systolic_blood_pressure_mm_hg: float | None = None

    total_cholesterol_mg_dl: float | None = None
    hdl_cholesterol_mg_dl: float | None = None
    non_hdl_cholesterol_mg_dl: float | None = None

    diabetes: bool = False

    prevention_context: PreventionContext = (
        PreventionContext.PRIMARY
    )

    established_cardiovascular_disease: bool = False

    country_code: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class SCORE2Config:
    """Configurações operacionais do motor."""

    minimum_age_years: float = 40.0
    maximum_age_years: float = 69.0

    minimum_systolic_bp_mm_hg: float = 100.0
    maximum_systolic_bp_mm_hg: float = 180.0

    minimum_non_hdl_mg_dl: float = 100.0
    maximum_non_hdl_mg_dl: float = 300.0

    lookup_mode: SCORE2LookupMode = (
        SCORE2LookupMode.NEAREST
    )

    allow_derived_non_hdl: bool = True

    table_version_required: bool = True

    equation_version: str = "SCORE2-2021"

    model_type: SCORE2ModelType = SCORE2ModelType.SCORE2


@dataclass(slots=True)
class SCORE2Validation:
    """Resultado da validação de aplicabilidade."""

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
# Estrutura da tabela
# ============================================================


@dataclass(frozen=True, slots=True)
class SCORE2TableRow:
    """
    Linha validada de uma tabela SCORE2.

    Cada linha representa uma combinação explícita de:

    - região;
    - sexo;
    - idade;
    - tabagismo;
    - pressão sistólica;
    - colesterol não HDL;
    - risco em dez anos.
    """

    region: SCORE2Region
    biological_sex: CardiovascularSex

    age_years: int
    current_smoker: bool

    systolic_blood_pressure_mm_hg: int
    non_hdl_cholesterol_mg_dl: float

    risk_percent_10_years: float


@dataclass(slots=True)
class SCORE2LookupTable:
    """Coleção versionada de linhas SCORE2."""

    rows: list[SCORE2TableRow] = field(
        default_factory=list,
    )

    version: str = ""
    source_name: str = ""
    source_reference: str = ""

    validated: bool = False

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )

    def is_usable(self) -> bool:
        """Verifica se a tabela pode ser usada."""

        return bool(
            self.validated
            and self.version.strip()
            and self.rows
        )


@dataclass(slots=True)
class SCORE2LookupResult:
    """Resultado interno da consulta à tabela."""

    found: bool

    risk_percent_10_years: float | None = None

    matched_row: SCORE2TableRow | None = None

    distance: float | None = None

    warnings: list[str] = field(
        default_factory=list,
    )


# ============================================================
# Motor principal
# ============================================================


class SCORE2RiskEngine:
    """Motor de validação e consulta do SCORE2."""

    def __init__(
        self,
        table: SCORE2LookupTable | None = None,
        config: SCORE2Config | None = None,
    ) -> None:
        self.table = table or SCORE2LookupTable()
        self.config = config or SCORE2Config()

    def assess(
        self,
        data: SCORE2Input,
    ) -> CardiovascularRiskEstimate:
        """
        Avalia a entrada e consulta uma tabela validada.

        O cálculo é bloqueado quando nenhuma tabela clínica
        validada foi disponibilizada.
        """

        validation = self.validate(data)

        common_metadata = {
            "model_type": self.config.model_type.value,
            "requested_region": data.region.value,
            "lookup_mode": self.config.lookup_mode.value,
            "table_version": self.table.version,
            "table_source": self.table.source_name,
            "table_validated": self.table.validated,
        }

        if not validation.valid:
            return CardiovascularRiskEstimate(
                equation=RiskEquationType.SCORE2,
                risk_category=(
                    CardiovascularRiskCategory.UNDETERMINED
                ),
                endpoint=(
                    "Evento cardiovascular fatal ou "
                    "não fatal em dez anos"
                ),
                population=(
                    f"European region: {data.region.value}"
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
                    **common_metadata,
                    "invalid_fields": list(
                        validation.invalid_fields
                    ),
                },
            )

        if not self.table.is_usable():
            return CardiovascularRiskEstimate(
                equation=RiskEquationType.SCORE2,
                risk_category=(
                    CardiovascularRiskCategory.UNDETERMINED
                ),
                endpoint=(
                    "Evento cardiovascular fatal ou "
                    "não fatal em dez anos"
                ),
                population=(
                    f"European region: {data.region.value}"
                ),
                valid=False,
                missing_fields=[],
                warnings=self._unique_strings(
                    validation.warnings
                    + [
                        "Nenhuma tabela SCORE2 validada e "
                        "versionada foi carregada. O sistema "
                        "não calculará risco com dados "
                        "clínicos incompletos ou presumidos."
                    ]
                ),
                calculation_version=(
                    self.config.equation_version
                ),
                metadata={
                    **common_metadata,
                    "calculation_blocked": True,
                    "block_reason": (
                        "validated_table_unavailable"
                    ),
                },
            )

        non_hdl = self.resolve_non_hdl(data)

        assert data.age_years is not None
        assert (
            data.systolic_blood_pressure_mm_hg
            is not None
        )
        assert non_hdl is not None

        lookup = self.lookup(
            age_years=data.age_years,
            biological_sex=data.biological_sex,
            region=data.region,
            current_smoker=data.current_smoker,
            systolic_blood_pressure_mm_hg=(
                data.systolic_blood_pressure_mm_hg
            ),
            non_hdl_cholesterol_mg_dl=non_hdl,
        )

        if not lookup.found:
            return CardiovascularRiskEstimate(
                equation=RiskEquationType.SCORE2,
                risk_category=(
                    CardiovascularRiskCategory.UNDETERMINED
                ),
                endpoint=(
                    "Evento cardiovascular fatal ou "
                    "não fatal em dez anos"
                ),
                population=(
                    f"European region: {data.region.value}"
                ),
                valid=False,
                missing_fields=[],
                warnings=self._unique_strings(
                    validation.warnings
                    + lookup.warnings
                    + [
                        "A tabela carregada não possui uma "
                        "combinação aplicável aos dados "
                        "informados."
                    ]
                ),
                calculation_version=(
                    self.table.version
                    or self.config.equation_version
                ),
                metadata={
                    **common_metadata,
                    "calculation_blocked": True,
                    "block_reason": (
                        "matching_table_row_unavailable"
                    ),
                    "non_hdl_cholesterol_mg_dl": non_hdl,
                },
            )

        risk_percent = lookup.risk_percent_10_years

        assert risk_percent is not None

        category = self.classify_risk(
            age_years=data.age_years,
            risk_percent=risk_percent,
        )

        warnings = list(validation.warnings)
        warnings.extend(lookup.warnings)

        if lookup.distance not in {None, 0.0}:
            warnings.append(
                "O risco foi obtido pela célula mais próxima "
                "da tabela. Os valores utilizados devem ser "
                "exibidos no relatório clínico."
            )

        matched_row = lookup.matched_row

        return CardiovascularRiskEstimate(
            equation=RiskEquationType.SCORE2,
            risk_percent_10_years=round(
                risk_percent,
                2,
            ),
            risk_category=category,
            endpoint=(
                "Evento cardiovascular fatal ou não fatal "
                "em dez anos"
            ),
            population=(
                f"European region: {data.region.value}"
            ),
            valid=True,
            missing_fields=[],
            warnings=self._unique_strings(warnings),
            calculation_version=(
                self.table.version
                or self.config.equation_version
            ),
            metadata={
                **common_metadata,
                "non_hdl_cholesterol_mg_dl": round(
                    non_hdl,
                    2,
                ),
                "lookup_distance": lookup.distance,
                "matched_age_years": (
                    matched_row.age_years
                    if matched_row
                    else None
                ),
                "matched_systolic_bp_mm_hg": (
                    matched_row
                    .systolic_blood_pressure_mm_hg
                    if matched_row
                    else None
                ),
                "matched_non_hdl_mg_dl": (
                    matched_row
                    .non_hdl_cholesterol_mg_dl
                    if matched_row
                    else None
                ),
                "current_smoker": data.current_smoker,
            },
        )

    def assess_integrated_input(
        self,
        data: CardiovascularAssessmentInput,
        *,
        region: SCORE2Region,
        country_code: str | None = None,
    ) -> CardiovascularRiskEstimate:
        """Converte a entrada cardiovascular integrada."""

        systolic = self._average_systolic_pressure(
            data
        )

        diabetes = (
            data.diabetes_status
            in {
                DiabetesStatus.TYPE_1,
                DiabetesStatus.TYPE_2,
                DiabetesStatus.OTHER,
            }
        )

        established_disease = any(
            (
                data.established_ascvd,
                data.prior_myocardial_infarction,
                data.prior_stroke_or_tia,
                data.peripheral_arterial_disease,
            )
        )

        return self.assess(
            SCORE2Input(
                age_years=data.age_years,
                biological_sex=data.biological_sex,
                region=region,
                current_smoker=(
                    data.smoking_status
                    == SmokingStatus.CURRENT
                ),
                systolic_blood_pressure_mm_hg=systolic,
                total_cholesterol_mg_dl=(
                    data.lipid_profile.total_cholesterol
                ),
                hdl_cholesterol_mg_dl=(
                    data.lipid_profile.hdl_cholesterol
                ),
                non_hdl_cholesterol_mg_dl=(
                    data.lipid_profile
                    .non_hdl_cholesterol
                ),
                diabetes=diabetes,
                prevention_context=(
                    data.prevention_context
                ),
                established_cardiovascular_disease=(
                    established_disease
                ),
                country_code=country_code,
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
        data: SCORE2Input,
    ) -> SCORE2Validation:
        """Valida aplicabilidade clínica e intervalos."""

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

            if self._valid_number(data.age_years):
                age = float(data.age_years)

                if age >= 70:
                    warnings.append(
                        "Para pessoas com 70 anos ou mais, "
                        "avaliar SCORE2-OP em vez do SCORE2."
                    )

        if (
            data.biological_sex
            not in {
                CardiovascularSex.MALE,
                CardiovascularSex.FEMALE,
            }
        ):
            missing.append("biological_sex")

        if data.region == SCORE2Region.UNDETERMINED:
            missing.append("region")

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

        non_hdl = self.resolve_non_hdl(data)

        if non_hdl is None:
            missing.append(
                "non_hdl_cholesterol_mg_dl"
            )
        elif not self._number_in_range(
            non_hdl,
            self.config.minimum_non_hdl_mg_dl,
            self.config.maximum_non_hdl_mg_dl,
        ):
            invalid.append(
                "non_hdl_cholesterol_mg_dl"
            )

        if data.diabetes:
            invalid.append("diabetes")

            warnings.append(
                "O SCORE2 convencional não é o modelo "
                "apropriado para pessoas com diabetes. "
                "Utilizar algoritmo específico aplicável."
            )

        if (
            data.prevention_context
            == PreventionContext.SECONDARY
            or data.established_cardiovascular_disease
        ):
            invalid.append("prevention_context")

            warnings.append(
                "O SCORE2 destina-se à prevenção primária "
                "em pessoas sem doença cardiovascular "
                "estabelecida."
            )

        if (
            data.prevention_context
            == PreventionContext.UNDETERMINED
        ):
            warnings.append(
                "Contexto preventivo não determinado. "
                "Confirmar ausência de doença "
                "cardiovascular prévia."
            )

        if data.country_code is None:
            warnings.append(
                "País não informado. A região SCORE2 deve "
                "ser confirmada por fonte oficial ou "
                "protocolo institucional."
            )

        return SCORE2Validation(
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
    # Consulta à tabela
    # ========================================================

    def lookup(
        self,
        *,
        age_years: float,
        biological_sex: CardiovascularSex,
        region: SCORE2Region,
        current_smoker: bool,
        systolic_blood_pressure_mm_hg: float,
        non_hdl_cholesterol_mg_dl: float,
    ) -> SCORE2LookupResult:
        """Seleciona a linha aplicável da tabela."""

        candidates = [
            row
            for row in self.table.rows
            if (
                row.region == region
                and row.biological_sex
                == biological_sex
                and row.current_smoker
                == current_smoker
            )
        ]

        if not candidates:
            return SCORE2LookupResult(
                found=False,
                warnings=[
                    "Nenhuma linha da tabela corresponde à "
                    "região, sexo e tabagismo informados."
                ],
            )

        exact = [
            row
            for row in candidates
            if (
                float(row.age_years)
                == float(age_years)
                and float(
                    row.systolic_blood_pressure_mm_hg
                )
                == float(
                    systolic_blood_pressure_mm_hg
                )
                and float(
                    row.non_hdl_cholesterol_mg_dl
                )
                == float(
                    non_hdl_cholesterol_mg_dl
                )
            )
        ]

        if exact:
            row = exact[0]

            return SCORE2LookupResult(
                found=True,
                risk_percent_10_years=(
                    row.risk_percent_10_years
                ),
                matched_row=row,
                distance=0.0,
            )

        if (
            self.config.lookup_mode
            == SCORE2LookupMode.EXACT
        ):
            return SCORE2LookupResult(
                found=False,
                warnings=[
                    "Não existe correspondência exata na "
                    "tabela SCORE2 carregada."
                ],
            )

        selected = self._select_nearest_row(
            candidates=candidates,
            age_years=age_years,
            systolic_blood_pressure_mm_hg=(
                systolic_blood_pressure_mm_hg
            ),
            non_hdl_cholesterol_mg_dl=(
                non_hdl_cholesterol_mg_dl
            ),
        )

        if selected is None:
            return SCORE2LookupResult(
                found=False,
            )

        row, distance = selected

        return SCORE2LookupResult(
            found=True,
            risk_percent_10_years=(
                row.risk_percent_10_years
            ),
            matched_row=row,
            distance=round(distance, 6),
        )

    def _select_nearest_row(
        self,
        *,
        candidates: Sequence[SCORE2TableRow],
        age_years: float,
        systolic_blood_pressure_mm_hg: float,
        non_hdl_cholesterol_mg_dl: float,
    ) -> tuple[SCORE2TableRow, float] | None:
        """Seleciona a célula mais próxima."""

        eligible = list(candidates)

        if (
            self.config.lookup_mode
            == SCORE2LookupMode.LOWER_BOUND
        ):
            eligible = [
                row
                for row in eligible
                if (
                    row.age_years <= age_years
                    and (
                        row.systolic_blood_pressure_mm_hg
                        <= systolic_blood_pressure_mm_hg
                    )
                    and (
                        row.non_hdl_cholesterol_mg_dl
                        <= non_hdl_cholesterol_mg_dl
                    )
                )
            ]

        elif (
            self.config.lookup_mode
            == SCORE2LookupMode.UPPER_BOUND
        ):
            eligible = [
                row
                for row in eligible
                if (
                    row.age_years >= age_years
                    and (
                        row.systolic_blood_pressure_mm_hg
                        >= systolic_blood_pressure_mm_hg
                    )
                    and (
                        row.non_hdl_cholesterol_mg_dl
                        >= non_hdl_cholesterol_mg_dl
                    )
                )
            ]

        if not eligible:
            return None

        def distance(
            row: SCORE2TableRow,
        ) -> float:
            age_component = (
                abs(row.age_years - age_years)
                / 5.0
            )

            systolic_component = (
                abs(
                    row.systolic_blood_pressure_mm_hg
                    - systolic_blood_pressure_mm_hg
                )
                / 20.0
            )

            non_hdl_component = (
                abs(
                    row.non_hdl_cholesterol_mg_dl
                    - non_hdl_cholesterol_mg_dl
                )
                / 40.0
            )

            return (
                age_component
                + systolic_component
                + non_hdl_component
            )

        row = min(
            eligible,
            key=distance,
        )

        return row, distance(row)

    # ========================================================
    # Classificação etária
    # ========================================================

    @staticmethod
    def classify_risk(
        *,
        age_years: float,
        risk_percent: float | None,
    ) -> CardiovascularRiskCategory:
        """
        Classifica o risco usando limiares etários do
        ecossistema SCORE2.

        Menores de 50 anos:
            baixo/moderado: <2,5%
            alto: 2,5 a <7,5%
            muito alto: >=7,5%

        50 a 69 anos:
            baixo/moderado: <5%
            alto: 5 a <10%
            muito alto: >=10%
        """

        if risk_percent is None:
            return (
                CardiovascularRiskCategory.UNDETERMINED
            )

        if age_years < 50:
            if risk_percent < 2.5:
                return (
                    CardiovascularRiskCategory.MODERATE
                )

            if risk_percent < 7.5:
                return CardiovascularRiskCategory.HIGH

            return (
                CardiovascularRiskCategory.VERY_HIGH
            )

        if risk_percent < 5.0:
            return CardiovascularRiskCategory.MODERATE

        if risk_percent < 10.0:
            return CardiovascularRiskCategory.HIGH

        return CardiovascularRiskCategory.VERY_HIGH

    # ========================================================
    # Colesterol não HDL
    # ========================================================

    def resolve_non_hdl(
        self,
        data: SCORE2Input,
    ) -> float | None:
        """Obtém ou calcula colesterol não HDL."""

        if self._valid_number(
            data.non_hdl_cholesterol_mg_dl
        ):
            value = float(
                data.non_hdl_cholesterol_mg_dl
            )

            if value >= 0:
                return value

        if not self.config.allow_derived_non_hdl:
            return None

        if not self._valid_number(
            data.total_cholesterol_mg_dl
        ):
            return None

        if not self._valid_number(
            data.hdl_cholesterol_mg_dl
        ):
            return None

        total = float(
            data.total_cholesterol_mg_dl
        )

        hdl = float(
            data.hdl_cholesterol_mg_dl
        )

        value = total - hdl

        if value < 0:
            return None

        return value

    # ========================================================
    # Utilidades
    # ========================================================

    @staticmethod
    def _average_systolic_pressure(
        data: CardiovascularAssessmentInput,
    ) -> float | None:
        """Calcula PAS média das medições válidas."""

        values = [
            measurement.systolic_mm_hg
            for measurement
            in data.blood_pressure_measurements
            if (
                SCORE2RiskEngine._valid_number(
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

        if not SCORE2RiskEngine._number_in_range(
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

        if not SCORE2RiskEngine._valid_number(value):
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
    def _unique_strings(
        values: Iterable[str],
    ) -> list[str]:
        """Remove textos vazios e duplicados."""

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