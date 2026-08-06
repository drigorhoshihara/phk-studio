"""
PHK Studio
Clinical Pharmacy Engine

QT Interval Assessment Engine.

Responsabilidades:

- validar QT, RR e frequência cardíaca;
- calcular QT corrigido por:
    - Bazett;
    - Fridericia;
    - Framingham;
    - Hodges;
- selecionar uma fórmula preferencial;
- classificar o QTc;
- estratificar risco de torsades de pointes;
- considerar eletrólitos, medicamentos e fatores clínicos;
- gerar alertas estruturados e metadados auditáveis.

Este módulo fornece suporte à decisão clínica.

A medição automática do QT deve ser revisada quando houver:

- fibrilação atrial;
- ritmo estimulado;
- bloqueio de ramo;
- QRS alargado;
- frequência cardíaca extrema;
- artefato;
- morfologia anormal da onda T;
- presença de onda U;
- dúvida sobre o final da repolarização.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Any, Iterable

from app.clinical_pharmacy_engine.assessment.cardiovascular.models import (
    CardiovascularAssessmentInput,
    CardiovascularSex,
    ECGData,
    QTAssessmentResult,
    QTClassification,
    QTCorrectionFormula,
    TorsadesRiskCategory,
)


# ============================================================
# Entrada e configuração
# ============================================================


@dataclass(slots=True)
class QTAssessmentInput:
    """Entrada normalizada para avaliação do QT."""

    biological_sex: CardiovascularSex = (
        CardiovascularSex.UNDETERMINED
    )

    qt_interval_ms: float | None = None
    rr_interval_ms: float | None = None
    heart_rate_bpm: float | None = None
    qrs_duration_ms: float | None = None

    provided_corrected_qt_ms: float | None = None

    atrial_fibrillation_present: bool = False
    ventricular_arrhythmia_present: bool = False
    bundle_branch_block_present: bool = False
    paced_rhythm: bool = False

    congenital_long_qt_syndrome: bool = False
    previous_torsades_de_pointes: bool = False
    previous_syncope_suspected_arrhythmic: bool = False

    bradycardia_present: bool = False
    structural_heart_disease: bool = False
    acute_myocardial_infarction: bool = False
    heart_failure_present: bool = False

    potassium_mmol_l: float | None = None
    magnesium_mg_dl: float | None = None
    calcium_mg_dl: float | None = None

    qt_prolonging_medications: list[str] = field(
        default_factory=list,
    )

    metadata: dict[str, Any] = field(
        default_factory=dict,
    )


@dataclass(slots=True)
class QTAssessmentConfig:
    """Configurações clínicas e matemáticas do motor."""

    preferred_formula: QTCorrectionFormula = (
        QTCorrectionFormula.FRIDERICIA
    )

    minimum_qt_ms: float = 200.0
    maximum_qt_ms: float = 800.0

    minimum_heart_rate_bpm: float = 20.0
    maximum_heart_rate_bpm: float = 250.0

    minimum_rr_ms: float = 240.0
    maximum_rr_ms: float = 3000.0

    wide_qrs_threshold_ms: float = 120.0

    male_prolonged_qtc_ms: float = 450.0
    female_prolonged_qtc_ms: float = 470.0
    undetermined_sex_prolonged_qtc_ms: float = 460.0

    borderline_margin_ms: float = 20.0

    markedly_prolonged_qtc_ms: float = 500.0
    extreme_qtc_ms: float = 550.0

    low_potassium_mmol_l: float = 3.5
    low_magnesium_mg_dl: float = 1.7
    low_calcium_mg_dl: float = 8.5

    severe_hypokalemia_mmol_l: float = 3.0
    severe_hypomagnesemia_mg_dl: float = 1.2

    bradycardia_threshold_bpm: float = 50.0

    prefer_fridericia_below_bpm: float = 60.0
    prefer_fridericia_above_bpm: float = 100.0


@dataclass(slots=True)
class QTValidationResult:
    """Resultado da validação dos dados de entrada."""

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
# Motor principal
# ============================================================


class QTIntervalAssessmentEngine:
    """Motor de cálculo e avaliação do intervalo QT."""

    def __init__(
        self,
        config: QTAssessmentConfig | None = None,
    ) -> None:
        self.config = config or QTAssessmentConfig()

    def assess(
        self,
        data: QTAssessmentInput,
    ) -> QTAssessmentResult:
        """Executa a avaliação completa do QT."""

        validation = self.validate(data)

        if not validation.valid:
            return QTAssessmentResult(
                raw_qt_ms=data.qt_interval_ms,
                heart_rate_bpm=data.heart_rate_bpm,
                classification=(
                    QTClassification.UNDETERMINED
                ),
                torsades_risk=(
                    TorsadesRiskCategory.UNDETERMINED
                ),
                qt_prolonging_medications=(
                    self._unique_strings(
                        data.qt_prolonging_medications
                    )
                ),
                valid=False,
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

        assert data.qt_interval_ms is not None

        heart_rate = self.resolve_heart_rate(data)
        rr_interval_ms = self.resolve_rr_interval(data)

        assert heart_rate is not None
        assert rr_interval_ms is not None

        calculations = self.calculate_all(
            qt_interval_ms=data.qt_interval_ms,
            heart_rate_bpm=heart_rate,
            rr_interval_ms=rr_interval_ms,
        )

        preferred_formula = self.select_preferred_formula(
            heart_rate_bpm=heart_rate,
            data=data,
        )

        preferred_qtc = calculations.get(
            preferred_formula
        )

        warnings = list(validation.warnings)

        if preferred_qtc is None:
            preferred_qtc = (
                data.provided_corrected_qt_ms
            )

            warnings.append(
                "Não foi possível obter QTc pela fórmula "
                "preferencial; foi utilizado o QTc informado."
            )

        classification = self.classify_qtc(
            qtc_ms=preferred_qtc,
            biological_sex=data.biological_sex,
        )

        electrolyte_risk = self.has_electrolyte_risk(
            data
        )

        torsades_risk = self.classify_torsades_risk(
            qtc_ms=preferred_qtc,
            classification=classification,
            data=data,
            electrolyte_risk=electrolyte_risk,
            heart_rate_bpm=heart_rate,
        )

        immediate_review = any(
            (
                classification
                in {
                    QTClassification
                    .MARKEDLY_PROLONGED,
                    QTClassification.EXTREME,
                },
                torsades_risk
                in {
                    TorsadesRiskCategory.HIGH,
                    TorsadesRiskCategory.VERY_HIGH,
                },
                data.previous_torsades_de_pointes,
                data.ventricular_arrhythmia_present,
            )
        )

        if (
            data.qrs_duration_ms is not None
            and data.qrs_duration_ms
            >= self.config.wide_qrs_threshold_ms
        ):
            warnings.append(
                "QRS alargado. O QTc pode superestimar a "
                "repolarização ventricular; avaliar JT/JTc "
                "ou método específico."
            )

        if data.bundle_branch_block_present:
            warnings.append(
                "Bloqueio de ramo informado. Interpretar "
                "QTc com cautela."
            )

        if data.paced_rhythm:
            warnings.append(
                "Ritmo estimulado informado. A correção "
                "convencional do QT pode não ser adequada."
            )

        if data.atrial_fibrillation_present:
            warnings.append(
                "Fibrilação atrial presente. Recomenda-se "
                "avaliar múltiplos ciclos cardíacos."
            )

        if heart_rate < 60 or heart_rate > 100:
            warnings.append(
                "Frequência cardíaca fora da faixa de "
                "60 a 100 bpm. Bazett pode apresentar "
                "distorção mais pronunciada."
            )

        if electrolyte_risk:
            warnings.append(
                "Foi identificado fator eletrolítico que "
                "pode aumentar o risco de prolongamento do "
                "QT e torsades de pointes."
            )

        if data.qt_prolonging_medications:
            warnings.append(
                "Existem medicamentos com potencial de "
                "prolongamento do QT. Realizar revisão de "
                "dose, associação, exposição e alternativas."
            )

        if (
            classification
            == QTClassification.MARKEDLY_PROLONGED
        ):
            warnings.append(
                "QTc igual ou superior ao limiar de "
                "prolongamento marcado. Revisão clínica e "
                "farmacoterapêutica prioritária."
            )

        if classification == QTClassification.EXTREME:
            warnings.append(
                "QTc em faixa extrema. Avaliação médica "
                "imediata deve ser considerada."
            )

        return QTAssessmentResult(
            raw_qt_ms=round(
                float(data.qt_interval_ms),
                2,
            ),
            heart_rate_bpm=round(
                heart_rate,
                2,
            ),
            qtc_bazett_ms=self._rounded(
                calculations.get(
                    QTCorrectionFormula.BAZETT
                )
            ),
            qtc_fridericia_ms=self._rounded(
                calculations.get(
                    QTCorrectionFormula.FRIDERICIA
                )
            ),
            qtc_framingham_ms=self._rounded(
                calculations.get(
                    QTCorrectionFormula.FRAMINGHAM
                )
            ),
            qtc_hodges_ms=self._rounded(
                calculations.get(
                    QTCorrectionFormula.HODGES
                )
            ),
            preferred_qtc_ms=self._rounded(
                preferred_qtc
            ),
            preferred_formula=preferred_formula,
            classification=classification,
            torsades_risk=torsades_risk,
            qt_prolonging_medications=(
                self._unique_strings(
                    data.qt_prolonging_medications
                )
            ),
            electrolyte_risk_present=(
                electrolyte_risk
            ),
            immediate_review_required=(
                immediate_review
            ),
            valid=True,
            warnings=self._unique_strings(warnings),
            metadata={
                "rr_interval_ms": round(
                    rr_interval_ms,
                    2,
                ),
                "provided_qtc_ms": (
                    data.provided_corrected_qt_ms
                ),
                "wide_qrs": bool(
                    data.qrs_duration_ms is not None
                    and data.qrs_duration_ms
                    >= self.config.wide_qrs_threshold_ms
                ),
                "qrs_duration_ms": (
                    data.qrs_duration_ms
                ),
                "medication_count": len(
                    self._unique_strings(
                        data.qt_prolonging_medications
                    )
                ),
                "congenital_long_qt_syndrome": (
                    data.congenital_long_qt_syndrome
                ),
                "previous_torsades_de_pointes": (
                    data.previous_torsades_de_pointes
                ),
            },
        )

    def assess_integrated_input(
        self,
        data: CardiovascularAssessmentInput,
        *,
        congenital_long_qt_syndrome: bool = False,
        previous_torsades_de_pointes: bool = False,
        previous_syncope_suspected_arrhythmic: (
            bool
        ) = False,
    ) -> QTAssessmentResult:
        """Converte a entrada cardiovascular integrada."""

        ecg = data.ecg

        heart_rate = (
            ecg.heart_rate_bpm
            if ecg.heart_rate_bpm is not None
            else self._average_heart_rate(data)
        )

        return self.assess(
            QTAssessmentInput(
                biological_sex=data.biological_sex,
                qt_interval_ms=ecg.qt_interval_ms,
                rr_interval_ms=ecg.rr_interval_ms,
                heart_rate_bpm=heart_rate,
                qrs_duration_ms=ecg.qrs_duration_ms,
                provided_corrected_qt_ms=(
                    ecg.corrected_qt_ms
                ),
                atrial_fibrillation_present=(
                    ecg.atrial_fibrillation_present
                ),
                ventricular_arrhythmia_present=(
                    ecg.ventricular_arrhythmia_present
                ),
                bundle_branch_block_present=(
                    ecg.bundle_branch_block_present
                ),
                paced_rhythm=ecg.paced_rhythm,
                congenital_long_qt_syndrome=(
                    congenital_long_qt_syndrome
                ),
                previous_torsades_de_pointes=(
                    previous_torsades_de_pointes
                ),
                previous_syncope_suspected_arrhythmic=(
                    previous_syncope_suspected_arrhythmic
                ),
                bradycardia_present=bool(
                    heart_rate is not None
                    and heart_rate
                    < self.config.bradycardia_threshold_bpm
                ),
                structural_heart_disease=(
                    data.heart_failure
                ),
                acute_myocardial_infarction=(
                    data.prior_myocardial_infarction
                ),
                heart_failure_present=(
                    data.heart_failure
                ),
                potassium_mmol_l=(
                    data.potassium_mmol_l
                ),
                magnesium_mg_dl=(
                    data.magnesium_mg_dl
                ),
                calcium_mg_dl=data.calcium_mg_dl,
                qt_prolonging_medications=list(
                    data.qt_prolonging_medications
                ),
                metadata={
                    **dict(data.metadata),
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
        data: QTAssessmentInput,
    ) -> QTValidationResult:
        """Valida os dados necessários ao cálculo."""

        missing: list[str] = []
        invalid: list[str] = []
        warnings: list[str] = []

        if data.qt_interval_ms is None:
            missing.append("qt_interval_ms")
        elif not self._number_in_range(
            data.qt_interval_ms,
            self.config.minimum_qt_ms,
            self.config.maximum_qt_ms,
        ):
            invalid.append("qt_interval_ms")

        heart_rate = self.resolve_heart_rate(data)
        rr_interval = self.resolve_rr_interval(data)

        if heart_rate is None and rr_interval is None:
            missing.append(
                "heart_rate_bpm_or_rr_interval_ms"
            )

        if (
            data.heart_rate_bpm is not None
            and not self._number_in_range(
                data.heart_rate_bpm,
                self.config.minimum_heart_rate_bpm,
                self.config.maximum_heart_rate_bpm,
            )
        ):
            invalid.append("heart_rate_bpm")

        if (
            data.rr_interval_ms is not None
            and not self._number_in_range(
                data.rr_interval_ms,
                self.config.minimum_rr_ms,
                self.config.maximum_rr_ms,
            )
        ):
            invalid.append("rr_interval_ms")

        if (
            data.qrs_duration_ms is not None
            and (
                not self._valid_number(
                    data.qrs_duration_ms
                )
                or data.qrs_duration_ms <= 0
                or data.qrs_duration_ms > 400
            )
        ):
            invalid.append("qrs_duration_ms")

        if (
            data.heart_rate_bpm is not None
            and data.rr_interval_ms is not None
            and self._valid_number(
                data.heart_rate_bpm
            )
            and self._valid_number(
                data.rr_interval_ms
            )
        ):
            derived_hr = (
                60000.0
                / float(data.rr_interval_ms)
            )

            difference = abs(
                derived_hr
                - float(data.heart_rate_bpm)
            )

            if difference > 10:
                warnings.append(
                    "Frequência cardíaca e intervalo RR "
                    "apresentam divergência relevante."
                )

        for field_name, value in {
            "potassium_mmol_l": data.potassium_mmol_l,
            "magnesium_mg_dl": data.magnesium_mg_dl,
            "calcium_mg_dl": data.calcium_mg_dl,
        }.items():
            if value is None:
                continue

            if (
                not self._valid_number(value)
                or float(value) < 0
            ):
                invalid.append(field_name)

        return QTValidationResult(
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
    # Fórmulas
    # ========================================================

    def calculate_all(
        self,
        *,
        qt_interval_ms: float,
        heart_rate_bpm: float,
        rr_interval_ms: float,
    ) -> dict[QTCorrectionFormula, float]:
        """Calcula QTc pelas quatro fórmulas."""

        rr_seconds = rr_interval_ms / 1000.0

        return {
            QTCorrectionFormula.BAZETT: (
                self.calculate_bazett(
                    qt_interval_ms,
                    rr_seconds,
                )
            ),
            QTCorrectionFormula.FRIDERICIA: (
                self.calculate_fridericia(
                    qt_interval_ms,
                    rr_seconds,
                )
            ),
            QTCorrectionFormula.FRAMINGHAM: (
                self.calculate_framingham(
                    qt_interval_ms,
                    rr_seconds,
                )
            ),
            QTCorrectionFormula.HODGES: (
                self.calculate_hodges(
                    qt_interval_ms,
                    heart_rate_bpm,
                )
            ),
        }

    @staticmethod
    def calculate_bazett(
        qt_ms: float,
        rr_seconds: float,
    ) -> float:
        """QTc = QT / raiz quadrada de RR."""

        if rr_seconds <= 0:
            raise ValueError(
                "RR deve ser maior que zero."
            )

        return qt_ms / sqrt(rr_seconds)

    @staticmethod
    def calculate_fridericia(
        qt_ms: float,
        rr_seconds: float,
    ) -> float:
        """QTc = QT / raiz cúbica de RR."""

        if rr_seconds <= 0:
            raise ValueError(
                "RR deve ser maior que zero."
            )

        return qt_ms / (rr_seconds ** (1.0 / 3.0))

    @staticmethod
    def calculate_framingham(
        qt_ms: float,
        rr_seconds: float,
    ) -> float:
        """QTc = QT + 154 × (1 - RR)."""

        return qt_ms + 154.0 * (1.0 - rr_seconds)

    @staticmethod
    def calculate_hodges(
        qt_ms: float,
        heart_rate_bpm: float,
    ) -> float:
        """QTc = QT + 1,75 × (FC - 60)."""

        return (
            qt_ms
            + 1.75
            * (heart_rate_bpm - 60.0)
        )

    # ========================================================
    # Seleção e classificação
    # ========================================================

    def select_preferred_formula(
        self,
        *,
        heart_rate_bpm: float,
        data: QTAssessmentInput,
    ) -> QTCorrectionFormula:
        """Seleciona a fórmula preferencial."""

        if (
            heart_rate_bpm
            < self.config.prefer_fridericia_below_bpm
            or heart_rate_bpm
            > self.config.prefer_fridericia_above_bpm
        ):
            return QTCorrectionFormula.FRIDERICIA

        return self.config.preferred_formula

    def classify_qtc(
        self,
        *,
        qtc_ms: float | None,
        biological_sex: CardiovascularSex,
    ) -> QTClassification:
        """Classifica o QTc."""

        if qtc_ms is None:
            return QTClassification.UNDETERMINED

        if qtc_ms >= self.config.extreme_qtc_ms:
            return QTClassification.EXTREME

        if (
            qtc_ms
            >= self.config.markedly_prolonged_qtc_ms
        ):
            return (
                QTClassification.MARKEDLY_PROLONGED
            )

        prolonged_threshold = (
            self._prolonged_threshold(
                biological_sex
            )
        )

        if qtc_ms >= prolonged_threshold:
            return QTClassification.PROLONGED

        if (
            qtc_ms
            >= prolonged_threshold
            - self.config.borderline_margin_ms
        ):
            return QTClassification.BORDERLINE

        return QTClassification.NORMAL

    def classify_torsades_risk(
        self,
        *,
        qtc_ms: float | None,
        classification: QTClassification,
        data: QTAssessmentInput,
        electrolyte_risk: bool,
        heart_rate_bpm: float,
    ) -> TorsadesRiskCategory:
        """Estratifica risco clínico simplificado."""

        if qtc_ms is None:
            return TorsadesRiskCategory.UNDETERMINED

        score = 0

        if classification == QTClassification.BORDERLINE:
            score += 1

        elif classification == QTClassification.PROLONGED:
            score += 2

        elif (
            classification
            == QTClassification.MARKEDLY_PROLONGED
        ):
            score += 4

        elif classification == QTClassification.EXTREME:
            score += 5

        medication_count = len(
            self._unique_strings(
                data.qt_prolonging_medications
            )
        )

        if medication_count == 1:
            score += 1
        elif medication_count >= 2:
            score += 2

        if electrolyte_risk:
            score += 2

        if (
            data.bradycardia_present
            or heart_rate_bpm
            < self.config.bradycardia_threshold_bpm
        ):
            score += 1

        if data.structural_heart_disease:
            score += 1

        if data.heart_failure_present:
            score += 1

        if data.acute_myocardial_infarction:
            score += 1

        if data.congenital_long_qt_syndrome:
            score += 3

        if data.previous_torsades_de_pointes:
            score += 4

        if data.previous_syncope_suspected_arrhythmic:
            score += 2

        if data.ventricular_arrhythmia_present:
            score += 4

        if score <= 1:
            return TorsadesRiskCategory.LOW

        if score <= 3:
            return TorsadesRiskCategory.MODERATE

        if score <= 6:
            return TorsadesRiskCategory.HIGH

        return TorsadesRiskCategory.VERY_HIGH

    # ========================================================
    # Eletrólitos
    # ========================================================

    def has_electrolyte_risk(
        self,
        data: QTAssessmentInput,
    ) -> bool:
        """Detecta eletrólitos abaixo dos limites."""

        return any(
            (
                data.potassium_mmol_l is not None
                and data.potassium_mmol_l
                < self.config.low_potassium_mmol_l,
                data.magnesium_mg_dl is not None
                and data.magnesium_mg_dl
                < self.config.low_magnesium_mg_dl,
                data.calcium_mg_dl is not None
                and data.calcium_mg_dl
                < self.config.low_calcium_mg_dl,
            )
        )

    # ========================================================
    # Conversões
    # ========================================================

    def resolve_heart_rate(
        self,
        data: QTAssessmentInput,
    ) -> float | None:
        """Obtém FC diretamente ou pelo intervalo RR."""

        if self._valid_number(
            data.heart_rate_bpm
        ):
            value = float(data.heart_rate_bpm)

            if (
                self.config.minimum_heart_rate_bpm
                <= value
                <= self.config.maximum_heart_rate_bpm
            ):
                return value

        if self._valid_number(
            data.rr_interval_ms
        ):
            rr = float(data.rr_interval_ms)

            if rr > 0:
                return 60000.0 / rr

        return None

    def resolve_rr_interval(
        self,
        data: QTAssessmentInput,
    ) -> float | None:
        """Obtém RR diretamente ou pela FC."""

        if self._valid_number(
            data.rr_interval_ms
        ):
            value = float(data.rr_interval_ms)

            if (
                self.config.minimum_rr_ms
                <= value
                <= self.config.maximum_rr_ms
            ):
                return value

        if self._valid_number(
            data.heart_rate_bpm
        ):
            heart_rate = float(
                data.heart_rate_bpm
            )

            if heart_rate > 0:
                return 60000.0 / heart_rate

        return None

    # ========================================================
    # Utilidades
    # ========================================================

    def _prolonged_threshold(
        self,
        biological_sex: CardiovascularSex,
    ) -> float:
        """Obtém limiar configurado por sexo."""

        if biological_sex == CardiovascularSex.MALE:
            return self.config.male_prolonged_qtc_ms

        if biological_sex == CardiovascularSex.FEMALE:
            return self.config.female_prolonged_qtc_ms

        return (
            self.config
            .undetermined_sex_prolonged_qtc_ms
        )

    @staticmethod
    def _average_heart_rate(
        data: CardiovascularAssessmentInput,
    ) -> float | None:
        """Calcula FC média das medições pressóricas."""

        values = [
            measurement.heart_rate_bpm
            for measurement
            in data.blood_pressure_measurements
            if (
                measurement.heart_rate_bpm is not None
                and QTIntervalAssessmentEngine
                ._valid_number(
                    measurement.heart_rate_bpm
                )
                and measurement.heart_rate_bpm > 0
            )
        ]

        if not values:
            return None

        return sum(values) / len(values)

    @staticmethod
    def _number_in_range(
        value: object,
        minimum: float,
        maximum: float,
    ) -> bool:
        """Valida número dentro do intervalo."""

        if not QTIntervalAssessmentEngine._valid_number(
            value
        ):
            return False

        numeric = float(value)

        return minimum <= numeric <= maximum

    @staticmethod
    def _valid_number(
        value: object,
    ) -> bool:
        """Valida número finito."""

        try:
            return isfinite(float(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _rounded(
        value: float | None,
    ) -> float | None:
        """Arredonda valor quando disponível."""

        if value is None:
            return None

        return round(value, 2)

    @staticmethod
    def _unique_strings(
        values: Iterable[str],
    ) -> list[str]:
        """Remove valores vazios e repetidos."""

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