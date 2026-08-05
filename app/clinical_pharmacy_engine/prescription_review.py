"""
PHK Studio
Clinical Pharmacy Engine

Prescription Review Engine.

Motor principal de revisão farmacêutica clínica.

Responsabilidades:

- análise estrutural da prescrição;
- detecção de duplicidades;
- avaliação preliminar de alergias;
- identificação de campos clínicos ausentes;
- análise de interações medicamentosas;
- classificação do risco;
- geração de intervenções farmacêuticas;
- suporte à validação profissional.

As análises automatizadas são apoio à decisão clínica.
Elas não substituem a avaliação do farmacêutico habilitado.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from app.clinical_pharmacy_engine.interaction_analyzer import (
    InteractionAnalyzer,
)
from app.clinical_pharmacy_engine.interactions.models import (
    InteractionAnalysisContext,
)
from app.clinical_pharmacy_engine.models import (
    AllergyRecord,
    DrugInteraction,
    DrugTherapyProblem,
    DrugTherapyProblemType,
    MedicationRecord,
    PatientProfile,
    PharmaceuticalIntervention,
    PrescriptionItem,
    PrescriptionRecord,
    PrescriptionReviewResult,
    SeverityLevel,
    UrgencyLevel,
)


class PrescriptionReviewer:
    """
    Orquestrador da revisão farmacêutica de prescrições.

    Coordena verificações estruturais, clínicas e de interação,
    consolidando os achados em um único resultado.
    """

    def __init__(
        self,
        interaction_analyzer: InteractionAnalyzer | None = None,
    ) -> None:
        self.interaction_analyzer = (
            interaction_analyzer
            or InteractionAnalyzer()
        )

    def review(
        self,
        patient: PatientProfile,
        prescription: PrescriptionRecord,
    ) -> PrescriptionReviewResult:
        """
        Executa a revisão clínica inicial da prescrição.
        """

        problems: list[DrugTherapyProblem] = []
        interventions: list[PharmaceuticalIntervention] = []

        problems.extend(
            self._validate_prescription_structure(
                patient=patient,
                prescription=prescription,
            )
        )

        problems.extend(
            self._detect_name_duplicates(
                patient=patient,
                prescription=prescription,
            )
        )

        problems.extend(
            self._detect_active_ingredient_duplicates(
                patient=patient,
                prescription=prescription,
            )
        )

        problems.extend(
            self._detect_allergy_conflicts(
                patient=patient,
                prescription=prescription,
            )
        )

        problems.extend(
            self._detect_missing_indications(
                patient=patient,
                prescription=prescription,
            )
        )

        problems.extend(
            self._detect_incomplete_directions(
                patient=patient,
                prescription=prescription,
            )
        )

        medications = self._build_medication_records(
            prescription,
        )

        interaction_context = (
            self._build_interaction_context(
                patient,
            )
        )

        interaction_result = (
            self.interaction_analyzer.analyze(
                medications=medications,
                context=interaction_context,
            )
        )

        interaction_problems = [
            self._interaction_to_problem(
                patient=patient,
                interaction=interaction,
            )
            for interaction in interaction_result.interactions
        ]

        problems.extend(interaction_problems)

        for problem in problems:
            interventions.append(
                self._build_intervention(
                    patient=patient,
                    problem=problem,
                )
            )

        overall_risk = self._calculate_overall_risk(
            problems,
        )

        high_risk_count = sum(
            problem.severity
            in {
                SeverityLevel.HIGH,
                SeverityLevel.CRITICAL,
            }
            for problem in problems
        )

        contraindication_count = sum(
            problem.problem_type
            == DrugTherapyProblemType.CONTRAINDICATION
            for problem in problems
        )

        return PrescriptionReviewResult(
            prescription_id=prescription.id,
            patient_id=patient.id,
            problems=problems,
            proposed_interventions=interventions,
            interactions=interaction_result.interactions,
            interaction_count=len(
                interaction_result.interactions,
            ),
            contraindication_count=contraindication_count,
            high_risk_count=high_risk_count,
            overall_risk=overall_risk,
            requires_urgent_review=(
                overall_risk
                in {
                    SeverityLevel.HIGH,
                    SeverityLevel.CRITICAL,
                }
            ),
            requires_medical_contact=(
                high_risk_count > 0
                or contraindication_count > 0
                or interaction_result.requires_prescriber_contact
            ),
            requires_referral=(
                overall_risk
                == SeverityLevel.CRITICAL
            ),
            pharmacist_validated=False,
        )

    def _validate_prescription_structure(
        self,
        patient: PatientProfile,
        prescription: PrescriptionRecord,
    ) -> list[DrugTherapyProblem]:
        """
        Verifica a integridade mínima da prescrição.
        """

        problems: list[DrugTherapyProblem] = []

        if prescription.patient_id != patient.id:
            problems.append(
                DrugTherapyProblem(
                    patient_id=patient.id,
                    title="Incompatibilidade de identificação",
                    description=(
                        "O identificador do paciente da prescrição "
                        "não corresponde ao paciente em análise."
                    ),
                    problem_type=DrugTherapyProblemType.OTHER,
                    severity=SeverityLevel.HIGH,
                    urgency=UrgencyLevel.ORANGE,
                    recommendation=(
                        "Confirmar a identidade do paciente e "
                        "a origem da prescrição antes da dispensação."
                    ),
                    detected_automatically=True,
                )
            )

        if not prescription.items:
            problems.append(
                DrugTherapyProblem(
                    patient_id=patient.id,
                    title="Prescrição sem itens",
                    description=(
                        "Nenhum medicamento foi identificado "
                        "na prescrição."
                    ),
                    problem_type=DrugTherapyProblemType.OTHER,
                    severity=SeverityLevel.HIGH,
                    urgency=UrgencyLevel.ORANGE,
                    recommendation=(
                        "Revisar o documento original e confirmar "
                        "se a extração dos itens foi concluída."
                    ),
                    detected_automatically=True,
                )
            )

        if not prescription.prescriber_name:
            problems.append(
                DrugTherapyProblem(
                    patient_id=patient.id,
                    title="Prescritor não identificado",
                    description=(
                        "O nome do prescritor não foi informado."
                    ),
                    problem_type=(
                        DrugTherapyProblemType.MONITORING_NEEDED
                    ),
                    severity=SeverityLevel.MODERATE,
                    urgency=UrgencyLevel.YELLOW,
                    recommendation=(
                        "Confirmar a identificação do prescritor "
                        "antes da validação farmacêutica."
                    ),
                    detected_automatically=True,
                )
            )

        if not prescription.prescriber_registration:
            problems.append(
                DrugTherapyProblem(
                    patient_id=patient.id,
                    title="Registro profissional ausente",
                    description=(
                        "O registro profissional do prescritor "
                        "não foi informado."
                    ),
                    problem_type=(
                        DrugTherapyProblemType.MONITORING_NEEDED
                    ),
                    severity=SeverityLevel.MODERATE,
                    urgency=UrgencyLevel.YELLOW,
                    recommendation=(
                        "Verificar o registro profissional "
                        "e a validade formal da prescrição."
                    ),
                    detected_automatically=True,
                )
            )

        return problems

    def _detect_name_duplicates(
        self,
        patient: PatientProfile,
        prescription: PrescriptionRecord,
    ) -> list[DrugTherapyProblem]:
        """
        Detecta repetição do mesmo nome de medicamento.
        """

        normalized_names = [
            self._normalize_text(item.medication_name)
            for item in prescription.items
            if item.medication_name
            and item.medication_name.strip()
        ]

        counts = Counter(normalized_names)
        problems: list[DrugTherapyProblem] = []

        for name, count in counts.items():
            if count < 2:
                continue

            problems.append(
                DrugTherapyProblem(
                    patient_id=patient.id,
                    title="Possível duplicidade de medicamento",
                    description=(
                        f"O medicamento '{name}' aparece "
                        f"{count} vezes na prescrição."
                    ),
                    problem_type=(
                        DrugTherapyProblemType.THERAPEUTIC_DUPLICATION
                    ),
                    involved_medications=[name],
                    severity=SeverityLevel.HIGH,
                    urgency=UrgencyLevel.ORANGE,
                    recommendation=(
                        "Revisar se os itens representam duplicidade, "
                        "esquemas distintos ou apresentações diferentes."
                    ),
                    detected_automatically=True,
                )
            )

        return problems

    def _detect_active_ingredient_duplicates(
        self,
        patient: PatientProfile,
        prescription: PrescriptionRecord,
    ) -> list[DrugTherapyProblem]:
        """
        Detecta repetição pelo princípio ativo informado.
        """

        active_ingredients = [
            self._normalize_text(
                item.active_ingredient,
            )
            for item in prescription.items
            if item.active_ingredient
            and item.active_ingredient.strip()
        ]

        counts = Counter(active_ingredients)
        problems: list[DrugTherapyProblem] = []

        for ingredient, count in counts.items():
            if count < 2:
                continue

            problems.append(
                DrugTherapyProblem(
                    patient_id=patient.id,
                    title="Duplicidade de princípio ativo",
                    description=(
                        f"O princípio ativo '{ingredient}' "
                        f"foi identificado em {count} itens."
                    ),
                    problem_type=(
                        DrugTherapyProblemType.THERAPEUTIC_DUPLICATION
                    ),
                    involved_medications=[ingredient],
                    severity=SeverityLevel.HIGH,
                    urgency=UrgencyLevel.ORANGE,
                    recommendation=(
                        "Confirmar se existe sobreposição terapêutica "
                        "ou associação não intencional."
                    ),
                    detected_automatically=True,
                )
            )

        return problems

    def _detect_allergy_conflicts(
        self,
        patient: PatientProfile,
        prescription: PrescriptionRecord,
    ) -> list[DrugTherapyProblem]:
        """
        Compara medicamentos e princípios ativos com alergias.

        A análise atual é textual e preliminar. Futuramente,
        deve consultar classes farmacológicas, excipientes
        e regras de reatividade cruzada.
        """

        active_allergies = [
            allergy
            for allergy in patient.allergies
            if allergy.active
        ]

        problems: list[DrugTherapyProblem] = []
        detected_keys: set[str] = set()

        for item in prescription.items:
            medication_terms = {
                self._normalize_text(
                    item.medication_name,
                ),
            }

            if item.active_ingredient:
                medication_terms.add(
                    self._normalize_text(
                        item.active_ingredient,
                    )
                )

            for allergy in active_allergies:
                if not self._matches_allergy(
                    allergy=allergy,
                    medication_terms=medication_terms,
                ):
                    continue

                duplicate_key = (
                    f"{self._normalize_text(allergy.substance)}:"
                    f"{self._normalize_text(item.medication_name)}"
                )

                if duplicate_key in detected_keys:
                    continue

                detected_keys.add(duplicate_key)

                problems.append(
                    DrugTherapyProblem(
                        patient_id=patient.id,
                        title="Possível conflito com alergia",
                        description=(
                            f"O item '{item.medication_name}' "
                            f"pode estar relacionado à alergia "
                            f"cadastrada para '{allergy.substance}'."
                        ),
                        problem_type=(
                            DrugTherapyProblemType.CONTRAINDICATION
                        ),
                        involved_medications=[
                            item.medication_name,
                        ],
                        severity=self._allergy_severity(
                            allergy,
                        ),
                        urgency=UrgencyLevel.RED,
                        recommendation=(
                            "Interromper a validação automática e "
                            "confirmar alergia, princípio ativo, "
                            "classe farmacológica e reatividade cruzada."
                        ),
                        detected_automatically=True,
                    )
                )

        return problems

    def _detect_missing_indications(
        self,
        patient: PatientProfile,
        prescription: PrescriptionRecord,
    ) -> list[DrugTherapyProblem]:
        """
        Identifica medicamentos sem indicação registrada.
        """

        problems: list[DrugTherapyProblem] = []
        detected_medications: set[str] = set()

        for item in prescription.items:
            if item.indication:
                continue

            normalized_name = self._normalize_text(
                item.medication_name,
            )

            if normalized_name in detected_medications:
                continue

            detected_medications.add(normalized_name)

            problems.append(
                DrugTherapyProblem(
                    patient_id=patient.id,
                    title="Indicação não registrada",
                    description=(
                        f"A indicação de '{item.medication_name}' "
                        "não foi informada."
                    ),
                    problem_type=(
                        DrugTherapyProblemType.MONITORING_NEEDED
                    ),
                    involved_medications=[
                        item.medication_name,
                    ],
                    severity=SeverityLevel.LOW,
                    urgency=UrgencyLevel.GREEN,
                    recommendation=(
                        "Confirmar a indicação clínica para avaliar "
                        "necessidade, efetividade e duração do tratamento."
                    ),
                    detected_automatically=True,
                )
            )

        return problems

    def _detect_incomplete_directions(
        self,
        patient: PatientProfile,
        prescription: PrescriptionRecord,
    ) -> list[DrugTherapyProblem]:
        """
        Detecta ausência de informações essenciais de uso.
        """

        problems: list[DrugTherapyProblem] = []
        detected_keys: set[str] = set()

        for item in prescription.items:
            missing_fields = self._missing_item_fields(
                item,
            )

            if not missing_fields:
                continue

            duplicate_key = (
                f"{self._normalize_text(item.medication_name)}:"
                f"{','.join(sorted(missing_fields))}"
            )

            if duplicate_key in detected_keys:
                continue

            detected_keys.add(duplicate_key)

            problems.append(
                DrugTherapyProblem(
                    patient_id=patient.id,
                    title="Posologia incompleta",
                    description=(
                        f"O item '{item.medication_name}' "
                        "possui campos ausentes: "
                        f"{', '.join(missing_fields)}."
                    ),
                    problem_type=(
                        DrugTherapyProblemType.INCORRECT_ADMINISTRATION
                    ),
                    involved_medications=[
                        item.medication_name,
                    ],
                    severity=SeverityLevel.MODERATE,
                    urgency=UrgencyLevel.YELLOW,
                    recommendation=(
                        "Confirmar os campos ausentes antes da "
                        "dispensação e da orientação ao paciente."
                    ),
                    detected_automatically=True,
                )
            )

        return problems

    @staticmethod
    def _build_medication_records(
        prescription: PrescriptionRecord,
    ) -> list[MedicationRecord]:
        """
        Converte itens prescritos para registros de medicamento.
        """

        return [
            MedicationRecord(
                name=item.medication_name,
                active_ingredient=item.active_ingredient,
                concentration=item.concentration,
                dosage_form=item.dosage_form,
                dose=item.dose,
                route=item.route,
                frequency=item.frequency,
                duration=item.duration,
                indication=item.indication,
            )
            for item in prescription.items
        ]

    @staticmethod
    def _build_interaction_context(
        patient: PatientProfile,
    ) -> InteractionAnalysisContext:
        """
        Monta o contexto clínico utilizado pelo analisador.
        """

        return InteractionAnalysisContext(
            patient_id=patient.id,
            age=patient.age,
            pregnancy=patient.pregnancy,
            breastfeeding=patient.breastfeeding,
            renal_function=patient.renal_function,
            hepatic_function=patient.hepatic_function,
            conditions=[
                condition.name
                for condition in patient.conditions
                if condition.active
            ],
            alcohol_use=patient.alcohol_use,
        )

    @staticmethod
    def _interaction_to_problem(
        patient: PatientProfile,
        interaction: DrugInteraction,
    ) -> DrugTherapyProblem:
        """
        Converte uma interação em problema farmacoterapêutico.
        """

        urgency = UrgencyLevel.GREEN

        if interaction.severity == SeverityLevel.MODERATE:
            urgency = UrgencyLevel.YELLOW

        elif interaction.severity == SeverityLevel.HIGH:
            urgency = UrgencyLevel.ORANGE

        elif interaction.severity == SeverityLevel.CRITICAL:
            urgency = UrgencyLevel.RED

        return DrugTherapyProblem(
            patient_id=patient.id,
            title="Interação medicamentosa identificada",
            description=(
                f"{interaction.medication_a} + "
                f"{interaction.medication_b}: "
                f"{interaction.clinical_effect}"
            ),
            problem_type=(
                DrugTherapyProblemType.DRUG_INTERACTION
            ),
            involved_medications=[
                interaction.medication_a,
                interaction.medication_b,
            ],
            severity=interaction.severity,
            urgency=urgency,
            recommendation=interaction.recommendation,
            detected_automatically=True,
        )

    def _build_intervention(
        self,
        patient: PatientProfile,
        problem: DrugTherapyProblem,
    ) -> PharmaceuticalIntervention:
        """
        Cria uma intervenção preliminar associada ao problema.
        """

        return PharmaceuticalIntervention(
            patient_id=patient.id,
            drug_therapy_problem_id=problem.id,
            description=(
                problem.recommendation
                or "Realizar revisão farmacêutica do problema."
            ),
            intervention_type=self._intervention_type(
                problem,
            ),
            rationale=problem.description,
            target_professional=(
                "pharmacist"
                if problem.severity
                in {
                    SeverityLevel.INFORMATIONAL,
                    SeverityLevel.LOW,
                    SeverityLevel.MODERATE,
                }
                else "prescriber"
            ),
            requires_follow_up=(
                problem.severity
                in {
                    SeverityLevel.HIGH,
                    SeverityLevel.CRITICAL,
                }
            ),
        )

    @staticmethod
    def _calculate_overall_risk(
        problems: Iterable[DrugTherapyProblem],
    ) -> SeverityLevel:
        """
        Retorna o maior risco encontrado.
        """

        risk_order = {
            SeverityLevel.INFORMATIONAL: 0,
            SeverityLevel.LOW: 1,
            SeverityLevel.MODERATE: 2,
            SeverityLevel.HIGH: 3,
            SeverityLevel.CRITICAL: 4,
        }

        highest = SeverityLevel.INFORMATIONAL

        for problem in problems:
            if (
                risk_order[problem.severity]
                > risk_order[highest]
            ):
                highest = problem.severity

        return highest

    @staticmethod
    def _normalize_text(
        value: str,
    ) -> str:
        """
        Normaliza texto para comparações simples.
        """

        return " ".join(
            value.strip().casefold().split()
        )

    def _matches_allergy(
        self,
        allergy: AllergyRecord,
        medication_terms: set[str],
    ) -> bool:
        """
        Verifica correspondência textual entre alergia e item.
        """

        allergy_term = self._normalize_text(
            allergy.substance,
        )

        return any(
            allergy_term == term
            or allergy_term in term
            or term in allergy_term
            for term in medication_terms
            if term
        )

    @staticmethod
    def _allergy_severity(
        allergy: AllergyRecord,
    ) -> SeverityLevel:
        """
        Define o risco farmacêutico associado à alergia.
        """

        if allergy.severity == SeverityLevel.CRITICAL:
            return SeverityLevel.CRITICAL

        return SeverityLevel.HIGH

    @staticmethod
    def _missing_item_fields(
        item: PrescriptionItem,
    ) -> list[str]:
        """
        Retorna os campos essenciais ausentes.
        """

        fields = {
            "dose": item.dose,
            "via": item.route,
            "frequência": item.frequency,
            "duração": item.duration,
        }

        return [
            field_name
            for field_name, value in fields.items()
            if value is None
            or not str(value).strip()
        ]

    @staticmethod
    def _intervention_type(
        problem: DrugTherapyProblem,
    ) -> str:
        """
        Sugere a categoria inicial da intervenção.
        """

        mapping = {
            DrugTherapyProblemType.THERAPEUTIC_DUPLICATION: (
                "therapy_review"
            ),
            DrugTherapyProblemType.CONTRAINDICATION: (
                "prescriber_contact"
            ),
            DrugTherapyProblemType.DRUG_INTERACTION: (
                "interaction_management"
            ),
            DrugTherapyProblemType.INCORRECT_ADMINISTRATION: (
                "schedule_review"
            ),
            DrugTherapyProblemType.MONITORING_NEEDED: (
                "monitoring_request"
            ),
        }

        return mapping.get(
            problem.problem_type,
            "other",
        )