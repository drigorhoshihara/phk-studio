"""
Regras iniciais de interação do PHK Studio.

Importante:
estas regras são demonstrativas e deverão ser validadas,
versionadas e vinculadas a fontes clínicas oficiais.
"""

from __future__ import annotations

from app.clinical_pharmacy_engine.interactions.models import (
    InteractionEvidenceLevel,
    InteractionManagement,
    InteractionRule,
)
from app.clinical_pharmacy_engine.models import (
    InteractionType,
    SeverityLevel,
)


DRUG_DRUG_RULES: list[InteractionRule] = [
    InteractionRule(
        agent_a="warfarina",
        agent_b="amiodarona",
        interaction_type=InteractionType.PHARMACOKINETIC,
        severity=SeverityLevel.HIGH,
        mechanism=(
            "A amiodarona pode reduzir o metabolismo "
            "da warfarina e aumentar sua exposição."
        ),
        clinical_effect=(
            "Aumento do INR e do risco de sangramento."
        ),
        recommendation=(
            "Revisar a associação, considerar ajuste "
            "da anticoagulação e monitorar INR."
        ),
        evidence_level=InteractionEvidenceLevel.HIGH,
        management=InteractionManagement.ADJUST_DOSE,
        monitoring_parameters=[
            "INR",
            "sinais de sangramento",
        ],
    ),
    InteractionRule(
        agent_a="claritromicina",
        agent_b="sinvastatina",
        interaction_type=InteractionType.PHARMACOKINETIC,
        severity=SeverityLevel.CRITICAL,
        mechanism=(
            "Inibição do CYP3A4, com aumento da "
            "exposição à sinvastatina."
        ),
        clinical_effect=(
            "Maior risco de miopatia e rabdomiólise."
        ),
        recommendation=(
            "Evitar a associação e discutir alternativa "
            "terapêutica com o prescritor."
        ),
        evidence_level=InteractionEvidenceLevel.HIGH,
        management=InteractionManagement.AVOID_COMBINATION,
        monitoring_parameters=[
            "dor muscular",
            "CK",
            "função renal",
        ],
        contraindicated=True,
    ),
    InteractionRule(
        agent_a="sertralina",
        agent_b="tramadol",
        interaction_type=InteractionType.PHARMACODYNAMIC,
        severity=SeverityLevel.HIGH,
        mechanism=(
            "Somação de atividade serotoninérgica."
        ),
        clinical_effect=(
            "Maior risco de toxicidade serotoninérgica "
            "e redução do limiar convulsivo."
        ),
        recommendation=(
            "Revisar a necessidade da associação e "
            "monitorar sinais neurológicos."
        ),
        evidence_level=InteractionEvidenceLevel.MODERATE,
        management=InteractionManagement.CONSIDER_ALTERNATIVE,
        monitoring_parameters=[
            "agitação",
            "clônus",
            "hiperreflexia",
            "hipertermia",
            "convulsões",
        ],
    ),
]


DRUG_FOOD_RULES: list[InteractionRule] = [
    InteractionRule(
        agent_a="levotiroxina",
        agent_b="leite",
        interaction_type=InteractionType.MEDICATION_FOOD,
        severity=SeverityLevel.MODERATE,
        mechanism=(
            "Minerais presentes no alimento podem "
            "reduzir a absorção da levotiroxina."
        ),
        clinical_effect=(
            "Possível redução do efeito terapêutico."
        ),
        recommendation=(
            "Orientar administração em jejum e separar "
            "o medicamento de alimentos ricos em cálcio."
        ),
        evidence_level=InteractionEvidenceLevel.MODERATE,
        management=(
            InteractionManagement.SEPARATE_ADMINISTRATION
        ),
        monitoring_parameters=[
            "TSH",
            "T4 livre",
        ],
    ),
    InteractionRule(
        agent_a="warfarina",
        agent_b="vitamina k",
        interaction_type=InteractionType.MEDICATION_FOOD,
        severity=SeverityLevel.MODERATE,
        mechanism=(
            "A vitamina K pode antagonizar o efeito "
            "anticoagulante da warfarina."
        ),
        clinical_effect=(
            "Redução do INR e do efeito anticoagulante."
        ),
        recommendation=(
            "Manter ingestão dietética consistente e "
            "monitorar INR em mudanças alimentares."
        ),
        evidence_level=InteractionEvidenceLevel.HIGH,
        management=InteractionManagement.MONITOR,
        monitoring_parameters=[
            "INR",
        ],
    ),
]


DRUG_ALCOHOL_RULES: list[InteractionRule] = [
    InteractionRule(
        agent_a="metronidazol",
        agent_b="alcool",
        interaction_type=InteractionType.MEDICATION_ALCOHOL,
        severity=SeverityLevel.HIGH,
        mechanism=(
            "Possível alteração do metabolismo do álcool."
        ),
        clinical_effect=(
            "Náusea, vômito, rubor, cefaleia e hipotensão."
        ),
        recommendation=(
            "Evitar bebidas alcoólicas durante o tratamento "
            "e pelo período recomendado após a última dose."
        ),
        evidence_level=InteractionEvidenceLevel.MODERATE,
        management=InteractionManagement.AVOID_COMBINATION,
    ),
]


DRUG_HERBAL_RULES: list[InteractionRule] = [
    InteractionRule(
        agent_a="warfarina",
        agent_b="ginkgo biloba",
        interaction_type=InteractionType.MEDICATION_HERBAL,
        severity=SeverityLevel.HIGH,
        mechanism=(
            "Possível somação de efeitos sobre hemostasia "
            "e agregação plaquetária."
        ),
        clinical_effect=(
            "Aumento do risco hemorrágico."
        ),
        recommendation=(
            "Evitar automedicação com o fitoterápico "
            "e revisar o risco individual."
        ),
        evidence_level=InteractionEvidenceLevel.LOW,
        management=InteractionManagement.CONSIDER_ALTERNATIVE,
        monitoring_parameters=[
            "sinais de sangramento",
            "INR",
        ],
    ),
]


DRUG_SUPPLEMENT_RULES: list[InteractionRule] = [
    InteractionRule(
        agent_a="levotiroxina",
        agent_b="calcio",
        interaction_type=InteractionType.MEDICATION_SUPPLEMENT,
        severity=SeverityLevel.MODERATE,
        mechanism=(
            "Formação de complexos e redução da absorção."
        ),
        clinical_effect=(
            "Possível redução da efetividade da levotiroxina."
        ),
        recommendation=(
            "Separar os horários de administração e "
            "monitorar função tireoidiana."
        ),
        evidence_level=InteractionEvidenceLevel.HIGH,
        management=(
            InteractionManagement.SEPARATE_ADMINISTRATION
        ),
        monitoring_parameters=[
            "TSH",
            "T4 livre",
        ],
    ),
]


DRUG_LAB_RULES: list[InteractionRule] = [
    InteractionRule(
        agent_a="biotina",
        agent_b="troponina",
        interaction_type=InteractionType.MEDICATION_LABORATORY,
        severity=SeverityLevel.HIGH,
        mechanism=(
            "Interferência em imunensaios dependentes "
            "de biotina."
        ),
        clinical_effect=(
            "Possibilidade de resultado laboratorial incorreto."
        ),
        recommendation=(
            "Informar o laboratório sobre o uso de biotina "
            "e avaliar suspensão conforme protocolo aplicável."
        ),
        evidence_level=InteractionEvidenceLevel.HIGH,
        management=InteractionManagement.URGENT_REVIEW,
    ),
]