"""
PHK Studio
Clinical Pharmacy Engine

Drug Monograph Builder.

Consolida evidências provenientes de múltiplas fontes em uma
monografia clínica estruturada, preservando rastreabilidade,
referências e necessidade de validação profissional.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from app.clinical_pharmacy_engine.knowledge_base.aggregator import (
    ClinicalKnowledgeAggregator,
)
from app.clinical_pharmacy_engine.knowledge_base.models import (
    KnowledgeDomain,
    KnowledgeEvidence,
    KnowledgeQuery,
    KnowledgeReference,
)
from app.clinical_pharmacy_engine.knowledge_base.monograph import (
    AdverseReactionEntry,
    DosageRecommendation,
    DoseAdjustment,
    DrugMonograph,
    MonographInteraction,
    MonographStatus,
)


def utc_now() -> datetime:
    """Retorna a data e hora atuais em UTC."""

    return datetime.now(timezone.utc)


class DrugMonographBuilder:
    """
    Constrói monografias clínicas a partir da Knowledge Base.

    A construção é automatizada, mas o resultado permanece
    sujeito à revisão e validação por profissional habilitado.
    """

    INTERACTION_DOMAINS = {
        KnowledgeDomain.DRUG_INTERACTION,
        KnowledgeDomain.FOOD_INTERACTION,
        KnowledgeDomain.ALCOHOL_INTERACTION,
        KnowledgeDomain.HERBAL_INTERACTION,
        KnowledgeDomain.SUPPLEMENT_INTERACTION,
    }

    def __init__(
        self,
        aggregator: ClinicalKnowledgeAggregator,
    ) -> None:
        self.aggregator = aggregator

    async def build(
        self,
        drug_name: str,
        source_codes: list[str] | None = None,
        use_cache: bool = True,
    ) -> DrugMonograph:
        """
        Pesquisa múltiplas fontes e constrói uma monografia.

        Args:
            drug_name:
                Nome do medicamento ou princípio ativo.

            source_codes:
                Fontes específicas que deverão ser consultadas.
                Quando omitido, o agregador utiliza todas as
                fontes disponíveis.

            use_cache:
                Define se o cache da Knowledge Base será usado.

        Returns:
            Monografia clínica consolidada.

        Raises:
            ValueError:
                Quando o nome informado for inválido.
        """

        normalized_name = self._normalize_text(
            drug_name,
        )

        if len(normalized_name) < 2:
            raise ValueError(
                "O nome do medicamento deve possuir "
                "pelo menos dois caracteres.",
            )

        query = KnowledgeQuery(
            term=drug_name.strip(),
            source_codes=source_codes or [],
            domains=[
                KnowledgeDomain.DRUG_MONOGRAPH,
                KnowledgeDomain.INDICATION,
                KnowledgeDomain.CONTRAINDICATION,
                KnowledgeDomain.DOSAGE,
                KnowledgeDomain.RENAL_ADJUSTMENT,
                KnowledgeDomain.HEPATIC_ADJUSTMENT,
                KnowledgeDomain.DRUG_INTERACTION,
                KnowledgeDomain.FOOD_INTERACTION,
                KnowledgeDomain.ALCOHOL_INTERACTION,
                KnowledgeDomain.HERBAL_INTERACTION,
                KnowledgeDomain.SUPPLEMENT_INTERACTION,
                KnowledgeDomain.ADVERSE_REACTION,
                KnowledgeDomain.PHARMACOVIGILANCE,
                KnowledgeDomain.PREGNANCY,
                KnowledgeDomain.LACTATION,
                KnowledgeDomain.PHARMACOKINETICS,
                KnowledgeDomain.PHARMACODYNAMICS,
                KnowledgeDomain.LABORATORY_INTERFERENCE,
                KnowledgeDomain.REGULATORY_ALERT,
                KnowledgeDomain.OTHER,
            ],
            limit_per_source=50,
        )

        search_result = await self.aggregator.search(
            query=query,
            use_cache=use_cache,
        )

        evidences = list(
            search_result.evidences,
        )

        monograph = DrugMonograph(
            preferred_name=drug_name.strip(),
            evidences=evidences,
            source_codes=list(
                search_result.successful_sources,
            ),
        )

        for evidence in evidences:
            self._apply_evidence(
                monograph=monograph,
                evidence=evidence,
            )

        self._finalize_monograph(
            monograph=monograph,
            search_result=search_result,
        )

        return monograph

    def _apply_evidence(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """
        Incorpora uma evidência normalizada à monografia.

        A classificação principal ocorre pelo KnowledgeDomain.
        Metadados e títulos são usados como classificação
        secundária para seções do DailyMed.
        """

        self._add_unique(
            monograph.source_codes,
            evidence.source_code,
        )

        monograph.references.extend(
            evidence.references,
        )

        self._apply_identifiers(
            monograph,
            evidence,
        )

        self._apply_structural_metadata(
            monograph,
            evidence.metadata,
        )

        domain = evidence.domain

        normalized_title = self._normalize_text(
            evidence.title or "",
        )

        if domain == KnowledgeDomain.DRUG_MONOGRAPH:
            self._apply_general_monograph(
                monograph,
                evidence,
            )

            self._apply_special_population_section(
                monograph,
                evidence,
                normalized_title,
            )

            return

        if domain == KnowledgeDomain.INDICATION:
            self._add_unique(
                monograph.indications,
                evidence.summary,
            )
            return

        if domain == KnowledgeDomain.CONTRAINDICATION:
            self._add_unique(
                monograph.contraindications,
                evidence.summary,
            )
            return

        if domain == KnowledgeDomain.DOSAGE:
            self._apply_dosage(
                monograph,
                evidence,
            )
            return

        if domain == KnowledgeDomain.PHARMACODYNAMICS:
            self._apply_pharmacodynamics(
                monograph,
                evidence,
                normalized_title,
            )
            return

        if domain == KnowledgeDomain.PHARMACOKINETICS:
            self._apply_pharmacokinetics(
                monograph,
                evidence,
            )
            return

        if domain in self.INTERACTION_DOMAINS:
            self._apply_interaction(
                monograph,
                evidence,
            )
            return

        if domain == KnowledgeDomain.ADVERSE_REACTION:
            self._apply_adverse_reaction(
                monograph,
                evidence,
            )
            return

        if domain == KnowledgeDomain.RENAL_ADJUSTMENT:
            self._apply_renal_adjustment(
                monograph,
                evidence,
            )
            return

        if domain == KnowledgeDomain.HEPATIC_ADJUSTMENT:
            self._apply_hepatic_adjustment(
                monograph,
                evidence,
            )
            return

        if domain == KnowledgeDomain.PREGNANCY:
            self._add_unique(
                monograph.pregnancy_information,
                evidence.summary,
            )
            return

        if domain == KnowledgeDomain.LACTATION:
            self._add_unique(
                monograph.lactation_information,
                evidence.summary,
            )
            return

        if domain == KnowledgeDomain.REGULATORY_ALERT:
            self._apply_regulatory_alert(
                monograph,
                evidence,
                normalized_title,
            )
            return

        if domain == KnowledgeDomain.LABORATORY_INTERFERENCE:
            self._add_unique(
                monograph.warnings,
                evidence.summary,
            )

            self._store_metadata_evidence(
                monograph=monograph,
                key="laboratory_interferences",
                evidence=evidence,
            )
            return

        if domain == KnowledgeDomain.PHARMACOVIGILANCE:
            self._apply_pharmacovigilance(
                monograph,
                evidence,
            )
            return

        if domain == KnowledgeDomain.OTHER:
            self._apply_other_section(
                monograph,
                evidence,
                normalized_title,
            )
            return

        self._store_metadata_evidence(
            monograph=monograph,
            key="unmapped_evidences",
            evidence=evidence,
        )

    def _apply_general_monograph(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Aplica dados gerais, nomes e sinônimos."""

        if (
            evidence.summary
            and not monograph.description
        ):
            monograph.description = evidence.summary

        self._add_unique(
            monograph.synonyms,
            evidence.subject,
        )

        for agent in evidence.related_agents:
            self._add_unique(
                monograph.synonyms,
                agent,
            )

        generic_name = self._optional_string(
            evidence.metadata.get(
                "generic_name",
            )
        )

        if (
            generic_name
            and not monograph.generic_name
        ):
            monograph.generic_name = generic_name

        brand_names = evidence.metadata.get(
            "brand_names",
        )

        if isinstance(brand_names, list):
            for brand_name in brand_names:
                self._add_unique(
                    monograph.brand_names,
                    self._optional_string(
                        brand_name,
                    ),
                )

        active_ingredients = evidence.metadata.get(
            "active_ingredients",
        )

        if isinstance(active_ingredients, list):
            for ingredient in active_ingredients:
                self._add_unique(
                    monograph.active_ingredients,
                    self._optional_string(
                        ingredient,
                    ),
                )

        therapeutic_classes = evidence.metadata.get(
            "therapeutic_classes",
        )

        if isinstance(therapeutic_classes, list):
            for therapeutic_class in therapeutic_classes:
                self._add_unique(
                    monograph.therapeutic_classes,
                    self._optional_string(
                        therapeutic_class,
                    ),
                )

        pharmacological_classes = evidence.metadata.get(
            "pharmacological_classes",
        )

        if isinstance(
            pharmacological_classes,
            list,
        ):
            for pharmacological_class in pharmacological_classes:
                self._add_unique(
                    monograph.pharmacological_classes,
                    self._optional_string(
                        pharmacological_class,
                    ),
                )

    def _apply_pharmacodynamics(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
        normalized_title: str,
    ) -> None:
        """Aplica farmacodinâmica e mecanismo de ação."""

        self._add_unique(
            monograph.pharmacodynamics,
            evidence.summary,
        )

        if (
            evidence.mechanism
            and not monograph.mechanism_of_action
        ):
            monograph.mechanism_of_action = (
                evidence.mechanism
            )

        if (
            "mechanism of action" in normalized_title
            and not monograph.mechanism_of_action
        ):
            monograph.mechanism_of_action = (
                evidence.summary
            )

    def _apply_dosage(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """
        Converte uma seção posológica em recomendação.

        Nesta etapa, o texto completo continua preservado.
        O Dose Engine poderá futuramente separar dose, via,
        frequência, duração e limites máximos.
        """

        indication = (
            self._optional_string(
                evidence.metadata.get(
                    "indication",
                )
            )
            or "general"
        )

        population = (
            self._optional_string(
                evidence.metadata.get(
                    "population",
                )
            )
            or "unspecified"
        )

        recommendation_text = (
            evidence.recommendation
            or evidence.summary
        )

        entry = DosageRecommendation(
            indication=indication,
            population=population,
            dose=recommendation_text,
            route=self._optional_string(
                evidence.metadata.get(
                    "route",
                )
            ),
            frequency=self._optional_string(
                evidence.metadata.get(
                    "frequency",
                )
            ),
            duration=self._optional_string(
                evidence.metadata.get(
                    "duration",
                )
            ),
            maximum_dose=self._optional_string(
                evidence.metadata.get(
                    "maximum_dose",
                )
            ),
            notes=self._metadata_string_list(
                evidence.metadata.get(
                    "notes",
                )
            ),
            references=list(
                evidence.references,
            ),
        )

        if not self._dosage_exists(
            monograph.dosage_recommendations,
            entry,
        ):
            monograph.dosage_recommendations.append(
                entry,
            )

    def _apply_renal_adjustment(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Adiciona recomendação de ajuste renal."""

        adjustment = DoseAdjustment(
            condition=(
                self._optional_string(
                    evidence.metadata.get(
                        "condition",
                    )
                )
                or "renal_impairment"
            ),
            recommendation=(
                evidence.recommendation
                or evidence.summary
            ),
            threshold=self._optional_string(
                evidence.metadata.get(
                    "threshold",
                )
            ),
            parameter=self._optional_string(
                evidence.metadata.get(
                    "parameter",
                )
            ),
            severity=evidence.severity,
            references=list(
                evidence.references,
            ),
        )

        if not self._adjustment_exists(
            monograph.renal_adjustments,
            adjustment,
        ):
            monograph.renal_adjustments.append(
                adjustment,
            )

    def _apply_hepatic_adjustment(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Adiciona recomendação de ajuste hepático."""

        adjustment = DoseAdjustment(
            condition=(
                self._optional_string(
                    evidence.metadata.get(
                        "condition",
                    )
                )
                or "hepatic_impairment"
            ),
            recommendation=(
                evidence.recommendation
                or evidence.summary
            ),
            threshold=self._optional_string(
                evidence.metadata.get(
                    "threshold",
                )
            ),
            parameter=self._optional_string(
                evidence.metadata.get(
                    "parameter",
                )
            ),
            severity=evidence.severity,
            references=list(
                evidence.references,
            ),
        )

        if not self._adjustment_exists(
            monograph.hepatic_adjustments,
            adjustment,
        ):
            monograph.hepatic_adjustments.append(
                adjustment,
            )

    @staticmethod
    def _apply_identifiers(
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Incorpora identificadores normalizados."""

        identifiers = evidence.raw_identifiers

        pubchem_cid = identifiers.get(
            "pubchem_cid",
        )

        if pubchem_cid:
            value = str(pubchem_cid)

            if value not in monograph.pubchem_cids:
                monograph.pubchem_cids.append(
                    value,
                )

        rxnorm_id = (
            identifiers.get("rxnorm_id")
            or identifiers.get("rxcui")
        )

        if rxnorm_id:
            value = str(rxnorm_id)

            if value not in monograph.rxnorm_ids:
                monograph.rxnorm_ids.append(
                    value,
                )

        atc_code = identifiers.get(
            "atc_code",
        )

        if atc_code:
            value = str(atc_code)

            if value not in monograph.atc_codes:
                monograph.atc_codes.append(
                    value,
                )

        if (
            identifiers.get("inchi")
            and not monograph.inchi
        ):
            monograph.inchi = str(
                identifiers["inchi"],
            )

        if (
            identifiers.get("inchikey")
            and not monograph.inchikey
        ):
            monograph.inchikey = str(
                identifiers["inchikey"],
            )

    @staticmethod
    def _apply_structural_metadata(
        monograph: DrugMonograph,
        metadata: dict[str, Any],
    ) -> None:
        """Aplica propriedades químicas e estruturais."""

        mapping = {
            "molecular_formula": "molecular_formula",
            "molecular_weight": "molecular_weight",
            "canonical_smiles": "canonical_smiles",
            "isomeric_smiles": "isomeric_smiles",
            "inchi": "inchi",
            "inchikey": "inchikey",
        }

        for metadata_key, attribute_name in mapping.items():
            value = metadata.get(
                metadata_key,
            )

            if (
                value is not None
                and getattr(
                    monograph,
                    attribute_name,
                ) is None
            ):
                setattr(
                    monograph,
                    attribute_name,
                    value,
                )

        chemical_metadata = {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "molecular_formula",
                "molecular_weight",
                "canonical_smiles",
                "isomeric_smiles",
                "inchi",
                "inchikey",
            }
        }

        if chemical_metadata:
            monograph.metadata.setdefault(
                "source_metadata",
                {},
            ).update(
                chemical_metadata,
            )

    @staticmethod
    def _apply_pharmacokinetics(
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Aplica dados farmacocinéticos."""

        key = (
            evidence.metadata.get(
                "pharmacokinetic_parameter",
            )
            or evidence.title
            or f"entry_{len(monograph.pharmacokinetics) + 1}"
        )

        normalized_key = str(key).strip()

        if not normalized_key:
            normalized_key = (
                f"entry_{len(monograph.pharmacokinetics) + 1}"
            )

        monograph.pharmacokinetics[
            normalized_key
        ] = evidence.summary

    def _apply_interaction(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Adiciona interação clínica à monografia."""

        interacting_agent = self._resolve_interacting_agent(
            monograph=monograph,
            evidence=evidence,
        )

        interaction = MonographInteraction(
            interacting_agent=interacting_agent,
            interaction_type=evidence.domain.value,
            severity=evidence.severity,
            mechanism=evidence.mechanism,
            clinical_effect=evidence.clinical_effect,
            recommendation=evidence.recommendation,
            evidence_level=(
                evidence.evidence_strength.value
            ),
            references=list(
                evidence.references,
            ),
        )

        if not self._interaction_exists(
            monograph.interactions,
            interaction,
        ):
            monograph.interactions.append(
                interaction,
            )

    @staticmethod
    def _apply_adverse_reaction(
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Adiciona reação adversa à monografia."""

        reaction = (
            evidence.metadata.get(
                "reaction",
            )
            or evidence.title
            or evidence.summary
        )

        entry = AdverseReactionEntry(
            reaction=str(reaction),
            frequency=DrugMonographBuilder._optional_string(
                evidence.metadata.get(
                    "frequency",
                )
            ),
            severity=evidence.severity,
            seriousness=DrugMonographBuilder._optional_string(
                evidence.metadata.get(
                    "seriousness",
                )
            ),
            onset=DrugMonographBuilder._optional_string(
                evidence.metadata.get(
                    "onset",
                )
            ),
            outcome=DrugMonographBuilder._optional_string(
                evidence.metadata.get(
                    "outcome",
                )
            ),
            source_code=evidence.source_code,
            references=list(
                evidence.references,
            ),
        )

        comparison_key = (
            entry.reaction.casefold(),
            (
                entry.source_code.casefold()
                if entry.source_code
                else ""
            ),
        )

        existing_keys = {
            (
                item.reaction.casefold(),
                (
                    item.source_code.casefold()
                    if item.source_code
                    else ""
                ),
            )
            for item in monograph.adverse_reactions
        }

        if comparison_key not in existing_keys:
            monograph.adverse_reactions.append(
                entry,
            )

    def _apply_regulatory_alert(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
        normalized_title: str,
    ) -> None:
        """Classifica alertas regulatórios e advertências."""

        text = (
            evidence.recommendation
            or evidence.summary
        )

        if (
            "boxed warning" in normalized_title
            or "warning box" in normalized_title
        ):
            self._add_unique(
                monograph.boxed_warnings,
                text,
            )

            self._add_unique(
                monograph.regulatory_alerts,
                text,
            )
            return

        if "precaution" in normalized_title:
            self._add_unique(
                monograph.precautions,
                text,
            )

        self._add_unique(
            monograph.warnings,
            text,
        )

        self._add_unique(
            monograph.regulatory_alerts,
            text,
        )

    def _apply_special_population_section(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
        normalized_title: str,
    ) -> None:
        """Classifica informações de populações especiais."""

        if (
            "pregnan" in normalized_title
            or "teratogenic" in normalized_title
        ):
            self._add_unique(
                monograph.pregnancy_information,
                evidence.summary,
            )

        if (
            "lactation" in normalized_title
            or "breast-feeding" in normalized_title
            or "breastfeeding" in normalized_title
            or "nursing mother" in normalized_title
        ):
            self._add_unique(
                monograph.lactation_information,
                evidence.summary,
            )

        if (
            "pediatric" in normalized_title
            or "children" in normalized_title
        ):
            self._add_unique(
                monograph.pediatric_information,
                evidence.summary,
            )

        if (
            "geriatric" in normalized_title
            or "elderly" in normalized_title
            or "older adult" in normalized_title
        ):
            self._add_unique(
                monograph.geriatric_information,
                evidence.summary,
            )

    def _apply_other_section(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
        normalized_title: str,
    ) -> None:
        """
        Classifica seções não mapeadas explicitamente.

        Isso é especialmente útil para rótulos OTC, cujos
        títulos podem não seguir o formato de bulas completas.
        """

        text = evidence.summary

        if (
            "pregnan" in normalized_title
            or "pregnant" in normalized_title
        ):
            self._add_unique(
                monograph.pregnancy_information,
                text,
            )
            return

        if (
            "breast-feeding" in normalized_title
            or "breastfeeding" in normalized_title
            or "lactation" in normalized_title
            or "nursing mother" in normalized_title
        ):
            self._add_unique(
                monograph.lactation_information,
                text,
            )
            return

        if (
            "pediatric" in normalized_title
            or "children" in normalized_title
        ):
            self._add_unique(
                monograph.pediatric_information,
                text,
            )
            return

        if (
            "geriatric" in normalized_title
            or "elderly" in normalized_title
        ):
            self._add_unique(
                monograph.geriatric_information,
                text,
            )
            return

        if (
            "storage" in normalized_title
            or "store at" in normalized_title
            or "handling" in normalized_title
            or "how supplied" in normalized_title
        ):
            self._add_unique(
                monograph.storage_information,
                text,
            )
            return

        if (
            "patient counseling" in normalized_title
            or "ask a doctor" in normalized_title
            or "directions" in normalized_title
            or "instructions for use" in normalized_title
        ):
            self._add_unique(
                monograph.patient_counseling,
                text,
            )
            return

        if (
            "warning" in normalized_title
            or "stop use" in normalized_title
            or "do not use" in normalized_title
        ):
            self._add_unique(
                monograph.warnings,
                text,
            )
            return

        if "inactive ingredient" in normalized_title:
            inactive_ingredients = (
                monograph.metadata.setdefault(
                    "inactive_ingredients",
                    [],
                )
            )

            self._add_unique(
                inactive_ingredients,
                text,
            )
            return

        if "active ingredient" in normalized_title:
            self._add_unique(
                monograph.active_ingredients,
                text,
            )
            return

        if (
            "clinical studies" in normalized_title
            or "clinical trial" in normalized_title
        ):
            self._store_metadata_evidence(
                monograph=monograph,
                key="clinical_studies",
                evidence=evidence,
            )
            return

        self._store_metadata_evidence(
            monograph=monograph,
            key="other_sections",
            evidence=evidence,
        )

    def _apply_pharmacovigilance(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Armazena dados gerais de farmacovigilância."""

        self._store_metadata_evidence(
            monograph=monograph,
            key="pharmacovigilance",
            evidence=evidence,
        )

        if evidence.severity:
            self._add_unique(
                monograph.warnings,
                evidence.summary,
            )

    def _finalize_monograph(
        self,
        monograph: DrugMonograph,
        search_result: Any,
    ) -> None:
        """Normaliza, deduplica e finaliza a monografia."""

        monograph.references = (
            self._deduplicate_references(
                monograph.references,
            )
        )

        monograph.source_codes = sorted(
            set(
                source_code
                for source_code in monograph.source_codes
                if source_code
            )
        )

        monograph.synonyms = self._deduplicate_strings(
            monograph.synonyms,
        )

        monograph.brand_names = self._deduplicate_strings(
            monograph.brand_names,
        )

        monograph.active_ingredients = (
            self._deduplicate_strings(
                monograph.active_ingredients,
            )
        )

        monograph.therapeutic_classes = (
            self._deduplicate_strings(
                monograph.therapeutic_classes,
            )
        )

        monograph.pharmacological_classes = (
            self._deduplicate_strings(
                monograph.pharmacological_classes,
            )
        )

        monograph.atc_codes = self._deduplicate_strings(
            monograph.atc_codes,
        )

        monograph.rxnorm_ids = self._deduplicate_strings(
            monograph.rxnorm_ids,
        )

        monograph.pubchem_cids = (
            self._deduplicate_strings(
                monograph.pubchem_cids,
            )
        )

        monograph.status = self._calculate_status(
            monograph,
        )

        monograph.requires_professional_review = True
        monograph.updated_at = utc_now()

        monograph.metadata[
            "failed_sources"
        ] = getattr(
            search_result,
            "failed_sources",
            {},
        )

        monograph.metadata[
            "unavailable_sources"
        ] = getattr(
            search_result,
            "unavailable_sources",
            [],
        )

        monograph.metadata[
            "warnings"
        ] = getattr(
            search_result,
            "warnings",
            [],
        )

        monograph.metadata[
            "successful_sources"
        ] = list(
            getattr(
                search_result,
                "successful_sources",
                [],
            )
        )

        monograph.metadata[
            "evidence_count"
        ] = monograph.evidence_count

        monograph.metadata[
            "source_count"
        ] = monograph.source_count

    @staticmethod
    def _calculate_status(
        monograph: DrugMonograph,
    ) -> MonographStatus:
        """
        Calcula a completude estrutural da monografia.

        REVIEW_REQUIRED indica boa cobertura automática,
        mas não substitui validação clínica profissional.
        """

        if monograph.evidence_count == 0:
            return MonographStatus.EMPTY

        sections = {
            "description": bool(
                monograph.description
            ),
            "identity": bool(
                monograph.rxnorm_ids
                or monograph.pubchem_cids
            ),
            "chemistry": bool(
                monograph.molecular_formula
                or monograph.canonical_smiles
            ),
            "indications": bool(
                monograph.indications
            ),
            "dosage": bool(
                monograph.dosage_recommendations
            ),
            "contraindications": bool(
                monograph.contraindications
            ),
            "warnings": bool(
                monograph.warnings
                or monograph.boxed_warnings
            ),
            "interactions": bool(
                monograph.interactions
            ),
            "adverse_reactions": bool(
                monograph.adverse_reactions
            ),
            "pharmacology": bool(
                monograph.pharmacodynamics
                or monograph.pharmacokinetics
                or monograph.mechanism_of_action
            ),
            "special_populations": bool(
                monograph.pregnancy_information
                or monograph.lactation_information
                or monograph.pediatric_information
                or monograph.geriatric_information
            ),
            "references": bool(
                monograph.references
            ),
        }

        completed_sections = sum(
            sections.values()
        )

        completion_ratio = (
            completed_sections
            / len(sections)
        )

        monograph.metadata[
            "completion_sections"
        ] = sections

        monograph.metadata[
            "completed_section_count"
        ] = completed_sections

        monograph.metadata[
            "total_section_count"
        ] = len(sections)

        monograph.metadata[
            "completion_ratio"
        ] = round(
            completion_ratio,
            4,
        )

        if completion_ratio >= 0.75:
            return MonographStatus.REVIEW_REQUIRED

        return MonographStatus.PARTIAL

    def _resolve_interacting_agent(
        self,
        monograph: DrugMonograph,
        evidence: KnowledgeEvidence,
    ) -> str:
        """Tenta identificar o agente que interage com o fármaco."""

        preferred_name = self._normalize_text(
            monograph.display_name,
        )

        for agent in evidence.related_agents:
            normalized_agent = self._normalize_text(
                agent,
            )

            if (
                normalized_agent
                and normalized_agent != preferred_name
            ):
                return agent.strip()

        metadata_agent = (
            evidence.metadata.get(
                "interacting_agent",
            )
            or evidence.metadata.get(
                "agent",
            )
        )

        return (
            self._optional_string(
                metadata_agent,
            )
            or "unspecified"
        )

    @staticmethod
    def _store_metadata_evidence(
        monograph: DrugMonograph,
        key: str,
        evidence: KnowledgeEvidence,
    ) -> None:
        """Preserva uma evidência em uma seção de metadados."""

        target = monograph.metadata.setdefault(
            key,
            [],
        )

        record = {
            "source_code": evidence.source_code,
            "domain": evidence.domain.value,
            "title": evidence.title,
            "summary": evidence.summary,
            "severity": evidence.severity,
            "recommendation": evidence.recommendation,
            "metadata": dict(
                evidence.metadata,
            ),
        }

        if record not in target:
            target.append(record)

    @staticmethod
    def _interaction_exists(
        interactions: list[MonographInteraction],
        candidate: MonographInteraction,
    ) -> bool:
        """Verifica duplicidade de interação."""

        candidate_key = (
            candidate.interacting_agent.casefold(),
            candidate.interaction_type.casefold(),
            (
                candidate.recommendation.casefold()
                if candidate.recommendation
                else ""
            ),
        )

        existing_keys = {
            (
                interaction.interacting_agent.casefold(),
                interaction.interaction_type.casefold(),
                (
                    interaction.recommendation.casefold()
                    if interaction.recommendation
                    else ""
                ),
            )
            for interaction in interactions
        }

        return candidate_key in existing_keys

    @staticmethod
    def _dosage_exists(
        dosages: list[DosageRecommendation],
        candidate: DosageRecommendation,
    ) -> bool:
        """Verifica duplicidade de recomendação posológica."""

        candidate_key = (
            candidate.indication.casefold(),
            candidate.population.casefold(),
            (
                candidate.dose.casefold()
                if candidate.dose
                else ""
            ),
        )

        existing_keys = {
            (
                dosage.indication.casefold(),
                dosage.population.casefold(),
                (
                    dosage.dose.casefold()
                    if dosage.dose
                    else ""
                ),
            )
            for dosage in dosages
        }

        return candidate_key in existing_keys

    @staticmethod
    def _adjustment_exists(
        adjustments: list[DoseAdjustment],
        candidate: DoseAdjustment,
    ) -> bool:
        """Verifica duplicidade de ajuste posológico."""

        candidate_key = (
            candidate.condition.casefold(),
            candidate.recommendation.casefold(),
        )

        existing_keys = {
            (
                adjustment.condition.casefold(),
                adjustment.recommendation.casefold(),
            )
            for adjustment in adjustments
        }

        return candidate_key in existing_keys

    @staticmethod
    def _metadata_string_list(
        value: Any,
    ) -> list[str]:
        """Normaliza metadado em lista textual."""

        if value is None:
            return []

        if isinstance(value, list):
            return [
                normalized
                for item in value
                if (
                    normalized
                    := DrugMonographBuilder._optional_string(
                        item,
                    )
                )
            ]

        normalized = (
            DrugMonographBuilder._optional_string(
                value,
            )
        )

        return [normalized] if normalized else []

    @staticmethod
    def _add_unique(
        target: list[str],
        value: str | None,
    ) -> None:
        """Adiciona texto somente se ainda não estiver presente."""

        if not value:
            return

        normalized_value = " ".join(
            value.strip().split()
        )

        if not normalized_value:
            return

        existing = {
            item.casefold()
            for item in target
        }

        if normalized_value.casefold() not in existing:
            target.append(
                normalized_value,
            )

    @staticmethod
    def _deduplicate_strings(
        values: Iterable[str],
    ) -> list[str]:
        """Remove textos duplicados preservando a ordem."""

        unique: list[str] = []
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
            unique.append(normalized)

        return unique

    @staticmethod
    def _deduplicate_references(
        references: Iterable[KnowledgeReference],
    ) -> list[KnowledgeReference]:
        """Remove referências duplicadas preservando proveniência."""

        unique: list[KnowledgeReference] = []
        seen: set[str] = set()

        for reference in references:
            if reference.doi:
                key = (
                    f"doi:{reference.doi.casefold()}"
                )

            elif reference.pmid:
                key = (
                    f"pmid:{reference.pmid}"
                )

            elif reference.url:
                key = (
                    f"url:{reference.url.casefold()}"
                )

            elif reference.title:
                key = (
                    "title:"
                    + " ".join(
                        reference.title
                        .casefold()
                        .split()
                    )
                )

            else:
                key = repr(reference)

            if key in seen:
                continue

            seen.add(key)
            unique.append(reference)

        return unique

    @staticmethod
    def _optional_string(
        value: object,
    ) -> str | None:
        """Converte valor opcional em texto limpo."""

        if value is None:
            return None

        normalized = " ".join(
            str(value).strip().split()
        )

        return normalized or None

    @staticmethod
    def _normalize_text(
        value: str | None,
    ) -> str:
        """Normaliza texto para comparações internas."""

        if not value:
            return ""

        return " ".join(
            value.strip().casefold().split()
        )