"""
PHK Studio
Clinical Pharmacy Engine

OpenFDA Knowledge Source.

Integra dados públicos da FDA provenientes de:

- drug labeling;
- FAERS adverse events;
- drug enforcement recalls.

Os dados são destinados a suporte informacional e exigem
revisão profissional antes de qualquer aplicação clínica.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from datetime import datetime
from typing import Any

import httpx

from app.clinical_pharmacy_engine.knowledge_base.models import (
    EvidenceStrength,
    KnowledgeDomain,
    KnowledgeEvidence,
    KnowledgeQuery,
    KnowledgeReference,
    KnowledgeSourceDescriptor,
    SourceAccessType,
    SourceStatus,
)
from app.clinical_pharmacy_engine.knowledge_base.source import (
    ClinicalKnowledgeSource,
)


class OpenFDAKnowledgeSource(ClinicalKnowledgeSource):
    """
    Fonte de conhecimento farmacêutico baseada na openFDA.

    O conector consulta três endpoints:

    1. drug/label
    2. drug/event
    3. drug/enforcement
    """

    BASE_URL = "https://api.fda.gov"

    LABEL_ENDPOINT = "/drug/label.json"
    EVENT_ENDPOINT = "/drug/event.json"
    ENFORCEMENT_ENDPOINT = "/drug/enforcement.json"

    descriptor = KnowledgeSourceDescriptor(
        code="openfda",
        name="openFDA",
        access_type=SourceAccessType.PUBLIC_API,
        status=SourceStatus.AVAILABLE,
        description=(
            "APIs públicas da FDA contendo rotulagem, "
            "eventos adversos do FAERS e recalls."
        ),
        base_url="https://open.fda.gov",
        supported_domains=[
            KnowledgeDomain.DRUG_MONOGRAPH,
            KnowledgeDomain.INDICATION,
            KnowledgeDomain.CONTRAINDICATION,
            KnowledgeDomain.DOSAGE,
            KnowledgeDomain.DRUG_INTERACTION,
            KnowledgeDomain.ADVERSE_REACTION,
            KnowledgeDomain.PHARMACOVIGILANCE,
            KnowledgeDomain.PREGNANCY,
            KnowledgeDomain.LACTATION,
            KnowledgeDomain.PHARMACOKINETICS,
            KnowledgeDomain.PHARMACODYNAMICS,
            KnowledgeDomain.REGULATORY_ALERT,
            KnowledgeDomain.OTHER,
        ],
        requires_credentials=False,
        license_notes=(
            "Dados públicos da FDA. Resultados podem ser "
            "incompletos, duplicados ou não validados e não "
            "devem substituir avaliação clínica profissional."
        ),
        country="US",
        language="en",
        version="openFDA REST",
    )

    LABEL_FIELD_MAP: dict[
        str,
        KnowledgeDomain,
    ] = {
        "indications_and_usage": (
            KnowledgeDomain.INDICATION
        ),
        "purpose": KnowledgeDomain.INDICATION,
        "contraindications": (
            KnowledgeDomain.CONTRAINDICATION
        ),
        "dosage_and_administration": (
            KnowledgeDomain.DOSAGE
        ),
        "drug_interactions": (
            KnowledgeDomain.DRUG_INTERACTION
        ),
        "adverse_reactions": (
            KnowledgeDomain.ADVERSE_REACTION
        ),
        "warnings": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "warnings_and_cautions": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "boxed_warning": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "pregnancy": KnowledgeDomain.PREGNANCY,
        "nursing_mothers": (
            KnowledgeDomain.LACTATION
        ),
        "mechanism_of_action": (
            KnowledgeDomain.PHARMACODYNAMICS
        ),
        "pharmacodynamics": (
            KnowledgeDomain.PHARMACODYNAMICS
        ),
        "pharmacokinetics": (
            KnowledgeDomain.PHARMACOKINETICS
        ),
        "clinical_pharmacology": (
            KnowledgeDomain.PHARMACODYNAMICS
        ),
        "description": (
            KnowledgeDomain.DRUG_MONOGRAPH
        ),
        "active_ingredient": (
            KnowledgeDomain.DRUG_MONOGRAPH
        ),
        "inactive_ingredient": (
            KnowledgeDomain.OTHER
        ),
        "storage_and_handling": (
            KnowledgeDomain.OTHER
        ),
        "how_supplied": KnowledgeDomain.OTHER,
        "information_for_patients": (
            KnowledgeDomain.OTHER
        ),
    }

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_label_records: int = 2,
        max_event_records: int = 100,
        max_recall_records: int = 20,
        max_section_characters: int = 12000,
        include_labels: bool = True,
        include_events: bool = True,
        include_recalls: bool = True,
        user_agent: str = "PHK-Studio/0.3",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds deve ser maior que zero.",
            )

        if max_label_records < 1:
            raise ValueError(
                "max_label_records deve ser maior que zero.",
            )

        if max_event_records < 1:
            raise ValueError(
                "max_event_records deve ser maior que zero.",
            )

        if max_recall_records < 1:
            raise ValueError(
                "max_recall_records deve ser maior que zero.",
            )

        if max_section_characters < 500:
            raise ValueError(
                "max_section_characters deve ser pelo "
                "menos 500.",
            )

        self.api_key = (
            api_key
            or os.getenv("OPENFDA_API_KEY")
        )

        self.timeout_seconds = timeout_seconds
        self.max_label_records = max_label_records
        self.max_event_records = max_event_records
        self.max_recall_records = max_recall_records
        self.max_section_characters = (
            max_section_characters
        )

        self.include_labels = include_labels
        self.include_events = include_events
        self.include_recalls = include_recalls
        self.user_agent = user_agent

    async def search(
        self,
        query: KnowledgeQuery,
    ) -> list[KnowledgeEvidence]:
        """Consulta os conjuntos openFDA habilitados."""

        term = query.normalized_term()

        if len(term) < 2:
            return []

        requested_limit = max(
            1,
            query.limit_per_source,
        )

        evidences: list[KnowledgeEvidence] = []

        async with httpx.AsyncClient(
            base_url=self.BASE_URL,
            timeout=self.timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            follow_redirects=True,
        ) as client:
            if self.include_labels:
                evidences.extend(
                    await self._search_labels(
                        client=client,
                        term=term,
                    )
                )

            if self.include_events:
                evidences.extend(
                    await self._search_adverse_events(
                        client=client,
                        term=term,
                    )
                )

            if self.include_recalls:
                evidences.extend(
                    await self._search_recalls(
                        client=client,
                        term=term,
                    )
                )

        return self._deduplicate_evidences(
            evidences,
        )[:requested_limit]

    async def health_check(self) -> bool:
        """Verifica se o endpoint de rotulagem responde."""

        try:
            async with httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=min(
                    self.timeout_seconds,
                    10.0,
                ),
                headers={
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    self.LABEL_ENDPOINT,
                    params=self._params(
                        {
                            "search": (
                                'openfda.generic_name:"aspirin"'
                            ),
                            "limit": 1,
                        }
                    ),
                )

            return response.status_code in {
                200,
                404,
            }

        except httpx.HTTPError:
            return False

    async def _search_labels(
        self,
        client: httpx.AsyncClient,
        term: str,
    ) -> list[KnowledgeEvidence]:
        """Pesquisa seções de rotulagem farmacêutica."""

        payload = await self._request_json(
            client=client,
            endpoint=self.LABEL_ENDPOINT,
            params={
                "search": self._drug_search_expression(
                    term,
                ),
                "limit": min(
                    self.max_label_records,
                    100,
                ),
            },
        )

        if payload is None:
            return []

        records = payload.get(
            "results",
            [],
        )

        if not isinstance(records, list):
            return []

        evidences: list[KnowledgeEvidence] = []

        for record in records:
            if not isinstance(record, dict):
                continue

            evidences.extend(
                self._label_record_to_evidences(
                    term=term,
                    record=record,
                )
            )

        return evidences

    async def _search_adverse_events(
        self,
        client: httpx.AsyncClient,
        term: str,
    ) -> list[KnowledgeEvidence]:
        """
        Pesquisa relatos FAERS e agrega termos de reação.

        O resultado representa sinal descritivo de relatos,
        não causalidade comprovada nem incidência.
        """

        payload = await self._request_json(
            client=client,
            endpoint=self.EVENT_ENDPOINT,
            params={
                "search": (
                    "patient.drug.openfda.generic_name:"
                    f'"{self._escape_search(term)}"'
                    " OR "
                    "patient.drug.medicinalproduct:"
                    f'"{self._escape_search(term)}"'
                ),
                "limit": min(
                    self.max_event_records,
                    100,
                ),
            },
        )

        if payload is None:
            return []

        records = payload.get(
            "results",
            [],
        )

        if not isinstance(records, list):
            return []

        reaction_counter: Counter[str] = Counter()
        serious_reports = 0
        death_reports = 0
        hospitalization_reports = 0
        report_ids: list[str] = []
        received_dates: list[datetime] = []

        for record in records:
            if not isinstance(record, dict):
                continue

            if str(record.get("serious")) == "1":
                serious_reports += 1

            if str(record.get("seriousnessdeath")) == "1":
                death_reports += 1

            if (
                str(
                    record.get(
                        "seriousnesshospitalization"
                    )
                )
                == "1"
            ):
                hospitalization_reports += 1

            report_id = self._clean_string(
                record.get("safetyreportid")
            )

            if report_id:
                report_ids.append(report_id)

            received_date = self._parse_fda_date(
                record.get("receivedate")
            )

            if received_date:
                received_dates.append(
                    received_date,
                )

            patient = record.get(
                "patient",
                {},
            )

            if not isinstance(patient, dict):
                continue

            reactions = patient.get(
                "reaction",
                [],
            )

            if not isinstance(reactions, list):
                continue

            for reaction in reactions:
                if not isinstance(reaction, dict):
                    continue

                term_value = self._clean_string(
                    reaction.get(
                        "reactionmeddrapt"
                    )
                )

                if term_value:
                    reaction_counter[
                        term_value
                    ] += 1

        if not reaction_counter:
            return []

        most_common = reaction_counter.most_common(
            25,
        )

        reaction_summary = "; ".join(
            f"{reaction}: {count}"
            for reaction, count in most_common
        )

        total_reports = len(records)

        summary = (
            f"FAERS descriptive signal for {term}. "
            f"Reports retrieved: {total_reports}. "
            f"Serious reports: {serious_reports}. "
            f"Hospitalization reports: "
            f"{hospitalization_reports}. "
            f"Reports mentioning death: {death_reports}. "
            f"Most frequently reported reaction terms: "
            f"{reaction_summary}. "
            "Spontaneous reporting data cannot establish "
            "causality, incidence, prevalence or comparative "
            "risk and may contain duplicate or incomplete "
            "reports."
        )

        reference = KnowledgeReference(
            title=(
                f"openFDA FAERS reports for {term}"
            ),
            url=(
                "https://open.fda.gov/apis/drug/event/"
            ),
            publisher=(
                "U.S. Food and Drug Administration"
            ),
            citation=(
                "openFDA Drug Adverse Event API, FAERS"
            ),
        )

        return [
            KnowledgeEvidence(
                source_code=self.code,
                domain=(
                    KnowledgeDomain.PHARMACOVIGILANCE
                ),
                subject=term,
                title=(
                    f"FAERS adverse-event signal: {term}"
                ),
                summary=summary,
                related_agents=[
                    term,
                ],
                evidence_strength=(
                    EvidenceStrength.MODERATE
                ),
                confidence=0.70,
                country="US",
                language="en",
                published_at=(
                    max(received_dates)
                    if received_dates
                    else None
                ),
                references=[
                    reference,
                ],
                raw_identifiers={
                    "faers_report_ids": report_ids[:50],
                },
                metadata={
                    "dataset": "FAERS",
                    "report_count": total_reports,
                    "serious_report_count": (
                        serious_reports
                    ),
                    "hospitalization_report_count": (
                        hospitalization_reports
                    ),
                    "death_report_count": death_reports,
                    "reaction_counts": dict(
                        most_common,
                    ),
                    "causality_established": False,
                    "incidence_available": False,
                    "signal_type": (
                        "descriptive_spontaneous_reports"
                    ),
                },
                requires_professional_review=True,
            )
        ]

    async def _search_recalls(
        self,
        client: httpx.AsyncClient,
        term: str,
    ) -> list[KnowledgeEvidence]:
        """Pesquisa recalls e ações de recolhimento."""

        payload = await self._request_json(
            client=client,
            endpoint=self.ENFORCEMENT_ENDPOINT,
            params={
                "search": (
                    'product_description:'
                    f'"{self._escape_search(term)}"'
                    " OR "
                    'openfda.generic_name:'
                    f'"{self._escape_search(term)}"'
                    " OR "
                    'openfda.brand_name:'
                    f'"{self._escape_search(term)}"'
                ),
                "limit": min(
                    self.max_recall_records,
                    100,
                ),
            },
        )

        if payload is None:
            return []

        records = payload.get(
            "results",
            [],
        )

        if not isinstance(records, list):
            return []

        evidences: list[KnowledgeEvidence] = []

        for record in records:
            if not isinstance(record, dict):
                continue

            evidence = self._recall_to_evidence(
                term=term,
                record=record,
            )

            if evidence is not None:
                evidences.append(evidence)

        return evidences

    def _label_record_to_evidences(
        self,
        term: str,
        record: dict[str, Any],
    ) -> list[KnowledgeEvidence]:
        """Transforma um registro de label em evidências."""

        openfda = record.get(
            "openfda",
            {},
        )

        if not isinstance(openfda, dict):
            openfda = {}

        generic_names = self._string_list(
            openfda.get("generic_name")
        )

        brand_names = self._string_list(
            openfda.get("brand_name")
        )

        manufacturer_names = self._string_list(
            openfda.get("manufacturer_name")
        )

        application_numbers = self._string_list(
            openfda.get("application_number")
        )

        product_ndcs = self._string_list(
            openfda.get("product_ndc")
        )

        rxcuis = self._string_list(
            openfda.get("rxcui")
        )

        uniis = self._string_list(
            openfda.get("unii")
        )

        spl_set_ids = self._string_list(
            openfda.get("spl_set_id")
        )

        label_name = (
            generic_names[0]
            if generic_names
            else (
                brand_names[0]
                if brand_names
                else term
            )
        )

        effective_time = self._parse_fda_date(
            record.get("effective_time")
        )

        reference_url = (
            "https://open.fda.gov/apis/drug/label/"
        )

        reference = KnowledgeReference(
            title=(
                f"openFDA drug label: {label_name}"
            ),
            url=reference_url,
            publisher=(
                "U.S. Food and Drug Administration"
            ),
            citation=(
                "openFDA Drug Labeling API"
            ),
        )

        evidences: list[KnowledgeEvidence] = []

        for field_name, domain in (
            self.LABEL_FIELD_MAP.items()
        ):
            text_values = self._string_list(
                record.get(field_name)
            )

            if not text_values:
                continue

            summary = self._normalize_whitespace(
                " ".join(text_values)
            )

            if not summary:
                continue

            metadata: dict[str, Any] = {
                "dataset": "openfda_label",
                "label_field": field_name,
                "generic_name": (
                    generic_names[0]
                    if generic_names
                    else None
                ),
                "brand_names": brand_names,
                "manufacturer_names": (
                    manufacturer_names
                ),
                "application_numbers": (
                    application_numbers
                ),
                "product_ndcs": product_ndcs,
                "unii": uniis,
                "spl_set_ids": spl_set_ids,
            }

            if field_name == "boxed_warning":
                metadata["boxed_warning"] = True

            if field_name == "active_ingredient":
                metadata[
                    "active_ingredients"
                ] = text_values

            evidences.append(
                KnowledgeEvidence(
                    source_code=self.code,
                    domain=domain,
                    subject=label_name,
                    title=field_name.replace(
                        "_",
                        " ",
                    ).title(),
                    summary=summary[
                        : self.max_section_characters
                    ],
                    related_agents=self._unique_strings(
                        [
                            term,
                            *generic_names,
                            *brand_names,
                        ]
                    ),
                    recommendation=(
                        summary[
                            : self.max_section_characters
                        ]
                        if domain
                        in {
                            KnowledgeDomain.DOSAGE,
                            KnowledgeDomain.CONTRAINDICATION,
                            KnowledgeDomain.DRUG_INTERACTION,
                            KnowledgeDomain.REGULATORY_ALERT,
                        }
                        else None
                    ),
                    evidence_strength=(
                        EvidenceStrength.HIGH
                    ),
                    confidence=0.93,
                    country="US",
                    language="en",
                    published_at=effective_time,
                    references=[
                        reference,
                    ],
                    raw_identifiers={
                        key: value
                        for key, value in {
                            "rxcui": rxcuis,
                            "unii": uniis,
                            "application_number": (
                                application_numbers
                            ),
                            "product_ndc": product_ndcs,
                            "spl_set_id": spl_set_ids,
                        }.items()
                        if value
                    },
                    metadata={
                        key: value
                        for key, value in metadata.items()
                        if value not in (None, "", [])
                    },
                    requires_professional_review=True,
                )
            )

        return evidences

    def _recall_to_evidence(
        self,
        term: str,
        record: dict[str, Any],
    ) -> KnowledgeEvidence | None:
        """Transforma um recall em alerta regulatório."""

        recall_number = self._clean_string(
            record.get("recall_number")
        )

        classification = self._clean_string(
            record.get("classification")
        )

        status = self._clean_string(
            record.get("status")
        )

        reason = self._clean_string(
            record.get("reason_for_recall")
        )

        product_description = self._clean_string(
            record.get("product_description")
        )

        recalling_firm = self._clean_string(
            record.get("recalling_firm")
        )

        distribution_pattern = self._clean_string(
            record.get("distribution_pattern")
        )

        action = self._clean_string(
            record.get("recall_initiation_date")
        )

        report_date = self._parse_fda_date(
            record.get("report_date")
        )

        if not any(
            [
                recall_number,
                reason,
                product_description,
            ]
        ):
            return None

        parts = [
            (
                f"Recall number: {recall_number}."
                if recall_number
                else None
            ),
            (
                f"Classification: {classification}."
                if classification
                else None
            ),
            (
                f"Status: {status}."
                if status
                else None
            ),
            (
                f"Product: {product_description}."
                if product_description
                else None
            ),
            (
                f"Reason: {reason}."
                if reason
                else None
            ),
            (
                f"Recalling firm: {recalling_firm}."
                if recalling_firm
                else None
            ),
            (
                "Distribution: "
                f"{distribution_pattern}."
                if distribution_pattern
                else None
            ),
        ]

        summary = " ".join(
            part
            for part in parts
            if part
        )

        reference = KnowledgeReference(
            title=(
                f"openFDA recall {recall_number or term}"
            ),
            url=(
                "https://open.fda.gov/apis/"
                "drug/enforcement/"
            ),
            publisher=(
                "U.S. Food and Drug Administration"
            ),
            citation=(
                "openFDA Drug Enforcement Reports API"
            ),
        )

        return KnowledgeEvidence(
            source_code=self.code,
            domain=KnowledgeDomain.REGULATORY_ALERT,
            subject=term,
            title=(
                f"Drug recall: "
                f"{recall_number or product_description or term}"
            ),
            summary=summary,
            related_agents=[
                term,
            ],
            recommendation=(
                "Review the recall status, affected lots, "
                "distribution and applicable regulatory or "
                "institutional actions before dispensing or "
                "using the affected product."
            ),
            evidence_strength=EvidenceStrength.HIGH,
            confidence=0.94,
            country="US",
            language="en",
            published_at=report_date,
            references=[
                reference,
            ],
            raw_identifiers={
                key: value
                for key, value in {
                    "recall_number": recall_number,
                    "recall_initiation_date": action,
                }.items()
                if value
            },
            metadata={
                key: value
                for key, value in {
                    "dataset": "drug_enforcement",
                    "classification": classification,
                    "status": status,
                    "reason_for_recall": reason,
                    "product_description": (
                        product_description
                    ),
                    "recalling_firm": recalling_firm,
                    "distribution_pattern": (
                        distribution_pattern
                    ),
                }.items()
                if value is not None
            },
            requires_professional_review=True,
        )

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Executa requisição tratando ausência de resultados."""

        response = await client.get(
            endpoint,
            params=self._params(params),
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        payload = response.json()

        if not isinstance(payload, dict):
            return None

        return payload

    def _params(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Acrescenta a chave da API quando configurada."""

        result = dict(params)

        if self.api_key:
            result["api_key"] = self.api_key

        return result

    @classmethod
    def _drug_search_expression(
        cls,
        term: str,
    ) -> str:
        """Cria consulta abrangente para rótulos."""

        escaped = cls._escape_search(term)

        return (
            f'openfda.generic_name:"{escaped}"'
            " OR "
            f'openfda.brand_name:"{escaped}"'
            " OR "
            f'openfda.substance_name:"{escaped}"'
        )

    @staticmethod
    def _escape_search(
        value: str,
    ) -> str:
        """Escapa caracteres básicos usados na expressão."""

        return (
            value
            .replace("\\", "\\\\")
            .replace('"', '\\"')
        )

    @staticmethod
    def _string_list(
        value: Any,
    ) -> list[str]:
        """Normaliza um campo escalar ou lista."""

        if value is None:
            return []

        values = (
            value
            if isinstance(value, list)
            else [value]
        )

        result: list[str] = []

        for item in values:
            if item is None:
                continue

            text = " ".join(
                str(item).strip().split()
            )

            if (
                text
                and text.casefold()
                not in {
                    existing.casefold()
                    for existing in result
                }
            ):
                result.append(text)

        return result

    @staticmethod
    def _clean_string(
        value: Any,
    ) -> str | None:
        """Converte valor para texto opcional."""

        if value is None:
            return None

        result = " ".join(
            str(value).strip().split()
        )

        return result or None

    @staticmethod
    def _parse_fda_date(
        value: Any,
    ) -> datetime | None:
        """Converte formatos de data usuais da FDA."""

        if value is None:
            return None

        text = str(value).strip()

        for pattern in (
            "%Y%m%d",
            "%Y-%m-%d",
            "%m/%d/%Y",
        ):
            try:
                return datetime.strptime(
                    text,
                    pattern,
                )

            except ValueError:
                continue

        return None

    @staticmethod
    def _normalize_whitespace(
        value: str,
    ) -> str:
        """Normaliza espaços e quebras de linha."""

        return re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

    @staticmethod
    def _unique_strings(
        values: list[str],
    ) -> list[str]:
        """Remove duplicatas preservando a ordem."""

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

    @staticmethod
    def _deduplicate_evidences(
        evidences: list[KnowledgeEvidence],
    ) -> list[KnowledgeEvidence]:
        """Remove evidências repetidas."""

        result: list[KnowledgeEvidence] = []
        seen: set[
            tuple[str, str, str]
        ] = set()

        for evidence in evidences:
            key = (
                evidence.domain.value,
                evidence.title.casefold(),
                evidence.summary.casefold(),
            )

            if key in seen:
                continue

            seen.add(key)
            result.append(evidence)

        return result