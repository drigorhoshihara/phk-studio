"""
PHK Studio
Clinical Pharmacy Engine

RxNorm Knowledge Source.

Normalização de nomes de medicamentos, ingredientes,
apresentações e conceitos clínicos por meio da API RxNav.
"""

from __future__ import annotations

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


class RxNormKnowledgeSource(ClinicalKnowledgeSource):
    """
    Fonte de normalização farmacológica baseada no RxNorm.

    Fluxo:

    1. pesquisa o termo pelo nome;
    2. recupera os RxCUIs correspondentes;
    3. consulta as propriedades dos conceitos;
    4. converte os resultados para KnowledgeEvidence.
    """

    BASE_URL = "https://rxnav.nlm.nih.gov/REST"

    descriptor = KnowledgeSourceDescriptor(
        code="rxnorm",
        name="RxNorm",
        access_type=SourceAccessType.PUBLIC_API,
        status=SourceStatus.AVAILABLE,
        description=(
            "Terminologia normalizada de medicamentos "
            "mantida pela National Library of Medicine."
        ),
        base_url="https://rxnav.nlm.nih.gov",
        supported_domains=[
            KnowledgeDomain.DRUG_MONOGRAPH,
            KnowledgeDomain.INDICATION,
            KnowledgeDomain.OTHER,
        ],
        requires_credentials=False,
        license_notes=(
            "A API pública fornece conceitos da terminologia "
            "RxNorm. Vocabulários vinculados podem possuir "
            "condições próprias de licenciamento."
        ),
        country="US",
        language="en",
        version="RxNav REST",
    )

    def __init__(
        self,
        timeout_seconds: float = 20.0,
        max_results: int = 10,
        user_agent: str = "PHK-Studio/0.3",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds deve ser maior que zero.",
            )

        if max_results < 1:
            raise ValueError(
                "max_results deve ser maior que zero.",
            )

        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.user_agent = user_agent

    async def search(
        self,
        query: KnowledgeQuery,
    ) -> list[KnowledgeEvidence]:
        term = query.normalized_term()

        if len(term) < 2:
            return []

        limit = min(
            query.limit_per_source,
            self.max_results,
        )

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            follow_redirects=True,
        ) as client:
            rxcuis = await self._find_rxcuis(
                client=client,
                term=term,
            )

            if not rxcuis:
                rxcuis = await self._approximate_match(
                    client=client,
                    term=term,
                    limit=limit,
                )

            evidences: list[KnowledgeEvidence] = []

            for rxcui in rxcuis[:limit]:
                properties = await self._get_properties(
                    client=client,
                    rxcui=rxcui,
                )

                if properties is None:
                    continue

                evidences.append(
                    self._to_evidence(
                        query=query,
                        rxcui=rxcui,
                        properties=properties,
                    )
                )

        return evidences

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(
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
                    f"{self.BASE_URL}/version.json",
                )

            return response.status_code == 200

        except httpx.HTTPError:
            return False

    async def _find_rxcuis(
        self,
        client: httpx.AsyncClient,
        term: str,
    ) -> list[str]:
        response = await client.get(
            f"{self.BASE_URL}/rxcui.json",
            params={
                "name": term,
                "search": 2,
            },
        )

        if response.status_code == 404:
            return []

        response.raise_for_status()

        payload = response.json()

        identifiers = (
            payload
            .get("idGroup", {})
            .get("rxnormId", [])
        )

        return self._normalize_identifiers(
            identifiers,
        )

    async def _approximate_match(
        self,
        client: httpx.AsyncClient,
        term: str,
        limit: int,
    ) -> list[str]:
        response = await client.get(
            f"{self.BASE_URL}/approximateTerm.json",
            params={
                "term": term,
                "maxEntries": limit,
                "option": 1,
            },
        )

        if response.status_code == 404:
            return []

        response.raise_for_status()

        payload = response.json()

        candidates = (
            payload
            .get("approximateGroup", {})
            .get("candidate", [])
        )

        identifiers = [
            candidate.get("rxcui")
            for candidate in candidates
            if candidate.get("rxcui")
        ]

        return self._normalize_identifiers(
            identifiers,
        )

    async def _get_properties(
        self,
        client: httpx.AsyncClient,
        rxcui: str,
    ) -> dict[str, Any] | None:
        response = await client.get(
            (
                f"{self.BASE_URL}/rxcui/"
                f"{rxcui}/properties.json"
            ),
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        payload = response.json()

        properties = payload.get(
            "properties",
        )

        if not isinstance(
            properties,
            dict,
        ):
            return None

        return properties

    def _to_evidence(
        self,
        query: KnowledgeQuery,
        rxcui: str,
        properties: dict[str, Any],
    ) -> KnowledgeEvidence:
        concept_name = (
            properties.get("name")
            or query.term
        )

        synonym = properties.get(
            "synonym",
        )

        term_type = properties.get(
            "tty",
        )

        language = properties.get(
            "language",
        )

        suppress = properties.get(
            "suppress",
        )

        summary_parts = [
            f"RxNorm concept: {concept_name}.",
            f"RxCUI: {rxcui}.",
        ]

        if term_type:
            summary_parts.append(
                f"Term type: {term_type}."
            )

        if synonym:
            summary_parts.append(
                f"Synonym: {synonym}."
            )

        related_agents = [
            str(concept_name),
            query.term,
        ]

        if synonym:
            related_agents.append(
                str(synonym),
            )

        record_url = (
            "https://mor.nlm.nih.gov/"
            f"RxNav/search?searchBy=RXCUI"
            f"&searchTerm={rxcui}"
        )

        return KnowledgeEvidence(
            source_code=self.code,
            domain=KnowledgeDomain.DRUG_MONOGRAPH,
            subject=str(concept_name),
            title=(
                f"RxNorm concept record: "
                f"{concept_name}"
            ),
            summary=" ".join(
                summary_parts,
            ),
            related_agents=self._unique_strings(
                related_agents,
            ),
            evidence_strength=EvidenceStrength.HIGH,
            confidence=0.97,
            country="US",
            language=(
                str(language).lower()
                if language
                else "en"
            ),
            references=[
                KnowledgeReference(
                    title=(
                        f"RxNorm concept {rxcui}: "
                        f"{concept_name}"
                    ),
                    url=record_url,
                    publisher=(
                        "U.S. National Library "
                        "of Medicine"
                    ),
                    citation=f"RxNorm RxCUI {rxcui}",
                )
            ],
            raw_identifiers={
                "rxnorm_id": str(rxcui),
                "rxcui": str(rxcui),
            },
            metadata={
                key: value
                for key, value in {
                    "rxnorm_name": concept_name,
                    "rxnorm_synonym": synonym,
                    "rxnorm_term_type": term_type,
                    "rxnorm_language": language,
                    "rxnorm_suppress": suppress,
                }.items()
                if value not in {
                    None,
                    "",
                }
            },
            requires_professional_review=True,
        )

    @staticmethod
    def _normalize_identifiers(
        identifiers: list[Any],
    ) -> list[str]:
        normalized: list[str] = []

        for identifier in identifiers:
            if identifier is None:
                continue

            value = str(
                identifier,
            ).strip()

            if (
                value
                and value not in normalized
            ):
                normalized.append(value)

        return normalized

    @staticmethod
    def _unique_strings(
        values: list[str],
    ) -> list[str]:
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