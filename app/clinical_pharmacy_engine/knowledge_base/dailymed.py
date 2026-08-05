"""
PHK Studio
Clinical Pharmacy Engine

DailyMed Knowledge Source.

Consulta documentos Structured Product Labeling, SPL,
publicados no DailyMed e converte suas seções em evidências
clínicas normalizadas para o Clinical Knowledge Engine.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

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


class DailyMedKnowledgeSource(ClinicalKnowledgeSource):
    """
    Fonte clínica baseada nos rótulos oficiais do DailyMed.

    Fluxo principal:

    1. pesquisa rótulos pelo nome do medicamento;
    2. classifica os registros encontrados;
    3. seleciona os documentos potencialmente mais úteis;
    4. baixa o SPL em XML;
    5. extrai suas seções;
    6. converte cada seção em KnowledgeEvidence.
    """

    BASE_URL = (
        "https://dailymed.nlm.nih.gov/"
        "dailymed/services/v2"
    )

    DOCUMENT_URL = (
        "https://dailymed.nlm.nih.gov/"
        "dailymed/drugInfo.cfm?setid={setid}"
    )

    HL7_NAMESPACE = {
        "hl7": "urn:hl7-org:v3",
    }

    descriptor = KnowledgeSourceDescriptor(
        code="dailymed",
        name="DailyMed",
        access_type=SourceAccessType.PUBLIC_API,
        status=SourceStatus.AVAILABLE,
        description=(
            "Base da National Library of Medicine contendo "
            "documentos Structured Product Labeling enviados "
            "à FDA."
        ),
        base_url="https://dailymed.nlm.nih.gov",
        supported_domains=[
            KnowledgeDomain.DRUG_MONOGRAPH,
            KnowledgeDomain.INDICATION,
            KnowledgeDomain.CONTRAINDICATION,
            KnowledgeDomain.DOSAGE,
            KnowledgeDomain.DRUG_INTERACTION,
            KnowledgeDomain.ADVERSE_REACTION,
            KnowledgeDomain.PREGNANCY,
            KnowledgeDomain.LACTATION,
            KnowledgeDomain.PHARMACOKINETICS,
            KnowledgeDomain.PHARMACODYNAMICS,
            KnowledgeDomain.REGULATORY_ALERT,
            KnowledgeDomain.OTHER,
        ],
        requires_credentials=False,
        license_notes=(
            "Preservar SET ID, versão do SPL, título, "
            "data de publicação e origem das seções."
        ),
        country="US",
        language="en",
        version="DailyMed REST v2",
    )

    SECTION_DOMAIN_MAP: dict[str, KnowledgeDomain] = {
        # Indicações e finalidade
        "indications and usage": (
            KnowledgeDomain.INDICATION
        ),
        "indications & usage": (
            KnowledgeDomain.INDICATION
        ),
        "indications": (
            KnowledgeDomain.INDICATION
        ),
        "uses": (
            KnowledgeDomain.INDICATION
        ),
        "purpose": (
            KnowledgeDomain.INDICATION
        ),

        # Contraindicações
        "contraindications": (
            KnowledgeDomain.CONTRAINDICATION
        ),
        "do not use": (
            KnowledgeDomain.CONTRAINDICATION
        ),

        # Posologia e administração
        "dosage and administration": (
            KnowledgeDomain.DOSAGE
        ),
        "dosage & administration": (
            KnowledgeDomain.DOSAGE
        ),
        "directions": (
            KnowledgeDomain.DOSAGE
        ),
        "recommended dosage": (
            KnowledgeDomain.DOSAGE
        ),
        "dose and administration": (
            KnowledgeDomain.DOSAGE
        ),
        "administration": (
            KnowledgeDomain.DOSAGE
        ),

        # Interações
        "drug interactions": (
            KnowledgeDomain.DRUG_INTERACTION
        ),
        "interactions": (
            KnowledgeDomain.DRUG_INTERACTION
        ),
        "ask a doctor or pharmacist before use": (
            KnowledgeDomain.DRUG_INTERACTION
        ),

        # Reações adversas
        "adverse reactions": (
            KnowledgeDomain.ADVERSE_REACTION
        ),
        "side effects": (
            KnowledgeDomain.ADVERSE_REACTION
        ),
        "undesirable effects": (
            KnowledgeDomain.ADVERSE_REACTION
        ),

        # Populações especiais
        "use in specific populations": (
            KnowledgeDomain.DRUG_MONOGRAPH
        ),
        "pregnancy": (
            KnowledgeDomain.PREGNANCY
        ),
        "pregnant": (
            KnowledgeDomain.PREGNANCY
        ),
        "lactation": (
            KnowledgeDomain.LACTATION
        ),
        "breast-feeding": (
            KnowledgeDomain.LACTATION
        ),
        "breastfeeding": (
            KnowledgeDomain.LACTATION
        ),
        "nursing mothers": (
            KnowledgeDomain.LACTATION
        ),
        "pediatric use": (
            KnowledgeDomain.DRUG_MONOGRAPH
        ),
        "geriatric use": (
            KnowledgeDomain.DRUG_MONOGRAPH
        ),

        # Farmacologia
        "clinical pharmacology": (
            KnowledgeDomain.PHARMACODYNAMICS
        ),
        "mechanism of action": (
            KnowledgeDomain.PHARMACODYNAMICS
        ),
        "pharmacodynamics": (
            KnowledgeDomain.PHARMACODYNAMICS
        ),
        "pharmacokinetics": (
            KnowledgeDomain.PHARMACOKINETICS
        ),

        # Alertas e precauções
        "boxed warning": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "warnings and precautions": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "warnings & precautions": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "warnings": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "warning": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "stop use": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "allergy alert": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "stomach bleeding warning": (
            KnowledgeDomain.REGULATORY_ALERT
        ),
        "reye's syndrome": (
            KnowledgeDomain.REGULATORY_ALERT
        ),

        # Informações gerais
        "description": (
            KnowledgeDomain.DRUG_MONOGRAPH
        ),
        "active ingredient": (
            KnowledgeDomain.DRUG_MONOGRAPH
        ),
        "inactive ingredients": (
            KnowledgeDomain.OTHER
        ),
        "how supplied": (
            KnowledgeDomain.OTHER
        ),
        "storage and handling": (
            KnowledgeDomain.OTHER
        ),
        "storage": (
            KnowledgeDomain.OTHER
        ),
        "other information": (
            KnowledgeDomain.OTHER
        ),
        "patient counseling information": (
            KnowledgeDomain.OTHER
        ),
        "instructions for use": (
            KnowledgeDomain.OTHER
        ),
    }

    INFORMATIVE_SECTION_MARKERS = {
        "indications and usage",
        "contraindications",
        "dosage and administration",
        "drug interactions",
        "adverse reactions",
        "clinical pharmacology",
        "mechanism of action",
        "pharmacokinetics",
        "pharmacodynamics",
        "use in specific populations",
        "warnings and precautions",
        "boxed warning",
    }

    OTC_SECTION_MARKERS = {
        "active ingredient",
        "purpose",
        "uses",
        "warnings",
        "directions",
        "inactive ingredients",
        "other information",
    }

    def __init__(
        self,
        timeout_seconds: float = 30.0,
        max_labels: int = 3,
        max_sections_per_label: int = 50,
        max_section_characters: int = 12000,
        label_search_multiplier: int = 4,
        user_agent: str = "PHK-Studio/0.3",
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds deve ser maior que zero.",
            )

        if max_labels < 1:
            raise ValueError(
                "max_labels deve ser maior que zero.",
            )

        if max_sections_per_label < 1:
            raise ValueError(
                "max_sections_per_label deve ser maior "
                "que zero.",
            )

        if max_section_characters < 500:
            raise ValueError(
                "max_section_characters deve ser pelo "
                "menos 500.",
            )

        if label_search_multiplier < 1:
            raise ValueError(
                "label_search_multiplier deve ser maior "
                "que zero.",
            )

        self.timeout_seconds = timeout_seconds
        self.max_labels = max_labels
        self.max_sections_per_label = (
            max_sections_per_label
        )
        self.max_section_characters = (
            max_section_characters
        )
        self.label_search_multiplier = (
            label_search_multiplier
        )
        self.user_agent = user_agent

    async def search(
        self,
        query: KnowledgeQuery,
    ) -> list[KnowledgeEvidence]:
        """
        Pesquisa rótulos e devolve evidências normalizadas.
        """

        term = query.normalized_term()

        if len(term) < 2:
            return []

        requested_limit = max(
            1,
            query.limit_per_source,
        )

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers={
                "Accept": (
                    "application/json, "
                    "application/xml, text/xml"
                ),
                "User-Agent": self.user_agent,
            },
            follow_redirects=True,
        ) as client:
            labels = await self._find_labels(
                client=client,
                term=term,
            )

            if not labels:
                return []

            ranked_labels = await self._rank_labels(
                client=client,
                labels=labels,
            )

            evidences: list[KnowledgeEvidence] = []

            for ranked_label in ranked_labels[
                : self.max_labels
            ]:
                label = ranked_label["label"]
                xml_content = ranked_label[
                    "xml_content"
                ]

                label_evidences = self._parse_spl(
                    query=query,
                    label=label,
                    xml_content=xml_content,
                )

                evidences.extend(
                    label_evidences,
                )

                if len(evidences) >= requested_limit:
                    break

        return evidences[:requested_limit]

    async def health_check(self) -> bool:
        """Verifica se a API do DailyMed está acessível."""

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
                    f"{self.BASE_URL}/spls.json",
                    params={
                        "drug_name": "aspirin",
                        "name_type": "both",
                        "pagesize": 1,
                        "page": 1,
                    },
                )

            return response.status_code == 200

        except httpx.HTTPError:
            return False

    async def _find_labels(
        self,
        client: httpx.AsyncClient,
        term: str,
    ) -> list[dict[str, Any]]:
        """Localiza rótulos candidatos pelo nome."""

        pagesize = min(
            max(
                self.max_labels
                * self.label_search_multiplier,
                10,
            ),
            100,
        )

        response = await client.get(
            f"{self.BASE_URL}/spls.json",
            params={
                "drug_name": term,
                "name_type": "both",
                "pagesize": pagesize,
                "page": 1,
            },
        )

        if response.status_code == 404:
            return []

        response.raise_for_status()

        payload = response.json()
        records = payload.get(
            "data",
            [],
        )

        if not isinstance(records, list):
            return []

        valid_records = [
            record
            for record in records
            if (
                isinstance(record, dict)
                and self._clean_string(
                    record.get("setid")
                )
            )
        ]

        return sorted(
            valid_records,
            key=self._label_sort_key,
            reverse=True,
        )

    async def _rank_labels(
        self,
        client: httpx.AsyncClient,
        labels: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Baixa os SPLs candidatos e prioriza os mais informativos.

        Um rótulo profissional com farmacologia, interação,
        contraindicação e reação adversa recebe pontuação maior
        do que um rótulo OTC resumido.
        """

        ranked: list[dict[str, Any]] = []

        candidate_limit = min(
            len(labels),
            max(
                self.max_labels
                * self.label_search_multiplier,
                self.max_labels,
            ),
        )

        for label in labels[:candidate_limit]:
            setid = self._clean_string(
                label.get("setid")
            )

            if not setid:
                continue

            try:
                xml_content = await self._get_spl_xml(
                    client=client,
                    setid=setid,
                )

            except httpx.HTTPError:
                continue

            if not xml_content:
                continue

            score = self._calculate_label_score(
                label=label,
                xml_content=xml_content,
            )

            ranked.append(
                {
                    "label": label,
                    "xml_content": xml_content,
                    "score": score,
                }
            )

        ranked.sort(
            key=lambda item: (
                item["score"],
                self._label_sort_key(
                    item["label"],
                ),
            ),
            reverse=True,
        )

        return ranked

    async def _get_spl_xml(
        self,
        client: httpx.AsyncClient,
        setid: str,
    ) -> bytes | None:
        """Recupera o documento SPL completo em XML."""

        response = await client.get(
            f"{self.BASE_URL}/spls/{setid}.xml",
        )

        if response.status_code == 404:
            return None

        response.raise_for_status()

        return response.content

    def _calculate_label_score(
        self,
        label: dict[str, Any],
        xml_content: bytes,
    ) -> float:
        """
        Calcula uma pontuação de utilidade clínica do SPL.
        """

        try:
            root = ElementTree.fromstring(
                xml_content,
            )

        except ElementTree.ParseError:
            return -1.0

        score = 0.0
        section_titles: list[str] = []

        sections = root.findall(
            ".//hl7:section",
            self.HL7_NAMESPACE,
        )

        for section in sections:
            title = self._section_title(
                section,
            )

            if not title:
                continue

            normalized_title = self._normalize_whitespace(
                title.casefold(),
            )

            section_titles.append(
                normalized_title,
            )

        for marker in self.INFORMATIVE_SECTION_MARKERS:
            if any(
                marker in title
                for title in section_titles
            ):
                score += 4.0

        for marker in self.OTC_SECTION_MARKERS:
            if any(
                marker in title
                for title in section_titles
            ):
                score += 0.5

        score += min(
            len(section_titles) * 0.15,
            5.0,
        )

        title = self._clean_string(
            label.get("title")
        )

        if title:
            normalized_label_title = (
                title.casefold()
            )

            if "professional" in normalized_label_title:
                score += 2.0

            if "prescription" in normalized_label_title:
                score += 2.0

        version_value = self._clean_string(
            label.get("spl_version")
        )

        try:
            version = int(
                version_value or 0,
            )

        except ValueError:
            version = 0

        score += min(
            version * 0.01,
            1.0,
        )

        return score

    def _parse_spl(
        self,
        query: KnowledgeQuery,
        label: dict[str, Any],
        xml_content: bytes,
    ) -> list[KnowledgeEvidence]:
        """Extrai evidências das seções de um SPL."""

        try:
            root = ElementTree.fromstring(
                xml_content,
            )

        except ElementTree.ParseError:
            return []

        setid = self._clean_string(
            label.get("setid")
        )

        spl_version = self._clean_string(
            label.get("spl_version")
        )

        published_date = self._parse_date(
            label.get("published_date")
        )

        label_title = (
            self._clean_string(
                label.get("title")
            )
            or self._find_document_title(root)
            or query.term
        )

        reference = KnowledgeReference(
            title=label_title,
            url=(
                self.DOCUMENT_URL.format(
                    setid=setid,
                )
                if setid
                else None
            ),
            publisher=(
                "U.S. National Library of Medicine, "
                "DailyMed"
            ),
            citation=(
                f"DailyMed SPL SET ID {setid}"
                if setid
                else "DailyMed SPL"
            ),
        )

        evidences: list[KnowledgeEvidence] = []

        sections = root.findall(
            ".//hl7:section",
            self.HL7_NAMESPACE,
        )

        seen_sections: set[
            tuple[str, str]
        ] = set()

        for section in sections[
            : self.max_sections_per_label
        ]:
            evidence = self._section_to_evidence(
                query=query,
                section=section,
                label_title=label_title,
                setid=setid,
                spl_version=spl_version,
                published_date=published_date,
                reference=reference,
            )

            if evidence is None:
                continue

            section_key = (
                evidence.domain.value,
                self._normalize_whitespace(
                    evidence.summary.casefold(),
                ),
            )

            if section_key in seen_sections:
                continue

            seen_sections.add(
                section_key,
            )

            evidences.append(
                evidence,
            )

        if not evidences:
            evidences.append(
                KnowledgeEvidence(
                    source_code=self.code,
                    domain=(
                        KnowledgeDomain.DRUG_MONOGRAPH
                    ),
                    subject=query.term,
                    title=label_title,
                    summary=(
                        "DailyMed SPL localizado, mas nenhuma "
                        "seção clínica reconhecida foi extraída."
                    ),
                    related_agents=[
                        query.term,
                        label_title,
                    ],
                    evidence_strength=(
                        EvidenceStrength.HIGH
                    ),
                    confidence=0.80,
                    country="US",
                    language="en",
                    published_at=published_date,
                    references=[
                        reference,
                    ],
                    raw_identifiers={
                        key: value
                        for key, value in {
                            "dailymed_setid": setid,
                            "spl_version": spl_version,
                        }.items()
                        if value
                    },
                    metadata={
                        "label_title": label_title,
                        "empty_clinical_extraction": True,
                    },
                    requires_professional_review=True,
                )
            )

        return evidences

    def _section_to_evidence(
        self,
        query: KnowledgeQuery,
        section: Element,
        label_title: str,
        setid: str | None,
        spl_version: str | None,
        published_date: datetime | None,
        reference: KnowledgeReference,
    ) -> KnowledgeEvidence | None:
        """Converte uma seção XML em KnowledgeEvidence."""

        section_title = self._section_title(
            section,
        )

        section_code = self._section_code(
            section,
        )

        text = self._extract_section_text(
            section,
        )

        if not text:
            return None

        normalized_title = self._normalize_whitespace(
            (section_title or "").casefold(),
        )

        domain = self._resolve_domain(
            normalized_title,
        )

        title = (
            section_title
            or (
                f"DailyMed SPL section {section_code}"
                if section_code
                else "DailyMed SPL section"
            )
        )

        recommendation = (
            text[: self.max_section_characters]
            if domain
            in {
                KnowledgeDomain.DOSAGE,
                KnowledgeDomain.CONTRAINDICATION,
                KnowledgeDomain.DRUG_INTERACTION,
                KnowledgeDomain.REGULATORY_ALERT,
            }
            else None
        )

        metadata = self._build_section_metadata(
            domain=domain,
            normalized_title=normalized_title,
            label_title=label_title,
            section_title=section_title,
            section_code=section_code,
            spl_version=spl_version,
        )

        related_agents = [
            query.term,
            label_title,
        ]

        return KnowledgeEvidence(
            source_code=self.code,
            domain=domain,
            subject=query.term,
            title=title,
            summary=text[
                : self.max_section_characters
            ],
            related_agents=self._unique_strings(
                related_agents,
            ),
            recommendation=recommendation,
            evidence_strength=EvidenceStrength.HIGH,
            confidence=self._domain_confidence(
                domain,
            ),
            country="US",
            language="en",
            published_at=published_date,
            references=[
                reference,
            ],
            raw_identifiers={
                key: value
                for key, value in {
                    "dailymed_setid": setid,
                    "spl_version": spl_version,
                    "section_code": section_code,
                }.items()
                if value
            },
            metadata=metadata,
            requires_professional_review=True,
        )

    def _build_section_metadata(
        self,
        domain: KnowledgeDomain,
        normalized_title: str,
        label_title: str,
        section_title: str | None,
        section_code: str | None,
        spl_version: str | None,
    ) -> dict[str, Any]:
        """Monta metadados clínicos da seção."""

        metadata: dict[str, Any] = {
            "label_title": label_title,
            "section_title": section_title,
            "section_code": section_code,
            "spl_version": spl_version,
            "source_document_type": "SPL",
        }

        if domain == KnowledgeDomain.ADVERSE_REACTION:
            metadata["reaction_source"] = (
                "official_product_label"
            )

        if domain == KnowledgeDomain.DOSAGE:
            metadata["population"] = (
                self._detect_population(
                    normalized_title,
                )
            )

        if domain == KnowledgeDomain.PHARMACOKINETICS:
            metadata[
                "pharmacokinetic_parameter"
            ] = (
                section_title
                or "pharmacokinetics"
            )

        if "boxed warning" in normalized_title:
            metadata["boxed_warning"] = True

        if "active ingredient" in normalized_title:
            metadata["ingredient_section"] = "active"

        if "inactive ingredient" in normalized_title:
            metadata["ingredient_section"] = "inactive"

        return {
            key: value
            for key, value in metadata.items()
            if value is not None
        }

    def _resolve_domain(
        self,
        normalized_title: str,
    ) -> KnowledgeDomain:
        """
        Resolve o domínio clínico pelo título da seção.

        Títulos mais específicos são avaliados primeiro para
        evitar que palavras genéricas capturem a seção errada.
        """

        title = self._normalize_whitespace(
            normalized_title.casefold(),
        )

        ordered_markers = sorted(
            self.SECTION_DOMAIN_MAP.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

        for marker, domain in ordered_markers:
            if marker.casefold() in title:
                return domain

        return KnowledgeDomain.OTHER

    def _find_document_title(
        self,
        root: Element,
    ) -> str | None:
        """Extrai o título principal do documento."""

        title_element = root.find(
            "./hl7:title",
            self.HL7_NAMESPACE,
        )

        if title_element is None:
            return None

        return self._extract_text(
            title_element,
        )

    def _section_title(
        self,
        section: Element,
    ) -> str | None:
        """Extrai o título de uma seção."""

        title_element = section.find(
            "./hl7:title",
            self.HL7_NAMESPACE,
        )

        if title_element is not None:
            title = self._extract_text(
                title_element,
            )

            if title:
                return title

        code_element = section.find(
            "./hl7:code",
            self.HL7_NAMESPACE,
        )

        if code_element is None:
            return None

        return self._clean_string(
            code_element.attrib.get(
                "displayName"
            )
        )

    def _section_code(
        self,
        section: Element,
    ) -> str | None:
        """Extrai o código LOINC ou código SPL da seção."""

        code_element = section.find(
            "./hl7:code",
            self.HL7_NAMESPACE,
        )

        if code_element is None:
            return None

        return self._clean_string(
            code_element.attrib.get(
                "code"
            )
        )

    def _extract_section_text(
        self,
        section: Element,
    ) -> str:
        """
        Extrai preferencialmente o conteúdo narrativo da seção.

        Evita repetir excessivamente o próprio título.
        """

        text_element = section.find(
            "./hl7:text",
            self.HL7_NAMESPACE,
        )

        if text_element is not None:
            text = self._extract_text(
                text_element,
            )

            if text:
                return text

        return self._extract_text(
            section,
        )

    def _extract_text(
        self,
        element: Element,
    ) -> str:
        """Extrai e normaliza todo o texto de um elemento."""

        raw_text = " ".join(
            part.strip()
            for part in element.itertext()
            if part and part.strip()
        )

        return self._normalize_whitespace(
            raw_text,
        )

    @staticmethod
    def _detect_population(
        normalized_title: str,
    ) -> str:
        """Tenta identificar a população clínica da seção."""

        if (
            "pediatric" in normalized_title
            or "children" in normalized_title
        ):
            return "pediatric"

        if (
            "geriatric" in normalized_title
            or "elderly" in normalized_title
        ):
            return "geriatric"

        if "pregnan" in normalized_title:
            return "pregnancy"

        return "unspecified"

    @staticmethod
    def _domain_confidence(
        domain: KnowledgeDomain,
    ) -> float:
        """Define confiança conforme clareza da classificação."""

        if domain == KnowledgeDomain.OTHER:
            return 0.82

        if domain == KnowledgeDomain.DRUG_MONOGRAPH:
            return 0.89

        return 0.94

    @staticmethod
    def _normalize_whitespace(
        value: str,
    ) -> str:
        """Remove espaços, quebras e tabulações repetidas."""

        return re.sub(
            r"\s+",
            " ",
            value,
        ).strip()

    @staticmethod
    def _clean_string(
        value: Any,
    ) -> str | None:
        """Converte um valor em texto limpo opcional."""

        if value is None:
            return None

        cleaned = str(value).strip()

        return cleaned or None

    @staticmethod
    def _parse_date(
        value: Any,
    ) -> datetime | None:
        """Converte datas conhecidas do DailyMed."""

        if value is None:
            return None

        text = str(value).strip()

        patterns = (
            "%b %d, %Y",
            "%B %d, %Y",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%Y%m%d",
        )

        for pattern in patterns:
            try:
                return datetime.strptime(
                    text,
                    pattern,
                )

            except ValueError:
                continue

        return None

    @classmethod
    def _label_sort_key(
        cls,
        label: dict[str, Any],
    ) -> tuple[datetime, int]:
        """Ordena por publicação e versão."""

        published = cls._parse_date(
            label.get("published_date")
        ) or datetime.min

        version_value = cls._clean_string(
            label.get("spl_version")
        )

        try:
            version = int(
                version_value or 0,
            )

        except ValueError:
            version = 0

        return published, version

    @staticmethod
    def _unique_strings(
        values: list[str],
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
            unique.append(
                normalized,
            )

        return unique