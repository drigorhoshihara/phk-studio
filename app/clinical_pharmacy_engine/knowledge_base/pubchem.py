"""
PHK Studio
Clinical Pharmacy Engine

PubChem Knowledge Source.

Conector público para consulta de informações químicas
e estruturais de medicamentos e outras substâncias.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

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


class PubChemKnowledgeSource(ClinicalKnowledgeSource):
    """
    Fonte de conhecimento químico baseada no PubChem PUG REST.

    O conector recupera propriedades estruturais e químicas.
    Ele não deve ser utilizado isoladamente para decisões
    clínicas, posológicas ou terapêuticas.
    """

    BASE_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    PROPERTY_NAMES = (
        "Title,"
        "MolecularFormula,"
        "MolecularWeight,"
        "CanonicalSMILES,"
        "IsomericSMILES,"
        "InChI,"
        "InChIKey,"
        "IUPACName,"
        "XLogP,"
        "TPSA,"
        "HBondDonorCount,"
        "HBondAcceptorCount,"
        "RotatableBondCount,"
        "ExactMass,"
        "MonoisotopicMass,"
        "Complexity,"
        "Charge"
    )

    descriptor = KnowledgeSourceDescriptor(
        code="pubchem",
        name="PubChem",
        access_type=SourceAccessType.PUBLIC_API,
        status=SourceStatus.AVAILABLE,
        description=(
            "Base pública do NIH/NCBI para informações "
            "químicas, estruturais e bioquímicas."
        ),
        base_url="https://pubchem.ncbi.nlm.nih.gov",
        supported_domains=[
            KnowledgeDomain.DRUG_MONOGRAPH,
            KnowledgeDomain.PHARMACOKINETICS,
            KnowledgeDomain.PHARMACODYNAMICS,
            KnowledgeDomain.TOXICOLOGY,
            KnowledgeDomain.OTHER,
        ],
        requires_credentials=False,
        license_notes=(
            "Dados públicos. A proveniência específica de cada "
            "registro deve ser preservada quando disponível."
        ),
        country="US",
        language="en",
        version="PUG REST",
    )

    def __init__(
        self,
        timeout_seconds: float = 20.0,
        max_results: int = 5,
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
        """
        Pesquisa um composto pelo nome e devolve evidências
        normalizadas para a Clinical Knowledge Base.
        """

        term = query.normalized_term()

        if len(term) < 2:
            return []

        limit = min(
            query.limit_per_source,
            self.max_results,
        )

        encoded_term = quote(
            term,
            safe="",
        )

        url = (
            f"{self.BASE_URL}/compound/name/"
            f"{encoded_term}/property/"
            f"{self.PROPERTY_NAMES}/JSON"
        )

        headers = {
            "Accept": "application/json",
            "User-Agent": self.user_agent,
        }

        async with httpx.AsyncClient(
            timeout=self.timeout_seconds,
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)

        if response.status_code == 404:
            return []

        response.raise_for_status()

        payload = response.json()

        properties = (
            payload
            .get("PropertyTable", {})
            .get("Properties", [])
        )

        evidences: list[KnowledgeEvidence] = []

        for property_data in properties[:limit]:
            evidence = self._to_evidence(
                query=query,
                property_data=property_data,
            )

            evidences.append(evidence)

        return evidences

    async def health_check(self) -> bool:
        """
        Verifica se a API pública do PubChem está respondendo.
        """

        url = (
            f"{self.BASE_URL}/compound/name/"
            "aspirin/property/Title/JSON"
        )

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
                response = await client.get(url)

            return response.status_code == 200

        except httpx.HTTPError:
            return False

    def _to_evidence(
        self,
        query: KnowledgeQuery,
        property_data: dict[str, Any],
    ) -> KnowledgeEvidence:
        cid = property_data.get("CID")

        preferred_name = (
            property_data.get("Title")
            or property_data.get("IUPACName")
            or query.term
        )

        formula = property_data.get(
            "MolecularFormula",
        )

        molecular_weight = property_data.get(
            "MolecularWeight",
        )

        canonical_smiles = (
            property_data.get("ConnectivitySMILES")
            or property_data.get("CanonicalSMILES")
        )

        isomeric_smiles = (
            property_data.get("SMILES")
            or property_data.get("IsomericSMILES")
        )

        inchikey = property_data.get(
            "InChIKey",
        )

        summary = self._build_summary(
            preferred_name=preferred_name,
            cid=cid,
            formula=formula,
            molecular_weight=molecular_weight,
            iupac_name=property_data.get(
                "IUPACName",
            ),
        )

        identifiers: dict[str, str] = {}

        if cid is not None:
            identifiers["pubchem_cid"] = str(cid)

        if inchikey:
            identifiers["inchikey"] = str(
                inchikey,
            )

        if property_data.get("InChI"):
            identifiers["inchi"] = str(
                property_data["InChI"],
            )

        metadata = {
            "molecular_formula": formula,
            "molecular_weight": molecular_weight,
            "exact_mass": property_data.get(
                "ExactMass",
            ),
            "monoisotopic_mass": property_data.get(
                "MonoisotopicMass",
            ),
            "iupac_name": property_data.get(
                "IUPACName",
            ),
            "canonical_smiles": canonical_smiles,
            "isomeric_smiles": isomeric_smiles,
            "inchi": property_data.get(
                "InChI",
            ),
            "inchikey": inchikey,
            "xlogp": property_data.get(
                "XLogP",
            ),
            "topological_polar_surface_area": (
                property_data.get("TPSA")
            ),
            "hydrogen_bond_donor_count": (
                property_data.get("HBondDonorCount")
            ),
            "hydrogen_bond_acceptor_count": (
                property_data.get("HBondAcceptorCount")
            ),
            "rotatable_bond_count": (
                property_data.get("RotatableBondCount")
            ),
            "complexity": property_data.get(
                "Complexity",
            ),
            "formal_charge": property_data.get(
                "Charge",
            ),
        }

        metadata = {
            key: value
            for key, value in metadata.items()
            if value is not None
        }

        record_url = None

        if cid is not None:
            record_url = (
                "https://pubchem.ncbi.nlm.nih.gov/"
                f"compound/{cid}"
            )

        return KnowledgeEvidence(
            source_code=self.code,
            domain=KnowledgeDomain.DRUG_MONOGRAPH,
            subject=str(preferred_name),
            title=(
                f"PubChem compound record: "
                f"{preferred_name}"
            ),
            summary=summary,
            related_agents=[
                str(preferred_name),
                query.term,
            ],
            evidence_strength=(
                EvidenceStrength.HIGH
            ),
            confidence=0.95,
            country="US",
            language="en",
            references=[
                KnowledgeReference(
                    title=(
                        f"PubChem Compound Record "
                        f"for {preferred_name}"
                    ),
                    url=record_url,
                    publisher=(
                        "National Center for "
                        "Biotechnology Information"
                    ),
                    citation=(
                        f"PubChem CID {cid}"
                        if cid is not None
                        else "PubChem compound record"
                    ),
                )
            ],
            raw_identifiers=identifiers,
            metadata=metadata,
            requires_professional_review=True,
        )

    @staticmethod
    def _build_summary(
        preferred_name: str,
        cid: int | str | None,
        formula: str | None,
        molecular_weight: str | float | None,
        iupac_name: str | None,
    ) -> str:
        parts = [
            f"Compound: {preferred_name}.",
        ]

        if cid is not None:
            parts.append(
                f"PubChem CID: {cid}.",
            )

        if iupac_name:
            parts.append(
                f"IUPAC name: {iupac_name}.",
            )

        if formula:
            parts.append(
                f"Molecular formula: {formula}.",
            )

        if molecular_weight is not None:
            parts.append(
                f"Molecular weight: "
                f"{molecular_weight} g/mol.",
            )

        return " ".join(parts)