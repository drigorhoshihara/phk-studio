"""
PHK Studio
Clinical Pharmacy Engine

Agregador de fontes clínicas.
"""

from __future__ import annotations

import asyncio
from time import perf_counter

from app.clinical_pharmacy_engine.knowledge_base.cache import (
    ClinicalKnowledgeCache,
)
from app.clinical_pharmacy_engine.knowledge_base.models import (
    KnowledgeEvidence,
    KnowledgeQuery,
    KnowledgeSearchResult,
    SourceExecutionResult,
    SourceStatus,
)
from app.clinical_pharmacy_engine.knowledge_base.registry import (
    ClinicalKnowledgeRegistry,
)


class ClinicalKnowledgeAggregator:
    """
    Consulta múltiplas fontes em paralelo e consolida evidências.
    """

    def __init__(
        self,
        registry: ClinicalKnowledgeRegistry,
        cache: ClinicalKnowledgeCache | None = None,
        timeout_seconds: float = 20.0,
        max_concurrent_sources: int = 5,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds deve ser maior que zero.",
            )

        if max_concurrent_sources < 1:
            raise ValueError(
                "max_concurrent_sources deve ser maior que zero.",
            )

        self.registry = registry

        self.cache = (
            cache
            or ClinicalKnowledgeCache()
        )

        self.timeout_seconds = timeout_seconds
        self.max_concurrent_sources = (
            max_concurrent_sources
        )

    async def search(
        self,
        query: KnowledgeQuery,
        use_cache: bool = True,
    ) -> KnowledgeSearchResult:
        normalized_term = query.normalized_term()

        if len(normalized_term) < 2:
            raise ValueError(
                "O termo de consulta deve possuir "
                "pelo menos dois caracteres.",
            )

        selected_codes = (
            self._normalize_source_codes(
                query.source_codes,
            )
            if query.source_codes
            else self.registry.list_codes()
        )

        semaphore = asyncio.Semaphore(
            self.max_concurrent_sources,
        )

        tasks = [
            self._search_source(
                source_code=source_code,
                query=query,
                semaphore=semaphore,
                use_cache=use_cache,
            )
            for source_code in selected_codes
        ]

        execution_results = await asyncio.gather(
            *tasks,
        )

        successful_sources: list[str] = []
        failed_sources: dict[str, str] = {}
        unavailable_sources: list[str] = []
        all_evidences: list[KnowledgeEvidence] = []

        for execution in execution_results:
            if execution.success:
                successful_sources.append(
                    execution.source_code,
                )

                all_evidences.extend(
                    execution.evidences,
                )

                continue

            if execution.error == "source_unavailable":
                unavailable_sources.append(
                    execution.source_code,
                )
            else:
                failed_sources[
                    execution.source_code
                ] = execution.error or "unknown_error"

        total_before = len(all_evidences)

        evidences = self._deduplicate(
            all_evidences,
        )

        evidences = self._rank(
            evidences,
            query,
        )

        warnings: list[str] = []

        if unavailable_sources:
            warnings.append(
                "Uma ou mais fontes estão indisponíveis "
                "ou ainda não foram configuradas."
            )

        if not evidences:
            warnings.append(
                "Nenhuma evidência foi encontrada nas fontes "
                "consultadas. Isso não exclui informação clínica "
                "ausente da base configurada."
            )

        return KnowledgeSearchResult(
            query=query,
            evidences=evidences,
            successful_sources=successful_sources,
            failed_sources=failed_sources,
            unavailable_sources=unavailable_sources,
            total_before_deduplication=total_before,
            total_results=len(evidences),
            warnings=warnings,
        )

    async def _search_source(
        self,
        source_code: str,
        query: KnowledgeQuery,
        semaphore: asyncio.Semaphore,
        use_cache: bool,
    ) -> SourceExecutionResult:
        started_at = perf_counter()

        try:
            source = self.registry.get(
                source_code,
            )

        except Exception as error:
            return SourceExecutionResult(
                source_code=source_code,
                success=False,
                error=(
                    f"{type(error).__name__}: {error}"
                ),
                elapsed_seconds=(
                    perf_counter() - started_at
                ),
            )

        if source.status != SourceStatus.AVAILABLE:
            return SourceExecutionResult(
                source_code=source_code,
                success=False,
                error="source_unavailable",
                elapsed_seconds=(
                    perf_counter() - started_at
                ),
            )

        if use_cache:
            cached = self.cache.get(
                source_code,
                query,
            )

            if cached is not None:
                return SourceExecutionResult(
                    source_code=source_code,
                    success=True,
                    evidences=cached,
                    elapsed_seconds=(
                        perf_counter() - started_at
                    ),
                    from_cache=True,
                )

        try:
            async with semaphore:
                evidences = await asyncio.wait_for(
                    source.search(query),
                    timeout=self.timeout_seconds,
                )

            if use_cache:
                self.cache.set(
                    source_code,
                    query,
                    evidences,
                )

            return SourceExecutionResult(
                source_code=source_code,
                success=True,
                evidences=evidences,
                elapsed_seconds=(
                    perf_counter() - started_at
                ),
            )

        except asyncio.CancelledError:
            raise

        except Exception as error:
            return SourceExecutionResult(
                source_code=source_code,
                success=False,
                error=(
                    f"{type(error).__name__}: {error}"
                ),
                elapsed_seconds=(
                    perf_counter() - started_at
                ),
            )

    @staticmethod
    def _normalize_source_codes(
        source_codes: list[str],
    ) -> list[str]:
        return list(
            dict.fromkeys(
                code.strip().casefold()
                for code in source_codes
                if code and code.strip()
            )
        )

    @staticmethod
    def _deduplicate(
        evidences: list[KnowledgeEvidence],
    ) -> list[KnowledgeEvidence]:
        unique: list[KnowledgeEvidence] = []
        seen: set[str] = set()

        for evidence in evidences:
            doi = evidence.raw_identifiers.get(
                "doi",
            )

            pmid = evidence.raw_identifiers.get(
                "pmid",
            )

            if doi:
                key = f"doi:{doi.casefold()}"

            elif pmid:
                key = f"pmid:{pmid}"

            else:
                normalized_title = " ".join(
                    evidence.title.casefold().split()
                )

                key = (
                    f"{evidence.domain.value}:"
                    f"{normalized_title}:"
                    f"{evidence.subject.casefold()}"
                )

            if key in seen:
                continue

            seen.add(key)
            unique.append(evidence)

        return unique

    @staticmethod
    def _rank(
        evidences: list[KnowledgeEvidence],
        query: KnowledgeQuery,
    ) -> list[KnowledgeEvidence]:
        query_terms = {
            term
            for term in query.normalized_term().split()
            if len(term) >= 2
        }

        strength_weight = {
            "very_high": 5.0,
            "high": 4.0,
            "moderate": 3.0,
            "low": 2.0,
            "very_low": 1.0,
            "unassessed": 0.0,
        }

        def score(
            evidence: KnowledgeEvidence,
        ) -> tuple[float, float, str]:
            searchable = " ".join(
                [
                    evidence.subject,
                    evidence.title,
                    evidence.summary,
                    evidence.clinical_effect or "",
                    evidence.mechanism or "",
                ]
            ).casefold()

            text_score = float(
                sum(
                    term in searchable
                    for term in query_terms
                )
            )

            evidence_score = strength_weight.get(
                evidence.evidence_strength.value,
                0.0,
            )

            return (
                text_score + evidence_score,
                evidence.confidence,
                evidence.title.casefold(),
            )

        return sorted(
            evidences,
            key=score,
            reverse=True,
        )