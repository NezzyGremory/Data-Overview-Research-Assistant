from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from app.nlp.domain_detector import DomainDetector
from app.nlp.keyword_extractor import KeywordExtractor

from app.research.deduplication import PaperDeduplicator
from app.research.gap_analyzer import ResearchGapAnalyzer
from app.research.landscape import ResearchLandscapeAnalyzer
from app.research.paper import Paper
from app.research.query_builder import ResearchQueryBuilder
from app.research.ranking import PaperRanker
from app.research.search import AcademicSearchEngine
from app.research.trend_analyzer import ResearchTrendAnalyzer


class ResearchIntelligenceEngine:
    """
    Research Intelligence Engine.

    Pipeline:

        Dataset
            ↓
        Keyword Extraction
            ↓
        Domain Detection
            ↓
        Query Building
            ↓
        Academic Search
            ↓
        Deduplication
            ↓
        Paper Ranking
            ↓
        Research Landscape
            ↓
        Research Trend
            ↓
        Potential Research Gap
            ↓
        Research Summary

    Catatan:
    - FACT:
        Data yang benar-benar berasal dari sumber akademik.
    - HEURISTIC:
        Hasil estimasi/perhitungan sistem.
    - POTENTIAL_GAP:
        Kandidat research gap, bukan klaim ilmiah mutlak.
    """

    def __init__(
        self,
        keyword_extractor=None,
        domain_detector=None,
        search_engine=None,
        ranker=None,
        deduplicator=None,
        query_builder=None,
        landscape_analyzer=None,
        trend_analyzer=None,
        gap_analyzer=None,
    ):
        self.keyword_extractor = keyword_extractor or KeywordExtractor()
        self.domain_detector = domain_detector or DomainDetector()
        self.search_engine = search_engine or AcademicSearchEngine()
        self.ranker = ranker or PaperRanker()
        self.deduplicator = deduplicator or PaperDeduplicator()
        self.query_builder = query_builder or ResearchQueryBuilder()
        self.landscape_analyzer = (
            landscape_analyzer or ResearchLandscapeAnalyzer()
        )
        self.trend_analyzer = trend_analyzer or ResearchTrendAnalyzer()
        self.gap_analyzer = gap_analyzer or ResearchGapAnalyzer()

    # =========================================================
    # MAIN
    # =========================================================

    def analyze(
        self,
        dataframe,
        fingerprint: Dict,
        ml_result: Dict | None = None,
        search_limit: int = 20,
        max_queries: int = 5,
        include_insights: bool = True,
    ) -> Dict:
        """Menjalankan seluruh pipeline research intelligence."""

        pipeline = {
            "keyword_extraction": {"status": "PENDING", "message": ""},
            "domain_detection": {"status": "PENDING", "message": ""},
            "query_building": {"status": "PENDING", "message": ""},
            "academic_search": {
                "status": "PENDING",
                "message": "",
                "queries": 0,
                "papers": 0,
            },
            "deduplication": {"status": "PENDING", "message": ""},
            "ranking": {"status": "PENDING", "message": ""},
            "landscape": {"status": "PENDING", "message": ""},
            "trend": {"status": "PENDING", "message": ""},
            "gap_analysis": {"status": "PENDING", "message": ""},
        }

        keyword_result: Dict[str, Any] = {}
        domain_result: Dict[str, Any] = {}
        queries: List[str] = []

        all_papers: List[Paper] = []
        unique_papers: List[Paper] = []
        paper_dicts: List[Dict[str, Any]] = []
        ranked_papers: List[Dict[str, Any]] = []

        source_counts: Dict[str, int] = {}
        provider_status_runs: Dict[str, List[str]] = {}
        search_errors: List[Dict[str, Any]] = []

        landscape: Dict[str, Any] = {}
        trend: Dict[str, Any] = {}
        gaps: Dict[str, Any] = {}

        # =====================================================
        # 1. KEYWORD EXTRACTION
        # =====================================================

        try:
            keyword_result = self._extract_keywords(
                dataframe=dataframe,
                fingerprint=fingerprint,
            )

            if not isinstance(keyword_result, dict):
                keyword_result = {
                    "status": "SUCCESS",
                    "keywords": self._normalize_list(keyword_result),
                }

            extracted_keywords = keyword_result.get("keywords", [])

            if isinstance(extracted_keywords, str):
                extracted_keywords = [extracted_keywords]

            if not isinstance(extracted_keywords, list):
                extracted_keywords = []

            keyword_result["keywords"] = extracted_keywords

            pipeline["keyword_extraction"] = {
                "status": "SUCCESS",
                "message": (
                    "Keyword extraction berhasil. "
                    f"{len(extracted_keywords)} keyword ditemukan."
                ),
                "keyword_count": len(extracted_keywords),
                "keywords": extracted_keywords,
            }

        except Exception as exc:
            pipeline["keyword_extraction"] = {
                "status": "ERROR",
                "message": str(exc),
                "keyword_count": 0,
                "keywords": [],
            }
            keyword_result = {
                "status": "ERROR",
                "keywords": [],
                "error": str(exc),
            }

        # =====================================================
        # 2. DOMAIN DETECTION
        # =====================================================

        try:
            domain_result = self._detect_domain(
                keyword_result=keyword_result,
                dataframe=dataframe,
                fingerprint=fingerprint,
            )

            if not isinstance(domain_result, dict):
                domain_result = {
                    "status": "ESTIMATION",
                    "primary_domain": str(domain_result),
                    "domains": [],
                    "possible_domains": [],
                }

            primary_domain = (
                domain_result.get("primary_domain")
                or domain_result.get("domain")
            )

            domains = domain_result.get("domains", [])
            if not isinstance(domains, list):
                domains = []

            domain_result["domains"] = domains

            # Compatibility alias for older UI/code.
            if "possible_domains" not in domain_result:
                domain_result["possible_domains"] = domains

            pipeline["domain_detection"] = {
                "status": "SUCCESS",
                "message": (
                    "Domain detection berhasil."
                    if primary_domain
                    else "Domain detector berjalan tetapi belum menemukan domain utama."
                ),
                "domain": primary_domain,
                "confidence": domain_result.get("confidence", 0.0),
                "domains": domains,
                "matched_keywords": domain_result.get(
                    "matched_keywords", []
                ),
            }

        except Exception as exc:
            pipeline["domain_detection"] = {
                "status": "ERROR",
                "message": str(exc),
                "domain": None,
                "confidence": 0.0,
                "domains": [],
            }

            domain_result = {
                "status": "ERROR",
                "primary_domain": None,
                "domains": [],
                "possible_domains": [],
                "confidence": 0.0,
                "error": str(exc),
            }

        # =====================================================
        # 3. QUERY BUILDING
        # =====================================================

        try:
            queries = self._build_queries(
                fingerprint=fingerprint,
                domain_result=domain_result,
                ml_result=ml_result,
                max_queries=max_queries,
                keyword_result=keyword_result,
            )

            queries = self._normalize_queries(queries)

            pipeline["query_building"] = {
                "status": "SUCCESS" if queries else "EMPTY",
                "message": (
                    f"{len(queries)} query berhasil dibuat."
                    if queries
                    else "Query builder tidak menghasilkan query."
                ),
                "query_count": len(queries),
                "queries": queries,
            }

        except Exception as exc:
            pipeline["query_building"] = {
                "status": "ERROR",
                "message": str(exc),
                "query_count": 0,
                "queries": [],
            }
            queries = []

        # =====================================================
        # 4. ACADEMIC SEARCH
        # =====================================================

        pipeline["academic_search"]["queries"] = len(queries)

        def search_query(query):
            try:
                return self._search(query=query, limit=search_limit)
            except Exception as exc:
                return {
                    "status": "ERROR",
                    "papers": [],
                    "sources": [],
                    "message": str(exc),
                    "error": str(exc),
                }

        # Run a small number of independent query searches together. Each
        # search already fans out to its providers, so keeping this at two
        # avoids an unbounded burst of requests while reducing serial waits.
        if len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(2, len(queries))) as executor:
                search_results = list(executor.map(search_query, queries))
        else:
            search_results = [search_query(query) for query in queries]

        for query, search_result in zip(queries, search_results):
            try:

                if not isinstance(search_result, dict):
                    search_result = {
                        "status": "ERROR",
                        "papers": [],
                        "sources": [],
                        "message": (
                            "Search engine mengembalikan format yang tidak valid."
                        ),
                    }

                search_status = search_result.get("status", "UNKNOWN")
                for provider, status in (search_result.get("provider_status") or {}).items():
                    provider_status_runs.setdefault(str(provider), []).append(str(status))
                papers = search_result.get("papers", [])

                if not isinstance(papers, list):
                    papers = []

                all_papers.extend(self._dicts_to_papers(papers))

                # Use provider-level candidate counts when available. This keeps
                # the research source distribution visible even when duplicate
                # papers from one provider are removed during deduplication.
                sources = search_result.get("source_counts")
                if sources is None:
                    sources = search_result.get("sources", [])

                if isinstance(sources, dict):
                    for source, count in sources.items():
                        try:
                            source_counts[source] = (
                                source_counts.get(source, 0) + int(count)
                            )
                        except (TypeError, ValueError):
                            pass

                elif isinstance(sources, list):
                    for source in sources:
                        if not source:
                            continue
                        source = str(source)
                        source_counts[source] = (
                            source_counts.get(source, 0) + 1
                        )

                if search_status == "ERROR":
                    search_errors.append(
                        {
                            "query": query,
                            "message": search_result.get(
                                "message",
                                search_result.get(
                                    "error",
                                    "Unknown search error.",
                                ),
                            ),
                        }
                    )

            except Exception as exc:
                search_errors.append(
                    {
                        "query": query,
                        "message": str(exc),
                    }
                )

        pipeline["academic_search"]["papers"] = len(all_papers)
        pipeline["academic_search"]["sources"] = source_counts
        pipeline["academic_search"]["provider_status"] = {
            provider: (
                "ENABLED" if "ENABLED" in statuses
                else "ERROR" if "ERROR" in statuses
                else statuses[-1]
            )
            for provider, statuses in provider_status_runs.items()
            if statuses
        }
        pipeline["academic_search"]["errors"] = search_errors

        if all_papers:
            pipeline["academic_search"]["status"] = "SUCCESS"
            pipeline["academic_search"]["message"] = (
                f"{len(all_papers)} paper berhasil dikumpulkan."
            )
        elif search_errors:
            pipeline["academic_search"]["status"] = "ERROR"
            pipeline["academic_search"]["message"] = (
                "Academic search tidak menghasilkan paper."
            )
        else:
            pipeline["academic_search"]["status"] = "EMPTY"
            pipeline["academic_search"]["message"] = (
                "Search selesai tetapi tidak ada paper."
            )

        # =====================================================
        # 5. DEDUPLICATION
        # =====================================================

        try:
            unique_papers = self.deduplicator.deduplicate(all_papers)

            if unique_papers is None:
                unique_papers = []

            pipeline["deduplication"] = {
                "status": "SUCCESS",
                "message": (
                    f"{len(unique_papers)} paper unik setelah deduplikasi."
                ),
                "input_count": len(all_papers),
                "output_count": len(unique_papers),
            }

        except Exception as exc:
            pipeline["deduplication"] = {
                "status": "ERROR",
                "message": str(exc),
                "input_count": len(all_papers),
                "output_count": len(all_papers),
            }
            unique_papers = all_papers

        # =====================================================
        # 6. CONVERT TO DICT
        # =====================================================

        paper_dicts = [
            self._paper_to_dict(paper)
            for paper in unique_papers
        ]

        # =====================================================
        # 7. RANKING
        # =====================================================

        try:
            ranked_papers = self.ranker.rank(
                papers=paper_dicts,
                fingerprint=fingerprint,
                domain_result=domain_result,
                ml_result=ml_result,
            )

            if ranked_papers is None:
                ranked_papers = []

            if not isinstance(ranked_papers, list):
                ranked_papers = list(ranked_papers)

            pipeline["ranking"] = {
                "status": "SUCCESS",
                "message": (
                    f"{len(ranked_papers)} paper berhasil diberi ranking."
                ),
            }

        except Exception as exc:
            pipeline["ranking"] = {
                "status": "ERROR",
                "message": str(exc),
            }

            ranked_papers = paper_dicts

            for paper in ranked_papers:
                if not isinstance(paper, dict):
                    continue

                paper.setdefault("relevance_score", 0)
                paper.setdefault("ranking_status", "FALLBACK")

        if include_insights:
            try:
                landscape = self.landscape_analyzer.analyze(papers=ranked_papers, ml_result=ml_result)
                if not isinstance(landscape, dict):
                    landscape = {"status": "SUCCESS", "result": landscape}
                pipeline["landscape"] = {"status": "SUCCESS", "message": "Research landscape berhasil dianalisis."}
            except Exception as exc:
                pipeline["landscape"] = {"status": "ERROR", "message": str(exc)}
                landscape = {"status": "ERROR", "error": str(exc), "summary": {}}

            try:
                trend = self.trend_analyzer.analyze(papers=ranked_papers, landscape=landscape)
                if not isinstance(trend, dict):
                    trend = {"status": "SUCCESS", "result": trend}
                pipeline["trend"] = {"status": "SUCCESS", "message": "Research trend berhasil dianalisis."}
            except Exception as exc:
                pipeline["trend"] = {"status": "ERROR", "message": str(exc)}
                trend = {"status": "ERROR", "error": str(exc)}

            try:
                gaps = self.gap_analyzer.analyze(papers=ranked_papers, ml_result=ml_result, landscape=landscape)
                if not isinstance(gaps, dict):
                    gaps = {"status": "SUCCESS", "result": gaps}
                pipeline["gap_analysis"] = {"status": "SUCCESS", "message": "Potential research gap berhasil dianalisis."}
            except Exception as exc:
                pipeline["gap_analysis"] = {"status": "ERROR", "message": str(exc)}
                gaps = {"status": "ERROR", "error": str(exc), "summary": {"gap_count": 0}}

            summary = self._build_summary(ranked_papers, landscape, trend, gaps)
            insights_status = "READY"
        else:
            for key, label in (("landscape", "Landscape"), ("trend", "Trend"), ("gap_analysis", "Research gap")):
                pipeline[key] = {
                    "status": "SKIPPED",
                    "message": f"Dihitung saat halaman {label} dibuka.",
                }
            landscape = {"status": "SKIPPED", "summary": {}}
            trend = {"status": "SKIPPED"}
            gaps = {"status": "SKIPPED", "summary": {"gap_count": 0}}
            summary = self._build_summary(ranked_papers, landscape, trend, gaps)
            summary["potential_gap_count"] = None
            summary["research_direction"] = "NOT_COMPUTED"
            insights_status = "NOT_COMPUTED"

        # =====================================================
        # 12. OVERALL STATUS
        # =====================================================

        overall_status = "SUCCESS" if ranked_papers else "PARTIAL"

        # =====================================================
        # 13. RETURN
        # =====================================================

        return {
            "status": overall_status,
            "keywords": keyword_result,
            "domain": domain_result,
            "queries": queries,
            "search": {
                "result_count": len(paper_dicts),
                "raw_result_count": len(all_papers),
                "unique_result_count": len(unique_papers),
                "sources": source_counts,
                "source_counts": source_counts,
                "errors": search_errors,
            },
            "papers": ranked_papers,
            "top_papers": ranked_papers[:10],
            "landscape": landscape,
            "trend": trend,
            "gaps": gaps,
            "summary": summary,
            "pipeline": pipeline,
            "insights_status": insights_status,
            "transparency": {
                "keyword_extraction": "HEURISTIC",
                "domain_detection": "ESTIMATION",
                "query_generation": "RECOMMENDATION",
                "paper_search": "FACT",
                "paper_ranking": "HEURISTIC",
                "landscape": "HEURISTIC",
                "trend": "HEURISTIC",
                "research_gap": "POTENTIAL_GAP",
            },
        }

    def analyze_insights(self, result: Dict[str, Any], ml_result: Dict | None = None) -> Dict[str, Any]:
        """Compute landscape, trend and gap output on demand without searching again."""
        if not isinstance(result, dict):
            result = {}
        papers = result.get("papers", [])
        papers = papers if isinstance(papers, list) else []
        pipeline = result.setdefault("pipeline", {})
        if not isinstance(pipeline, dict):
            pipeline = {}
            result["pipeline"] = pipeline

        try:
            landscape = self.landscape_analyzer.analyze(papers=papers, ml_result=ml_result)
            if not isinstance(landscape, dict):
                landscape = {"status": "SUCCESS", "result": landscape}
            pipeline["landscape"] = {"status": "SUCCESS", "message": "Research landscape berhasil dianalisis."}
        except Exception as exc:
            landscape = {"status": "ERROR", "error": str(exc), "summary": {}}
            pipeline["landscape"] = {"status": "ERROR", "message": str(exc)}

        try:
            trend = self.trend_analyzer.analyze(papers=papers, landscape=landscape)
            if not isinstance(trend, dict):
                trend = {"status": "SUCCESS", "result": trend}
            pipeline["trend"] = {"status": "SUCCESS", "message": "Research trend berhasil dianalisis."}
        except Exception as exc:
            trend = {"status": "ERROR", "error": str(exc)}
            pipeline["trend"] = {"status": "ERROR", "message": str(exc)}

        try:
            gaps = self.gap_analyzer.analyze(papers=papers, ml_result=ml_result, landscape=landscape)
            if not isinstance(gaps, dict):
                gaps = {"status": "SUCCESS", "result": gaps}
            pipeline["gap_analysis"] = {"status": "SUCCESS", "message": "Potential research gap berhasil dianalisis."}
        except Exception as exc:
            gaps = {"status": "ERROR", "error": str(exc), "summary": {"gap_count": 0}}
            pipeline["gap_analysis"] = {"status": "ERROR", "message": str(exc)}

        result.update({
            "landscape": landscape,
            "trend": trend,
            "gaps": gaps,
            "summary": self._build_summary(papers, landscape, trend, gaps),
            "pipeline": pipeline,
            "insights_status": "READY",
        })
        return result

    # =========================================================
    # KEYWORD ADAPTER
    # =========================================================

    def _extract_keywords(self, dataframe, fingerprint):
        extractor = self.keyword_extractor

        try:
            result = extractor.extract(
                dataframe=dataframe,
                fingerprint=fingerprint,
            )
            if result is not None:
                return result
        except TypeError:
            pass

        try:
            result = extractor.extract(dataframe, fingerprint)
            if result is not None:
                return result
        except TypeError:
            pass

        try:
            result = extractor.extract(fingerprint)
            if result is not None:
                return result
        except TypeError:
            pass

        representation = (
            fingerprint.get("representation", fingerprint)
            if isinstance(fingerprint, dict)
            else {}
        )

        if not isinstance(representation, dict):
            representation = {}

        return {
            "status": "FALLBACK",
            "keywords": self._normalize_list(
                representation.get("keywords", [])
            ),
            "status_type": "HEURISTIC",
        }

    # =========================================================
    # DOMAIN ADAPTER
    # =========================================================

    def _detect_domain(
        self,
        keyword_result: Dict[str, Any] | None = None,
        dataframe=None,
        fingerprint: Dict | None = None,
    ):
        """
        Menghubungkan KeywordExtractor/Fingerprint ke DomainDetector.

        DomainDetector final menerima keyword sebagai sumber utama,
        tetapi engine tetap memberi dataframe + fingerprint bila detector
        versi yang lebih baru membutuhkannya.

        Urutan pemanggilan:
        1. detect(keywords=..., dataframe=..., fingerprint=...)
        2. detect(keywords=...)
        3. detect(dataframe=..., fingerprint=...)
        4. detect(fingerprint)
        """

        detector = self.domain_detector
        keyword_result = (
            keyword_result if isinstance(keyword_result, dict) else {}
        )
        fingerprint = (
            fingerprint if isinstance(fingerprint, dict) else {}
        )

        keywords = keyword_result.get("keywords", [])

        if not keywords:
            representation = fingerprint.get(
                "representation",
                fingerprint,
            )
            if isinstance(representation, dict):
                keywords = representation.get("keywords", [])

        keywords = self._normalize_list(keywords)

        # Primary call for the final flexible detector.
        try:
            result = detector.detect(
                keywords=keywords,
                dataframe=dataframe,
                fingerprint=fingerprint,
            )
            if result is not None:
                return result
        except TypeError:
            pass

        # Compatibility with detector that only accepts keywords.
        try:
            result = detector.detect(keywords)
            if result is not None:
                return result
        except TypeError:
            pass

        # Compatibility with older/custom detectors.
        try:
            result = detector.detect(
                dataframe=dataframe,
                fingerprint=fingerprint,
            )
            if result is not None:
                return result
        except TypeError:
            pass

        try:
            result = detector.detect(dataframe, fingerprint)
            if result is not None:
                return result
        except TypeError:
            pass

        try:
            result = detector.detect(fingerprint)
            if result is not None:
                return result
        except TypeError:
            pass

        return {
            "status": "FALLBACK",
            "primary_domain": None,
            "domains": [],
            "possible_domains": [],
            "confidence": 0.0,
            "matched_keywords": [],
            "status_type": "ESTIMATION",
            "message": (
                "Domain detector tidak menghasilkan hasil yang kompatibel."
            ),
        }

    # =========================================================
    # QUERY BUILDER
    # =========================================================

    def _build_queries(
        self,
        fingerprint,
        domain_result,
        ml_result,
        max_queries,
        keyword_result=None,
    ):
        builder = self.query_builder

        try:
            result = builder.build(
                fingerprint=fingerprint,
                domain_result=domain_result,
                ml_result=ml_result,
                keyword_result=keyword_result,
                max_queries=max_queries,
            )
            return self._extract_queries(result)[:max_queries]
        except TypeError:
            pass

        try:
            result = builder.build(
                fingerprint,
                domain_result,
                ml_result,
                keyword_result,
                max_queries,
            )
            return self._extract_queries(result)[:max_queries]
        except TypeError:
            pass

        try:
            result = builder.build(
                fingerprint=fingerprint,
                domain_result=domain_result,
                ml_result=ml_result,
                max_queries=max_queries,
            )
            extracted = self._extract_queries(result)
            if extracted:
                return extracted[:max_queries]
        except TypeError:
            pass

        fallback_queries = self._build_fallback_queries(
            fingerprint=fingerprint,
            domain_result=domain_result,
            ml_result=ml_result,
            keyword_result=keyword_result,
        )

        return fallback_queries[:max_queries]

    # =========================================================
    # EXTRACT QUERIES
    # =========================================================

    @staticmethod
    def _extract_queries(result) -> List[str]:
        if result is None:
            return []

        if isinstance(result, dict):
            queries = result.get("queries", [])
        elif isinstance(result, str):
            queries = [result]
        elif isinstance(result, (list, tuple, set)):
            queries = list(result)
        else:
            queries = []

        return ResearchIntelligenceEngine._normalize_queries(queries)

    # =========================================================
    # FALLBACK QUERY BUILDER
    # =========================================================

    @staticmethod
    def _build_fallback_queries(
        fingerprint,
        domain_result,
        ml_result,
        keyword_result,
    ) -> List[str]:
        queries = []

        keywords = []
        if isinstance(keyword_result, dict):
            keywords = keyword_result.get("keywords", [])

        if not keywords:
            representation = (
                fingerprint.get("representation", fingerprint)
                if isinstance(fingerprint, dict)
                else {}
            )
            if isinstance(representation, dict):
                keywords = representation.get("keywords", [])

        domain = ""
        if isinstance(domain_result, dict):
            domain = (
                domain_result.get("primary_domain", "")
                or domain_result.get("domain", "")
                or ""
            )

        task = ""
        if isinstance(ml_result, dict):
            primary_task = ml_result.get("primary_task", "")
            if isinstance(primary_task, dict):
                task = (
                    primary_task.get("task", "")
                    or primary_task.get("name", "")
                    or ""
                )
            else:
                task = str(primary_task).strip() if primary_task else ""

        if isinstance(keywords, str):
            keywords = [keywords]

        if not isinstance(keywords, (list, tuple, set)):
            keywords = []

        keywords = [
            str(k).strip()
            for k in keywords
            if k
        ]
        keywords = [k for k in keywords if len(k) >= 2]

        if keywords:
            queries.append(" ".join(keywords[:5]))

        if domain and keywords:
            queries.append(f"{domain} {' '.join(keywords[:4])}")

        if task and keywords:
            queries.append(f"{task} {' '.join(keywords[:4])}")

        if domain and task:
            queries.append(f"{domain} {task}")

        columns = (
            fingerprint.get("columns", [])
            if isinstance(fingerprint, dict)
            else []
        )

        if isinstance(columns, list):
            column_names = []

            for column in columns[:6]:
                if isinstance(column, dict):
                    name = column.get("name") or column.get("column")
                else:
                    name = column

                if name:
                    column_names.append(str(name))

            if column_names:
                queries.append(" ".join(column_names))

        return ResearchIntelligenceEngine._normalize_queries(queries)

    # =========================================================
    # SEARCH ADAPTER
    # =========================================================

    def _search(self, query, limit):
        engine = self.search_engine

        try:
            result = engine.search(
                query=query,
                limit=limit,
            )

            if result is None:
                return {
                    "status": "ERROR",
                    "papers": [],
                    "sources": [],
                    "message": "Search engine mengembalikan None.",
                }

            return result
        except TypeError:
            pass

        try:
            result = engine.search(query, limit)

            if result is None:
                return {
                    "status": "ERROR",
                    "papers": [],
                    "sources": [],
                    "message": "Search engine mengembalikan None.",
                }

            return result
        except Exception as exc:
            return {
                "status": "ERROR",
                "papers": [],
                "sources": [],
                "message": str(exc),
                "error": str(exc),
            }

    # =========================================================
    # PAPER CONVERSION
    # =========================================================

    @staticmethod
    def _dicts_to_papers(papers: List[Any]) -> List[Paper]:
        result = []

        if not isinstance(papers, list):
            return result

        for item in papers:
            if isinstance(item, Paper):
                result.append(item)
                continue

            if not isinstance(item, dict):
                continue

            try:
                authors = item.get("authors", [])
                if not isinstance(authors, list):
                    authors = [str(authors)]

                keywords = item.get("keywords", [])
                if not isinstance(keywords, list):
                    keywords = [str(keywords)]

                issn = item.get("issn", [])
                if isinstance(issn, str):
                    issn = [issn]
                elif not isinstance(issn, list):
                    issn = []

                citation_count = item.get("citation_count", 0)

                try:
                    citation_count = int(citation_count or 0)
                except (TypeError, ValueError):
                    citation_count = 0

                result.append(
                    Paper(
                        title=str(item.get("title", "") or ""),
                        authors=authors,
                        abstract=str(item.get("abstract", "") or ""),
                        year=item.get("year"),
                        doi=item.get("doi"),
                        venue=item.get("venue"),
                        url=item.get("url"),
                        citation_count=citation_count,
                        source=str(
                            item.get("source", "unknown") or "unknown"
                        ),
                        external_id=item.get("external_id"),
                        keywords=keywords,
                        issn=issn,
                    )
                )
            except Exception:
                continue

        return result

    # =========================================================
    # PAPER TO DICT
    # =========================================================

    @staticmethod
    def _paper_to_dict(paper: Paper) -> Dict[str, Any]:
        if isinstance(paper, dict):
            return dict(paper)

        if hasattr(paper, "to_dict"):
            result = paper.to_dict()
            if isinstance(result, dict):
                return result

        return {
            "title": getattr(paper, "title", ""),
            "authors": getattr(paper, "authors", []),
            "abstract": getattr(paper, "abstract", ""),
            "year": getattr(paper, "year", None),
            "doi": getattr(paper, "doi", None),
            "venue": getattr(paper, "venue", None),
            "url": getattr(paper, "url", None),
            "citation_count": getattr(paper, "citation_count", 0),
            "source": getattr(paper, "source", "unknown"),
            "external_id": getattr(paper, "external_id", None),
            "keywords": getattr(paper, "keywords", []),
            "issn": getattr(paper, "issn", []),
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    @staticmethod
    def _build_summary(
        ranked_papers,
        landscape,
        trend,
        gaps,
    ):
        if not isinstance(ranked_papers, list):
            ranked_papers = []

        if not isinstance(landscape, dict):
            landscape = {}

        if not isinstance(trend, dict):
            trend = {}

        if not isinstance(gaps, dict):
            gaps = {}

        top_paper = (
            ranked_papers[0]
            if ranked_papers
            and isinstance(ranked_papers[0], dict)
            else None
        )

        gap_summary = gaps.get("summary", {})
        if not isinstance(gap_summary, dict):
            gap_summary = {}

        landscape_summary = landscape.get("summary", {})
        if not isinstance(landscape_summary, dict):
            landscape_summary = {}

        return {
            "papers_found": len(ranked_papers),
            "top_paper": top_paper,
            "top_relevance_score": (
                top_paper.get("relevance_score")
                if top_paper
                else None
            ),
            "potential_gap_count": gap_summary.get("gap_count", 0),
            "research_direction": trend.get(
                "recent_direction",
                "UNKNOWN",
            ),
            "dominant_method": landscape_summary.get(
                "dominant_method"
            ),
            "latest_publication_year": landscape_summary.get(
                "latest_publication_year"
            ),
        }

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _normalize_queries(queries) -> List[str]:
        if queries is None:
            return []

        if isinstance(queries, str):
            queries = [queries]

        if not isinstance(queries, (list, tuple, set)):
            return []

        result = []
        seen = set()

        for query in queries:
            if query is None:
                continue

            if isinstance(query, dict):
                query = (
                    query.get("query")
                    or query.get("text")
                    or ""
                )

            query = str(query).strip()
            if not query:
                continue

            normalized = " ".join(query.split())
            key = normalized.lower()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result

    @staticmethod
    def _normalize_list(value) -> List[str]:
        if value is None:
            return []

        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []

        if isinstance(value, (list, tuple, set)):
            result = []

            for item in value:
                if item is None:
                    continue

                if isinstance(item, dict):
                    item = (
                        item.get("keyword")
                        or item.get("term")
                        or item.get("name")
                        or item.get("text")
                        or ""
                    )

                text = str(item).strip()
                if text:
                    result.append(text)

            return result

        return []
