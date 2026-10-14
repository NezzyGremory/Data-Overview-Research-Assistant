from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set

try:
    from app.nlp.similarity import FingerprintSimilarity
except ImportError:
    FingerprintSimilarity = None


class PaperRanker:
    """
    Explainable heuristic ranking untuk mengukur relevansi paper
    terhadap dataset research profile.

    Komponen skor utama:
    - Semantic Similarity : 40%
    - Dataset Name Match  : 25%
    - Schema Match        : 20%
    - Keyword Match       : 10%
    - Metadata Match      : 5%

    Metadata Match terdiri dari:
    - Domain
    - Target
    - ML Method
    - Metadata completeness

    Catatan penting:
    - Relevance score bukan accuracy.
    - Relevance score bukan bukti bahwa paper menggunakan dataset yang sama.
    - Dataset Usage Confidence dipisahkan dari relevance score.
    - Semantic similarity tidak boleh dianggap sebagai bukti identitas dataset.

    Status:
    HEURISTIC
    """

    WEIGHTS = {
        "semantic": 0.40,
        "dataset_name": 0.25,
        "schema": 0.20,
        "keyword": 0.10,
        "metadata": 0.05,
    }

    CONFIDENCE_LEVELS = (
        "VERY HIGH",
        "HIGH",
        "MEDIUM",
        "LOW",
        "TOPIC ONLY",
    )

    DOMAIN_ALIASES = {
        "transportation": {
            "transport",
            "transportation",
            "passenger",
            "vehicle",
            "mobility",
            "traffic",
            "travel",
            "aviation",
            "railway",
            "road",
        },
        "healthcare": {
            "health",
            "healthcare",
            "medical",
            "medicine",
            "clinical",
            "patient",
            "disease",
            "hospital",
            "diagnosis",
        },
        "finance": {
            "finance",
            "financial",
            "bank",
            "banking",
            "credit",
            "investment",
            "stock",
            "market",
        },
        "business": {
            "business",
            "customer",
            "sales",
            "marketing",
            "retail",
            "company",
            "commerce",
        },
        "education": {
            "education",
            "student",
            "school",
            "university",
            "learning",
            "academic",
            "teaching",
        },
        "environment": {
            "environment",
            "climate",
            "pollution",
            "weather",
            "air",
            "water",
            "ecology",
        },
        "agriculture": {
            "agriculture",
            "agricultural",
            "crop",
            "farming",
            "plant",
            "soil",
            "yield",
        },
        "technology": {
            "technology",
            "software",
            "computer",
            "network",
            "cyber",
            "system",
            "application",
        },
        "social_science": {
            "social",
            "society",
            "social science",
            "behavior",
            "population",
            "demographic",
        },
    }

    TARGET_ALIASES = {
        "survived": {
            "survived",
            "survival",
            "survivor",
            "survivors",
            "mortality",
            "mortality prediction",
            "survival prediction",
        },
        "price": {
            "price",
            "pricing",
            "cost",
            "value",
            "valuation",
        },
        "class": {
            "class",
            "classification",
            "category",
            "categorization",
            "label",
        },
        "target": {
            "target",
            "prediction",
            "predictive",
            "outcome",
        },
    }

    METHOD_ALIASES = {
        "random_forest_classifier": {
            "random forest",
            "random forest classifier",
            "random forests",
            "rf",
            "rf classifier",
        },
        "random_forest_regressor": {
            "random forest",
            "random forest regression",
            "random forest regressor",
            "random forests",
        },
        "logistic_regression": {
            "logistic regression",
            "logistic",
            "logit",
        },
        "decision_tree_classifier": {
            "decision tree",
            "decision tree classifier",
            "classification tree",
        },
        "decision_tree_regressor": {
            "decision tree",
            "decision tree regression",
            "regression tree",
        },
        "svm_classifier": {
            "svm",
            "support vector machine",
            "support vector classifier",
        },
        "knn_classifier": {
            "knn",
            "k nearest neighbor",
            "k-nearest neighbor",
            "nearest neighbor",
        },
        "gradient_boosting_regressor": {
            "gradient boosting",
            "gradient boost",
            "gradient boosting regression",
        },
        "linear_regression": {
            "linear regression",
            "linear model",
        },
        "kmeans": {
            "kmeans",
            "k-means",
            "k means",
        },
        "dbscan": {
            "dbscan",
            "density based clustering",
        },
        "agglomerative_clustering": {
            "agglomerative clustering",
            "hierarchical clustering",
        },
        "isolation_forest": {
            "isolation forest",
        },
        "local_outlier_factor": {
            "local outlier factor",
            "lof",
        },
        "one_class_svm": {
            "one class svm",
            "one-class svm",
        },
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Inisialisasi PaperRanker.

        weights bersifat configurable tetapi total harus 1.0.
        """

        self.weights = dict(self.WEIGHTS)

        if weights:
            self.weights.update(weights)

        total_weight = sum(
            self.weights.get(key, 0.0)
            for key in self.WEIGHTS
        )

        if abs(total_weight - 1.0) > 0.0001:
            raise ValueError(
                "Total ranking weight harus sama dengan 1.0."
            )

        self.fingerprint_similarity = None

        if FingerprintSimilarity is not None:
            try:
                self.fingerprint_similarity = (
                    FingerprintSimilarity()
                )
            except Exception:
                self.fingerprint_similarity = None

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def rank(
        self,
        papers: List[Dict],
        fingerprint: Dict,
        domain_result: Dict | None = None,
        ml_result: Dict | None = None,
    ) -> List[Dict]:

        dataset_keywords = self._extract_keywords(
            fingerprint
        )

        dataset_domain = self._extract_domain(
            domain_result
        )

        target = self._extract_target(
            ml_result,
            fingerprint,
        )

        methods = self._extract_methods(
            ml_result
        )

        dataset_name = self._extract_dataset_name(
            fingerprint
        )

        dataset_representation = self._get_representation(
            fingerprint
        )

        ranked = []

        for paper in papers:
            scored = self._score_paper(
                paper=paper,
                fingerprint=fingerprint,
                dataset_representation=dataset_representation,
                dataset_name=dataset_name,
                dataset_keywords=dataset_keywords,
                dataset_domain=dataset_domain,
                target=target,
                methods=methods,
            )

            ranked.append(scored)

        ranked.sort(
            key=lambda item: (
                item.get(
                    "relevance_score",
                    0.0,
                ),
                item.get(
                    "dataset_usage_confidence_score",
                    0.0,
                ),
            ),
            reverse=True,
        )

        for index, item in enumerate(
            ranked,
            start=1,
        ):
            item["rank"] = index

        return ranked

    # ==========================================================
    # PAPER SCORING
    # ==========================================================

    def _score_paper(
        self,
        paper: Dict,
        fingerprint: Dict,
        dataset_representation: Dict,
        dataset_name: str | None,
        dataset_keywords: Set[str],
        dataset_domain: str | None,
        target: str | None,
        methods: Set[str],
    ) -> Dict:

        title = str(
            paper.get(
                "title",
                "",
            )
            or ""
        )

        abstract = str(
            paper.get(
                "abstract",
                "",
            )
            or ""
        )

        paper_keywords = self._normalize_list(
            paper.get(
                "keywords",
                [],
            )
        )

        paper_text = self._normalize(
            " ".join(
                [
                    title,
                    abstract,
                    " ".join(paper_keywords),
                ]
            )
        )

        # ------------------------------------------------------
        # 1. SEMANTIC SIMILARITY
        # ------------------------------------------------------

        semantic_score = self._semantic_similarity(
            fingerprint=fingerprint,
            dataset_keywords=dataset_keywords,
            dataset_domain=dataset_domain,
            target=target,
            methods=methods,
            paper_text=paper_text,
        )

        # ------------------------------------------------------
        # 2. DATASET NAME MATCH
        # ------------------------------------------------------

        dataset_name_score = self._dataset_name_score(
            dataset_name=dataset_name,
            paper=paper,
            paper_text=paper_text,
        )

        # ------------------------------------------------------
        # 3. SCHEMA MATCH
        # ------------------------------------------------------

        schema_score = self._schema_score(
            fingerprint=fingerprint,
            paper=paper,
            paper_text=paper_text,
        )

        # ------------------------------------------------------
        # 4. KEYWORD MATCH
        # ------------------------------------------------------

        keyword_score = self._keyword_score(
            keywords=dataset_keywords,
            text=paper_text,
        )

        # ------------------------------------------------------
        # 5. METADATA SUB-SCORES
        # ------------------------------------------------------

        # Diagnostic scores.
        #
        # Ketiga skor ini tetap disediakan agar sistem dapat
        # menjelaskan kenapa sebuah paper dianggap relevan.
        #
        # PENTING:
        # Ketiganya TIDAK menjadi bobot tambahan.
        # Mereka hanya menjadi bagian dari Metadata Match 5%.

        domain_score = self._domain_score(
            dataset_domain,
            paper_text,
        )

        target_score = self._target_score(
            target,
            paper_text,
        )

        method_score = self._methods_score(
            methods,
            paper_text,
        )

        metadata_score = self._metadata_score(
            paper=paper,
            dataset_domain=dataset_domain,
            target=target,
            methods=methods,
            paper_text=paper_text,
        )

        # ------------------------------------------------------
        # FINAL SCORE
        # ------------------------------------------------------

        final_score = (
            semantic_score
            * self.weights["semantic"]
            + dataset_name_score
            * self.weights["dataset_name"]
            + schema_score
            * self.weights["schema"]
            + keyword_score
            * self.weights["keyword"]
            + metadata_score
            * self.weights["metadata"]
        )

        final_score = max(
            0.0,
            min(
                100.0,
                final_score,
            ),
        )

        # ------------------------------------------------------
        # DATASET USAGE CONFIDENCE
        # ------------------------------------------------------

        usage = self._dataset_usage_confidence(
            dataset_name_score=dataset_name_score,
            schema_score=schema_score,
            keyword_score=keyword_score,
            semantic_score=semantic_score,
            paper=paper,
            paper_text=paper_text,
        )

        # ------------------------------------------------------
        # EXPLANATION
        # ------------------------------------------------------

        reasons = self._build_reasons(
            semantic_score=semantic_score,
            dataset_name_score=dataset_name_score,
            schema_score=schema_score,
            keyword_score=keyword_score,
            metadata_score=metadata_score,
            usage=usage,
        )

        return {
            **paper,

            "relevance_score": round(
                final_score,
                2,
            ),

            # ==================================================
            # SCORE BREAKDOWN
            # ==================================================
            #
            # Lima komponen pertama adalah komponen ranking utama.
            #
            # domain / target / method adalah diagnostic scores
            # untuk backward compatibility dan explainability.
            #
            # Mereka TIDAK menambah total bobot menjadi >100%.
            #
            "score_breakdown": {
                # Ranking utama
                "semantic": round(
                    semantic_score,
                    2,
                ),
                "dataset_name": round(
                    dataset_name_score,
                    2,
                ),
                "schema": round(
                    schema_score,
                    2,
                ),
                "keyword": round(
                    keyword_score,
                    2,
                ),
                "metadata": round(
                    metadata_score,
                    2,
                ),

                # Diagnostic / compatibility
                "domain": round(
                    domain_score,
                    2,
                ),
                "target": round(
                    target_score,
                    2,
                ),
                "method": round(
                    method_score,
                    2,
                ),
            },

            "score_weights": {
                "semantic": self.weights["semantic"],
                "dataset_name": self.weights["dataset_name"],
                "schema": self.weights["schema"],
                "keyword": self.weights["keyword"],
                "metadata": self.weights["metadata"],
            },

            "dataset_usage_confidence": (
                usage["confidence"]
            ),

            "dataset_usage_confidence_score": (
                usage["confidence_score"]
            ),

            "dataset_usage_evidence": (
                usage["evidence"]
            ),

            "dataset_usage_limitations": (
                usage["limitations"]
            ),

            "reasons": reasons,

            "ranking_status": "HEURISTIC",
        }

    # ==========================================================
    # SEMANTIC SIMILARITY
    # ==========================================================

    def _semantic_similarity(
        self,
        fingerprint: Dict,
        dataset_keywords: Set[str],
        dataset_domain: str | None,
        target: str | None,
        methods: Set[str],
        paper_text: str,
    ) -> float:

        if not paper_text:
            return 0.0

        profile_terms = set(
            dataset_keywords
        )

        if dataset_domain:
            normalized_domain = self._normalize(
                dataset_domain
            )

            profile_terms.add(
                normalized_domain
            )

            profile_terms.update(
                self.DOMAIN_ALIASES.get(
                    normalized_domain,
                    set(),
                )
            )

        if target:
            normalized_target = self._normalize(
                target
            )

            profile_terms.add(
                normalized_target
            )

            profile_terms.update(
                self.TARGET_ALIASES.get(
                    normalized_target,
                    set(),
                )
            )

        for method in methods:
            normalized_method = self._normalize(
                method
            )

            profile_terms.add(
                normalized_method
            )

            profile_terms.update(
                self.METHOD_ALIASES.get(
                    normalized_method,
                    set(),
                )
            )

        profile_terms = {
            self._normalize(term)
            for term in profile_terms
            if self._normalize(term)
        }

        if not profile_terms:
            return 0.0

        matched = 0.0
        total_weight = 0.0

        paper_tokens = set(
            paper_text.split()
        )

        for term in profile_terms:

            if not term:
                continue

            weight = 1.0

            words = term.split()

            if len(words) >= 3:
                weight = 1.5
            elif len(words) == 2:
                weight = 1.25

            total_weight += weight

            if self._concept_match(
                term,
                paper_text,
            ):
                matched += weight
                continue

            if len(term) >= 5:

                best_ratio = 0.0

                for token in paper_tokens:

                    if len(token) < 4:
                        continue

                    # SequenceMatcher cannot reach 0.90 when the strings are
                    # too different in length. Skip those pairs before its
                    # comparatively expensive character matching step.
                    if (
                        2 * min(len(term), len(token))
                        / (len(term) + len(token))
                        < 0.90
                    ):
                        continue

                    ratio = SequenceMatcher(
                        None,
                        term,
                        token,
                    ).ratio()

                    best_ratio = max(
                        best_ratio,
                        ratio,
                    )

                    if best_ratio >= 0.90:
                        break

                if best_ratio >= 0.90:
                    matched += weight * 0.75

        if total_weight <= 0:
            return 0.0

        return min(
            100.0,
            (matched / total_weight) * 100,
        )

    # ==========================================================
    # DATASET NAME MATCH
    # ==========================================================

    def _dataset_name_score(
        self,
        dataset_name: str | None,
        paper: Dict,
        paper_text: str,
    ) -> float:

        if not dataset_name:
            return 0.0

        normalized_name = self._normalize(
            dataset_name
        )

        if not normalized_name:
            return 0.0

        title = self._normalize(
            paper.get(
                "title",
                "",
            )
            or ""
        )

        abstract = self._normalize(
            paper.get(
                "abstract",
                "",
            )
            or ""
        )

        if self._phrase_in_text(
            normalized_name,
            title,
        ):
            return 100.0

        if self._phrase_in_text(
            normalized_name,
            abstract,
        ):
            return 90.0

        name_tokens = {
            token
            for token in normalized_name.split()
            if len(token) >= 3
        }

        if not name_tokens:
            return 0.0

        text_tokens = set(
            paper_text.split()
        )

        matched = name_tokens & text_tokens

        token_score = (
            len(matched)
            / len(name_tokens)
        ) * 100

        sequence_score = SequenceMatcher(
            None,
            normalized_name,
            paper_text[:5000],
        ).ratio() * 100

        return min(
            85.0,
            token_score * 0.75
            + sequence_score * 0.25,
        )

    # ==========================================================
    # SCHEMA MATCH
    # ==========================================================

    def _schema_score(
        self,
        fingerprint: Dict,
        paper: Dict,
        paper_text: str,
    ) -> float:

        paper_fingerprint = (
            paper.get(
                "fingerprint"
            )
            or paper.get(
                "dataset_fingerprint"
            )
        )

        if (
            paper_fingerprint
            and self.fingerprint_similarity is not None
        ):
            try:
                comparison = (
                    self.fingerprint_similarity.compare(
                        fingerprint,
                        paper_fingerprint,
                    )
                )

                return max(
                    0.0,
                    min(
                        100.0,
                        float(
                            comparison.get(
                                "overall_score",
                                0.0,
                            )
                        ),
                    ),
                )

            except Exception:
                pass

        representation = self._get_representation(
            fingerprint
        )

        columns = representation.get(
            "column_signature",
            [],
        )

        if not columns:
            return 0.0

        column_names = set()

        for column in columns:

            if not isinstance(
                column,
                dict,
            ):
                continue

            candidates = [
                column.get(
                    "normalized_name"
                ),
                column.get(
                    "name"
                ),
            ]

            for candidate in candidates:

                normalized = self._normalize(
                    candidate
                )

                if normalized:
                    column_names.add(
                        normalized
                    )

        if not column_names:
            return 0.0

        meaningful_columns = {
            name
            for name in column_names
            if len(name) >= 3
        }

        if not meaningful_columns:
            return 0.0

        matched = 0

        for column_name in meaningful_columns:

            if self._concept_match(
                column_name,
                paper_text,
            ):
                matched += 1

        direct_match_score = (
            matched
            / len(meaningful_columns)
        ) * 100

        semantic_types = set()

        for column in columns:

            if not isinstance(
                column,
                dict,
            ):
                continue

            semantic_type = self._normalize(
                column.get(
                    "semantic_type",
                    "",
                )
            )

            if semantic_type:
                semantic_types.add(
                    semantic_type
                )

        type_matches = 0

        for semantic_type in semantic_types:

            if self._phrase_in_text(
                semantic_type,
                paper_text,
            ):
                type_matches += 1

        if semantic_types:
            type_score = (
                type_matches
                / len(semantic_types)
            ) * 100
        else:
            type_score = 0.0

        return min(
            100.0,
            direct_match_score * 0.80
            + type_score * 0.20,
        )

    # ==========================================================
    # KEYWORD MATCH
    # ==========================================================

    def _keyword_score(
        self,
        keywords: Set[str],
        text: str,
    ) -> float:

        if not keywords:
            return 0.0

        matched = 0

        for keyword in keywords:

            if self._concept_match(
                keyword,
                text,
            ):
                matched += 1

        return min(
            100.0,
            matched
            / len(keywords)
            * 100,
        )

    # ==========================================================
    # METADATA MATCH
    # ==========================================================

    def _metadata_score(
        self,
        paper: Dict,
        dataset_domain: str | None,
        target: str | None,
        methods: Set[str],
        paper_text: str,
    ) -> float:
        """
        Metadata Match adalah komponen 5% dari ranking utama.

        Sub-komponen:
        - domain
        - target
        - ML method
        - metadata completeness

        Nilai yang dikembalikan tetap 0-100.
        """

        scores = []

        if dataset_domain:
            scores.append(
                self._domain_score(
                    dataset_domain,
                    paper_text,
                )
            )

        if target:
            scores.append(
                self._target_score(
                    target,
                    paper_text,
                )
            )

        if methods:
            scores.append(
                self._methods_score(
                    methods,
                    paper_text,
                )
            )

        metadata_fields = [
            paper.get("year"),
            paper.get("doi"),
            paper.get("url"),
            paper.get("venue"),
        ]

        completeness = (
            sum(
                1
                for value in metadata_fields
                if value
            )
            / len(metadata_fields)
        ) * 100

        scores.append(
            completeness
        )

        if not scores:
            return 0.0

        return min(
            100.0,
            sum(scores)
            / len(scores),
        )

    # ==========================================================
    # DATASET USAGE CONFIDENCE
    # ==========================================================

    def _dataset_usage_confidence(
        self,
        dataset_name_score: float,
        schema_score: float,
        keyword_score: float,
        semantic_score: float,
        paper: Dict,
        paper_text: str,
    ) -> Dict:

        evidence = []
        limitations = []

        explicit_same_dataset = bool(
            paper.get(
                "dataset_name"
            )
            or paper.get(
                "dataset_names"
            )
            or paper.get(
                "dataset_id"
            )
            or paper.get(
                "dataset_identifier"
            )
        )

        if dataset_name_score >= 95:
            evidence.append(
                "Nama dataset memiliki kecocokan sangat tinggi."
            )
        elif dataset_name_score >= 75:
            evidence.append(
                "Nama dataset memiliki kecocokan tinggi."
            )
        elif dataset_name_score >= 50:
            evidence.append(
                "Sebagian komponen nama dataset ditemukan pada paper."
            )

        if schema_score >= 80:
            evidence.append(
                "Informasi schema menunjukkan kemiripan sangat tinggi."
            )
        elif schema_score >= 50:
            evidence.append(
                "Sebagian informasi schema memiliki kemiripan."
            )

        if keyword_score >= 70:
            evidence.append(
                "Keyword dataset memiliki kecocokan kuat dengan paper."
            )
        elif keyword_score >= 40:
            evidence.append(
                "Sebagian keyword dataset ditemukan pada paper."
            )

        if semantic_score >= 80:
            evidence.append(
                "Konteks penelitian sangat relevan dengan dataset profile."
            )
        elif semantic_score >= 50:
            evidence.append(
                "Konteks penelitian memiliki relevansi cukup tinggi."
            )

        # ------------------------------------------------------
        # VERY HIGH
        # ------------------------------------------------------

        if (
            explicit_same_dataset
            and dataset_name_score >= 90
            and schema_score >= 80
        ):
            confidence = "VERY HIGH"
            confidence_score = 95.0

        # ------------------------------------------------------
        # HIGH
        # ------------------------------------------------------

        elif (
            dataset_name_score >= 90
            and schema_score >= 60
        ):
            confidence = "HIGH"
            confidence_score = 80.0

        elif (
            dataset_name_score >= 80
            and keyword_score >= 60
            and schema_score >= 50
        ):
            confidence = "HIGH"
            confidence_score = 75.0

        # ------------------------------------------------------
        # MEDIUM
        # ------------------------------------------------------

        elif (
            dataset_name_score >= 60
            and (
                schema_score >= 40
                or keyword_score >= 50
            )
        ):
            confidence = "MEDIUM"
            confidence_score = 55.0

        elif (
            schema_score >= 60
            and keyword_score >= 50
        ):
            confidence = "MEDIUM"
            confidence_score = 50.0

        # ------------------------------------------------------
        # LOW
        # ------------------------------------------------------

        elif (
            dataset_name_score >= 35
            or schema_score >= 35
            or keyword_score >= 35
        ):
            confidence = "LOW"
            confidence_score = 30.0

        # ------------------------------------------------------
        # TOPIC ONLY
        # ------------------------------------------------------

        else:
            confidence = "TOPIC ONLY"
            confidence_score = 10.0

        # ------------------------------------------------------
        # LIMITATIONS
        # ------------------------------------------------------

        if not explicit_same_dataset:
            limitations.append(
                "Tidak ditemukan identifier dataset eksplisit pada metadata paper."
            )

        if dataset_name_score < 90:
            limitations.append(
                "Kecocokan nama dataset belum cukup kuat untuk membuktikan identitas dataset."
            )

        if schema_score < 80:
            limitations.append(
                "Informasi schema paper terbatas atau belum cukup kuat."
            )

        limitations.append(
            "Semantic similarity tidak dianggap sebagai bukti bahwa paper menggunakan dataset yang sama."
        )

        if confidence in {
            "VERY HIGH",
            "HIGH",
        }:
            limitations.append(
                "Confidence tetap bersifat heuristik dan perlu diverifikasi dari paper asli."
            )

        return {
            "confidence": confidence,
            "confidence_score": confidence_score,
            "evidence": evidence,
            "limitations": limitations,
        }

    # ==========================================================
    # REASONS
    # ==========================================================

    def _build_reasons(
        self,
        semantic_score: float,
        dataset_name_score: float,
        schema_score: float,
        keyword_score: float,
        metadata_score: float,
        usage: Dict,
    ) -> List[str]:

        reasons = []

        if semantic_score >= 70:
            reasons.append(
                "Semantic/context similarity tinggi."
            )
        elif semantic_score >= 40:
            reasons.append(
                "Semantic/context similarity sedang."
            )

        if dataset_name_score >= 80:
            reasons.append(
                "Dataset name match tinggi."
            )
        elif dataset_name_score >= 50:
            reasons.append(
                "Sebagian nama dataset memiliki kecocokan."
            )

        if schema_score >= 70:
            reasons.append(
                "Schema match tinggi."
            )
        elif schema_score >= 40:
            reasons.append(
                "Sebagian informasi schema memiliki kecocokan."
            )

        if keyword_score >= 70:
            reasons.append(
                "Keyword match tinggi."
            )
        elif keyword_score >= 40:
            reasons.append(
                "Keyword match sedang."
            )

        if metadata_score >= 70:
            reasons.append(
                "Metadata research cukup lengkap dan relevan."
            )

        reasons.append(
            "Dataset usage confidence: "
            + usage["confidence"]
        )

        if not reasons:
            reasons.append(
                "Relevansi paper relatif rendah berdasarkan evidence yang tersedia."
            )

        return reasons

    # ==========================================================
    # DOMAIN
    # ==========================================================

    def _domain_score(
        self,
        domain: str | None,
        text: str,
    ) -> float:

        if not domain:
            return 0.0

        domain = self._normalize(
            domain
        )

        aliases = self.DOMAIN_ALIASES.get(
            domain,
            {domain},
        )

        matched = [
            alias
            for alias in aliases
            if self._phrase_in_text(
                alias,
                text,
            )
        ]

        if not matched:
            return 0.0

        return min(
            100.0,
            50.0 + len(matched) * 15.0,
        )

    # ==========================================================
    # TARGET
    # ==========================================================

    def _target_score(
        self,
        target: str | None,
        text: str,
    ) -> float:

        if not target:
            return 0.0

        target = self._normalize(
            target
        )

        aliases = set(
            self.TARGET_ALIASES.get(
                target,
                {target},
            )
        )

        aliases.add(
            target
        )

        if target.endswith("ed"):
            aliases.add(
                target[:-2]
            )

        if target.endswith("ing"):
            aliases.add(
                target[:-3]
            )

        if target.endswith("s"):
            aliases.add(
                target[:-1]
            )

        matched = sum(
            1
            for alias in aliases
            if alias.strip()
            and self._concept_match(
                alias,
                text,
            )
        )

        if matched == 0:
            return 0.0

        return min(
            100.0,
            60.0 + matched * 20.0,
        )

    # ==========================================================
    # METHODS
    # ==========================================================

    def _methods_score(
        self,
        methods: Set[str],
        text: str,
    ) -> float:

        if not methods:
            return 0.0

        matched_methods = 0

        for method in methods:

            method = self._normalize(
                method
            )

            aliases = set(
                self.METHOD_ALIASES.get(
                    method,
                    set(),
                )
            )

            method_name = method.replace(
                "_",
                " ",
            )

            aliases.add(
                method_name
            )

            if " classifier" in method_name:
                aliases.add(
                    method_name.replace(
                        " classifier",
                        "",
                    )
                )

            if " regressor" in method_name:
                aliases.add(
                    method_name.replace(
                        " regressor",
                        "",
                    )
                )

            tokens = method_name.split()

            if len(tokens) >= 2:
                if tokens[-1] in {
                    "classifier",
                    "regressor",
                }:
                    aliases.add(
                        " ".join(
                            tokens[:-1]
                        )
                    )

            if any(
                self._phrase_in_text(
                    alias,
                    text,
                )
                for alias in aliases
                if alias.strip()
            ):
                matched_methods += 1

        return min(
            100.0,
            matched_methods
            / len(methods)
            * 100,
        )

    # ==========================================================
    # EXTRACTION
    # ==========================================================

    def _extract_keywords(
        self,
        fingerprint: Dict,
    ) -> Set[str]:

        representation = self._get_representation(
            fingerprint
        )

        keywords = representation.get(
            "keywords",
            [],
        )

        if isinstance(
            keywords,
            dict,
        ):
            keywords = keywords.keys()

        if isinstance(
            keywords,
            str,
        ):
            keywords = [
                keywords
            ]

        return {
            self._normalize(
                str(keyword)
            )
            for keyword in keywords
            if str(keyword).strip()
        }

    @staticmethod
    def _extract_domain(
        domain_result: Dict | None,
    ) -> str | None:

        if not domain_result:
            return None

        return (
            domain_result.get(
                "primary_domain"
            )
            or domain_result.get(
                "domain"
            )
        )

    @staticmethod
    def _extract_target(
        ml_result: Dict | None,
        fingerprint: Dict,
    ) -> str | None:

        if ml_result:

            primary = ml_result.get(
                "primary_task",
                {},
            )

            if isinstance(
                primary,
                dict,
            ):
                target = primary.get(
                    "target"
                )

                if target:
                    return str(
                        target
                    )

        representation = fingerprint.get(
            "representation",
            fingerprint,
        )

        if not isinstance(
            representation,
            dict,
        ):
            return None

        candidates = representation.get(
            "target_candidates",
            [],
        )

        if candidates:

            first = candidates[0]

            if isinstance(
                first,
                dict,
            ):
                return first.get(
                    "name"
                )

            return str(
                first
            )

        return None

    def _extract_methods(
        self,
        ml_result: Dict | None,
    ) -> Set[str]:

        if not ml_result:
            return set()

        recommendation = ml_result.get(
            "recommendation",
            {},
        )

        if not isinstance(
            recommendation,
            dict,
        ):
            return set()

        recommendations = recommendation.get(
            "recommendations",
            [],
        )

        methods = set()

        for item in recommendations[:5]:

            if not isinstance(
                item,
                dict,
            ):
                continue

            method_id = (
                item.get(
                    "method_id"
                )
                or item.get(
                    "method"
                )
            )

            if method_id:
                methods.add(
                    self._normalize(
                        method_id
                    )
                )

        return methods

    # ==========================================================
    # DATASET NAME EXTRACTION
    # ==========================================================

    def _extract_dataset_name(
        self,
        fingerprint: Dict,
    ) -> str | None:

        representation = self._get_representation(
            fingerprint
        )

        candidates = [
            representation.get(
                "dataset_name"
            ),
            representation.get(
                "name"
            ),
            fingerprint.get(
                "dataset_name"
            ),
            fingerprint.get(
                "name"
            ),
            fingerprint.get(
                "file_name"
            ),
            fingerprint.get(
                "filename"
            ),
        ]

        for candidate in candidates:

            if candidate:

                value = str(
                    candidate
                ).strip()

                if value:
                    return value

        return None

    # ==========================================================
    # REPRESENTATION
    # ==========================================================

    @staticmethod
    def _get_representation(
        fingerprint: Dict,
    ) -> Dict:

        if not isinstance(
            fingerprint,
            dict,
        ):
            return {}

        representation = fingerprint.get(
            "representation",
            fingerprint,
        )

        if not isinstance(
            representation,
            dict,
        ):
            return {}

        return representation

    # ==========================================================
    # NORMALIZATION
    # ==========================================================

    @staticmethod
    def _normalize_list(
        values,
    ) -> List[str]:

        if values is None:
            return []

        if isinstance(
            values,
            str,
        ):
            values = [
                values
            ]

        result = []

        for value in values:

            normalized = PaperRanker._normalize(
                value
            )

            if normalized:
                result.append(
                    normalized
                )

        return result

    @staticmethod
    def _normalize(
        text,
    ) -> str:

        if text is None:
            return ""

        text = str(
            text
        ).lower()

        text = re.sub(
            r"[^a-z0-9\s-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ==========================================================
    # CONCEPT MATCHING
    # ==========================================================

    @staticmethod
    def _concept_match(
        keyword: str,
        text: str,
    ) -> bool:

        keyword = PaperRanker._normalize(
            keyword
        )

        if not keyword:
            return False

        text = PaperRanker._normalize(
            text
        )

        if not text:
            return False

        if PaperRanker._phrase_in_text(
            keyword,
            text,
        ):
            return True

        words = keyword.split()

        if len(words) != 1:
            return False

        word = words[0]

        if len(word) < 3:
            return False

        variants = {
            word,
            word.rstrip("s"),
            word + "s",
            word + "ing",
            word + "ed",
            word + "ion",
        }

        irregular = {
            "survived": {
                "survive",
                "survival",
                "survivor",
                "survivors",
            },
            "survival": {
                "survive",
                "survived",
                "survivor",
                "survivors",
            },
            "passenger": {
                "passengers",
            },
            "vehicle": {
                "vehicles",
            },
            "student": {
                "students",
            },
            "patient": {
                "patients",
            },
        }

        variants.update(
            irregular.get(
                word,
                set(),
            )
        )

        return any(
            PaperRanker._phrase_in_text(
                variant,
                text,
            )
            for variant in variants
        )

    # ==========================================================
    # PHRASE MATCHING
    # ==========================================================

    @staticmethod
    def _phrase_in_text(
        phrase: str,
        text: str,
    ) -> bool:

        phrase = PaperRanker._normalize(
            phrase
        )

        text = PaperRanker._normalize(
            text
        )

        if not phrase or not text:
            return False

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(
                phrase
            )
            + r"(?![a-z0-9])"
        )

        return re.search(
            pattern,
            text,
        ) is not None
