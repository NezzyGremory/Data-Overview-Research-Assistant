from __future__ import annotations

from collections import Counter
from typing import Dict, List


class ResearchLandscapeAnalyzer:
    """
    Menganalisis landscape penelitian dari paper yang
    telah ditemukan.

    Hasil bersifat HEURISTIC karena hanya berdasarkan
    kumpulan paper yang berhasil ditemukan.
    """

    def analyze(
        self,
        papers: List[Dict],
        ml_result: Dict | None = None,
    ) -> Dict:

        if not papers:
            return {
                "status": "NO_DATA",
                "paper_count": 0,
                "publication_years": {},
                "top_venues": [],
                "sources": [],
                "methods": [],
                "research_activity": [],
                "summary": {
                    "papers_analyzed": 0,
                    "latest_publication_year": None,
                    "dominant_method": None,
                    "method_count": 0,
                },
                "status_type": "HEURISTIC",
            }

        years = Counter()
        venues = Counter()
        sources = Counter()
        methods = Counter()

        for paper in papers:

            year = paper.get("year")

            if isinstance(
                year,
                int,
            ):
                years[str(year)] += 1

            venue = paper.get(
                "venue"
            )

            if venue:
                venues[str(venue)] += 1

            source = paper.get(
                "source"
            )

            if source:
                provider_names = [
                    name.strip()
                    for name in str(source).split(",")
                    if name.strip()
                ]
                for provider_name in provider_names:
                    sources[provider_name] += 1

            self._extract_methods(
                paper,
                methods,
            )

        total = len(papers)

        return {
            "status": "SUCCESS",
            "paper_count": total,

            "publication_years": dict(
                sorted(
                    years.items(),
                    key=lambda item: item[0],
                )
            ),

            "top_venues": self._percentages(
                venues,
                total,
            ),

            "sources": self._percentages(
                sources,
                sum(sources.values()),
            ),

            "methods": self._percentages(
                methods,
                total,
            ),

            "research_activity": (
                self._build_activity(years)
            ),

            "summary": self._summary(
                total=total,
                years=years,
                methods=methods,
            ),

            "status_type": "HEURISTIC",
        }

    @staticmethod
    def _extract_methods(
        paper: Dict,
        counter: Counter,
    ) -> None:

        text = (
            f"{paper.get('title', '')} "
            f"{paper.get('abstract', '')} "
            f"{' '.join(map(str, paper.get('keywords', []))) if isinstance(paper.get('keywords', []), list) else paper.get('keywords', '')}"
        ).lower()

        method_aliases = {
            "Random Forest": [
                "random forest",
            ],
            "Logistic Regression": [
                "logistic regression",
            ],
            "Decision Tree": [
                "decision tree",
            ],
            "SVM": [
                "support vector machine",
                "svm",
            ],
            "KNN": [
                "k-nearest",
                "knn",
                "k nearest",
            ],
            "Gradient Boosting": [
                "gradient boosting",
            ],
            "Linear Regression": [
                "linear regression",
            ],
            "K-Means": [
                "k-means",
                "kmeans",
            ],
            "DBSCAN": [
                "dbscan",
            ],
            "Agglomerative Clustering": [
                "agglomerative clustering",
                "hierarchical clustering",
            ],
            "Isolation Forest": [
                "isolation forest",
            ],
            "Local Outlier Factor": [
                "local outlier factor",
                "lof",
            ],
            "One-Class SVM": [
                "one class svm",
                "one-class svm",
            ],
        }

        for method, aliases in method_aliases.items():

            if any(
                alias in text
                for alias in aliases
            ):
                counter[method] += 1

    @staticmethod
    def _percentages(
        counter: Counter,
        total: int,
    ) -> List[Dict]:

        if total == 0:
            return []

        result = []

        for name, count in counter.most_common():

            result.append(
                {
                    "name": name,
                    "count": count,
                    "percentage": round(
                        count / total * 100,
                        2,
                    ),
                }
            )

        return result

    @staticmethod
    def _build_activity(
        years: Counter,
    ) -> List[Dict]:

        return [
            {
                "year": year,
                "paper_count": count,
            }
            for year, count in sorted(
                years.items()
            )
        ]

    @staticmethod
    def _summary(
        total: int,
        years: Counter,
        methods: Counter,
    ) -> Dict:

        latest_year = (
            max(
                years.keys()
            )
            if years
            else None
        )

        dominant_method = (
            methods.most_common(1)[0][0]
            if methods
            else None
        )

        return {
            "papers_analyzed": total,
            "latest_publication_year": latest_year,
            "dominant_method": dominant_method,
            "method_count": len(methods),
        }
