"""Explicit, assumption-aware inferential tests for tabular datasets.

The analyzer never chooses a test from column dtypes. Callers must name the
test and its variables; the result records hypotheses, assumptions and limits.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


class InferentialStatisticsAnalyzer:
    ALPHA = 0.05
    MAX_NORMALITY_SAMPLE = 5000

    TESTS = {
        "shapiro": "Shapiro-Wilk",
        "anderson": "Anderson-Darling",
        "welch_t": "Welch independent t-test",
        "mann_whitney": "Mann-Whitney U",
        "paired_t": "Paired t-test",
        "wilcoxon": "Wilcoxon signed-rank",
        "anova": "One-way ANOVA",
        "welch_anova": "Welch ANOVA",
        "kruskal": "Kruskal-Wallis",
        "levene": "Levene variance test",
        "bartlett": "Bartlett variance test",
        "chi_square": "Chi-square association test",
        "fisher": "Fisher exact test",
        "pearson": "Pearson correlation",
        "spearman": "Spearman correlation",
        "kendall": "Kendall tau correlation",
    }

    def analyze(
        self,
        dataframe: pd.DataFrame,
        test: str,
        column: str | None = None,
        second_column: str | None = None,
        group_column: str | None = None,
    ) -> dict[str, Any]:
        if dataframe is None or dataframe.empty:
            return self._error("Dataset kosong.")
        if test not in self.TESTS:
            return self._error(f"Uji statistik '{test}' belum didukung.")

        try:
            if test in {"shapiro", "anderson"}:
                return self._finish(self._normality(dataframe, test, column))
            if test in {"welch_t", "mann_whitney", "anova", "welch_anova", "kruskal", "levene", "bartlett"}:
                return self._finish(self._group_test(dataframe, test, column, group_column))
            if test in {"paired_t", "wilcoxon"}:
                return self._finish(self._paired_test(dataframe, test, column, second_column))
            if test in {"chi_square", "fisher"}:
                return self._finish(self._categorical_test(dataframe, test, column, second_column))
            return self._finish(self._correlation(dataframe, test, column, second_column))
        except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
            return self._error(str(exc), test=test)

    def _normality(self, dataframe, test, column):
        series = self._numeric(dataframe, column)
        values = self._finite(series)
        if len(values) < 3:
            raise ValueError("Uji normalitas membutuhkan minimal 3 observasi numerik finite.")
        assumptions = [
            self._assumption("Observasi", "Observasi dianggap mewakili sampel yang diteliti; mekanisme sampling tidak dapat diverifikasi dari tabel."),
            self._assumption("Ukuran sampel", "Shapiro-Wilk dibatasi 5.000 observasi; untuk data lebih besar dipakai sampel acak deterministik." if test == "shapiro" else "Anderson-Darling memakai distribusi normal sebagai acuan."),
        ]
        if test == "shapiro":
            sample = values
            sampled = len(sample) > self.MAX_NORMALITY_SAMPLE
            if sampled:
                sample = np.random.default_rng(42).choice(sample, self.MAX_NORMALITY_SAMPLE, replace=False)
            result = stats.shapiro(sample)
            statistic, p_value = float(result.statistic), float(result.pvalue)
            hypotheses = ("Data berasal dari distribusi normal.", "Data menyimpang dari distribusi normal.")
            note = "Shapiro-Wilk p-value sensitif pada ukuran sampel dan bukan ukuran besar penyimpangan."
            if sampled:
                assumptions.append(self._assumption("Sampling uji", f"Dipakai sampel deterministik 5.000 dari {len(values):,} nilai.", "WARNING"))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                result = stats.anderson(values, dist="norm")
            statistic = float(result.statistic)
            critical = {f"{float(level):g}%": float(value) for level, value in zip(result.significance_level, result.critical_values)}
            p_value = None
            hypotheses = ("Data mengikuti distribusi normal.", "Data menyimpang dari distribusi normal.")
            note = "Anderson-Darling memberi nilai kritis, bukan satu p-value; bandingkan statistik dengan nilai kritis pada taraf yang dipilih."
            assumptions.append(self._assumption("Parameter acuan", "Rata-rata dan simpangan baku diestimasi dari sampel; nilai kritis normal dipakai."))
        output = self._base(test, [column], len(values), hypotheses, assumptions)
        output.update({"statistic": statistic, "p_value": p_value, "interpretation": note})
        if test == "anderson":
            output["critical_values"] = critical
            output["decision"] = "TOLAK_H0" if statistic > critical.get("5%", math.inf) else "GAGAL_MENOLAK_H0"
            output["decision_alpha"] = 0.05
        return output

    def _group_test(self, dataframe, test, column, group_column):
        values = self._numeric(dataframe, column)
        if group_column not in dataframe.columns:
            raise ValueError("Pilih kolom kelompok yang tersedia.")
        if group_column == column:
            raise ValueError("Kolom pengukuran dan kelompok harus berbeda.")
        pair = pd.DataFrame({"value": values, "group": dataframe[group_column]}).dropna()
        finite = np.isfinite(pd.to_numeric(pair["value"], errors="coerce"))
        pair = pair.loc[finite]
        grouped = list(pair.groupby("group", sort=True, observed=True))
        groups = [part["value"].to_numpy(dtype=float) for _, part in grouped]
        labels = [str(label) for label, _ in grouped]
        if len(groups) < 2:
            raise ValueError("Kolom kelompok harus memiliki sedikitnya dua kelompok dengan nilai pengukuran.")
        if any(len(group) < 2 for group in groups):
            raise ValueError("Setiap kelompok membutuhkan minimal 2 observasi yang dapat dipakai.")
        if test in {"welch_t", "mann_whitney"} and len(groups) != 2:
            raise ValueError(f"{self.TESTS[test]} membutuhkan tepat dua kelompok; pilih ANOVA/Kruskal untuk lebih banyak kelompok.")
        assumptions = [
            self._assumption("Independensi", "Kelompok dianggap terdiri dari observasi independen; struktur berulang/berpasangan tidak dapat disimpulkan otomatis."),
            self._assumption("Pemilihan kelompok", "Kelompok dipilih pengguna; hasil tidak membuktikan hubungan kausal."),
        ]
        hypotheses = ("Rata-rata/nilai distribusi sama antar kelompok.", "Sedikitnya satu kelompok berbeda.")
        statistic = p_value = degrees = None
        confidence_interval = None
        effect_size = None
        details: dict[str, Any] = {"group_sizes": dict(zip(labels, map(len, groups)))}
        if test == "welch_t":
            first, second = groups
            result = stats.ttest_ind(first, second, equal_var=False)
            v1, v2 = np.var(first, ddof=1), np.var(second, ddof=1)
            term1, term2 = v1 / len(first), v2 / len(second)
            degrees = (term1 + term2) ** 2 / (term1**2 / (len(first) - 1) + term2**2 / (len(second) - 1))
            difference = float(np.mean(first) - np.mean(second))
            critical = stats.t.ppf(1 - self.ALPHA / 2, degrees)
            margin = critical * math.sqrt(term1 + term2)
            confidence_interval = [difference - margin, difference + margin]
            pooled_sd = math.sqrt(((len(first) - 1) * v1 + (len(second) - 1) * v2) / (len(first) + len(second) - 2))
            if pooled_sd > 0:
                cohen_d = difference / pooled_sd
                correction = 1 - 3 / (4 * (len(first) + len(second)) - 9)
                effect_size = self._effect("Hedges g", cohen_d * correction, "Perbedaan rata-rata yang distandarkan; bukan ukuran kepentingan praktis.")
            statistic, p_value = float(result.statistic), float(result.pvalue)
            hypotheses = ("Rata-rata populasi kedua kelompok sama.", "Rata-rata populasi kedua kelompok berbeda.")
            assumptions.extend([
                self._assumption("Distribusi", "Uji t mengasumsikan data dalam tiap kelompok cukup mendekati normal, terutama untuk sampel kecil; tinjau grafik/uji normalitas."),
                self._assumption("Varians", "Welch tidak mengasumsikan varians yang sama."),
            ])
            details["mean_difference_first_minus_second"] = difference
        elif test == "mann_whitney":
            result = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided", method="auto")
            statistic, p_value = float(result.statistic), float(result.pvalue)
            delta = 2 * statistic / (len(groups[0]) * len(groups[1])) - 1
            effect_size = self._effect("Cliff's delta", delta, "Ukuran dominasi pasangan observasi; tidak langsung menyatakan beda median bila bentuk distribusi berbeda.")
            hypotheses = ("Distribusi kedua kelompok sama.", "Distribusi kedua kelompok berbeda.")
            assumptions.append(self._assumption("Bentuk distribusi", "Interpretasi sebagai perbedaan lokasi/median memerlukan bentuk distribusi yang sebanding."))
        elif test in {"anova", "welch_anova"}:
            if len(groups) < 3:
                raise ValueError(f"{self.TESTS[test]} membutuhkan sedikitnya tiga kelompok.")
            equal_var_result = stats.levene(*groups, center="median")
            assumptions.append(self._assumption("Kesamaan varians", f"Levene p={float(equal_var_result.pvalue):.4g}; {'bukti varians berbeda terdeteksi' if equal_var_result.pvalue < self.ALPHA else 'tidak ditemukan bukti kuat varians berbeda'}.", "WARNING" if test == "anova" and equal_var_result.pvalue < self.ALPHA else "CHECKED"))
            if test == "anova":
                result = stats.f_oneway(*groups)
                all_values = np.concatenate(groups)
                grand_mean = float(np.mean(all_values))
                ss_between = sum(len(group) * (float(np.mean(group)) - grand_mean) ** 2 for group in groups)
                ss_total = float(np.sum((all_values - grand_mean) ** 2))
                if ss_total > 0:
                    effect_size = self._effect("Eta-squared", ss_between / ss_total, "Proporsi variasi sampel yang terkait dengan perbedaan kelompok.")
                degrees = {"numerator": len(groups) - 1, "denominator": len(all_values) - len(groups)}
                assumptions.append(self._assumption("Residual", "ANOVA klasik mengasumsikan residual dalam kelompok mendekati normal."))
            else:
                statistic, p_value, degrees = self._welch_anova(groups)
                assumptions.append(self._assumption("Varians", "Welch ANOVA dipilih untuk membolehkan varians berbeda."))
                result = None
            if result is not None:
                statistic, p_value = float(result.statistic), float(result.pvalue)
        elif test == "kruskal":
            result = stats.kruskal(*groups, nan_policy="omit")
            statistic, p_value = float(result.statistic), float(result.pvalue)
            degrees = len(groups) - 1
            epsilon = max(0.0, (statistic - len(groups) + 1) / (len(pair) - len(groups)))
            effect_size = self._effect("Epsilon-squared", epsilon, "Estimasi proporsi variasi peringkat terkait perbedaan kelompok; interpretasikan hati-hati.")
            assumptions.append(self._assumption("Observasi", "Uji non-parametrik tetap mengasumsikan observasi independen dan variabel terukur minimal ordinal."))
        else:
            center = "mean" if test == "bartlett" else "median"
            result = (stats.bartlett if test == "bartlett" else stats.levene)(*groups, **({} if test == "bartlett" else {"center": center}))
            statistic, p_value = float(result.statistic), float(result.pvalue)
            hypotheses = ("Varians semua kelompok sama.", "Sedikitnya satu varians kelompok berbeda.")
            assumptions.append(self._assumption("Normalitas", "Bartlett sensitif terhadap penyimpangan normalitas; Levene berbasis median lebih tahan." if test == "bartlett" else "Levene berbasis median lebih tahan terhadap pencilan dan non-normalitas."))
        output = self._base(test, [column, group_column], len(pair), hypotheses, assumptions)
        output.update({"statistic": statistic, "degrees_of_freedom": degrees, "p_value": p_value, "confidence_interval": confidence_interval, "effect_size": effect_size, "group_labels": labels, "details": details})
        return output

    def _paired_test(self, dataframe, test, column, second_column):
        first = self._numeric(dataframe, column)
        second = self._numeric(dataframe, second_column)
        pair = pd.DataFrame({"first": first, "second": second}).dropna()
        finite_mask = np.isfinite(pair.to_numpy(dtype=float)).all(axis=1)
        pair = pair.loc[finite_mask]
        difference = pair["first"].to_numpy() - pair["second"].to_numpy()
        n = len(difference)
        if n < 2:
            raise ValueError("Uji berpasangan membutuhkan minimal 2 pasangan numerik finite.")
        assumptions = [self._assumption("Pasangan", "Baris dianggap mewakili pasangan yang sama; pastikan urutan/ID pasangan benar sebelum menjalankan uji.", "WARNING"), self._assumption("Pasangan independen", "Selisih antar pasangan dianggap independen.")]
        if test == "paired_t":
            if float(np.std(difference, ddof=1)) == 0:
                output = self._base(
                    test,
                    [column, second_column],
                    n,
                    ("Rata-rata selisih pasangan sama dengan nol.", "Rata-rata selisih pasangan berbeda dari nol."),
                    assumptions,
                )
                output.update({
                    "status": "NOT_EVALUABLE",
                    "message": "Simpangan baku selisih bernilai nol; statistik t dan interval kepercayaan tidak terdefinisi.",
                    "details": {"mean_difference_first_minus_second": float(np.mean(difference))},
                })
                return output
            if n >= 3:
                normality = stats.shapiro(difference[:self.MAX_NORMALITY_SAMPLE])
                assumptions.append(self._assumption("Normalitas selisih", f"Shapiro-Wilk pada selisih: p={float(normality.pvalue):.4g}; periksa juga grafik Q-Q, terutama untuk sampel kecil.", "WARNING" if normality.pvalue < self.ALPHA else "CHECKED"))
            result = stats.ttest_rel(pair["first"], pair["second"])
            degrees = n - 1
            mean_difference = float(np.mean(difference))
            standard_error = float(stats.sem(difference))
            margin = float(stats.t.ppf(1 - self.ALPHA / 2, degrees) * standard_error)
            sd = float(np.std(difference, ddof=1))
            effect_size = self._effect("Cohen dz", mean_difference / sd if sd > 0 else None, "Rata-rata selisih dibagi simpangan baku selisih.")
            confidence_interval = [mean_difference - margin, mean_difference + margin]
            statistic, p_value = float(result.statistic), float(result.pvalue)
            hypotheses = ("Rata-rata selisih pasangan sama dengan nol.", "Rata-rata selisih pasangan berbeda dari nol.")
        else:
            if not np.any(difference != 0):
                raise ValueError("Wilcoxon tidak dapat dihitung karena semua selisih bernilai nol.")
            result = stats.wilcoxon(pair["first"], pair["second"], alternative="two-sided", method="auto")
            statistic, p_value = float(result.statistic), float(result.pvalue)
            nonzero = difference != 0
            ranks = stats.rankdata(np.abs(difference[nonzero]))
            positive_rank_sum = float(ranks[difference[nonzero] > 0].sum())
            effect_size = self._effect("Matched rank-biserial correlation", 2 * positive_rank_sum / float(ranks.sum()) - 1, "Arah dan kekuatan dominasi tanda selisih berpasangan.")
            degrees = None
            confidence_interval = None
            hypotheses = ("Distribusi selisih berpasangan simetris terhadap nol.", "Distribusi selisih berpasangan tidak simetris terhadap nol.")
            assumptions.append(self._assumption("Distribusi selisih", "Wilcoxon signed-rank memerlukan selisih berpasangan yang setidaknya ordinal dan cukup simetris untuk interpretasi lokasi."))
        output = self._base(test, [column, second_column], n, hypotheses, assumptions)
        output.update({"statistic": statistic, "degrees_of_freedom": degrees, "p_value": p_value, "confidence_interval": confidence_interval, "effect_size": effect_size, "details": {"mean_difference_first_minus_second": float(np.mean(difference))}})
        return output

    def _categorical_test(self, dataframe, test, column, second_column):
        if column not in dataframe.columns or second_column not in dataframe.columns:
            raise ValueError("Pilih dua kolom kategori yang tersedia.")
        if column == second_column:
            raise ValueError("Pilih dua kolom kategori yang berbeda.")
        pair = dataframe[[column, second_column]].dropna()
        table = pd.crosstab(pair[column], pair[second_column])
        if table.shape[0] < 2 or table.shape[1] < 2:
            raise ValueError("Setiap kolom kategori harus memiliki minimal dua nilai teramati.")
        if test == "fisher" and table.shape != (2, 2):
            raise ValueError("Fisher exact test memerlukan tabel kontingensi 2x2.")
        assumptions = [self._assumption("Observasi", "Setiap baris dianggap satu observasi independen; hitungan berulang/berpasangan memerlukan metode lain.")]
        hypotheses = ("Kedua variabel kategori independen.", "Kedua variabel kategori berasosiasi.")
        if test == "fisher":
            result = stats.fisher_exact(table.to_numpy(), alternative="two-sided")
            statistic, p_value = float(result.statistic), float(result.pvalue)
            effect_size = self._effect("Odds ratio", statistic, "Odds ratio sampel; dapat bernilai tak hingga bila ada sel nol. CI exact belum dihitung.")
            degrees = None
            assumptions.append(self._assumption("Tabel", "Dua kategori per variabel; Fisher exact tidak memakai pendekatan chi-square."))
        else:
            chi2, p_value, degrees, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
            statistic = float(chi2)
            n = int(table.to_numpy().sum())
            denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
            effect_size = self._effect("Cramér's V", math.sqrt(float(chi2) / denominator) if denominator else None, "Ukuran asosiasi kategori, tanpa arah.")
            min_expected = float(expected.min())
            assumptions.append(self._assumption("Frekuensi harapan", f"Minimum hitungan harapan {min_expected:.2f}; pendekatan chi-square lebih dapat diandalkan bila frekuensi harapan tidak terlalu kecil.", "WARNING" if min_expected < 5 else "CHECKED"))
        output = self._base(test, [column, second_column], len(pair), hypotheses, assumptions)
        output.update({"statistic": statistic, "degrees_of_freedom": degrees, "p_value": float(p_value), "confidence_interval": None, "effect_size": effect_size, "details": {"contingency_table": table.astype(int).to_dict()}})
        return output

    def _correlation(self, dataframe, test, column, second_column):
        first = self._numeric(dataframe, column)
        second = self._numeric(dataframe, second_column)
        pair = pd.DataFrame({"first": first, "second": second}).dropna()
        pair = pair.loc[np.isfinite(pair.to_numpy(dtype=float)).all(axis=1)]
        n = len(pair)
        if n < 3:
            raise ValueError("Uji korelasi membutuhkan minimal 3 pasangan numerik finite.")
        if pair["first"].nunique() < 2 or pair["second"].nunique() < 2:
            raise ValueError("Korelasi tidak terdefinisi untuk kolom konstan.")
        functions = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}
        result = functions[test](pair["first"], pair["second"])
        coefficient, p_value = float(result.statistic), float(result.pvalue)
        assumptions = [self._assumption("Pasangan observasi", "Setiap baris menjadi pasangan; periksa bahwa kedua kolom memang diamati pada unit yang sama."), self._assumption("Independensi", "Pasangan antar baris dianggap independen."), self._assumption("Interpretasi", "Korelasi tidak menunjukkan sebab-akibat.")]
        interval = None
        if test == "pearson":
            assumptions.append(self._assumption("Bentuk hubungan", "Pearson mengukur hubungan linear; inferensi klasik mengasumsikan pasangan independen dan mendekati bivariat normal."))
            if n > 3 and abs(coefficient) < 1:
                z = np.arctanh(coefficient)
                margin = stats.norm.ppf(1 - self.ALPHA / 2) / math.sqrt(n - 3)
                interval = [float(np.tanh(z - margin)), float(np.tanh(z + margin))]
            effect = self._effect("Pearson r", coefficient, "Kekuatan dan arah hubungan linear.")
        elif test == "spearman":
            effect = self._effect("Spearman rho", coefficient, "Kekuatan dan arah hubungan monoton berbasis peringkat.")
        else:
            effect = self._effect("Kendall tau-b", coefficient, "Kekuatan asosiasi ordinal berbasis pasangan konkordan dan diskordan.")
        output = self._base(test, [column, second_column], n, ("Tidak ada asosiasi menurut koefisien yang diuji.", "Terdapat asosiasi menurut koefisien yang diuji."), assumptions)
        output.update({"statistic": coefficient, "degrees_of_freedom": n - 2 if test == "pearson" else None, "p_value": p_value, "confidence_interval": interval, "effect_size": effect})
        return output

    @staticmethod
    def _welch_anova(groups):
        """Welch's one-way F statistic with Satterthwaite denominator df."""
        sizes = np.asarray([len(group) for group in groups], dtype=float)
        means = np.asarray([np.mean(group) for group in groups], dtype=float)
        variances = np.asarray([np.var(group, ddof=1) for group in groups], dtype=float)
        if np.any(variances <= 0) or not np.isfinite(variances).all():
            raise ValueError("Welch ANOVA membutuhkan variasi positif pada setiap kelompok.")
        weights = sizes / variances
        weight_sum = float(weights.sum())
        weighted_mean = float(np.sum(weights * means) / weight_sum)
        k = len(groups)
        adjustment = float(np.sum((1 - weights / weight_sum) ** 2 / (sizes - 1)))
        numerator = float(np.sum(weights * (means - weighted_mean) ** 2) / (k - 1))
        statistic = numerator / (1 + (2 * (k - 2) / (k**2 - 1)) * adjustment)
        df_num = float(k - 1)
        df_den = float((k**2 - 1) / (3 * adjustment)) if adjustment > 0 else math.inf
        p_value = float(stats.f.sf(statistic, df_num, df_den))
        return statistic, p_value, {"numerator": df_num, "denominator": df_den}

    def _base(self, test, variables, sample_size, hypotheses, assumptions):
        return {
            "status": "COMPLETED", "test_id": test, "test_name": self.TESTS[test],
            "variables": [str(value) for value in variables if value is not None],
            "sample_size": int(sample_size), "alpha": self.ALPHA,
            "hypotheses": {"null": hypotheses[0], "alternative": hypotheses[1]},
            "assumptions": assumptions, "statistic": None,
            "degrees_of_freedom": None, "p_value": None,
            "confidence_interval": None, "effect_size": None,
            "decision": None, "interpretation": "",
        }

    def _finish(self, result):
        for key in ("statistic", "p_value"):
            value = result.get(key)
            if isinstance(value, (int, float)) and not np.isfinite(value):
                result.update({
                    "status": "NOT_EVALUABLE", "statistic": None, "p_value": None,
                    "decision": None,
                    "message": "Statistik uji tidak terdefinisi untuk nilai pada dataset ini; hasil tidak dirangking atau diinterpretasikan.",
                })
                return result
        p_value = result.get("p_value")
        if p_value is not None:
            result["decision"] = "TOLAK_H0" if p_value < self.ALPHA else "GAGAL_MENOLAK_H0"
            result["interpretation"] = (
                "Hasil memberikan bukti statistik terhadap H0 pada taraf 5%; "
                "besar efek dan asumsi tetap perlu ditinjau."
                if p_value < self.ALPHA else
                "Data belum memberi bukti statistik yang cukup untuk menolak H0 pada taraf 5%; ini bukan bukti bahwa H0 benar."
            )
            result["interpretation"] += " P-value tidak dikoreksi untuk pengujian hipotesis lain pada dataset ini."
        return result

    @classmethod
    def _assumption(cls, name, detail, status="USER_ASSUMED"):
        return {"name": name, "status": status, "detail": detail}

    @staticmethod
    def _effect(name, value, interpretation):
        if value is not None and not np.isfinite(value):
            value = None
        return {"name": name, "value": float(value) if value is not None else None, "interpretation": interpretation}

    @staticmethod
    def _numeric(dataframe, column):
        if column not in dataframe.columns:
            raise ValueError("Pilih kolom numerik yang tersedia.")
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise ValueError(f"Kolom '{column}' harus bertipe numerik untuk uji ini.")
        if pd.api.types.is_bool_dtype(dataframe[column]):
            raise ValueError(f"Kolom '{column}' bertipe boolean; pilih kolom numerik yang sesuai.")
        return pd.to_numeric(dataframe[column], errors="coerce")

    @staticmethod
    def _finite(series):
        values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
        return values[np.isfinite(values)]

    @classmethod
    def _error(cls, message, test=None):
        return {"status": "ERROR", "test_id": test, "message": message, "test_name": cls.TESTS.get(test, "Uji statistik"), "variables": [], "sample_size": 0, "assumptions": [], "decision": None}
