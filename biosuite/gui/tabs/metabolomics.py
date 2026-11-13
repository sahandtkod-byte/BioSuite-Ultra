"""
Metabolomics Analysis tab: peak detection, feature tables, alignment, ANOVA, demo PCA.
"""
import math
import os
from tkinter import filedialog

import customtkinter as ctk


class MetabolomicsTabMixin:
    """Provides the Metabolomics Analysis tab."""

    def _build_metabolomics_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['metabolomics'] = f
        self._section_header(f, "Metabolomics Analysis")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, "Peak Detection & Feature Analysis", "sub").pack(anchor='w', pady=(0, 6))

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 6))
        self.meta_path = self._input_entry(file_row, "Intensity matrix (CSV)...")
        self.meta_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Load", self._meta_load).pack(side='right', padx=(0, 4))
        self._action_button(file_row, "Detect Peaks", self._meta_detect).pack(side='right')

        param_row = ctk.CTkFrame(inner, fg_color='transparent')
        param_row.pack(fill='x', pady=(0, 6))
        self._label(param_row, "Min SNR:", "body").pack(side='left', padx=(0, 4))
        self.meta_snr = self._input_entry(param_row, "3.0", width=60)
        self.meta_snr.pack(side='left', padx=(0, 12))
        self._label(param_row, "Min Peak Width:", "body").pack(side='left', padx=(0, 4))
        self.meta_width = self._input_entry(param_row, "5", width=60)
        self.meta_width.pack(side='left')

        self._label(inner, "Or use demo data:", "small").pack(anchor='w', pady=(4, 4))
        demo_row = ctk.CTkFrame(inner, fg_color='transparent')
        demo_row.pack(fill='x', pady=(0, 8))
        self._action_button(demo_row, "Run Demo Analysis", self._meta_demo).pack(side='left')
        self._action_button(demo_row, "PCA Plot", self._meta_pca, color_key='success').pack(side='left', padx=(8, 0))

        self._label(inner, "Feature table, alignment & ANOVA", "sub").pack(anchor='w', pady=(4, 6))
        adv_row = ctk.CTkFrame(inner, fg_color='transparent')
        adv_row.pack(fill='x', pady=(0, 6))
        self._action_button(adv_row, "Detect Features", self._meta_features).pack(side='left', padx=(0, 6))
        self._action_button(adv_row, "Align Features", self._meta_align).pack(side='left', padx=(0, 6))
        self._action_button(adv_row, "ANOVA", self._meta_anova).pack(side='left')

        tol_row = ctk.CTkFrame(inner, fg_color='transparent')
        tol_row.pack(fill='x', pady=(0, 6))
        self._label(tol_row, "m/z tolerance:", "small").pack(side='left', padx=(0, 4))
        self.meta_mz_tol = self._input_entry(tol_row, "0.01", width=70)
        self.meta_mz_tol.pack(side='left', padx=(0, 12))
        self._label(tol_row, "RT tolerance:", "small").pack(side='left', padx=(0, 4))
        self.meta_rt_tol = self._input_entry(tol_row, "30", width=70)
        self.meta_rt_tol.pack(side='left', padx=(0, 12))
        self._label(tol_row, "Groups (comma-separated):", "small").pack(side='left', padx=(0, 4))
        self.meta_groups = self._input_entry(tol_row, "A,A,B,B", width=120)
        self.meta_groups.pack(side='left')

        ctk.CTkFrame(inner, height=1, fg_color=T['border']).pack(fill='x', pady=(4, 8))

        self._label(inner, "Results", "sub").pack(anchor='w', pady=(0, 6))
        self.meta_result = self._text_box(inner, height=200)
        self.meta_result.pack(fill='both', expand=True)

    def _meta_load(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.meta_path.delete(0, 'end')
            self.meta_path.insert(0, path)
            self._set_status(f"Loaded: {os.path.basename(path)}")

    def _meta_detect(self):
        import numpy as np
        import pandas as pd
        path = self.meta_path.get().strip()
        if not path:
            self._msg_warning("Input Required", "Please load a CSV intensity matrix.")
            return
        try:
            min_snr = float(self.meta_snr.get() or 3.0)
            min_width = int(self.meta_width.get() or 5)
        except ValueError:
            self._msg_error(
                "Error", "Min SNR must be a number and Min Peak Width a whole number.")
            return
        if not math.isfinite(min_snr) or min_snr < 0:
            self._msg_error("Error", "Min SNR must be a non-negative finite number.")
            return
        if min_width < 1:
            self._msg_error("Error", "Min Peak Width must be at least 1.")
            return
        try:
            from ...core.metabolomics import detect_peaks
            df = pd.read_csv(path)
            matrix = df.select_dtypes(include=[np.number]).values
            if not matrix.size:
                self._msg_error("Error", "The CSV holds no numeric intensity columns.")
                return
            all_features = []
            for i in range(matrix.shape[0]):
                peaks = detect_peaks(matrix[i], min_snr=min_snr,
                                     min_peak_width=min_width)
                all_features.extend(peaks)
            self.meta_result.delete("1.0", "end")
            self.meta_result.insert("1.0",
                f"Detected {len(all_features)} peaks across {matrix.shape[0]} samples\n\n"
                f"First 10 features:\n" +
                "\n".join(f"  RT={f.rt:.1f}, Intensity={f.intensity:.0f}, SNR={f.snr:.1f}"
                          for f in all_features[:10]))
            self._set_status(f"Detected {len(all_features)} peaks")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _meta_demo(self):
        try:
            import numpy as np

            from ...core.metabolomics import MetabolomicsReport, detect_peaks
            rng = np.random.default_rng(42)
            # Simulate chromatogram
            x = np.linspace(0, 100, 1000)
            signal = (np.exp(-((x - 20)**2) / 10) * 1000 +
                     np.exp(-((x - 50)**2) / 8) * 800 +
                     np.exp(-((x - 75)**2) / 12) * 600 +
                     rng.normal(0, 10, 1000))
            peaks = detect_peaks(signal)
            report = MetabolomicsReport(
                total_features=len(peaks),
                detected_peaks=len(peaks),
                message=f"Demo analysis: {len(peaks)} peaks detected in simulated chromatogram")
            from ...core.metabolomics import format_metabolomics_report
            self.meta_result.delete("1.0", "end")
            self.meta_result.insert("1.0", format_metabolomics_report(report))
            self._set_status("Demo metabolomics analysis complete")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _meta_pca(self):
        try:
            import numpy as np

            from ...core.metabolomics import pca_feature_matrix
            rng = np.random.default_rng(42)
            # Generate sample data
            matrix = rng.random((10, 50)) * 100
            coords, var = pca_feature_matrix(matrix, n_components=2)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(coords[:, 0], coords[:, 1], c=self.T['accent'], s=60, alpha=0.7)
            ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
            ax.set_title('Metabolomics PCA')
            ax.grid(True, alpha=0.3)
            self._record_plot(fig, "Metabolomics PCA")
            self._show_plot_from_figure(fig, "Metabolomics PCA")
            self._set_status("PCA plot displayed")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _meta_read_matrix(self):
        import numpy as np
        import pandas as pd
        path = self.meta_path.get().strip()
        if not path:
            self._msg_warning("Input Required", "Please load a CSV intensity matrix.")
            return None
        try:
            df = pd.read_csv(path)
        except Exception as e:
            self._msg_error("Error", str(e))
            return None
        matrix = df.select_dtypes(include=[np.number]).values
        if not matrix.size:
            self._msg_error("Error", "The CSV holds no numeric intensity columns.")
            return None
        return matrix

    def _meta_features(self):
        matrix = self._meta_read_matrix()
        if matrix is None:
            return
        try:
            from ...core.metabolomics import detect_features_from_matrix
            features = detect_features_from_matrix(matrix)
            lines = [f"Detected {len(features)} features across {matrix.shape[0]} samples"]
            if features.empty:
                lines.append("No peaks passed the detection thresholds.")
            else:
                lines.append("")
                lines.append(features.head(10).to_string(index=False))
            self.meta_result.delete("1.0", "end")
            self.meta_result.insert("1.0", "\n".join(lines))
            self._set_status(f"Detected {len(features)} features")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _meta_align(self):
        matrix = self._meta_read_matrix()
        if matrix is None:
            return
        try:
            mz_tol = float(self.meta_mz_tol.get() or 0.01)
            rt_tol = float(self.meta_rt_tol.get() or 30)
        except ValueError:
            self._msg_error("Error", "Tolerances must be numbers.")
            return
        try:
            from ...core.metabolomics import align_features, detect_features_from_matrix
            per_sample = [detect_features_from_matrix(matrix[i:i + 1])
                          for i in range(matrix.shape[0])]
            aligned = align_features(per_sample, mz_tolerance=mz_tol, rt_tolerance=rt_tol)
            lines = [f"Aligned matrix: {aligned.shape[0]} features x {aligned.shape[1]} samples"]
            if not aligned.empty:
                lines.append("")
                lines.append(aligned.head(10).to_string())
            self.meta_result.delete("1.0", "end")
            self.meta_result.insert("1.0", "\n".join(lines))
            self._set_status(f"Aligned {aligned.shape[0]} features")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _meta_anova(self):
        matrix = self._meta_read_matrix()
        if matrix is None:
            return
        labels = [g.strip() for g in self.meta_groups.get().split(',') if g.strip()]
        if not labels:
            self._msg_warning("Input Required",
                              "Enter one group label per sample row, comma-separated.")
            return
        if len(labels) != matrix.shape[0]:
            self._msg_error("Error", f"Got {len(labels)} group labels for {matrix.shape[0]} samples.")
            return
        try:
            from ...core.metabolomics import anova_test
            results = anova_test(matrix, labels)
            significant = int((results['p_value'] < 0.05).sum())
            lines = [
                f"ANOVA across {len(set(labels))} groups, {matrix.shape[1]} features",
                f"Significant features (p<0.05): {significant}",
                "",
                results.to_string(index=False),
            ]
            self.meta_result.delete("1.0", "end")
            self.meta_result.insert("1.0", "\n".join(lines))
            self._set_status(f"ANOVA: {significant} significant features")
        except Exception as e:
            self._msg_error("Error", str(e))

