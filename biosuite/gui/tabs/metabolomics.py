"""
Metabolomics Analysis tab: Peak detection, feature alignment, PCA.
"""
import os
import customtkinter as ctk
from tkinter import filedialog

from ..themes import FONT_FAMILY, FONT_SMALL


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
            from ...core.metabolomics import detect_peaks, format_metabolomics_report
            df = pd.read_csv(path)
            matrix = df.select_dtypes(include=[np.number]).values
            all_features = []
            for i in range(matrix.shape[0]):
                peaks = detect_peaks(matrix[i],
                    min_snr=float(self.meta_snr.get() or 3.0),
                    min_peak_width=int(self.meta_width.get() or 5))
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
            from ...core.metabolomics import detect_peaks, MetabolomicsReport
            np.random.seed(42)
            # Simulate chromatogram
            x = np.linspace(0, 100, 1000)
            signal = (np.exp(-((x - 20)**2) / 10) * 1000 +
                     np.exp(-((x - 50)**2) / 8) * 800 +
                     np.exp(-((x - 75)**2) / 12) * 600 +
                     np.random.normal(0, 10, 1000))
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
            from ...core.metabolomics import pca_feature_matrix, detect_peaks
            np.random.seed(42)
            # Generate sample data
            matrix = np.random.rand(10, 50) * 100
            coords, var = pca_feature_matrix(matrix, n_components=2)
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(coords[:, 0], coords[:, 1], c='cyan', s=60, alpha=0.7)
            ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
            ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
            ax.set_title('Metabolomics PCA')
            ax.grid(True, alpha=0.3)
            self._record_plot(fig, "Metabolomics PCA")
            self._show_plot_figure(fig)
            self._set_status("PCA plot displayed")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _show_plot_figure(self, fig):
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        win = ctk.CTkToplevel(self)
        win.title("Plot")
        win.geometry("800x600")
        win.configure(fg_color=self.T['bg'])
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        def on_close():
            canvas.get_tk_widget().destroy()
            fig.clear()
            plt.close(fig)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)
        btn = ctk.CTkButton(win, text="Close", command=on_close,
                           fg_color=self.T['accent'], text_color='#000000')
        btn.pack(pady=8)
