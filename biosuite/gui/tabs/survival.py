"""
Survival Analysis tab: Kaplan-Meier curves, log-rank test, Cox PH.
"""
import os
import customtkinter as ctk
from tkinter import filedialog

from ..themes import FONT_FAMILY, FONT_SMALL


class SurvivalTabMixin:
    """Provides the Survival Analysis tab."""

    def _build_survival_frame(self):
        T = self.T
        f = ctk.CTkFrame(self.content, fg_color=T['bg'])
        self.frames['survival'] = f
        self._section_header(f, "Survival Analysis")
        card = self._card(f)
        card.pack(fill='both', expand=True)
        inner = ctk.CTkFrame(card, fg_color='transparent')
        inner.pack(fill='both', expand=True, padx=16, pady=12)

        self._label(inner, "Kaplan-Meier Survival Analysis", "sub").pack(anchor='w', pady=(0, 6))

        file_row = ctk.CTkFrame(inner, fg_color='transparent')
        file_row.pack(fill='x', pady=(0, 6))
        self.surv_path = self._input_entry(file_row, "CSV with 'time' and 'event' columns...")
        self.surv_path.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(file_row, "Load", self._surv_load).pack(side='right', padx=(0, 4))
        self._action_button(file_row, "Run Analysis", self._surv_run).pack(side='right')

        self._label(inner, "Or use demo data:", "small").pack(anchor='w', pady=(4, 4))
        self._action_button(inner, "Run with Demo Data", self._surv_demo).pack(anchor='w', pady=(0, 8))

        ctk.CTkFrame(inner, height=1, fg_color=T['border']).pack(fill='x', pady=(4, 8))

        self._label(inner, "Results", "sub").pack(anchor='w', pady=(0, 6))
        self.surv_result = self._text_box(inner, height=250)
        self.surv_result.pack(fill='both', expand=True)

    def _surv_load(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self.surv_path.delete(0, 'end')
            self.surv_path.insert(0, path)
            self._set_status(f"Loaded: {os.path.basename(path)}")

    def _surv_run(self):
        import pandas as pd
        path = self.surv_path.get().strip()
        if not path:
            self._msg_warning("Input Required", "Please load a CSV file with time and event columns.")
            return
        try:
            from ..core.survival import kaplan_meier, log_rank_test, format_survival_report
            df = pd.read_csv(path)
            if 'time' not in df.columns or 'event' not in df.columns:
                self._msg_error("Error", "CSV must have 'time' and 'event' columns.")
                return
            result = kaplan_meier(df['time'].values, df['event'].values)
            report = format_survival_report(result)
            self.surv_result.delete("1.0", "end")
            self.surv_result.insert("1.0", report)
            self._set_status("Survival analysis complete")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _surv_demo(self):
        try:
            import numpy as np
            from ..core.survival import kaplan_meier, format_survival_report
            np.random.seed(42)
            times = np.random.exponential(30, 50)
            events = np.random.binomial(1, 0.6, 50)
            result = kaplan_meier(times, events)
            report = format_survival_report(result)
            self.surv_result.delete("1.0", "end")
            self.surv_result.insert("1.0", report)
            self._set_status("Demo survival analysis complete")
        except Exception as e:
            self._msg_error("Error", str(e))
