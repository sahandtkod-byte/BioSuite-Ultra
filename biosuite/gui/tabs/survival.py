"""
Survival Analysis tab: Kaplan-Meier, log-rank group comparison and Cox summary.
"""
import os
from tkinter import filedialog

import customtkinter as ctk


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

        self._label(inner, "Compare two groups (log-rank)", "sub").pack(anchor='w', pady=(4, 6))
        group_row = ctk.CTkFrame(inner, fg_color='transparent')
        group_row.pack(fill='x', pady=(0, 6))
        self._label(group_row, "Group column:", "small").pack(side='left', padx=(0, 6))
        self.surv_group_col = self._input_entry(group_row, "group")
        self.surv_group_col.insert(0, "group")
        self.surv_group_col.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(group_row, "Compare Groups", self._surv_logrank).pack(side='right')

        self._label(inner, "Cox summary (median split)", "sub").pack(anchor='w', pady=(0, 6))
        cox_row = ctk.CTkFrame(inner, fg_color='transparent')
        cox_row.pack(fill='x', pady=(0, 6))
        self._label(cox_row, "Covariate column:", "small").pack(side='left', padx=(0, 6))
        self.surv_cov_col = self._input_entry(cox_row, "covariate")
        self.surv_cov_col.insert(0, "covariate")
        self.surv_cov_col.pack(side='left', fill='x', expand=True, padx=(0, 8))
        self._action_button(cox_row, "Run Cox Summary", self._surv_cox).pack(side='right')

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
            from ...core.survival import format_km_result, kaplan_meier
            df = pd.read_csv(path)
            if 'time' not in df.columns or 'event' not in df.columns:
                self._msg_error("Error", "CSV must have 'time' and 'event' columns.")
                return
            result = kaplan_meier(df['time'].values, df['event'].values)
            report = format_km_result(result)
            self.surv_result.delete("1.0", "end")
            self.surv_result.insert("1.0", report)
            self._set_status("Survival analysis complete")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _surv_demo(self):
        try:
            import numpy as np

            from ...core.survival import format_km_result, kaplan_meier
            rng = np.random.default_rng(42)
            times = rng.exponential(30, 50)
            events = rng.binomial(1, 0.6, 50)
            result = kaplan_meier(times, events)
            report = format_km_result(result)
            self.surv_result.delete("1.0", "end")
            self.surv_result.insert("1.0", report)
            self._set_status("Demo survival analysis complete")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _surv_read_csv(self, required):
        path = self.surv_path.get().strip()
        if not path:
            self._msg_warning("Input Required", "Please load a CSV file with time and event columns.")
            return None
        try:
            import pandas as pd
            df = pd.read_csv(path)
        except Exception as e:
            self._msg_error("Error", str(e))
            return None
        missing = [c for c in required if c not in df.columns]
        if missing:
            self._msg_error("Error", f"CSV is missing column(s): {', '.join(missing)}.")
            return None
        return df

    def _surv_logrank(self):
        from ...core.survival import kaplan_meier, log_rank_test
        group_col = self.surv_group_col.get().strip() or "group"
        df = self._surv_read_csv(["time", "event", group_col])
        if df is None:
            return
        try:
            groups = df[group_col].dropna().unique()
            if len(groups) != 2:
                self._msg_error("Error", f"Log-rank needs exactly 2 groups in '{group_col}', "
                                         f"found {len(groups)}.")
                return
            lines = ["=== Log-Rank Group Comparison ==="]
            pairs = []
            for name in groups:
                sub = df[df[group_col] == name]
                km = kaplan_meier(sub["time"].values, sub["event"].values)
                lines.append(f"Group {name!r}: n={len(sub)}, events={km.number_events}, "
                             f"median survival={km.median_survival}")
                pairs.append((sub["time"].values, sub["event"].values))
            result = log_rank_test(pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1])
            lines.append(f"Chi-square statistic: {result['statistic']}")
            lines.append(f"p-value: {result['p_value']}")
            self.surv_result.delete("1.0", "end")
            self.surv_result.insert("1.0", "\n".join(lines))
            self._set_status("Log-rank comparison complete")
        except Exception as e:
            self._msg_error("Error", str(e))

    def _surv_cox(self):
        from ...core.survival import cox_ph_summary
        cov_col = self.surv_cov_col.get().strip() or "covariate"
        df = self._surv_read_csv(["time", "event", cov_col])
        if df is None:
            return
        try:
            result = cox_ph_summary(df["time"].values, df["event"].values, df[cov_col].values)
        except Exception as e:
            self._msg_error("Error", str(e))
            return
        ratio = result["hazard_ratio"]
        lines = [
            "=== Cox Proportional Hazards (median split) ===",
            f"Hazard ratio: {ratio if ratio is not None else 'not estimable'}",
            f"p-value: {result['p_value']}",
            f"Median survival, high group: {result['median_high_group']}",
            f"Median survival, low group: {result['median_low_group']}",
            f"Split: {result['split']}",
            result["note"],
        ]
        if result["message"]:
            lines.append(result["message"])
        self.surv_result.delete("1.0", "end")
        self.surv_result.insert("1.0", "\n".join(lines))
        self._set_status("Cox summary complete")
