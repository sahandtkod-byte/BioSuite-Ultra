"""
Modern Tkinter GUI with center dialogs, theme switching, and professional look.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import os
import sys
import builtins
import pandas as pd
import numpy as np
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bioplatter.core.utils import config, set_theme, load_dataframe_safe, center_window, save_config
from bioplatter.core.sequence import read_fasta, gc_content, reverse_complement, translate
from bioplatter.plotting.biological_plots import *
from bioplatter.plotting.math_plots import *
from bioplatter.plotting.specialized_plots import *

class BioSuiteApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("BioSuite – Modern Bioinformatic Platform")
        self.geometry("1100x800")
        self.minsize(900, 650)
        self.apply_theme()
        self.create_menu()
        self.create_widgets()
        # Apply modern style to ttk widgets
        self.setup_styles()

    def setup_styles(self):
        style = ttk.Style()
        if config['theme'] == 'light':
            style.configure('TNotebook.Tab', padding=[12, 4], font=('Segoe UI', 10))
            style.configure('TButton', padding=[8, 4], font=('Segoe UI', 9))
            style.configure('TLabel', font=('Segoe UI', 9))
            style.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'))
        else:
            style.configure('TNotebook.Tab', padding=[12, 4], font=('Segoe UI', 10))
            style.configure('TButton', padding=[8, 4], font=('Segoe UI', 9))
            style.configure('TLabel', font=('Segoe UI', 9))
            style.configure('TLabelframe.Label', font=('Segoe UI', 9, 'bold'))

    def apply_theme(self):
        import sv_ttk
        sv_ttk.set_theme("dark" if config['theme'] == 'dark' else "light")
        if config['theme'] == 'light':
            bg = '#f0f2f5'
            fg = '#1e1e2e'
            btn_bg = '#0078d4'
            tk_bg = '#f0f2f5'
        else:
            bg = '#1e1e2e'
            fg = '#e0e0e0'
            btn_bg = '#2d2d3a'
            tk_bg = '#1e1e2e'
        self.configure(bg=tk_bg)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background=bg, foreground=fg, fieldbackground=bg)
        style.configure('TFrame', background=bg)
        style.configure('TLabel', background=bg, foreground=fg)
        style.configure('TButton', background=btn_bg, foreground=fg)
        style.map('TButton', background=[('active', '#3a3a4a')])
        style.configure('TNotebook', background=bg, tabmargins=[2,5,2,0])
        style.configure('TNotebook.Tab', background=btn_bg, foreground=fg, padding=[12,4])
        style.map('TNotebook.Tab', background=[('selected', '#0078d4')])

    def create_menu(self):
        menubar = tk.Menu(self, bg=config['theme']=='light' and '#f0f2f5' or '#1e1e2e', fg=config['theme']=='light' and '#1e1e2e' or '#e0e0e0')
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open FASTA...", command=self.open_fasta)
        file_menu.add_command(label="Open CSV/Excel...", command=self.open_datafile)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        analyze_menu = tk.Menu(menubar, tearoff=0)
        analyze_menu.add_command(label="GC Content", command=self.gc_calc_seq)
        analyze_menu.add_command(label="Reverse Complement", command=self.revcomp_seq)
        analyze_menu.add_command(label="Translate", command=self.translate_seq_frame)
        analyze_menu.add_command(label="FASTA Viewer", command=self.view_fasta)
        menubar.add_cascade(label="Analyze", menu=analyze_menu)

        plots_menu = tk.Menu(menubar, tearoff=0)
        plots_menu.add_command(label="Volcano Plot", command=lambda: self.run_plot('volcano'))
        plots_menu.add_command(label="PCA Plot", command=lambda: self.run_plot('pca'))
        plots_menu.add_command(label="Manhattan Plot", command=lambda: self.run_plot('manhattan'))
        plots_menu.add_command(label="Boxplot", command=lambda: self.run_plot('boxplot'))
        plots_menu.add_separator()
        plots_menu.add_command(label="Show All Plots", command=lambda: self.notebook.select(self.plot_frame))
        menubar.add_cascade(label="Plots", menu=plots_menu)

        theme_menu = tk.Menu(menubar, tearoff=0)
        theme_menu.add_command(label="Light Theme", command=lambda: self.switch_theme('light'))
        theme_menu.add_command(label="Dark Theme", command=lambda: self.switch_theme('dark'))
        menubar.add_cascade(label="Theme", menu=theme_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="User Guide", command=self.show_help)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    def switch_theme(self, theme):
        config['theme'] = theme
        save_config(config)
        set_theme(theme)
        self.apply_theme()
        self.setup_styles()
        # Refresh notebook tabs style
        style = ttk.Style()
        style.theme_use('clam')
        self.update()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Plot Selector Tab
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="📊 Advanced Plots")
        self.setup_plot_selector()

        # Sequence Tools Tab
        self.seq_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.seq_frame, text="🧬 Sequence Tools")
        self.setup_sequence_tools()

        # Status bar
        self.status = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=2)

    def setup_plot_selector(self):
        # Left panel: search and categories
        left_frame = ttk.Frame(self.plot_frame, width=280)
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        left_frame.pack_propagate(False)

        ttk.Label(left_frame, text="🔍 Search", font=('Segoe UI', 10, 'bold')).pack(pady=(10,5))
        self.search_entry = ttk.Entry(left_frame)
        self.search_entry.pack(padx=10, pady=(0,10), fill='x')
        self.search_entry.bind('<KeyRelease>', lambda e: self.update_plot_list())

        ttk.Label(left_frame, text="📂 Categories", font=('Segoe UI', 10, 'bold')).pack(pady=(5,5))
        self.cat_box = tk.Listbox(left_frame, bg='#353545' if config['theme']=='dark' else '#f0f0f0',
                                  fg='white' if config['theme']=='dark' else 'black',
                                  relief=tk.FLAT, highlightthickness=0, selectbackground='#0078d4')
        self.cat_box.pack(padx=10, pady=(0,10), fill='both', expand=True)
        categories = ['All Plots', 'Advanced Biological', 'Basic Biological',
                      'Mathematical', 'Specialized', 'Additional', 'New Plots']
        for c in categories: self.cat_box.insert('end', c)
        self.cat_box.selection_set(0)
        self.cat_box.bind('<<ListboxSelect>>', lambda e: self.update_plot_list())

        # Right panel: plot list
        right_frame = ttk.Frame(self.plot_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=5, pady=5)

        self.count_label = ttk.Label(right_frame, text="📊 Available: 0", anchor='w')
        self.count_label.pack(fill='x')

        list_frame = ttk.Frame(right_frame)
        list_frame.pack(fill='both', expand=True)
        scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scroll.pack(side='right', fill='y')
        self.plot_listbox = tk.Listbox(list_frame, yscrollcommand=scroll.set,
                                       bg='#2a2a3e' if config['theme']=='dark' else 'white',
                                       fg='#e0e0e0' if config['theme']=='dark' else 'black',
                                       selectbackground='#0078d4', relief=tk.FLAT,
                                       font=('Consolas', 10))
        self.plot_listbox.pack(side='left', fill='both', expand=True)
        scroll.config(command=self.plot_listbox.yview)

        self.plots_data = [
            ('🌋 Volcano Plot', 'volcano', 'Advanced Biological'),
            ('📊 PCA Plot', 'pca', 'Advanced Biological'),
            ('🧬 Manhattan Plot', 'manhattan', 'Advanced Biological'),
            ('📈 MA Plot', 'ma', 'Advanced Biological'),
            ('🔄 Venn Diagram', 'venn', 'Advanced Biological'),
            ('📊 Barplot', 'barplot', 'Basic Biological'),
            ('📦 Boxplot', 'boxplot', 'Basic Biological'),
            ('🔥 Heatmap', 'heatmap', 'Basic Biological'),
            ('⚫ Scatter Plot', 'scatter', 'Basic Biological'),
            ('📉 Time Series', 'timeseries', 'Basic Biological'),
            ('📐 Sine', 'sine', 'Mathematical'),
            ('📐 Cosine', 'cosine', 'Mathematical'),
            ('📐 Linear', 'linear', 'Mathematical'),
            ('📐 Quadratic', 'quadratic', 'Mathematical'),
            ('📐 Cubic', 'cubic', 'Mathematical'),
            ('📐 Exponential', 'exponential', 'Mathematical'),
            ('📐 Logistic', 'logistic', 'Mathematical'),
            ('🧬 GSEA Plot', 'gsea', 'Specialized'),
            ('🧬 Motif Logo', 'motif', 'Specialized'),
            ('🧬 Sankey Diagram', 'sankey', 'Specialized'),
            ('📊 QQ-plot', 'qq', 'Additional'),
            ('🔥 Clustered Heatmap', 'clustered_heatmap', 'Additional'),
            ('🎯 Circos Plot', 'circos', 'Additional'),
            ('🔬 Alignment Viewer', 'alignment', 'Additional'),
            ('🗺️ UMAP Plot', 'umap', 'Additional'),
            ('🎻 Violin Plot', 'violin', 'New Plots'),
            ('☁️ Raincloud Plot', 'raincloud', 'New Plots'),
            ('🏔️ Ridge Plot', 'ridge', 'New Plots'),
            ('🔘 Dot Plot', 'dotplot', 'New Plots')
        ]
        self.plot_func_map = {
            'volcano': volcano_plot, 'pca': pca_plot, 'manhattan': manhattan_plot,
            'ma': ma_plot, 'venn': venn_diagram, 'barplot': barplot_custom,
            'boxplot': boxplot_custom, 'heatmap': heatmap_custom, 'scatter': scatter_custom,
            'timeseries': timeseries_plot, 'sine': sine_plot, 'cosine': cosine_plot,
            'linear': linear_plot, 'quadratic': quadratic_plot, 'cubic': cubic_plot,
            'exponential': exponential_plot, 'logistic': logistic_plot, 'gsea': gsea_plot,
            'motif': motif_logo, 'sankey': sankey_diagram, 'qq': qq_plot,
            'clustered_heatmap': clustered_heatmap, 'circos': circos_plot,
            'alignment': alignment_viewer, 'umap': umap_plot,
            'violin': violin_plot, 'raincloud': raincloud_plot, 'ridge': ridge_plot, 'dotplot': dot_plot
        }
        self.update_plot_list()

        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="🎨 Generate Plot", command=self.generate_plot).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="❓ Help", command=self.show_help).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="📁 Export All", command=self.export_all_plots).pack(side='left', padx=5)
        self.plot_listbox.bind('<Double-Button-1>', lambda e: self.generate_plot())

    def update_plot_list(self):
        term = self.search_entry.get().strip().lower()
        sel_idx = self.cat_box.curselection()
        sel_cat = self.cat_box.get(sel_idx[0]) if sel_idx else 'All Plots'
        self.plot_listbox.delete(0, 'end')
        cnt = 0
        for name, pid, cat in self.plots_data:
            if (sel_cat == 'All Plots' or cat == sel_cat) and (term == '' or term in name.lower()):
                self.plot_listbox.insert('end', name)
                cnt += 1
        self.count_label.config(text=f"📊 Available: {cnt}")

    def generate_plot(self):
        sel = self.plot_listbox.curselection()
        if not sel:
            messagebox.showwarning("No Selection", "Please select a plot.")
            return
        plot_name = self.plot_listbox.get(sel[0])
        plot_id = next(p[1] for p in self.plots_data if p[0] == plot_name)
        self.run_plot(plot_id)

    def run_plot(self, plot_id):
        original_input = builtins.input
        _last_df = None

        def gui_input(prompt):
            nonlocal _last_df
            if 'Load data from file?' in prompt:
                return 'y' if messagebox.askyesno("Confirm", prompt) else 'n'
            if 'Show data summary' in prompt or 'Use default' in prompt:
                return 'y' if messagebox.askyesno("Confirm", prompt) else 'n'
            if 'Save this plot?' in prompt or 'Save as HTML' in prompt:
                return 'y' if messagebox.askyesno("Save", prompt) else 'n'
            if 'Correlation type' in prompt:
                res = simpledialog.askstring("Correlation", "Pearson or Spearman?", initialvalue='pearson')
                return res if res else 'pearson'
            if 'File path' in prompt:
                path = filedialog.askopenfilename(title=prompt, filetypes=[("CSV","*.csv"),("Excel","*.xlsx")])
                if path:
                    try:
                        _last_df = pd.read_csv(path) if path.endswith('.csv') else pd.read_excel(path)
                    except Exception as e:
                        messagebox.showerror("Error", str(e))
                return path if path else ''
            if 'column' in prompt.lower():
                if _last_df is not None:
                    cols = list(_last_df.columns)
                    dlg = tk.Toplevel(self)
                    dlg.title("Select Column")
                    center_window(dlg, 400, 180)
                    dlg.configure(bg='#1e1e2e' if config['theme']=='dark' else '#f0f2f5')
                    ttk.Label(dlg, text=prompt).pack(pady=10)
                    combo = ttk.Combobox(dlg, values=cols, state='readonly')
                    combo.pack(pady=5)
                    result = ""
                    def ok():
                        nonlocal result
                        result = combo.get()
                        dlg.destroy()
                    ttk.Button(dlg, text="OK", command=ok).pack(pady=10)
                    center_window(dlg, 400, 180)
                    dlg.transient(self)
                    dlg.grab_set()
                    self.wait_window(dlg)
                    return result
                else:
                    return simpledialog.askstring("Column Name", prompt)
            default = ''
            match = re.search(r'\(default ([^)]+)\)', prompt)
            if match: default = match.group(1)
            dlg = tk.Toplevel(self)
            dlg.title("Parameter Input")
            center_window(dlg, 450, 180)
            dlg.configure(bg='#1e1e2e' if config['theme']=='dark' else '#f0f2f5')
            ttk.Label(dlg, text=prompt, wraplength=380).pack(pady=15, padx=15)
            entry = ttk.Entry(dlg, width=30)
            entry.insert(0, default)
            entry.pack(pady=5)
            result = ""
            def ok():
                nonlocal result
                result = entry.get()
                dlg.destroy()
            ttk.Button(dlg, text="OK", command=ok).pack(pady=10)
            dlg.transient(self)
            dlg.grab_set()
            self.wait_window(dlg)
            return result

        builtins.input = gui_input
        try:
            func = self.plot_func_map.get(plot_id)
            if func: func()
            else: messagebox.showerror("Error", f"Plot '{plot_id}' not found.")
        except Exception as e:
            messagebox.showerror("Plot Error", str(e))
        finally:
            builtins.input = original_input

    def export_all_plots(self):
        folder = filedialog.askdirectory(title="Select Export Folder")
        if folder:
            try:
                export_all_to_folder(folder)
                messagebox.showinfo("Export Complete", f"All plots saved to:\n{folder}")
            except Exception as e:
                messagebox.showerror("Export Failed", str(e))

    # ---------- Sequence Tools ----------
    def setup_sequence_tools(self):
        left_frame = ttk.Frame(self.seq_frame)
        left_frame.pack(side='left', fill='both', expand=True, padx=10, pady=10)

        ttk.Label(left_frame, text="✏️ Sequence Input", font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        self.seq_input = tk.Text(left_frame, height=12, width=60, font=('Consolas', 10),
                                 wrap=tk.NONE, relief=tk.FLAT, borderwidth=1)
        self.seq_input.pack(fill='both', expand=True, pady=5)
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill='x', pady=5)
        ttk.Button(btn_frame, text="Load FASTA", command=self.load_fasta_to_input).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Clear", command=lambda: self.seq_input.delete(1.0, tk.END)).pack(side='left', padx=2)

        right_frame = ttk.Frame(self.seq_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)

        ttk.Label(right_frame, text="📋 Results", font=('Segoe UI', 12, 'bold')).pack(anchor='w')
        self.seq_result = tk.Text(right_frame, height=12, width=60, font=('Consolas', 10),
                                  wrap=tk.WORD, relief=tk.FLAT, borderwidth=1)
        self.seq_result.pack(fill='both', expand=True, pady=5)

        action_frame = ttk.Frame(self.seq_frame)
        action_frame.pack(side='bottom', fill='x', pady=10, padx=10)
        ttk.Button(action_frame, text="GC%", command=self.gc_calc_seq).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Reverse Complement", command=self.revcomp_seq).pack(side='left', padx=5)
        ttk.Button(action_frame, text="Translate (Frame 1)", command=self.translate_seq_frame).pack(side='left', padx=5)

    def load_fasta_to_input(self):
        path = filedialog.askopenfilename(filetypes=[("FASTA", "*.fasta *.fa"), ("All", "*.*")])
        if path:
            seqs = read_fasta(path)
            if seqs:
                content = ""
                for name, seq in seqs:
                    content += f">{name}\n{seq}\n"
                self.seq_input.delete(1.0, tk.END)
                self.seq_input.insert(tk.END, content)
                self.status.config(text=f"Loaded {len(seqs)} sequences from {os.path.basename(path)}")
            else:
                messagebox.showerror("Error", "Could not read FASTA file.")

    def get_current_sequence(self):
        text = self.seq_input.get(1.0, tk.END).strip()
        lines = text.split('\n')
        seq = ''.join(line.strip() for line in lines if not line.startswith('>')).replace(' ', '')
        return seq

    def gc_calc_seq(self):
        seq = self.get_current_sequence()
        if not seq:
            messagebox.showwarning("No Sequence", "Please enter or load a sequence.")
            return
        gc = gc_content(seq)
        self.seq_result.delete(1.0, tk.END)
        self.seq_result.insert(tk.END, f"Sequence length: {len(seq)}\n")
        self.seq_result.insert(tk.END, f"GC content: {gc:.2f}%\n")
        self.seq_result.insert(tk.END, f"A: {seq.upper().count('A')}, T: {seq.upper().count('T')}\n")
        self.seq_result.insert(tk.END, f"G: {seq.upper().count('G')}, C: {seq.upper().count('C')}\n")

    def revcomp_seq(self):
        seq = self.get_current_sequence()
        if not seq:
            messagebox.showwarning("No Sequence", "Please enter or load a sequence.")
            return
        rc = reverse_complement(seq)
        self.seq_result.delete(1.0, tk.END)
        self.seq_result.insert(tk.END, f"Original: {seq[:100]}{'...' if len(seq)>100 else ''}\n")
        self.seq_result.insert(tk.END, f"Reverse Complement:\n{rc[:100]}{'...' if len(rc)>100 else ''}\n")

    def translate_seq_frame(self):
        seq = self.get_current_sequence()
        if not seq:
            messagebox.showwarning("No Sequence", "Please enter or load a sequence.")
            return
        prot = translate(seq, frame=1)
        self.seq_result.delete(1.0, tk.END)
        self.seq_result.insert(tk.END, f"Translation (frame 1):\n{prot[:200]}{'...' if len(prot)>200 else ''}\n")
        self.seq_result.insert(tk.END, "(Stop codons shown as '*')")

    def view_fasta(self):
        path = filedialog.askopenfilename(filetypes=[("FASTA", "*.fasta *.fa"), ("All", "*.*")])
        if path:
            seqs = read_fasta(path)
            if seqs:
                viewer = tk.Toplevel(self)
                viewer.title(f"FASTA Viewer - {os.path.basename(path)}")
                viewer.geometry("800x600")
                center_window(viewer, 800, 600)
                viewer.configure(bg='#1e1e2e' if config['theme']=='dark' else '#f0f2f5')
                text = tk.Text(viewer, wrap=tk.NONE, font=('Courier', 9))
                scroll_y = tk.Scrollbar(viewer, orient=tk.VERTICAL, command=text.yview)
                scroll_x = tk.Scrollbar(viewer, orient=tk.HORIZONTAL, command=text.xview)
                text.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
                scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
                scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
                text.pack(fill=tk.BOTH, expand=True)
                content = ""
                for name, seq in seqs:
                    content += f">{name}\n"
                    for i in range(0, len(seq), 60):
                        content += f"{seq[i:i+60]}\n"
                text.insert(tk.END, content)
                text.config(state=tk.DISABLED)
            else:
                messagebox.showerror("Error", "Could not read FASTA file.")

    def open_fasta(self):
        self.load_fasta_to_input()
        self.notebook.select(self.seq_frame)

    def open_datafile(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx")])
        if path:
            df = load_dataframe_safe(path)
            if df is not None:
                self.status.config(text=f"Loaded {df.shape[0]} rows, {df.shape[1]} cols from {os.path.basename(path)}")

    def show_help(self):
        messagebox.showinfo("Help",
            "BioSuite – Modern Bioinformatic Platform\n\n"
            "Plots:\n- Select from 30+ biological and mathematical plots\n"
            "- Double-click or press Generate, then answer popup dialogs\n\n"
            "Sequence Tools:\n- Enter or load FASTA, then use buttons for GC%, revcomp, translation\n\n"
            "Theme:\n- Change between Light and Dark from the Theme menu\n\n"
            "Export:\n- Export All to Folder saves all plots as PNG\n- Batch PDF exports to a single PDF file")

    def show_about(self):
        messagebox.showinfo("About BioSuite", "Bio-Platter Pro v11.0 Integrated\n(c) 2025\nAll plots are self-contained.\nNo internet required.")