"""
Custom themed dialogs and splash screen for BioSuite GUI.
No native Windows widgets — all cyberpunk-themed.
"""
import customtkinter as ctk
import tkinter as tk

from .themes import FONT_FAMILY, FONT_BODY, FONT_BUTTON, FONT_MONO, FONT_SMALL


class _BaseDialog(ctk.CTkToplevel):
    """Base class for all custom themed dialogs."""

    def __init__(self, parent, T, title="Dialog", width=420, height=220):
        super().__init__(parent)
        self.T = T
        self.result = None
        self.title(title)
        self.overrideredirect(True)
        self.attributes('-topmost', True)

        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - width) // 2
        y = (sh - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.configure(fg_color=T['bg'])

        self._outer = ctk.CTkFrame(self, fg_color=T.get('dialog_border', T['border']),
                                    corner_radius=16)
        self._outer.pack(fill='both', expand=True, padx=2, pady=2)

        self._card = ctk.CTkFrame(self._outer, fg_color=T.get('dialog_bg', T['card']),
                                   corner_radius=14)
        self._card.pack(fill='both', expand=True, padx=2, pady=2)

        self._body = ctk.CTkFrame(self._card, fg_color='transparent')
        self._body.pack(fill='both', expand=True, padx=24, pady=(20, 16))

        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.after(50, self._do_grab)
        self.bind('<Escape>', lambda e: self._on_cancel())

    def _do_grab(self):
        try:
            self.grab_set()
        except RuntimeError:
            pass

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()


class BioMessageDialog(_BaseDialog):
    """Themed info / warning / error dialog with icon and OK button."""

    def __init__(self, parent, T, title="Message", message="", msg_type="info"):
        self._msg_type = msg_type
        h = 180 if len(message) < 80 else 220
        super().__init__(parent, T, title=title, width=440, height=h)

        top = ctk.CTkFrame(self._body, fg_color='transparent')
        top.pack(fill='x', pady=(0, 16))

        icons = {'info': '\u2139', 'warning': '\u26A0', 'error': '\u2716', 'success': '\u2714'}
        colors = {'info': T['accent'], 'warning': '#f59e0b', 'error': T['danger'], 'success': T['success']}
        icon = icons.get(msg_type, '\u2139')
        color = colors.get(msg_type, T['accent'])

        ctk.CTkLabel(top, text=icon, font=(FONT_FAMILY, 28),
                      text_color=color, width=40).pack(side='left', padx=(0, 12), anchor='n')
        ctk.CTkLabel(top, text=message, font=FONT_BODY,
                      text_color=T['text'], wraplength=340, justify='left').pack(side='left', fill='x', expand=True)

        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="OK", width=100, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000' if msg_type != 'error' else '#ffffff',
                       hover_color=T['accent_dim'],
                       command=self._on_ok).pack(anchor='e')
        self.bind('<Return>', lambda e: self._on_ok())

    def _on_ok(self):
        self.result = 'ok'
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()


class BioConfirmDialog(_BaseDialog):
    """Themed yes/no confirmation dialog."""

    def __init__(self, parent, T, title="Confirm", message=""):
        h = 180 if len(message) < 80 else 220
        super().__init__(parent, T, title=title, width=440, height=h)

        top = ctk.CTkFrame(self._body, fg_color='transparent')
        top.pack(fill='x', pady=(0, 16))

        ctk.CTkLabel(top, text='\u2753', font=(FONT_FAMILY, 28),
                      text_color=T['accent'], width=40).pack(side='left', padx=(0, 12), anchor='n')
        ctk.CTkLabel(top, text=message, font=FONT_BODY,
                      text_color=T['text'], wraplength=340, justify='left').pack(side='left', fill='x', expand=True)

        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['border_light'],
                       text_color=T['text'], hover_color=T['border'],
                       command=self._on_no).pack(side='right', padx=(8, 0))
        ctk.CTkButton(btn_row, text="Yes", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._on_yes).pack(side='right')
        self.bind('<Return>', lambda e: self._on_yes())
        self.bind('<Escape>', lambda e: self._on_no())

    def _on_yes(self):
        self.result = True
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()

    def _on_no(self):
        self.result = False
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()


class BioInputDialog(_BaseDialog):
    """Themed text input dialog with placeholder and OK/Cancel."""

    def __init__(self, parent, T, title="Input", prompt="", default=""):
        super().__init__(parent, T, title=title, width=460, height=230)

        ctk.CTkLabel(self._body, text=prompt, font=FONT_BODY,
                      text_color=T['text'], wraplength=400, justify='left').pack(anchor='w', pady=(0, 10))

        self._entry = ctk.CTkEntry(self._body, height=40, font=(FONT_MONO, 13),
                                    corner_radius=8, fg_color=T['input_bg'],
                                    border_color=T['border'], text_color=T['text'],
                                    placeholder_text_color=T['text_muted'])
        self._entry.pack(fill='x', pady=(0, 16))
        if default:
            self._entry.insert(0, default)
        self._entry.select_range(0, 'end')
        self.after(80, lambda: self._entry.focus_force())

        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['border_light'],
                       text_color=T['text'], hover_color=T['border'],
                       command=self._on_cancel).pack(side='right', padx=(8, 0))
        ctk.CTkButton(btn_row, text="OK", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._on_ok).pack(side='right')
        self.bind('<Return>', lambda e: self._on_ok())

    def _on_ok(self):
        self.result = self._entry.get().strip()
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()


class BioFilePickerDialog(_BaseDialog):
    """Themed file selection dialog with browse button."""

    def __init__(self, parent, T, title="Select File", filetypes=None, prompt="Choose a file:"):
        super().__init__(parent, T, title=title, width=500, height=200)

        self._filetypes = filetypes or [("All", "*.*")]

        ctk.CTkLabel(self._body, text=prompt, font=FONT_BODY,
                      text_color=T['text'], wraplength=440, justify='left').pack(anchor='w', pady=(0, 10))

        entry_row = ctk.CTkFrame(self._body, fg_color='transparent')
        entry_row.pack(fill='x', pady=(0, 16))
        self._entry = ctk.CTkEntry(entry_row, height=40, font=(FONT_MONO, 12),
                                    corner_radius=8, fg_color=T['input_bg'],
                                    border_color=T['border'], text_color=T['text'],
                                    placeholder_text_color=T['text_muted'])
        self._entry.pack(side='left', fill='x', expand=True, padx=(0, 8))
        ctk.CTkButton(entry_row, text="Browse", width=80, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent_dim'],
                       text_color='#000000', hover_color=T['accent'],
                       command=self._browse).pack(side='right')

        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['border_light'],
                       text_color=T['text'], hover_color=T['border'],
                       command=self._on_cancel).pack(side='right', padx=(8, 0))
        ctk.CTkButton(btn_row, text="OK", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._on_ok).pack(side='right')
        self.after(80, lambda: self._entry.focus_force())

    def _browse(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=self._filetypes)
        if path:
            self._entry.delete(0, 'end')
            self._entry.insert(0, path)

    def _on_ok(self):
        self.result = self._entry.get().strip() or None
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()


class BioDropdownDialog(_BaseDialog):
    """Themed dropdown selection dialog."""

    def __init__(self, parent, T, title="Select", prompt="", options=None, default=None):
        super().__init__(parent, T, title=title, width=440, height=250)

        ctk.CTkLabel(self._body, text=prompt, font=FONT_BODY,
                      text_color=T['text'], wraplength=380, justify='left').pack(anchor='w', pady=(0, 10))

        self._combo = ctk.CTkComboBox(self._body, values=options or [], height=36,
                                       font=(FONT_FAMILY, 12), corner_radius=8,
                                       fg_color=T['input_bg'], border_color=T['border'],
                                       button_color=T['accent'], button_hover_color=T['accent_dim'],
                                       dropdown_fg_color=T['card'], dropdown_hover_color=T['border'],
                                       dropdown_text_color=T['text'], text_color=T['text'])
        self._combo.pack(fill='x', pady=(0, 16))
        if default:
            self._combo.set(default)
        elif options:
            self._combo.set(options[0])

        btn_row = ctk.CTkFrame(self._body, fg_color='transparent')
        btn_row.pack(fill='x')
        ctk.CTkButton(btn_row, text="Cancel", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['border_light'],
                       text_color=T['text'], hover_color=T['border'],
                       command=self._on_cancel).pack(side='right', padx=(8, 0))
        ctk.CTkButton(btn_row, text="OK", width=90, height=36, corner_radius=8,
                       font=FONT_BUTTON, fg_color=T['accent'],
                       text_color='#000000', hover_color=T['accent_dim'],
                       command=self._on_ok).pack(side='right')
        self.after(80, lambda: self._combo.focus_force())

    def _on_ok(self):
        self.result = self._combo.get()
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()

    def _on_cancel(self):
        self.result = None
        try:
            self.grab_release()
        except RuntimeError:
            pass
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
#  SPLASH SCREEN — Professional glassy design
# ═══════════════════════════════════════════════════════════════════════════════

class BioSplashScreen(ctk.CTkToplevel):
    """Animated splash screen with glassy cyberpunk aesthetic."""

    def __init__(self, parent, T):
        super().__init__(parent)
        self.T = T
        self.overrideredirect(True)
        self.attributes('-topmost', True)

        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        w, h = 520, 340
        x, y = (sw - w) // 2, (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.configure(fg_color=T['bg'])

        # Outer glow border — accent color with transparency feel
        outer = ctk.CTkFrame(self, fg_color=T.get('dialog_border', T['accent']),
                              corner_radius=24)
        outer.pack(fill='both', expand=True, padx=4, pady=4)

        # Glass card — slightly lighter than bg
        card = ctk.CTkFrame(outer, fg_color=T.get('dialog_bg', T['card']),
                             corner_radius=20)
        card.pack(fill='both', expand=True, padx=3, pady=3)

        body = ctk.CTkFrame(card, fg_color='transparent')
        body.pack(fill='both', expand=True, padx=44, pady=32)

        # App title — large, accent color
        title = ctk.CTkLabel(body, text="BIOSUITE", font=(FONT_FAMILY, 36, 'bold'),
                              text_color=T['accent'])
        title.pack(pady=(0, 2))

        # Subtitle
        ctk.CTkLabel(body, text="Ultra v4.0", font=(FONT_FAMILY, 13),
                      text_color=T['text_dim']).pack(pady=(0, 4))
        ctk.CTkLabel(body, text="Bioinformatic Platform", font=FONT_SMALL,
                      text_color=T['text_muted']).pack(pady=(0, 32))

        # Status text — dimmed
        self._status = ctk.CTkLabel(body, text="Initializing...", font=FONT_SMALL,
                                     text_color=T['text_dim'])
        self._status.pack(pady=(0, 14))

        # Sleek progress bar — wider, thinner, accent colored
        progress_frame = ctk.CTkFrame(body, fg_color='transparent')
        progress_frame.pack(fill='x')
        self._progress = ctk.CTkProgressBar(progress_frame, width=340, height=5,
                                              corner_radius=3, border_width=0,
                                              fg_color=T['border'],
                                              progress_color=T['accent'])
        self._progress.pack()
        self._progress.set(0)

        # Version tag at bottom
        ctk.CTkLabel(body, text="v4.0  ·  Pure Python", font=(FONT_FAMILY, 9),
                      text_color=T['text_muted']).pack(side='bottom', pady=(20, 0))

    def update_status(self, text, progress):
        self._status.configure(text=text)
        self._progress.set(progress)
        self.update_idletasks()

    def animate_out(self):
        self._fade_out()

    def _fade_out(self):
        alpha = self.attributes('-alpha')
        if alpha > 0.05:
            self.attributes('-alpha', alpha - 0.1)
            self.after(15, self._fade_out)
        else:
            self.destroy()
