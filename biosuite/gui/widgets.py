"""
Reusable GUI widgets for BioSuite Ultra.

Currently provides a lightweight themed Tooltip that works with any
tkinter/customtkinter widget and follows the active theme.
"""
import tkinter as tk


class Tooltip:
    """Hover tooltip for any widget. Themed, auto-positioned, zero-cost when hidden.

    Usage:
        Tooltip(widget, "Explain what this button does", theme=T)

    The tooltip appears after a short delay and disappears on leave or click.
    """

    DELAY_MS = 450
    PAD_X = 10
    OFFSET_Y = 6

    def __init__(self, widget, text, theme=None, delay=None):
        self._widget = widget
        self._text = text
        self._T = theme or {}
        self._delay = delay if delay is not None else self.DELAY_MS
        self._tip = None
        self._after_id = None
        widget.bind('<Enter>', self._schedule, add='+')
        widget.bind('<Leave>', self._hide, add='+')
        widget.bind('<ButtonPress>', self._hide, add='+')

    def _schedule(self, event=None):
        self._cancel()
        self._after_id = self._widget.after(self._delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self._widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None:
            return
        try:
            if not self._widget.winfo_exists():
                return
        except Exception:
            return
        T = self._T
        tip = tk.Toplevel(self._widget)
        tip.overrideredirect(True)
        tip.attributes('-topmost', True)
        bg = T.get('card', '#111c11')
        border = T.get('border_light', '#2a5a2a')
        fg = T.get('text', '#e0ffe8')
        frame = tk.Frame(tip, bg=border, bd=0)
        frame.pack()
        label = tk.Label(frame, text=self._text, justify='left',
                         bg=bg, fg=fg, font=('Segoe UI', 10),
                         padx=self.PAD_X, pady=6, wraplength=280)
        label.pack(padx=1, pady=1)
        # Position: below the widget, clamped to screen
        self._widget.update_idletasks()
        x = self._widget.winfo_rootx()
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + self.OFFSET_Y
        tip.update_idletasks()
        sw = tip.winfo_screenwidth()
        if x + tip.winfo_reqwidth() > sw:
            x = sw - tip.winfo_reqwidth() - 8
        tip.geometry(f"+{max(0, x)}+{y}")
        self._tip = tip

    def _hide(self, event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def attach_tooltip(widget, text, theme=None):
    """Attach a themed tooltip; ignores empty text. Returns the Tooltip (or None)."""
    if not text:
        return None
    try:
        return Tooltip(widget, text, theme)
    except Exception:
        return None
