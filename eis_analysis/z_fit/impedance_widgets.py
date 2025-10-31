# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re

from PyQt5.QtWidgets import (
    QMenu,
    QAction,
    QLineEdit,
    QMessageBox,
)

from ..z_system.definitions import FUNC_MAP, SYMBOL_MAP, COMPONENT_MAP, parse_system_key


class ImpedanceEntryWidget(QLineEdit):
    """
    A subclass of VarEntryWidget tailored for the impedance GUI.
    Parses input to handle variable types, modes, and special forms.
    """

    def __init__(self, acceptable_values=None, parent=None):
        """
        Initialize the ImpedanceEntryWidget.

        Args:
            acceptable_values (dict): A dictionary of acceptable input values where keys are symbols and values are their transformations.
            var_mapping (dict): A dictionary mapping variable types (e.g., "Z") to their aliases (e.g., "impedance").
            parent (QWidget): The parent widget.
        """
        # Preload with mode_mapping and form_mapping
        # self._default = ("Z'", "impedance.real")
        self._parsed_text = "impedance.real"
        super().__init__(parent)

        self.acceptable_values = {
            "variables": SYMBOL_MAP.copy(),
            "functions": FUNC_MAP.copy(),
            "modes": COMPONENT_MAP.copy(),
        }

        if isinstance(acceptable_values, dict):
            self.acceptable_values |= acceptable_values

        self.setPlaceholderText("Z'")
        self.editingFinished.connect(self._remove_spaces)

    def contextMenuEvent(self, event):
        """
        Override the context menu event to provide a custom menu for inserting acceptable values.
        """
        menu = QMenu(self)

        # Add acceptable values to the context menu
        if self.acceptable_values:
            for key, value in self.acceptable_values.items():
                if isinstance(value, dict):
                    sub_menu = QMenu(key.replace("_", " ").title(), self)
                    for symbol, transf in value.items():
                        action = QAction(f"{symbol} → {transf}", self)
                        action.triggered.connect(lambda _, s=symbol: self.insert_symbol(s))
                        sub_menu.addAction(action)
                    menu.addMenu(sub_menu)
                else:
                    action = QAction(f"{key} → {value}", self)
                    action.triggered.connect(lambda _, s=key: self.insert_symbol(s))
                    menu.addAction(action)

        menu.exec_(event.globalPos())

    def insert_symbol(self, symbol):
        """
        Insert the selected symbol at the cursor position or append it to the text.

        Args:
            symbol (str): The symbol to insert.
        """
        old_text = super().text()

        # Determine wrapping logic
        open_sym, close_sym = symbol, ""
        if symbol == "||":
            open_sym, close_sym = "|", "|"
        elif symbol in FUNC_MAP:
            open_sym, close_sym = f"{symbol}(", ")"

        if self.hasSelectedText():
            sec_start = self.selectionStart()
            sec_end = sec_start + self.selectionLength()
        else:
            sec_start = self.cursorPosition()
            sec_end = len(old_text) - sec_start if close_sym else sec_start

        center_text = old_text[sec_start:sec_end] if close_sym else ""
        new_text = old_text[:sec_start] + open_sym + center_text + close_sym + old_text[sec_end:]
        new_cursor_pos = sec_end + len(open_sym) + len(close_sym)

        self.blockSignals(True)
        self.setText(new_text)
        self.blockSignals(False)

        self.clearFocus()
        self.setFocus()

        self.setCursorPosition(new_cursor_pos)

        # if open_sym and close_sym:
        #     if self.hasSelectedText():
        #         sec_start = self.selectionStart()
        #         sec_end = sec_start + self.selectionLength()
        #     else:
        #         sec_start = self.cursorPosition()
        #         sec_end = len(old_text) - sec_start
        #     new_text = (
        #         old_text[:sec_start]
        #         + open_sym
        #         + old_text[sec_start:sec_end]
        #         + close_sym
        #         + old_text[sec_end:]
        #     )
        #     new_cursor_pos = sec_end + len(open_sym) + len(close_sym)
        # else:
        #     cursor_position = self.cursorPosition()
        #     new_text = old_text[:cursor_position] + symbol + old_text[cursor_position:]
        #     new_cursor_pos = cursor_position + len(symbol)

    def parse_input(self, text=None):
        """
        Parse the current input text and return the transformed form.

        Returns:
            str: The transformed form of the input text.
        """
        text = text if isinstance(text, str) else super().text().strip()

        if not text:
            self._parsed_text = parse_system_key(self.placeholderText())
        else:
            self._parsed_text = parse_system_key(text)

    def get_raw_text(self):
        """
        Get the raw (unparsed) text entered by the user.
        """
        text = super().text()
        if not text:
            # return self._default[0]
            return self.placeholderText()
        return text

    def _remove_spaces(self):
        """
        Process the text based on the selected mode and update the line edit.
        """
        text = super().text()
        if not text:
            return

        pattern = r"\s*(" + "|".join(re.escape(char) for char in ["(", ")"]) + r")\s*"
        processed_text = re.sub(pattern, r"\1", text)
        # Update the text in the line edit
        self.setText(processed_text)

    def setText(self, text):
        """
        Override the setText method to ensure the last valid text is updated.
        """
        try:
            text = text.strip()
            if text != self.placeholderText().strip():
                if text:
                    self._parsed_text = parse_system_key(text)
                super().setText(text)
        except Exception as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def text(self):
        """
        Override the text() method to return the parsed input.
        """
        return self._parsed_text

        # form_mapping = {
        #     "ln": "ln ",
        #     "log": "log10 ",
        #     "ƒₛₚₗ": "interpolated ",
        #     "ƒₛₘ": "smoothed ",
        #     "∂": "derivative ",
        # }
        # self.var_mapping = {
        #     "f": "frequency",
        #     "ω": "angular_frequency",
        #     "Z": "impedance",
        #     "Y": "admittance",
        #     "M": "modulus",
        #     "C": "capacitance",
        #     "χ": "susceptibility",
        #     "εᵣ": "relative_permittivity",
        #     "εᵣ_dc": "relative_permittivity_corrected",
        #     "σ": "conductivity",
        #     "σ_dc": "dc_conductivity",
        #     "ρ": "resistivity",
        # }
        # self.mode_mapping = {
        #     "'": "real",
        #     "''": "imag",
        #     '"': "imag",
        #     "θ": "phase",
        #     "tan(δ)": "slope",
        # }

    #     # Sort keys by length to ensure longer keys are matched first
    #     sorted_var_keys = sorted(self.var_mapping.keys(), key=len, reverse=True)
    #     sorted_mode_keys = sorted(self.mode_mapping.keys(), key=len, reverse=True)

    #     # Regex to match variable types and modes
    #     type_pattern = rf"(\b{'|'.join(re.escape(t) for t in sorted_var_keys)}\b)"
    #     mode_pattern = rf"({'|'.join(re.escape(m) for m in sorted_mode_keys)})"

    #     # Handle '|' characters.  If no mode -> magnitude, else abs(value with mode)
    #     text = re.sub(
    #         rf"\|{type_pattern}\|",
    #         lambda match: f"{self.var_mapping[match.group(1)]}.mag",
    #         text,
    #     )
    #     text = re.sub(
    #         rf"\|{type_pattern}{mode_pattern}\|",
    #         lambda match: f"abs({self.var_mapping[match.group(1)]}.{self.mode_mapping[match.group(2)]})",
    #         text,
    #     )

    #     # Handle other types with modes
    #     # Ensure mode is followed by space, end of string, or closing parenthesis
    #     text = re.sub(
    #         rf"{type_pattern}\s*{mode_pattern}",
    #         lambda match: f"{self.var_mapping[match.group(1)]}.{self.mode_mapping[match.group(2)]}",
    #         text,
    #     )

    #     # Close any unclosed parentheses
    #     if text.count("(") != text.count(")"):
    #         raise ValueError("Unmatched parentheses in input text.")

    #     # Pass the parsed text to the parent class for further transformation
    #     text = self._base_parse_input(text)
    #     if not text:
    #         # return self._default[1]
    #         return self.parse_input(self.placeholderText())

    #     return text

    # def _base_parse_input(self, text=None):
    #     """
    #     Parse the current input text and return the transformed form.

    #     Returns:
    #         str: The transformed form of the input text.
    #     """
    #     result = text if isinstance(text, str) else super().text().strip()

    #     # Replace each symbol in the text with its transformation
    #     for key, value in self.acceptable_values.items():
    #         if isinstance(value, dict):
    #             for symbol, transformation in value.items():
    #                 result = result.replace(symbol, transformation)
    #         else:
    #             result = result.replace(key, value)

    #     # Normalize spaces
    #     result = re.sub(r"\s+", " ", result)  # Replace multiple spaces with a single space
    #     result = re.sub(r"\s*\(\s*", "(", result)  # Replace ' (' with '('
    #     result = re.sub(r"\s*\)\s*", ")", result)  # Replace ' )' with ')'

    #     # Return the fully transformed text
    #     return result.strip()
