# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 17:05:01 2018.

@author: JClenney

General function file
"""
import re

from PyQt5.QtGui import QValidator
from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtWidgets import (
    QMenu,
    QAction,
    QLineEdit,
    QMessageBox,
    QInputDialog,
)

from ..widgets.widget_helpers import SelectionHelper, TextHistoryManager
from ..data_treatment.circuits import CKTS, CKT_FUNCS
from ..impedance_supplement.elements import ELEMENTS, ELEMENT_PAIR_MAP
from ..impedance_supplement.model_ops import (
    shift_elem_num,
    get_valid_model,
    clean_model_basic,
    parse_model_groups,
    get_valid_sub_model,
    clean_model_elements,
)
from ..impedance_supplement.model_eval import (
    model_compare,
    validate_model,
    extract_ckt_elements,
)

# from ..data_treatment.impedance_supplament import (
#     ELEMENTS,
#     ELEMENT_PAIR_MAP,
#     model_compare,
#     shift_elem_num,
#     validate_model,
#     # get_valid_model,
#     clean_model_basic,
#     parse_model_groups,
#     # get_valid_sub_model,
#     clean_model_elements,
#     extract_ckt_elements,
# )


# def get_valid_model(input_str):
#     if not input_str:
#         return "", []
#     try:
#         if validate_model(input_str):
#             model, elems = clean_model_elements(input_str)
#             return model, elems
#         return "", []
#     except Exception:
#         return "", []


# def get_valid_sub_model(main, sub_model):
#     if not sub_model or not main:
#         return "", []
#     try:
#         if validate_model(sub_model):
#             sub_model, s_elems = clean_model_elements(sub_model)
#             if sub_model not in main:
#                 return "", []
#             main, m_elems = clean_model_elements(main)
#             if not all(e in m_elems for e in s_elems):
#                 return "", []
#             return sub_model, s_elems
#         return "", []
#     except Exception:
#         return "", []


class ModelSelectionHelper(SelectionHelper):
    def expand_selection(self, text, sel_start, sel_len):
        sel_end = sel_start + sel_len

        def balanced_and_new(ns, ne):
            s = text[ns:ne]
            return (
                s.count("p(") == s.count(")")
                and s.find("p(") <= s.find(")")
                and (ns != sel_start or ne != sel_end)
                and s[0].isalpha()
                and (s[-1].isdigit() or s[-1] == ")")
            )

        new_start, new_end = sel_start, sel_end
        range_map = {}
        for ss in range(sel_start, -1, -1):
            # Incriment towards 0
            if not text[ss].isalpha() or (ss and text[ss - 1] == "("):
                # skip if current doesnt start with letter or is just inside "("
                continue
            for se in range(sel_end, len(text) + 1):
                # Incriment towards len(text), ie the end
                if text[se - 1] in "-,p(" or (se < len(text) and text[se] == ","):
                    # skip if ends with - , or p( or next is ,
                    continue
                if balanced_and_new(ss, se):
                    range_map[text[ss:se]] = (ss, se)
            if range_map and ss < sel_start - 2:
                break

        if range_map:
            # pick the shortest range
            new_start, new_end = range_map[min(range_map.keys(), key=len)]

        if balanced_and_new(new_start, new_end):
            return new_start, new_end - new_start
        else:
            return 0, len(text)


class ModelValidator(QValidator):
    def __init__(self, parent=None, **_):
        super().__init__(parent)

    def validate(self, input_str, pos):
        if validate_model(input_str):
            return QValidator.Acceptable, input_str, pos
        else:
            return QValidator.Intermediate, input_str, pos


class ModelLineEdit(QLineEdit):
    """
    A QLineEdit subclass with additional functionality for validating input and handling errors.
    """

    # Signal to connect an "on error" function
    onError = pyqtSignal(Exception)
    validityChanged = pyqtSignal(bool)  # True if model is valid
    replaceElement = pyqtSignal(str, str)  # old element, new element
    convertSection = pyqtSignal(str, str, str, type(validate_model), dict)
    # sortModel = pyqtSignal(list, str, str)  # groups, direction, sort_by

    def __init__(self, *args, ignore_error=True, default="", **kwargs):
        """
        Initialize the ModelLineEdit.

        Args:
            ignore_error (bool): Whether to ignore errors if no onError function is defined. Default is True.
        """
        self.ignore_error = ignore_error
        self.models = CKTS.copy()

        super().__init__(*args, **kwargs)
        self.validator = ModelValidator(self)

        self.setValidator(self.validator)
        self._elements = []
        self._sub_models = []
        self._sub_model_elems = []
        self.models = CKTS.copy()

        model, _ = get_valid_model(default or CKTS["Randles"])
        if not model:
            model = CKTS["Randles"]

        super().setText(model)

        self.selector = ModelSelectionHelper(self)
        self.history_manager = TextHistoryManager(self, set_method=super().setText, max_len=30)

        self.textChanged.connect(self._check_text)
        self.installEventFilter(self)

    @property
    def elements(self):
        if self.hasAcceptableInput():
            self._elements = extract_ckt_elements(self.text())
        return self._elements

    @property
    def sub_models(self):
        if not all(e in self.elements for e in self._sub_model_elems):
            self._sub_models = []
            self._sub_model_elems = []
        if not self._sub_models:
            text = self.text() if self.hasAcceptableInput() else self.history_manager.current
            self.sub_models = parse_model_groups(text, "infer")
        return self._sub_models

    @sub_models.setter
    def sub_models(self, value):
        if isinstance(value, (tuple, list)):
            self._sub_models = list(value)
        self._sub_model_elems = []
        for sm in self._sub_models:
            self._sub_model_elems += extract_ckt_elements(sm)

    def _check_text(self):
        if self.hasAcceptableInput():
            self.validityChanged.emit(True)
            self.history_manager.update_history(self.text())
        else:
            self.validityChanged.emit(False)

    def eventFilter(self, source, event):
        """
        Event filter to handle key press events for deleting items.
        """
        if event.type() == QEvent.KeyPress:
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self.setText(clean_model_basic(self.text()))
                self.keyPressEvent(event)
                return True
        return False

    def setText(self, text):
        """
        Override the setText method to ensure the last valid text is updated.
        """
        try:
            state, model, _ = self.validator.validate(text, 0)
            if state == QValidator.Acceptable:
                super().setText(model)
                self._check_text()
            else:
                # Do not set invalid text programically; emit error or ignore
                self.validityChanged.emit(False)
        except Exception as e:
            if self.receivers(self.onError):
                self.onError.emit(e)

            elif not self.ignore_error:
                # Re-raise the error if ignore_error is False and no onError function is connected
                raise e

    def contextMenuEvent(self, event):
        """
        Override the context menu event to add custom options.
        """

        # Create standard context menu
        menu = super().createStandardContextMenu()

        self.history_manager.create_or_patch_menu(menu, insert=True)

        menu.addSeparator()

        # Create "Insert model" action with submenu
        insert_element_menu = QMenu("Insert element", self)
        for element in sorted(ELEMENTS, key=len):
            action = QAction(element, self)
            action.triggered.connect(lambda _, value=f"{element}0": self._insert_at_cursor(value))
            insert_element_menu.addAction(action)
        menu.addMenu(insert_element_menu)

        # Create "Replace Element" action
        replace_element_action = QAction("Replace Element", self)
        replace_element_action.triggered.connect(self._replace_element)
        replace_element_action.setEnabled(bool(self.selectedText()))
        menu.addAction(replace_element_action)

        menu.addSeparator()

        # Create "Set model" action with submenu
        set_model_menu = QMenu("Set model", self)
        for model_name, model_value in self.models.items():
            action = QAction(f"{model_name}: {model_value}", self)
            action.triggered.connect(lambda _, value=model_value: self.setText(value))
            set_model_menu.addAction(action)
        menu.addMenu(set_model_menu)

        # Create "Insert model" action with submenu
        insert_model_menu = QMenu("Insert model", self)
        for model_name, model_value in self.models.items():
            action = QAction(f"{model_name}: {model_value}", self)
            action.triggered.connect(lambda _, value=model_value: self._insert_at_cursor(value))
            insert_model_menu.addAction(action)
        menu.addMenu(insert_model_menu)

        # Create "Convert Section" action
        convert_section_action = QAction("Convert Section", self)
        convert_section_action.triggered.connect(self._convert_section)
        convert_section_action.setEnabled(bool(self.selectedText()))
        menu.addAction(convert_section_action)

        menu.addSeparator()

        # Create "Save model" action
        save_model_action = QAction("Save model", self)
        save_model_action.triggered.connect(self._save_model)
        menu.addAction(save_model_action)

        # Create "Reset sub-models" action
        define_submodels_action = QAction("Reset sub-models", self)
        define_submodels_action.triggered.connect(lambda _: setattr(self, "sub_models", []))
        menu.addAction(define_submodels_action)

        # Create "Define sub-models" action
        define_submodels_action = QAction("Define sub-models", self)
        define_submodels_action.triggered.connect(self._define_submodels)
        menu.addAction(define_submodels_action)

        menu.exec_(event.globalPos())

    def _insert_at_cursor(self, text):
        """
        Insert text at the cursor position.
        """
        cursor_pos = self.cursorPosition()
        current_text = self.text()
        if not current_text:
            self.setText(text)
            self.setCursorPosition(len(text))
            return

        pre_char = current_text[cursor_pos - 1] if cursor_pos > 0 else ""
        post_char = current_text[cursor_pos] if cursor_pos < len(current_text) else ""

        if pre_char and post_char and pre_char not in "()-," and post_char not in "p-,)":
            proceed = QMessageBox.question(
                self,
                "Insertion Warning",
                f"Inserting here (within {pre_char}{post_char}) may create an invalid model. Proceed?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if not proceed:
                return

        prefix = ""
        if pre_char and pre_char not in "(-,":
            prefix = "," if post_char in [")", ","] else "-"

        suffix = ""
        if post_char and post_char not in ")-,":
            suffix = "," if pre_char in ["(", ","] else "-"

        new_text = prefix + shift_elem_num(text, 0, current_text) + suffix

        new_text = current_text[:cursor_pos] + new_text + current_text[cursor_pos:]
        super().setText(new_text)
        self.setCursorPosition(cursor_pos + len(new_text))

    def _save_model(self):
        """
        Save the current model with a user-provided name.
        Only saves valid models and displays a warning if invalid.
        """
        if self.hasAcceptableInput():
            name, ok = QInputDialog.getText(self, "Save Model", "Model name:")
            if ok and name:
                self.models[name] = self.text()
        else:
            QMessageBox.warning(
                self, "Invalid Model", "Current model is invalid and cannot be saved."
            )

    def _define_submodels(self):
        """
        Define sub-models by editing a semicolon-separated string.
        """
        # self.commit_model()
        self._check_text()

        current_submodels = "; ".join(self.sub_models)
        new_submodels, ok = QInputDialog.getText(
            self,
            "Define Sub-models",
            "Edit sub-models (separate with semicolons):",
            text=current_submodels,
        )

        if ok and new_submodels:
            # Split by semicolon and remove empty strings
            sub_list = []
            for s_model in new_submodels.split(";"):
                if not s_model.strip():
                    continue
                try:
                    text, elements = clean_model_elements(s_model)
                    validate_model(text, raise_err=True)

                except Exception as e:
                    QMessageBox.warning(
                        self,
                        "Invalid Sub-model",
                        f"Sub-model '{s_model.strip()}' is invalid: {str(e)}",
                    )
                    return
                for elem in elements:
                    if elem not in self.elements:
                        QMessageBox.warning(
                            self,
                            "Invalid Sub-model",
                            f"Element '{elem}' in sub-model '{s_model.strip()}' not found in main model.",
                        )
                        return
                sub_list.append(text)

            if sub_list:
                self.sub_models = sub_list

    def _replace_element(self):
        """
        Replace selected element with another element type.
        Requires an element to be selected in the text editor.
        Emits replaceElement signal with old and new element types.
        """
        # Get the currently selected text
        text = self.selectedText()
        if not text:
            QMessageBox.warning(self, "No Selection", "Please select an element to replace.")
            return

        # Try to extract an element name from the selection
        text_match = re.search(r"([A-Za-z]+)(_?\d+)", text)
        if not text_match:
            QMessageBox.warning(
                self, "Invalid Selection", "Selection does not contain a valid element."
            )
            return

        if text_match.group(0) not in self.elements:
            QMessageBox.warning(
                self, "Invalid Element", f"'{text_match.group(0)}' is not a valid element type."
            )
            return

        elements = sorted(list(ELEMENTS), key=len)
        if text_match.group(1) in elements:
            elements.remove(text_match.group(1))
            elements.append(text_match.group(1))
        if text_match.group(1) in ELEMENT_PAIR_MAP:
            pair_elem = ELEMENT_PAIR_MAP[text_match.group(1)]
            if pair_elem in elements:
                elements.remove(pair_elem)
                elements.insert(0, pair_elem)

        # Create a dialog to select the new element type
        new_element, ok = QInputDialog.getItem(
            self,
            "Replace Element",
            f"Replace {text_match.group(1)} with:",
            elements,
            0,
            False,
        )

        if ok and new_element and new_element != text_match.group(1):
            # Emit the signal with the old and new element types
            new_element = new_element + text_match.group(2)
            test_model = get_valid_model(self.text().replace(text_match.group(0), new_element))[0]
            if test_model:
                self.replaceElement.emit(text_match.group(0), new_element)
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Replacement",
                    f"Replacing '{text_match.group(0)}' with '{new_element}' results in an invalid model.",
                )

    def _convert_section(self):
        """
        Convert selected section to a different circuit model.
        Handles CPE/ICPE elements and uses CKT_FUNCS for conversion mapping.
        """
        selected_text = self.selectedText()
        if not selected_text:
            QMessageBox.warning(self, "No Selection", "Please select a section to convert.")
            return

        kwargs = {}

        current_model = self.text()
        residual = current_model.replace(selected_text, "")
        text, elements = get_valid_sub_model(current_model, clean_model_basic(selected_text))
        if not text:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                f"Selection is not a valid circuit model: {selected_text}",
            )
            return

        recommended = ""
        key_map = {}
        for key, info in CKT_FUNCS.items():
            key_map[f"{key}: {info[1]} → {info[2]}"] = key
            if not recommended and model_compare(text, info[1]):
                recommended = f"{key}: {info[1]} → {info[2]}"
                # break
        # # Build descriptive conversion options from CKT_FUNCS
        # key_map = {f"{key}: {info[1]} → {info[2]}": key for key, info in CKT_FUNCS.items()}
        key_list = list(key_map.keys())
        if recommended:
            key_list.remove(recommended)
            key_list.insert(0, recommended)
        # Show dialog for user to select conversion type
        conv_desc, ok = QInputDialog.getItem(
            self,
            "Select Conversion",
            "Convert section to:",
            key_list,
            0,
            False,
        )
        if not ok or not conv_desc:
            return

        conv_key = key_map[conv_desc]
        func, old_model, new_model, get_val = CKT_FUNCS[conv_key]
        # Pass min number from text for renumbering to func
        old_model = shift_elem_num(old_model, text, residual)
        new_model = shift_elem_num(new_model, text, residual)

        old_elems = get_valid_model(old_model)[1]
        for elem in elements:
            el = elem.replace("ICPE", "R") if "ICPE" in elem else elem.replace("CPE", "C")
            if el not in old_elems:
                QMessageBox.warning(
                    self,
                    "Incompatible Elements",
                    f"Element '{elem}' in selection is not compatible with conversion to '{conv_key}'.",
                )
                return
        # Check if the selected text has enough elements for the conversion
        if len(old_elems) != len(elements):
            QMessageBox.warning(
                self,
                "Parameter Mismatch",
                f"Conversion to '{conv_key}' requires {len(old_elems)} elements, "
                f"but the selection has {len(elements)} elements.",
            )
            return

        # Validate that the replacement would create a valid model
        test_model = get_valid_model(current_model.replace(text, new_model))[0]
        if not test_model:
            QMessageBox.warning(
                self,
                "Invalid Conversion",
                f"Converting to '{conv_key}' would create an invalid model.",
            )
            return

        if get_val:
            vals = get_val.split(",")
            for v in vals:
                v = v.strip()
                val, ok = QInputDialog.getText(
                    self,
                    "Input Required",
                    f"Enter value for {v} (in Ohm, F, S, etc.):",
                    text=str(1.0),
                )
                if not ok:
                    return
                try:
                    kwargs[v] = float(val)
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid Input",
                        f"Value '{val}' for parameter '{v}' is not a valid number.",
                    )
                    return

        self.convertSection.emit(text, old_model, new_model, func, kwargs)

    # def _sort_model(self):
    #     """
    #     Sort the model components by their impedance characteristics.
    #     Shows dialogs to select sorting options.
    #     """
    #     current_model = self.text()
    #     if not self.hasAcceptableInput():
    #         QMessageBox.warning(self, "Invalid Model", "Current model is invalid and cannot be sorted.")
    #         return

    #     # Get series groups
    #     series_groups = parse_model_groups(current_model, use_numbers=False)
    #     if len(series_groups) <= 1:
    #         QMessageBox.information(self, "Sorting", "Model has only one series component, no sorting needed.")
    #         return

    #     # First dialog for sort direction
    #     direction, ok1 = QInputDialog.getItem(
    #         self,
    #         "Sort Direction",
    #         "Choose sort order:",
    #         ["Ascending", "Descending"],
    #         0,
    #         False
    #     )

    #     if not ok1:
    #         return

    #     # Second dialog for sort criteria
    #     sort_by, ok2 = QInputDialog.getItem(
    #         self,
    #         "Sort Criteria",
    #         "Sort by:",
    #         ["R (real)", "C (imag)", "RC (arc peak)"],
    #         0,
    #         False
    #     )

    #     if not ok2:
    #         return

    #     # Emit signal for the main GUI to handle the sorting
    #     self.sortModel.emit(series_groups, direction, sort_by)
