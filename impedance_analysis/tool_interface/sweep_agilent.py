# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 15:09:18 2023

@author: j2cle
"""

from vb2py.vbfunctions import *
from vb2py.vbdebug import *

"""===================================
 4294A Step1
===================================
===================================
 4294A Step2 (Get Data)
===================================
===================================
 4294A Step2 (Get Image)
===================================
"""

rm = VisaComLib.ResourceManager()
instrument = VisaComLib.FormattedIO488()
COLOR_RED = 3
COLOR_BLUE = 5
xData = vbObjectInitialize(objtype=float)


def Step1():
    strAddr = str

    strRes = str
    # VB2PY (UntranslatedCode) On Error GoTo ErrorHandler
    Range["C22"].Value = vbNullString
    Range["D22"].Value = vbNullString
    strAddr = setAddrString()
    # ===================================
    # Open Instrument
    # ===================================
    if setVisaCom(strAddr):
        strRes = cmdGet("*IDN?")
        Range("C22").Select()
        Selection.Font.ColorIndex = COLOR_BLUE
        Range["C22"].Value = "OK"
        Range["D22"].Value = strRes
        if Sheets(1).Cells(15, 3).Value == "GPIB":
            execLocal((strAddr))
    else:
        Range("C22").Select()
        Selection.Font.ColorIndex = COLOR_RED
        Range["C22"].Value = "ERROR"
        Range["D22"].Value = "Configuration Error."
        return
    return
    # ===================================
    # ErrorHandler
    # ===================================
    # Display the error message
    MsgBox("*** Error : " + Error, vbExclamation)
    sys.exit(0)


def Step2_GetData():
    strAddr = str

    strSheetName = str

    cnt = int
    # VB2PY (UntranslatedCode) On Error GoTo ErrorHandler
    Sheets[1].Cells[34, 3] = vbNullString
    strSheetName = "MasterSheet"
    if not ExistSheet(strSheetName, cnt):
        Sheets[1].Cells[34, 3] = "MasterSheet not found."
        return
    strSheetName = Sheets(1).Cells(29, 3)
    if not ExistSheet(strSheetName, cnt):
        if not CreateExcelSheet(strSheetName):
            Sheets[1].Cells[34, 3] = "Invalid Excel Sheet Name."
            return
    else:
        Sheets[1].Cells[34, 3] = "Duplicated Sheet Name."
        return
    strAddr = setAddrString()
    # ===================================
    # Open Instrument
    # ===================================
    if setVisaCom(strAddr):
        Sheets[1].Cells[32, 4] = "Please wait..."
        GetSettings()()
        GetMeasData()()
        MakeChart()((strSheetName))
        if Sheets(1).Cells(15, 3).Value == "GPIB":
            execLocal((strAddr))
        Sheets[1].Cells[32, 4] = vbNullString
    return
    # ===================================
    # ErrorHandler
    # ===================================
    # Display the error message
    MsgBox("*** Error : " + Error, vbExclamation)
    if Cells(15, 3).Value == "GPIB":
        execLocal((strAddr))
    Sheets[1].Cells[32, 4] = vbNullString
    sys.exit(0)


def Step2_GetImage():
    strAddr = str

    strMeas = str

    tmpImg2 = str

    intFileNo = int

    bArray = vbObjectInitialize(objtype=Byte)

    strSheetName = str

    lngFileSize = Long()

    i = Variant()

    iBufCnt = int

    cnt = int

    strRange = str

    vErr = Variant()

    MAX_BUFF_SIZE = 16384

    SAVE_FILENAME = "agt4294a.tif"
    # VB2PY (UntranslatedCode) On Error GoTo ErrorHandler
    Sheets[1].Cells[34, 3] = vbNullString
    strSheetName = "MasterSheet"
    if not ExistSheet(strSheetName, cnt):
        Sheets[1].Cells[34, 3] = "MasterSheet not found."
        return
    strSheetName = Sheets(1).Cells(29, 3)
    if not ExistSheet(strSheetName, cnt):
        if not CreateExcelSheet(strSheetName):
            Sheets[1].Cells[34, 3] = "Invalid Excel Sheet Name."
            return
    strAddr = setAddrString()
    # ===================================
    # Open Instrument
    # ===================================
    if setVisaCom(strAddr):
        Sheets[1].Cells[32, 7] = "Please wait..."
        strMeas = cmdGet("MEAS?")
        cmdSet()(("STOD MEMO"))
        cmdSet()(("SAVDTIF " + '"' + "\\" + SAVE_FILENAME + '"'))
        vErr = Split(cmdGet("OUTPERRO?"), ",")
        if not Val(vErr(0)) == 0:
            cmdSet()(("PURG " + '"' + "\\" + SAVE_FILENAME + '"'))
            cmdSet()(("SAVDTIF " + '"' + "\\" + SAVE_FILENAME + '"'))
        cmdSet()(("ROPEN " + '"' + "\\" + SAVE_FILENAME + '"'))
        lngFileSize = cmdGet("FSIZE? " + '"' + "\\" + SAVE_FILENAME + '"')
        iBufCnt = Int(lngFileSize / MAX_BUFF_SIZE)
        if lngFileSize % MAX_BUFF_SIZE > 0:
            iBufCnt = iBufCnt + 1
        intFileNo = FreeFile()
        tmpImg2 = SAVE_FILENAME
        VBFiles.openFile(
            intFileNo, tmpImg2, "b"
        )  # VB2PY (UnknownFileMode) 'Access', 'Write'
        for i in vbForRange(1, iBufCnt):
            instrument.WriteString("READ?")
            bArray = instrument.ReadIEEEBlock(BinaryType_UI1)
            Put(intFileNo, VBGetMissingArgument(Put, 1), bArray)
        VBFiles.closeFile(intFileNo)
        Sheets(cnt).Select()
        Sheets(cnt).Pictures.Insert(tmpImg2).Select()
        if strMeas == "COMP":
            strRange = "N2"
        else:
            strRange = "L8"
        Selection.Left = Range(strRange).Left
        Selection.Top = Range(strRange).Top
        Sheets(1).Select()
        cmdSet()(("PURG " + '"' + "\\" + SAVE_FILENAME + '"'))
        if not Dir(tmpImg2) == vbNullString:
            Kill(tmpImg2)
        if Cells(15, 3).Value == "GPIB":
            execLocal((strAddr))
        Sheets[1].Cells[32, 7] = vbNullString
    return
    # ===================================
    # ErrorHandler
    # ===================================
    # Display the error message
    MsgBox("*** Error : " + Error, vbExclamation)
    if not tmpImg2 == vbNullString and not Dir(tmpImg2) == vbNullString:
        VBFiles.closeFile(intFileNo)
        Kill(tmpImg2)
    if Cells(15, 3).Value == "GPIB":
        execLocal((strAddr))
    Sheets[1].Cells[32, 7] = vbNullString
    sys.exit(0)


def setVisaCom(strAddr):
    # VB2PY (UntranslatedCode) On Error GoTo setVisaCom_Err
    fn_return_value = False
    rm = VisaComLib.ResourceManager()
    instrument = VisaComLib.FormattedIO488()
    instrument.IO = rm.Open(strAddr)
    instrument.IO.timeout = 30000
    instrument.IO.TerminationCharacterEnabled = True
    fn_return_value = True
    return fn_return_value
    return fn_return_value


def setAddrString():
    fn_return_value = vbNullString
    select_variable_0 = Cells(15, 3).Value
    if select_variable_0 == "GPIB":
        fn_return_value = "GPIB0::" + Cells(17, 3).Value + "::INSTR"
    elif select_variable_0 == "LAN":
        fn_return_value = "TCPIP0::" + Cells(17, 3).Value + "::5025::SOCKET"
    else:
        fn_return_value = vbNullString
    return fn_return_value


def CreateExcelSheet(strName):
    vStr = Variant()
    # VB2PY (UntranslatedCode) On Error GoTo CreateExcelSheet_Error
    fn_return_value = False
    Sheets("MasterSheet").Copy(after=Sheets(Sheets.count))
    Sheets[ActiveSheet.Name].Name = strName
    Sheets(1).Select()
    fn_return_value = True
    return fn_return_value
    Application.DisplayAlerts = False
    ActiveSheet.Delete()
    Sheets(1).Select()
    Application.DisplayAlerts = True
    return fn_return_value


def ExistSheet(strName, iNo):
    i = Variant()

    cnt = int
    fn_return_value = False
    for i in vbForRange(1, Sheets.count):
        if Sheets(i).Name == strName:
            fn_return_value = True
            break
    iNo = i
    return fn_return_value


def GetSettings():
    i = int

    cnt = int

    vRes = Variant()

    vParam = Variant()

    strRes = str

    strRes2 = str
    cnt = Sheets.count
    with_variable0 = Sheets(cnt)
    # --- Date & Time
    with_variable0.Range["B1"].Value = Now
    # --- FW Revision
    strRes = cmdGet("*IDN?")
    vRes = Split(strRes, ",")
    with_variable0.Range["B2"].Value = "FW: " + Left(vRes(3), Len(vRes(3)) - 1)
    # --- Measurement Function ---
    cmdSet()(("FORM4"))
    strRes = cmdGet("MEAS?")
    with_variable0.Range["D3"].Value = ConvParam(strRes)
    vParam = Split(with_variable0.Range("D3").Value, "-")
    with_variable0.Range["L5"].Value = Trim(vParam(0))
    with_variable0.Range["L6"].Value = Trim(vParam(1))
    with_variable0.Range["D10"].Value = "Data Trace A Real"
    with_variable0.Range["E10"].Value = "Data Trace A Imag"
    with_variable0.Range["F10"].Value = "Data Trace B Real"
    with_variable0.Range["G10"].Value = "Data Trace B Imag"
    # ------ Adapter Setting -------------------
    strRes = cmdGet("E4TP?")
    select_variable_1 = strRes
    if select_variable_1 == "OFF":
        with_variable0.Range["D4"].Value = "NONE"
    elif select_variable_1 == "M1":
        with_variable0.Range["D4"].Value = "4TP 1M"
    elif select_variable_1 == "M2":
        with_variable0.Range["D4"].Value = "4TP 2M"
    elif select_variable_1 == "APC7":
        with_variable0.Range["D4"].Value = "7mm, 42942A"
    elif select_variable_1 == "PROBE":
        with_variable0.Range["D4"].Value = "Probe, 42941A"
    else:
        with_variable0.Range["D4"].Value = "Error"
    # ------ Sweep Type -------------------------
    strRes = cmdGet("SWPT?")
    with_variable0.Range["D5"] = strRes
    # ------ Number of Points -------------------------
    strRes = cmdGet("POIN?")
    with_variable0.Range["D6"] = Val(strRes)
    # ------ Point Delay -------------------------
    strRes = cmdGet("PDELT?")
    with_variable0.Range["D7"] = Val(strRes) + " [Sec]"
    # ------ Sweep Delay -------------------------
    strRes = cmdGet("SDELT?")
    with_variable0.Range["D8"] = Val(strRes) + " [Sec]"
    # ------ Signal Level -----------------------
    strRes = cmdGet("POWE?")
    strRes2 = cmdGet("POWMOD?")
    if strRes2 == "VOLT":
        with_variable0.Range["H3"] = Val(strRes) + " [V]"
    else:
        with_variable0.Range["H3"] = Val(strRes) + " [A]"
    # ------ DC Bias -------------------------
    strRes = cmdGet("DCO?")
    if strRes == "1" or strRes == "ON":
        strRes = cmdGet("DCMOD?")
        with_variable0.Range["H4"] = strRes
    else:
        with_variable0.Range["H4"] = "Off"
    # ------ Bandwidth -------------------------
    strRes = cmdGet("BWFACT?")
    with_variable0.Range["H5"] = Val(strRes)
    # ------ Sweep Averaging -------------------------
    strRes = cmdGet("AVER?")
    strRes2 = cmdGet("AVERFACT?")
    if strRes == "0":
        with_variable0.Range["H6"] = "Off"
    else:
        with_variable0.Range["H6"] = strRes2
    # ------ Point Averaging -------------------------
    strRes = cmdGet("PAVER?")
    strRes2 = cmdGet("PAVERFACT?")
    if strRes == "0":
        with_variable0.Range["H7"] = "Off"
    else:
        with_variable0.Range["H7"] = strRes2
    # ------ Sweep Parameter -------------------------
    strRes = cmdGet("SWPP?")
    select_variable_2 = strRes
    if select_variable_2 == "FREQ":
        with_variable0.Range["C10"] = "Frequency"
    elif select_variable_2 == "OLEV":
        with_variable0.Range["C10"] = "OSC Level"
    elif select_variable_2 == "DCB":
        with_variable0.Range["C10"] = "DC Bias"
    else:
        with_variable0.Range["C10"] = "Error"
    return fn_return_value


def GetMeasData():
    i = Variant()

    j = Variant()

    k = int

    varRes = Variant()

    cnt = int

    lNop = Long()

    strRes = str

    strRes2 = str

    strMeas = str

    sData_A = str

    sData_B = str

    vData_A = Variant()

    vData_B = Variant()
    cnt = Sheets.count
    with_variable1 = Sheets(cnt)
    with_variable1.Range["B11:B812"].NumberFormatLocal = "0"
    with_variable1.Range["C11:G812"].NumberFormatLocal = "0.0000E+00"
    # ------ Number of Points -------------------------
    lNop = cmdGet("POIN?")
    # --- Measurement Function ---
    strMeas = cmdGet("MEAS?")
    strRes = cmdGet("OUTPSWPRM?")
    varRes = Split(strRes, ",")
    cmdSet()(("TRAC A"))
    sData_A = cmdGet("OUTPDTRC?")
    cmdSet()(("TRAC B"))
    sData_B = cmdGet("OUTPDTRC?")
    vData_A = Split(sData_A, ",")
    vData_B = Split(sData_B, ",")
    for i in vbForRange(1, lNop):
        with_variable1.Range["B" + 10 + i] = i
        with_variable1.Range["C" + 10 + i] = CSng(varRes(i - 1))
        with_variable1.Range["D" + 10 + i] = CSng(vData_A(2 * i - 2))
        with_variable1.Range["E" + 10 + i] = CSng(vData_A(2 * i - 1))
        with_variable1.Range["F" + 10 + i] = CSng(vData_B(2 * i - 2))
        with_variable1.Range["G" + 10 + i] = CSng(vData_B(2 * i - 1))
    cmdSet()(("TRAC A"))
    return fn_return_value


def MakeChart(strSheetName):
    i = int

    lNop = Long()

    strMeas = str

    cnt = int

    strRange = str

    strGname = str

    r1 = Range()

    r2 = Range()
    lNop = cmdGet("POIN?")
    strMeas = cmdGet("MEAS?")
    cnt = Sheets.count
    Sheets(cnt).Select()
    if strMeas != "COMP":
        ActiveSheet.ChartObjects(1).Activate()
        with_variable2 = ActiveChart
        with_variable2.SeriesCollection[1].XValues = Sheets(cnt).Range(
            "N11:N" + lNop + 10
        )
        with_variable2.SeriesCollection[1].Values = Sheets(cnt).Range(
            "D11:D" + lNop + 10
        )
        with_variable2.SeriesCollection[2].XValues = Sheets(cnt).Range(
            "N11:N" + lNop + 10
        )
        with_variable2.SeriesCollection[2].Values = Sheets(cnt).Range(
            "F11:F" + lNop + 10
        )
        with_variable2.ChartTitle.Text = Sheets(cnt).Range("D3")
        with_variable2.Axes(xlValue, xlSecondary).Select()
        Selection.TickLabelPosition = xlNextToAxis
        Range("A1").Select()
    else:
        for i in vbForRange(1, 2):
            if i == 1:
                strRange = "D11:E"
            else:
                strRange = "F11:G"
            Range(strRange + lNop + 10).Select()
            Charts.Add()
            ActiveChart.ChartType = xlXYScatterLinesNoMarkers
            ActiveChart.SetSourceData(
                Source=Sheets(strSheetName).Range(strRange + lNop + 10),
                PlotBy=xlColumns,
            )
            ActiveChart.Location(Where=xlLocationAsObject, Name=strSheetName)
            with_variable3 = ActiveChart
            with_variable3.HasTitle = True
            with_variable3.HasLegend = False
            if i == 1:
                with_variable3.ChartTitle.Characters.Text = "Complex Z"
            else:
                with_variable3.ChartTitle.Characters.Text = "Complex Y"
            with_variable3.Axes[xlCategory, xlPrimary].HasTitle = True
            with_variable3.Axes[
                xlCategory, xlPrimary
            ].AxisTitle.Characters.Text = "Real"
            with_variable3.Axes[xlValue, xlPrimary].HasTitle = True
            with_variable3.Axes[xlValue, xlPrimary].AxisTitle.Characters.Text = "Imag"
            # ----------------------
            # X-axis
            # ----------------------
            with_variable4 = ActiveChart.Axes(xlCategory)
            with_variable4.HasMajorGridlines = True
            with_variable4.HasMinorGridlines = False
            with_variable4.MinimumScaleIsAuto = True
            with_variable4.MaximumScaleIsAuto = True
            with_variable4.MinorUnitIsAuto = True
            with_variable4.MajorUnitIsAuto = True
            with_variable4.Crosses = xlCustom
            with_variable4.CrossesAt = with_variable4.MinimumScale
            with_variable4.ReversePlotOrder = False
            with_variable4.ScaleType = xlLinear
            with_variable4.DisplayUnit = xlNone
            with_variable5 = Selection.Border
            with_variable5.ColorIndex = 16
            with_variable5.Weight = xlHairline
            with_variable5.LineStyle = xlContinuous
            # ----------------------
            # Y-axis
            # ----------------------
            with_variable6 = ActiveChart.Axes(xlValue)
            with_variable6.HasMajorGridlines = True
            with_variable6.HasMinorGridlines = False
            with_variable6.MinimumScaleIsAuto = True
            with_variable6.MaximumScaleIsAuto = True
            with_variable6.MinorUnitIsAuto = True
            with_variable6.MajorUnitIsAuto = True
            with_variable6.Crosses = xlCustom
            with_variable6.CrossesAt = with_variable6.MinimumScale
            with_variable6.ReversePlotOrder = False
            with_variable6.ScaleType = xlLinear
            with_variable6.DisplayUnit = xlNone
            with_variable7 = Selection.Border
            with_variable7.ColorIndex = 16
            with_variable7.Weight = xlHairline
            with_variable7.LineStyle = xlContinuous
        # ------------------------
        # Delete default chart.
        # ------------------------
        ActiveSheet.ChartObjects(1).Activate()
        ActiveChart.Parent.Delete()
        # ---------------------------
        # Move chart around.
        # ---------------------------
        with_variable8 = ActiveSheet
        # --- Chart 1 ---
        with_variable9 = with_variable8.ChartObjects(1)
        with_variable9.Activate()
        with_variable9.Top = ActiveSheet.Range("B31").Top
        with_variable9.Left = ActiveSheet.Range("B31").Left
        with_variable9.Height = 280
        with_variable9.Width = 450
        ActiveChart.Axes(xlCategory).Select()
        Selection.TickLabels.NumberFormatLocal = "0.00"
        ActiveChart.Axes(xlValue).Select()
        Selection.TickLabels.NumberFormatLocal = "0.00E+00"
        # --- Chart 2 ---
        with_variable10 = with_variable8.ChartObjects(2)
        with_variable10.Activate()
        with_variable10.Top = ActiveSheet.Range("J31").Top
        with_variable10.Left = ActiveSheet.Range("J31").Left
        with_variable10.Height = 280
        with_variable10.Width = 450
        ActiveChart.Axes(xlCategory).Select()
        Selection.TickLabels.NumberFormatLocal = "0.00"
        ActiveChart.Axes(xlValue).Select()
        Selection.TickLabels.NumberFormatLocal = "0.00E+00"
        Range("A1").Select()
    Sheets(1).Select()
    return fn_return_value


def ConvParam(strType):
    fn_return_value = vbNullString
    select_variable_3 = strType
    if select_variable_3 == "IMPH":
        fn_return_value = Chr(124) + "Z" + Chr(124) + " - theta"
    elif select_variable_3 == "IRIM":
        fn_return_value = "R - X"
    elif select_variable_3 == "LSR":
        fn_return_value = "Ls - Rs"
    elif select_variable_3 == "LSQ":
        fn_return_value = "Ls - Q"
    elif select_variable_3 == "CSR":
        fn_return_value = "Cs - R"
    elif select_variable_3 == "CSQ":
        fn_return_value = "Cs - Q"
    elif select_variable_3 == "CSD":
        fn_return_value = "Cs - D"
    elif select_variable_3 == "AMPH":
        fn_return_value = Chr(124) + "Y" + Chr(124) + " - theta"
    elif select_variable_3 == "ARIM":
        fn_return_value = "G - B"
    elif select_variable_3 == "LPG":
        fn_return_value = "Lp - G"
    elif select_variable_3 == "LPQ":
        fn_return_value = "Lp - Q"
    elif select_variable_3 == "CPG":
        fn_return_value = "Cp - G"
    elif select_variable_3 == "CPQ":
        fn_return_value = "Cp - Q"
    elif select_variable_3 == "CPD":
        fn_return_value = "Cp - D"
    elif select_variable_3 == "COMP":
        fn_return_value = "Comp Z - Comp Y"
    elif select_variable_3 == "IMLS":
        fn_return_value = Chr(124) + "Z" + Chr(124) + " - Ls"
    elif select_variable_3 == "IMCS":
        fn_return_value = Chr(124) + "Z" + Chr(124) + " - Cs"
    elif select_variable_3 == "IMLP":
        fn_return_value = Chr(124) + "Z" + Chr(124) + " - Lp"
    elif select_variable_3 == "IMCP":
        fn_return_value = Chr(124) + "Z" + Chr(124) + " - Cp"
    elif select_variable_3 == "IMRS":
        fn_return_value = Chr(124) + "Z" + Chr(124) + " - Rs"
    elif select_variable_3 == "IMQ":
        fn_return_value = Chr(124) + "Z" + Chr(124) + " - Q"
    elif select_variable_3 == "IMD":
        fn_return_value = Chr(124) + "Z" + Chr(124) + " - D"
    elif select_variable_3 == "LPR":
        fn_return_value = "Lp - Rp"
    elif select_variable_3 == "CPR":
        fn_return_value = "Cp - Rp"
    else:
        fn_return_value = "ERROR-ERROR"
    return fn_return_value


def cmdSet(sCmd):
    lRet = Long()
    instrument.WriteString(sCmd)
    return fn_return_value


def cmdGet(sCmd):
    str = str
    fn_return_value = vbNullString
    str = vbNullString
    instrument.WriteString(sCmd)
    str = instrument.ReadString()
    fn_return_value = Left(str, Len(str) - 1)
    return fn_return_value


def execLocal(strAddr):
    gpib = IGpib()
    # VB2PY (UntranslatedCode) On Error GoTo execLocal_Err
    gpib = rm.Open(strAddr)
    gpib.ControlREN((GPIB_REN_GTL))
    gpib.Close()


# VB2PY (UntranslatedCode) Option Explicit
