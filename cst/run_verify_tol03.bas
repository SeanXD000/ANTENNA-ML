' ============================================================
' CST macro：製程容差驗證：verify_best3 的 ±0.3 mm 最差角落（模型估帶內最差 -10.98 dB）
' ------------------------------------------------------------
' 用法：CST 上方 Home → Macros → Edit/Open VBA Macros...（或 Macros → New VBA Macro），
'       貼上本檔全部內容後按 Run。
' 輸出：OUT_FILE（預設 C:\attena\antenna-ml\data\verify_tol03.txt），
'       格式與 CST 參數掃描匯出相同（每組以 #Parameters = {...} 開頭），antenna_ml.py 可直接讀。
' 每跑完一組就寫入檔案，中途中斷也不會遺失已完成的結果；
' 要從第 N 組接著跑，把 START_RUN 改成 N 再執行即可。
' ============================================================
Option Explicit

Const OUT_FILE As String = "C:\attena\antenna-ml\data\verify_tol03.txt"
Const START_RUN As Integer = 1
Const N_RUNS As Integer = 1
Const S11_TREE As String = "1D Results\S-Parameters\S1,1"

Sub Main()
    ' 名目 verify_best3 = 36.97, 12.502, 22.944, 13.932；最差方向 W_slot1 +0.3、W_slot2 -0.3、W_slot3 +0.3、Ymax +0.3
    Dim P(1 To 1, 1 To 4) As Double
    SetRow P, 1, 37.27, 12.202, 23.244, 14.232

    ' 固定參數：和原本的掃描相同，先全部設好，避免沿用上一次留下的值
    StoreParameter "H_cu", 0.035
    StoreParameter "H_sub", 1.6
    StoreParameter "L_slot", 1.5
    StoreParameter "L_sub", 50
    StoreParameter "W_feed", 1.5
    StoreParameter "W_sub", 52

    Dim i As Integer
    For i = START_RUN To N_RUNS
        ReportInformationToWindow "Tolerance run " & i & "/" & N_RUNS & " 開始"
        StoreParameter "W_slot1", P(i, 1)
        StoreParameter "W_slot2", P(i, 2)
        StoreParameter "W_slot3", P(i, 3)
        StoreParameter "Ymax", P(i, 4)
        RebuildOnParametricChange False, True
        Solver.Start
        AppendS11 OUT_FILE
        ReportInformationToWindow "Tolerance run " & i & "/" & N_RUNS & " 完成，已寫入 " & OUT_FILE
    Next i
    MsgBox "全部完成：" & OUT_FILE
End Sub

Sub SetRow(P() As Double, r As Integer, a As Double, b As Double, c As Double, d As Double)
    P(r, 1) = a: P(r, 2) = b: P(r, 3) = c: P(r, 4) = d
End Sub

' 數字轉字串一律用小數點（Str 不受 Windows 地區設定影響）
Function Num(x As Double) As String
    Num = Trim(Str(x))
End Function

Sub AppendS11(path As String)
    Dim names As Variant, k As Integer, header As String
    names = Array("H_cu", "H_sub", "L_slot", "L_sub", "W_feed", "W_slot1", "W_slot2", "W_slot3", "W_sub", "Ymax")
    header = "#Parameters = {"
    For k = 0 To UBound(names)
        If k > 0 Then header = header & "; "
        header = header & names(k) & "=" & Num(RestoreDoubleParameter(names(k)))
    Next k
    header = header & "}"

    ' 讀目前這次模擬的 S1,1（複數），換算成 dB
    Dim res As Object
    Set res = Resulttree.GetResultFromTreeItem(S11_TREE, "3D:RunID:0")
    If res Is Nothing Then
        MsgBox "找不到 " & S11_TREE & "，請確認導覽樹中的 S 參數名稱"
        End
    End If

    Dim fn As Integer, n As Long, j As Long, re As Double, im As Double, mag As Double
    fn = FreeFile
    Open path For Append As #fn
    Print #fn, header
    Print #fn, "#""Frequency / GHz""" & vbTab & """S1,1 [Magnitude / dB]"""
    Print #fn, "#-----------------------------------------"
    n = res.GetN
    For j = 0 To n - 1
        re = res.GetYRe(j)
        im = res.GetYIm(j)
        mag = Sqr(re * re + im * im)
        If mag < 1E-12 Then mag = 1E-12
        Print #fn, Num(res.GetX(j)) & vbTab & Num(20 * Log(mag) / Log(10))
    Next j
    Close #fn
End Sub
