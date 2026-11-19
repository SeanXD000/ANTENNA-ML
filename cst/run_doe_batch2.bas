' ============================================================
' CST macro：依 doe_batch2.csv 的 25 組參數，自動逐組模擬並匯出 S11
' ------------------------------------------------------------
' 用法：CST 上方 Home → Macros → Edit/Open VBA Macros...（或 Macros → New VBA Macro），
'       貼上本檔全部內容後按 Run。
' 輸出：OUT_FILE（預設 C:\attena\antenna-ml\data\doe_batch2.txt），
'       格式與 CST 參數掃描匯出相同（每組以 #Parameters = {...} 開頭），antenna_ml.py 可直接讀。
' 每跑完一組就寫入檔案，中途中斷也不會遺失已完成的結果；
' 要從第 N 組接著跑，把 START_RUN 改成 N 再執行即可。
' ============================================================
Option Explicit

Const OUT_FILE As String = "C:\attena\antenna-ml\data\doe_batch2.txt"
Const START_RUN As Integer = 1
Const N_RUNS As Integer = 25
Const S11_TREE As String = "1D Results\S-Parameters\S1,1"

Sub Main()
    ' 25 組參數：W_slot1, W_slot2, W_slot3, Ymax（與 doe_batch2.csv 相同）
    Dim P(1 To 25, 1 To 4) As Double
    SetRow P, 1, 36.1, 14.2, 21.7, 13.3
    SetRow P, 2, 36.4, 11.9, 22.3, 15.5
    SetRow P, 3, 36.6, 14, 23.1, 12.2
    SetRow P, 4, 36.7, 11.1, 24.6, 14.2
    SetRow P, 5, 37, 15.8, 20.3, 12.8
    SetRow P, 6, 37.2, 11.3, 20.5, 14
    SetRow P, 7, 37.4, 13.3, 25.6, 11.7
    SetRow P, 8, 37.9, 10.3, 22.8, 13.4
    SetRow P, 9, 37.9, 15.5, 25.4, 16.5
    SetRow P, 10, 38.4, 12.4, 23.2, 16.1
    SetRow P, 11, 38.4, 13.4, 24.6, 14.4
    SetRow P, 12, 38.8, 12.4, 21.5, 15
    SetRow P, 13, 39.1, 15.7, 21.3, 11.8
    SetRow P, 14, 39.2, 13, 25.2, 16.8
    SetRow P, 15, 39.5, 15, 23.7, 12.1
    SetRow P, 16, 39.8, 10.9, 20.8, 15.6
    SetRow P, 17, 39.9, 12, 23.9, 15.2
    SetRow P, 18, 40.3, 13.7, 22.4, 11.1
    SetRow P, 19, 40.5, 14.7, 20.2, 13
    SetRow P, 20, 40.6, 12.7, 22.1, 13.8
    SetRow P, 21, 41, 15.3, 25.8, 16.7
    SetRow P, 22, 41.3, 10.6, 25, 14.6
    SetRow P, 23, 41.4, 14.4, 23.6, 15.9
    SetRow P, 24, 41.6, 11.6, 21.1, 11.3
    SetRow P, 25, 42, 10.2, 24.1, 12.5

    ' 固定參數：和原本的掃描相同，先全部設好，避免沿用上一次留下的值
    StoreParameter "H_cu", 0.035
    StoreParameter "H_sub", 1.6
    StoreParameter "L_slot", 1.5
    StoreParameter "L_sub", 50
    StoreParameter "W_feed", 1.5
    StoreParameter "W_sub", 52

    Dim i As Integer
    For i = START_RUN To N_RUNS
        ReportInformationToWindow "DOE run " & i & "/" & N_RUNS & " 開始"
        StoreParameter "W_slot1", P(i, 1)
        StoreParameter "W_slot2", P(i, 2)
        StoreParameter "W_slot3", P(i, 3)
        StoreParameter "Ymax", P(i, 4)
        RebuildOnParametricChange False, True
        Solver.Start
        AppendS11 OUT_FILE
        ReportInformationToWindow "DOE run " & i & "/" & N_RUNS & " 完成，已寫入 " & OUT_FILE
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
