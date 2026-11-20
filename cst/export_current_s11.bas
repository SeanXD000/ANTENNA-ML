' ============================================================
' CST macro：不重跑模擬，只把「目前專案最後一次模擬」的 S1,1 匯出成 antenna_ml.py 讀得懂的格式
' ------------------------------------------------------------
' 用法：模擬跑完後，Home → Macros → New VBA Macro，貼上本檔全部內容後按 Run。
' 輸出：OUT_FILE（附加寫入，每執行一次多一組 #Parameters 區塊；重跑前先刪掉舊檔）
' 用途：50 Ω 正規化（Normalize to fixed impedance）的結果另存在 data_50ohm\，
'       不放 data\，避免和以饋入線阻抗為參考的訓練資料混在一起。
' 若出現「找不到 1D Results\S-Parameters\S1,1」，請在導覽樹看 S 參數的實際名稱，改 S11_TREE。
' ============================================================
Option Explicit

Const OUT_FILE As String = "C:\attena\antenna-ml\data_50ohm\verify_best3_50ohm_newport.txt"
Const S11_TREE As String = "1D Results\S-Parameters\S1,1"

Sub Main()
    AppendS11 OUT_FILE
    MsgBox "已匯出：" & OUT_FILE
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
