Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.SetOutputToWaveFile("english_test.wav")
$speaker.Speak("Hello and welcome to the speech recognition service test.")
$speaker.Dispose()
Write-Host "Created english_test.wav successfully"
