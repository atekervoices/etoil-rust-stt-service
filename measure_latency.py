import json
import ssl
import sys
import time
import wave
import numpy as np
import requests
import websocket

SERVER_URL = "https://stt-staging.atekervoices.com"
AUDIO_FILE = "test_audio/eng_1.wav"

def load_resampled_16k(wav_path):
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if sr != 16000:
        ratio = sr / 16000.0
        new_len = int(round(len(samples) / ratio))
        resampled = np.interp(np.arange(new_len) * ratio, np.arange(len(samples)), samples)
    else:
        resampled = samples

    pcm16 = (np.clip(resampled, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    duration = len(pcm16) / 32000.0
    return pcm16, duration

def test_health():
    print("=" * 70)
    print(" 1. HEALTH CHECK & ENDPOINT REACHABILITY")
    print("=" * 70)
    t0 = time.perf_counter()
    resp = requests.get(f"{SERVER_URL}/health", timeout=10)
    rtt_ms = (time.perf_counter() - t0) * 1000
    print(f"Status Code:        {resp.status_code}")
    print(f"Network RTT:        {rtt_ms:.1f} ms")
    data = resp.json()
    print(f"Model:              {data.get('model')}")
    print(f"Chunk Duration:     {data.get('chunk_duration_ms')} ms ({data.get('chunk_size_samples')} samples)\n")

def test_sync_transcribe(wav_path):
    print("=" * 70)
    print(" 2. SYNCHRONOUS HTTP TRANSCRIPTION (/v1/transcribe)")
    print("=" * 70)
    with open(wav_path, "rb") as f:
        wav_bytes = f.read()

    t0 = time.perf_counter()
    resp = requests.post(
        f"{SERVER_URL}/v1/transcribe",
        data=wav_bytes,
        headers={"Content-Type": "audio/wav"},
        params={"language": "en"},
        timeout=30,
    )
    total_time_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code == 200:
        data = resp.json()
        proc_time_ms = data.get("processing_time", 0.0) * 1000
        dur_s = data.get("audio_duration", 0.0)
        rtf = data.get("real_time_factor", 0.0)
        text = data.get("text", "")
        print(f"Audio Duration:     {dur_s:.2f} s")
        print(f"Total Turnaround:   {total_time_ms:.1f} ms")
        print(f"Server Proc Time:   {proc_time_ms:.1f} ms")
        print(f"Server RTF:         {rtf:.4f}")
        print(f"Transcript:         \"{text}\"\n")
    else:
        print(f"Error {resp.status_code}: {resp.text}\n")

def test_realtime_websocket(pcm16, audio_duration):
    print("=" * 70)
    print(" 3. REALTIME WEBSOCKET STREAMING LATENCY (/v1/realtime)")
    print("=" * 70)
    ws_url = SERVER_URL.replace("https://", "wss://").replace("http://", "ws://") + "/v1/realtime?language=en"

    t_connect_start = time.perf_counter()
    ws = websocket.create_connection(ws_url, sslopt={"cert_reqs": ssl.CERT_NONE})
    t_connected = time.perf_counter()

    ready_raw = ws.recv()
    t_ready = time.perf_counter()
    ttfb_ms = (t_ready - t_connect_start) * 1000
    print(f"WebSocket Connect:  {(t_connected - t_connect_start)*1000:.1f} ms")
    print(f"Handshake TTFB:     {ttfb_ms:.1f} ms (received ready event)")

    # 560ms chunks (8960 samples @ 16-bit = 17920 bytes)
    chunk_bytes = 8960 * 2
    chunks = [pcm16[i:i + chunk_bytes] for i in range(0, len(pcm16), chunk_bytes)]
    print(f"Audio Chunks:       {len(chunks)} chunks of 560ms ({audio_duration:.2f}s total audio)")
    print("\n--- Streaming Audio at Real-Time Cadence (560ms intervals) ---")

    t_stream_start = time.perf_counter()
    first_token_latency_ms = None
    first_token_speech_ms = None
    partials = []

    for idx, chunk in enumerate(chunks):
        t_chunk_send = time.perf_counter()
        ws.send_binary(chunk)
        speech_sent_ms = (idx + 1) * 560

        # Receive any messages ready
        deadline = t_chunk_send + 0.560
        while time.perf_counter() < deadline:
            timeout_remain = max(0.01, deadline - time.perf_counter())
            ws.settimeout(timeout_remain)
            try:
                raw_msg = ws.recv()
                t_recv = time.perf_counter()
                msg = json.loads(raw_msg)
                elapsed_from_start_ms = (t_recv - t_stream_start) * 1000

                if msg.get("type") in ("transcript", "partial") and msg.get("text"):
                    text = msg.get("text").strip()
                    if first_token_latency_ms is None:
                        first_token_latency_ms = elapsed_from_start_ms
                        first_token_speech_ms = speech_sent_ms
                        print(f" [CHUNK {idx+1}] >>> TTFT: {first_token_latency_ms:.1f} ms from stream start (speech sent: {speech_sent_ms} ms)")
                        print(f"           First Partial: \"{text}\"")
                    else:
                        print(f" [CHUNK {idx+1}] Step @ +{elapsed_from_start_ms:.0f}ms: \"{text}\"")
                    partials.append((elapsed_from_start_ms, text))
            except (websocket.WebSocketTimeoutException, TimeoutError):
                break

    print("\n--- Sending Flush Signal ---")
    t_flush = time.perf_counter()
    ws.send(json.dumps({"type": "flush"}))
    ws.settimeout(10.0)

    final_text = ""
    ttff_ms = None
    while True:
        try:
            raw_msg = ws.recv()
            t_recv = time.perf_counter()
            msg = json.loads(raw_msg)
            if msg.get("type") == "final" or (msg.get("type") == "transcript" and msg.get("is_final")):
                ttff_ms = (t_recv - t_flush) * 1000
                final_text = msg.get("text")
                break
        except Exception as e:
            print("Flush timeout:", e)
            break

    ws.close()
    print(f"Flush-to-Final TTFF: {ttff_ms:.1f} ms")
    print(f"Final Transcript:    \"{final_text}\"")
    print("=" * 70)

if __name__ == "__main__":
    pcm16, dur = load_resampled_16k(AUDIO_FILE)
    test_health()
    test_sync_transcribe(AUDIO_FILE)
    test_realtime_websocket(pcm16, dur)
