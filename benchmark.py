import requests
import sys
import time
import wave
import websocket
import json
import base64
import threading
from pathlib import Path
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_wav_as_raw_pcm(wav_file_path):
    """Load WAV file and extract raw PCM data with metadata."""
    try:
        with wave.open(str(wav_file_path), 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            n_frames = wav_file.getnframes()
            raw_pcm_data = wav_file.readframes(n_frames)
            duration = n_frames / sample_rate

            print(f"Audio file info:")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {channels}")
            print(f"  Sample width: {sample_width} bytes ({sample_width * 8}-bit)")
            print(f"  Frames: {n_frames}")
            print(f"  Duration: {duration:.3f} seconds")
            print(f"  Raw PCM size: {len(raw_pcm_data)} bytes")

            return raw_pcm_data, sample_rate, channels, sample_width, duration

    except Exception as e:
        print(f"Error loading WAV file: {e}")
        return None, None, None, None, None


def send_single_request(server_url, raw_pcm_data, sample_rate, request_id=0):
    """Send one single file transcription request."""
    headers = {"Content-Type": "application/octet-stream"}
    params = {"sample_rate": sample_rate, "source_lang": "en", "target_lang": "en"}

    start = time.perf_counter()
    response = requests.post(
        f"{server_url}/v1/transcribe/canary",
        data=raw_pcm_data,
        headers=headers,
        params=params,
        timeout=60,
    )
    latency = time.perf_counter() - start

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    body = response.json()
    return {
        "request_id": request_id,
        "latency": latency,
        "server_processing_time": body.get("processing_time", 0),
        "text": body.get("text", ""),
    }


def send_batch_request(server_url, raw_pcm_data, sample_rate, request_id=0):
    """Send one batch transcription request with single file."""
    import io
    
    # Create a file-like object from the raw PCM data
    audio_file = io.BytesIO(raw_pcm_data)
    
    files = {'files': ('test_audio.wav', audio_file, 'audio/wav')}
    params = {"sample_rate": sample_rate, "source_lang": "en", "target_lang": "en"}

    start = time.perf_counter()
    response = requests.post(
        f"{server_url}/v1/transcribe/batch",
        files=files,
        params=params,
        timeout=60,
    )
    latency = time.perf_counter() - start

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

    body = response.json()
    # Extract first result from batch response
    if body.get("results") and len(body["results"]) > 0:
        result = body["results"][0]
        return {
            "request_id": request_id,
            "latency": latency,
            "server_processing_time": result.get("processing_time", 0),
            "text": result.get("text", ""),
        }
    else:
        raise RuntimeError("No results in batch response")


def send_websocket_request(server_url, raw_pcm_data, sample_rate, request_id=0):
    """Send one WebSocket transcription request."""
    ws_url = server_url.replace('http://', 'ws://').replace('https://', 'wss://') + '/ws'
    
    result = {
        "request_id": request_id,
        "latency": 0,
        "server_processing_time": 0,
        "text": "",
        "received_final": False,
    }
    
    def on_message(ws, message):
        nonlocal result
        try:
            data = json.loads(message)
            print(f"  WebSocket message: {data.get('type', 'unknown')} - {str(data)[:100]}")
            
            if data.get('type') == 'final':
                result['text'] = data.get('text', '')
                result['latency'] = time.perf_counter() - start_time
                result['received_final'] = True
                print(f"  Got final transcription: {result['text'][:50]}...")
            elif data.get('type') == 'partial':
                # Collect partial results to build complete transcription
                partial_text = data.get('text', '')
                if partial_text:
                    result['partial_text'] = result.get('partial_text', '') + partial_text
                    result['latency'] = time.perf_counter() - start_time
                    print(f"  Partial: {partial_text}")
            elif data.get('type') == 'ready':
                print(f"  WebSocket ready for transcription")
            elif data.get('type') == 'error':
                result['text'] = f"WebSocket error: {data.get('message', 'Unknown error')}"
                print(f"  WebSocket error: {data.get('message')}")
        except Exception as e:
            print(f"  Error parsing WebSocket message: {e}")
    
    def on_error(ws, error):
        nonlocal result
        result['text'] = f"WebSocket connection error: {error}"
        print(f"  WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        print(f"  WebSocket closed: {close_status_code} - {close_msg}")
    
    def on_open(ws):
        print(f"  WebSocket connected, sending config...")
        # Send configuration
        config = {
            "type": "config",
            "sample_rate": sample_rate,
            "source_lang": "en",
            "target_lang": "en"
        }
        ws.send(json.dumps(config))
        
        def send_audio_chunks():
            # Wait for ready message
            time.sleep(0.5)
            
            print(f"  Sending audio data in chunks ({len(raw_pcm_data)} bytes)...")
            # Send audio in chunks like the working test
            chunk_size = 1024
            for i in range(0, len(raw_pcm_data), chunk_size):
                chunk = raw_pcm_data[i:i+chunk_size]
                chunk_base64 = base64.b64encode(chunk).decode('utf-8')
                
                audio_msg = {
                    "type": "audio",
                    "data": chunk_base64,
                    "sample_rate": sample_rate
                }
                
                ws.send(json.dumps(audio_msg))
                print(f"  Sent audio chunk {i//chunk_size + 1}: {len(chunk)} bytes")
                
                # Small delay between chunks
                time.sleep(0.1)
            
            # Wait for final processing
            def close_connection():
                time.sleep(3)  # Wait for final processing
                if not result['received_final']:
                    # Use collected partial text as the final result
                    if result.get('partial_text'):
                        result['text'] = result['partial_text']
                        print(f"  Using collected partial text as final: {result['text'][:50]}...")
                    else:
                        result['text'] = "Timeout - no transcription received"
                ws.close()
            
            threading.Thread(target=close_connection, daemon=True).start()
        
        threading.Thread(target=send_audio_chunks, daemon=True).start()
    
    start_time = time.perf_counter()
    
    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Run WebSocket with timeout
    print(f"  Connecting to {ws_url}...")
    ws.run_forever()
    
    if not result['text'] or 'error' in result['text'].lower():
        raise RuntimeError(f"WebSocket transcription failed: {result['text']}")
    
    return result


def warmup(server_url, endpoint, raw_pcm_data, sample_rate):
    """Single warmup request so first-run JIT / CUDA overhead is excluded."""
    print(f"  Warmup request to {endpoint} ...")
    
    if endpoint == "canary":
        r = send_single_request(server_url, raw_pcm_data, sample_rate)
    elif endpoint == "batch":
        r = send_batch_request(server_url, raw_pcm_data, sample_rate)
    elif endpoint == "websocket":
        r = send_websocket_request(server_url, raw_pcm_data, sample_rate)
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    
    print(
        f"  Warmup done: {r['latency']*1000:.1f}ms latency "
        f"({r['server_processing_time']*1000:.1f}ms server)"
    )
    print(f"  Text: \"{r['text'][:120]}{'...' if len(r['text']) > 120 else ''}\"")
    return r["text"]


def run_sequential(server_url, endpoint, raw_pcm_data, sample_rate, n):
    """Send *n* requests one-by-one."""
    results = []
    for i in range(n):
        if endpoint == "canary":
            results.append(send_single_request(server_url, raw_pcm_data, sample_rate, request_id=i))
        elif endpoint == "batch":
            results.append(send_batch_request(server_url, raw_pcm_data, sample_rate, request_id=i))
        elif endpoint == "websocket":
            results.append(send_websocket_request(server_url, raw_pcm_data, sample_rate, request_id=i))
        else:
            raise ValueError(f"Unknown endpoint: {endpoint}")
    return results


def run_concurrent(server_url, endpoint, raw_pcm_data, sample_rate, n):
    """Fire *n* requests simultaneously."""
    results = [None] * n
    
    # Choose the right function based on endpoint
    if endpoint == "canary":
        func = send_single_request
    elif endpoint == "batch":
        func = send_batch_request
    elif endpoint == "websocket":
        func = send_websocket_request
    else:
        raise ValueError(f"Unknown endpoint: {endpoint}")
    
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {
            pool.submit(func, server_url, raw_pcm_data, sample_rate, i): i
            for i in range(n)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
    return results


def print_round_detail(round_num, results, reference_text=None):
    """Print per-request latency table for a single round."""
    lats = [r["latency"] * 1000 for r in results]
    mismatches = 0
    print(f"\n  Round {round_num}:")
    print(f"    {'Req':>4}  {'Latency':>10}  {'Server':>10}  {'Match':>5}")
    print(f"    {'----':>4}  {'----------':>10}  {'----------':>10}  {'-----':>5}")
    for r in results:
        if reference_text is not None:
            ok = r["text"] == reference_text
            if not ok:
                mismatches += 1
            flag = "ok" if ok else "FAIL"
        else:
            flag = "-"
        print(
            f"    #{r['request_id']:<3d}  "
            f"{r['latency']*1000:>8.1f}ms  "
            f"{r['server_processing_time']*1000:>8.1f}ms  "
            f"{flag:>5}"
        )
    mn = min(lats)
    avg = statistics.mean(lats)
    mx = max(lats)
    print(f"    {'':>4}  {'----------':>10}")
    print(f"    min   {mn:>8.1f}ms")
    print(f"    avg   {avg:>8.1f}ms")
    print(f"    max   {mx:>8.1f}ms")
    if reference_text is not None and mismatches > 0:
        print(f"    ** {mismatches}/{len(results)} responses differ from reference **")
    return lats, mismatches


def print_summary_table(
    seq_all_lats, con_all_lats,
    seq_wall_times, con_wall_times,
    num_requests,
    seq_total_mismatches, con_total_mismatches,
    seq_total_reqs, con_total_reqs,
):
    """Print the final side-by-side comparison."""
    seq_avg_wall = statistics.mean(seq_wall_times)
    con_avg_wall = statistics.mean(con_wall_times)
    seq_tp = num_requests / seq_avg_wall
    con_tp = num_requests / con_avg_wall

    col_w = 16  # column width

    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':<30} {'Sequential':>{col_w}} {'Concurrent':>{col_w}}")
    print(f"  {'-'*62}")

    def row(label, seq_val, con_val, unit="ms"):
        print(f"  {label:<30} {seq_val:>{col_w-3}.1f} {unit}  {con_val:>{col_w-3}.1f} {unit}")

    row("Avg wall time / round",  seq_avg_wall * 1000,  con_avg_wall * 1000)
    row("Throughput",              seq_tp,                con_tp,               unit="req/s")
    print(f"  {'-'*62}")
    row("Per-req latency  (min)",  min(seq_all_lats),     min(con_all_lats))
    row("Per-req latency  (p50)",  statistics.median(seq_all_lats), statistics.median(con_all_lats))
    row("Per-req latency  (avg)",  statistics.mean(seq_all_lats),   statistics.mean(con_all_lats))
    row("Per-req latency  (p95)",  sorted(seq_all_lats)[int(len(seq_all_lats)*0.95)],
                                    sorted(con_all_lats)[int(len(con_all_lats)*0.95)])
    row("Per-req latency  (max)",  max(seq_all_lats),     max(con_all_lats))

    if len(seq_all_lats) > 1 and len(con_all_lats) > 1:
        row("Per-req latency  (std)",
            statistics.stdev(seq_all_lats), statistics.stdev(con_all_lats))

    speedup = seq_avg_wall / con_avg_wall if con_avg_wall > 0 else 0
    tp_gain = ((con_tp - seq_tp) / seq_tp * 100) if seq_tp > 0 else 0

    print(f"\n  Wall-time speedup: {speedup:.2f}x")
    print(f"  Throughput gain:   {tp_gain:+.1f}%")

    if speedup > 1.0:
        print(
            f"\n  Concurrent batching is ~{speedup:.1f}x faster in wall time "
            f"for {num_requests} simultaneous requests."
        )
    else:
        print(
            f"\n  No speedup observed. This may indicate the service isn't optimized "
            f"for concurrent processing or requests are queuing."
        )

    # -- accuracy report -----------------------------------------------
    print(f"\n{'='*70}")
    print("ACCURACY CHECK")
    print(f"{'='*70}")
    seq_ok = seq_total_reqs - seq_total_mismatches
    con_ok = con_total_reqs - con_total_mismatches
    print(f"  Sequential:  {seq_ok}/{seq_total_reqs} match reference", end="")
    print("  PASS" if seq_total_mismatches == 0 else f"  FAIL ({seq_total_mismatches} mismatches)")
    print(f"  Concurrent:  {con_ok}/{con_total_reqs} match reference", end="")
    print("  PASS" if con_total_mismatches == 0 else f"  FAIL ({con_total_mismatches} mismatches)")

    if seq_total_mismatches == 0 and con_total_mismatches == 0:
        print(f"\n  All {seq_total_reqs + con_total_reqs} responses are identical — "
              f"concurrent processing does not affect output accuracy.")


def run_batch_comparison(server_url, audio_file, endpoint, num_requests, num_rounds):
    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"Audio file not found: {audio_file}")
        print("Please create a test_audio.wav file or specify a different audio file.")
        return False

    raw_pcm, sr, _, _, audio_dur = load_wav_as_raw_pcm(audio_path)
    if raw_pcm is None:
        return False

    print(f"\nBenchmark config:")
    print(f"  Server URL:          {server_url}")
    print(f"  Endpoint:            /v1/transcribe/{endpoint}")
    print(f"  Requests per round:  {num_requests}")
    print(f"  Rounds:              {num_rounds}")
    print(f"  Audio duration:      {audio_dur:.3f}s")

    # -- warmup (also establishes the reference transcription) ----------
    try:
        reference_text = warmup(server_url, endpoint, raw_pcm, sr)
    except Exception as e:
        print(f"Warmup failed: {e}")
        return False

    print(f"\n  Reference text for accuracy check:")
    print(f"    \"{reference_text[:200]}{'...' if len(reference_text) > 200 else ''}\"")

    print(f"\n  Warmup concurrent batch ({num_requests} requests) ...")
    try:
        wall_start = time.perf_counter()
        run_concurrent(server_url, endpoint, raw_pcm, sr, num_requests)
        wall_time = time.perf_counter() - wall_start
        print(f"  Warmup concurrent done: {wall_time*1000:.0f}ms wall")
    except Exception as e:
        print(f"  Warmup concurrent failed: {e}")
        return False

    # -- sequential ----------------------------------------------------
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL  ({num_requests} requests x {num_rounds} rounds)")
    print(f"{'='*70}")

    seq_all_lats = []
    seq_wall_times = []
    seq_total_mismatches = 0
    seq_total_reqs = 0

    for rnd in range(1, num_rounds + 1):
        wall_start = time.perf_counter()
        results = run_sequential(server_url, endpoint, raw_pcm, sr, num_requests)
        wall_time = time.perf_counter() - wall_start

        seq_wall_times.append(wall_time)
        lats, mismatches = print_round_detail(rnd, results, reference_text)
        seq_all_lats.extend(lats)
        seq_total_mismatches += mismatches
        seq_total_reqs += len(results)
        print(f"    wall  {wall_time*1000:>8.1f}ms   ({num_requests / wall_time:.1f} req/s)")

    # -- concurrent ----------------------------------------------------
    print(f"\n{'='*70}")
    print(f"CONCURRENT  ({num_requests} requests x {num_rounds} rounds)")
    print(f"{'='*70}")

    con_all_lats = []
    con_wall_times = []
    con_total_mismatches = 0
    con_total_reqs = 0

    for rnd in range(1, num_rounds + 1):
        wall_start = time.perf_counter()
        results = run_concurrent(server_url, endpoint, raw_pcm, sr, num_requests)
        wall_time = time.perf_counter() - wall_start

        con_wall_times.append(wall_time)
        lats, mismatches = print_round_detail(rnd, results, reference_text)
        con_all_lats.extend(lats)
        con_total_mismatches += mismatches
        con_total_reqs += len(results)
        print(f"    wall  {wall_time*1000:>8.1f}ms   ({num_requests / wall_time:.1f} req/s)")

    # -- summary -------------------------------------------------------
    print_summary_table(
        seq_all_lats, con_all_lats,
        seq_wall_times, con_wall_times,
        num_requests,
        seq_total_mismatches, con_total_mismatches,
        seq_total_reqs, con_total_reqs,
    )

    if con_total_mismatches > 0:
        print(f"\n  WARNING: {con_total_mismatches} concurrent responses differ from reference.")
        print(f"  This indicates concurrent processing may affect transcription accuracy.")
        return False

    return True


def main():
    server_url = "http://localhost:8080"
    audio_file = "test_audio.wav"
    endpoint = "canary"  # canary, batch, or websocket
    num_requests = 8
    num_rounds = 3

    if len(sys.argv) >= 2:
        endpoint = sys.argv[1]
    if len(sys.argv) >= 3:
        num_requests = int(sys.argv[2])
    if len(sys.argv) >= 4:
        num_rounds = int(sys.argv[3])
    if len(sys.argv) >= 5:
        audio_file = sys.argv[4]

    print(f"Canary STT Service - Sequential vs Concurrent Benchmark")
    print(f"{'='*70}")
    print(f"Endpoints: canary (single file), batch (multipart), websocket (streaming)")
    print(f"{'='*70}\n")

    try:
        success = run_batch_comparison(
            server_url, audio_file, endpoint, num_requests, num_rounds,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
