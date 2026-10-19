import asyncio
import json
import websockets
import numpy as np
import librosa
import sys

async def main():
    uri = "ws://127.0.0.1:8099/v1/realtime?language=lug"
    print(f"Connecting to {uri}...", flush=True)
    async with websockets.connect(uri) as ws:
        ready = await ws.recv()
        print(f"SERVER READY: {ready}", flush=True)

        audio, sr = librosa.load("../test.wav", sr=16000)
        pcm16 = (audio * 32767).astype(np.int16).tobytes()
        chunk_size = 3200  # 100ms
        num_chunks = len(pcm16) // chunk_size
        print(f"Streaming {num_chunks} chunks ({len(audio)/16000:.2f}s audio)...", flush=True)

        for i in range(0, len(pcm16), chunk_size):
            chunk = pcm16[i:i+chunk_size]
            await ws.send(chunk)
            await asyncio.sleep(0.05)
            # check messages
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    print(f"STREAM MSG: {msg}", flush=True)
                except asyncio.TimeoutError:
                    break

        print("Finished sending audio. Sending flush...", flush=True)
        await ws.send(json.dumps({"type": "flush"}))

        for _ in range(5):
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=3.0)
                print(f"FINAL MSG: {msg}", flush=True)
            except asyncio.TimeoutError:
                break

        print("Done!", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
