import os
import sys
import numpy as np
import onnxruntime as ort
import soundfile as sf
import sentencepiece as spm
import librosa

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def compute_mel(audio_16k):
    preemph = np.empty_like(audio_16k)
    preemph[0] = audio_16k[0]
    preemph[1:] = audio_16k[1:] - 0.97 * audio_16k[:-1]

    mel = librosa.feature.melspectrogram(
        y=preemph,
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window='hann',
        center=True,
        pad_mode='reflect',
        power=2.0,
        n_mels=128,
        norm='slaney',
        htk=False,
    )
    return np.log(mel + 5.9604645e-8)

def decode_audio(model_dir, audio_path, prompt_idx, lang_name):
    sp = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "tokenizer.model"))
    vocab_size = sp.get_piece_size()
    blank_id = vocab_size

    enc_sess = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(os.path.join(model_dir, "decoder_joint.onnx"), providers=["CPUExecutionProvider"])

    # Load and resample audio to 16kHz
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        num_samples = int(len(audio) * 16000 / sr)
        audio = scipy.signal.resample(audio, num_samples)
        sr = 16000

    mel = compute_mel(audio)
    total_mel_frames = mel.shape[1]
    print(f"\n--- Decoding {lang_name} (prompt {prompt_idx}) | Mel shape: {mel.shape} ({total_mel_frames} frames) ---")

    # Initial states
    cache_ch = np.zeros((24, 1, 56, 1024), dtype=np.float32)
    cache_tm = np.zeros((24, 1, 1024, 8), dtype=np.float32)
    cache_len = np.zeros((1,), dtype=np.int64)

    state_1 = np.zeros((2, 1, 640), dtype=np.float32)
    state_2 = np.zeros((2, 1, 640), dtype=np.float32)
    last_token = blank_id

    # Streaming params from config: chunk_size = 2 output frames = 16 mel frames, pre_encode_cache = 9
    chunk_size = 16
    pre_encode_cache = 9
    expected_size = pre_encode_cache + chunk_size

    emitted_tokens = []
    chunk_idx = 0
    buffer_idx = 0

    while buffer_idx < total_mel_frames:
        chunk_end = min(buffer_idx + chunk_size, total_mel_frames)
        main_len = chunk_end - buffer_idx

        chunk_data = np.zeros((1, 128, expected_size), dtype=np.float32)
        if chunk_idx > 0 and buffer_idx >= pre_encode_cache:
            cache_start = buffer_idx - pre_encode_cache
            chunk_data[0, :, :pre_encode_cache] = mel[:, cache_start:cache_start+pre_encode_cache]

        chunk_data[0, :, pre_encode_cache:pre_encode_cache+main_len] = mel[:, buffer_idx:chunk_end]

        chunk_length = pre_encode_cache + main_len
        enc_inputs = {
            "processed_signal": chunk_data,
            "processed_signal_length": np.array([chunk_length], dtype=np.int64),
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_tm,
            "cache_last_channel_len": cache_len,
            "prompt_index": np.array([prompt_idx], dtype=np.int64),
        }

        enc_outs = enc_sess.run(None, enc_inputs)
        encoded = enc_outs[0]      # [1, 1024, enc_len]
        enc_len = enc_outs[1][0]   # scalar
        cache_ch = enc_outs[2]
        cache_tm = enc_outs[3]
        cache_len = enc_outs[4]

        # Greedy RNNT decode on encoder frames
        for t in range(int(enc_len)):
            frame = encoded[:, :, t:t+1] # [1, 1024, 1]

            for symbol_step in range(10):
                dec_inputs = {
                    "encoder_outputs": frame,
                    "targets": np.array([[last_token]], dtype=np.int32),
                    "target_length": np.array([1], dtype=np.int32),
                    "input_states_1": state_1,
                    "input_states_2": state_2,
                }
                dec_outs = dec_sess.run(None, dec_inputs)
                logits = dec_outs[0].squeeze()
                new_state_1 = dec_outs[2]
                new_state_2 = dec_outs[3]
                blank_val = logits[blank_id]
                top_indices = np.argsort(logits)[::-1][:5]
                top_tokens = [(idx, logits[idx], repr(sp.id_to_piece(int(idx)))) for idx in top_indices]

                max_idx = top_indices[0]
                max_val = logits[max_idx]

                # If this frame has audio (margin is small), print top tokens
                if blank_val - top_tokens[1][1] < 10.0 and t == 0:
                    cand_str = ", ".join([f"{t[2]} ({t[1]:.2f})" for t in top_tokens])
                    print(f"  Chunk {chunk_idx:2d} t={t}: {cand_str}")

                if max_idx == blank_id:
                    break

                emitted_tokens.append(int(max_idx))
                piece = repr(sp.id_to_piece(int(max_idx)))
                print(f"    [EMIT] token {max_idx} ({piece}) with logit {max_val:.2f} (blank was {blank_val:.2f})")
                last_token = int(max_idx)
                state_1 = new_state_1
                state_2 = new_state_2

        buffer_idx += chunk_size
        chunk_idx += 1

    transcript = sp.decode(emitted_tokens)
    print(f"\nFinal transcript for {lang_name}: '{transcript}' (total tokens: {len(emitted_tokens)})")
    return transcript

def main():
    model_dir = "nemotron-3.5-asr-streaming-luganda-onnx"
    audio_path = "audio/audio (1).wav"

    # Test Luganda (slot 63)
    decode_audio(model_dir, audio_path, prompt_idx=63, lang_name="Luganda")

    # Test English (slot 0)
    decode_audio(model_dir, "audio/test.wav", prompt_idx=0, lang_name="English")

if __name__ == "__main__":
    main()
