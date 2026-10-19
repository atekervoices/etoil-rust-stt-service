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

def test_decode(audio_path, prompt_idx, blank_penalty):
    model_dir = "nemotron-3.5-asr-streaming-luganda-onnx"
    sp = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "tokenizer.model"))
    blank_id = sp.get_piece_size()

    enc_sess = ort.InferenceSession(os.path.join(model_dir, "encoder.onnx"), providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(os.path.join(model_dir, "decoder_joint.onnx"), providers=["CPUExecutionProvider"])

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))

    mel = compute_mel(audio)
    total_mel_frames = mel.shape[1]

    chunk_size = 56 # 560ms
    pre_encode_cache = 9
    expected_size = pre_encode_cache + chunk_size

    cache_ch = np.zeros((24, 1, 56, 1024), dtype=np.float32)
    cache_tm = np.zeros((24, 1, 1024, 8), dtype=np.float32)
    cache_len = np.zeros((1,), dtype=np.int64)

    state_1 = np.zeros((2, 1, 640), dtype=np.float32)
    state_2 = np.zeros((2, 1, 640), dtype=np.float32)
    last_token = blank_id

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

        enc_inputs = {
            "processed_signal": chunk_data,
            "processed_signal_length": np.array([pre_encode_cache + main_len], dtype=np.int64),
            "cache_last_channel": cache_ch,
            "cache_last_time": cache_tm,
            "cache_last_channel_len": cache_len,
            "prompt_index": np.array([prompt_idx], dtype=np.int64),
        }
        enc_outs = enc_sess.run(None, enc_inputs)
        encoded = enc_outs[0]
        enc_len = enc_outs[1][0]
        cache_ch, cache_tm, cache_len = enc_outs[2], enc_outs[3], enc_outs[4]

        for t in range(int(enc_len)):
            frame = encoded[:, :, t:t+1]
            for _ in range(10):
                dec_inputs = {
                    "encoder_outputs": frame,
                    "targets": np.array([[last_token]], dtype=np.int32),
                    "target_length": np.array([1], dtype=np.int32),
                    "input_states_1": state_1,
                    "input_states_2": state_2,
                }
                dec_outs = dec_sess.run(None, dec_inputs)
                logits = dec_outs[0].squeeze().copy()

                # Apply blank penalty
                logits[blank_id] -= blank_penalty

                max_idx = np.argmax(logits)
                if max_idx == blank_id:
                    break

                piece = sp.id_to_piece(int(max_idx))
                emitted_tokens.append(int(max_idx))
                last_token = int(max_idx)
                state_1, state_2 = dec_outs[2], dec_outs[3]

        buffer_idx += chunk_size
        chunk_idx += 1

    transcript = sp.decode(emitted_tokens)
    print(f"Blank penalty: {blank_penalty:.1f} -> Transcript: '{transcript}'")

for bp in [0.0, 2.0, 2.5, 3.0, 3.5, 4.0]:
    test_decode("audio/audio (1).wav", prompt_idx=63, blank_penalty=bp)
