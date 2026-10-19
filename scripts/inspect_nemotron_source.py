import inspect
from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import NemotronAsrStreamingForRNNT, NemotronAsrStreamingEncoder

print("=== NemotronAsrStreamingForRNNT.forward ===")
print(inspect.getsource(NemotronAsrStreamingForRNNT.forward))

print("\n=== NemotronAsrStreamingEncoder.forward ===")
print(inspect.getsource(NemotronAsrStreamingEncoder.forward))
