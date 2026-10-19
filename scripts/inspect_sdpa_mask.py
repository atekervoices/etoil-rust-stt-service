import torch
import transformers.masking_utils as mu

print("mu functions:", [x for x in dir(mu) if "mask" in x])

# Let's inspect create_bidirectional_mask or whatever sdpa_mask calls
import inspect
print(inspect.getsource(mu.sdpa_mask))
