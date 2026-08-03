# reference model for the CUDA implementation, prayeth it work

import numpy as np

def softmax(x, axis=-1):
  x_max = np.max(x, axis=axis, keepdims=True)
  e_x = np.exp(x - x_max)
  return e_x / np.sum(e_x, axis=axis, keepdims=True)

# prefill section
def prefill_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
  '''
  S = number of tokens in prompt
  d_k = dimensions/embeddings for each vector (size of vector/floats per token)
  '''

  S, d_k = Q.shape;

  scores = np.matmul(Q, K.T) * (1/np.sqrt(d_k));
  mask = np.triu(np.ones((S,S), dtype=bool), k=1)
  scores[mask] = -np.inf
  weights = softmax(scores)
  return np.matmul(weights, V)

# decode section
class KV_Cache:
  def __init__(self, d_k: int):
    # start empty
    self.k_cache = np.empty((0, d_k))
    self.v_cache = np.empty((0, d_k))
  def append(self, k_row: np.ndarray, v_row: np.ndarray):
    self.k_cache = np.vstack([self.k_cache, k_row])
    self.v_cache = np.vstack([self.v_cache, v_row])

def decode_attention_step(q_row: np.ndarray, k_row: np.ndarray, v_row: np.ndarray, cache: KV_Cache) -> np.ndarray:
  '''
  q,k,v_row are row with length d_k
  '''
  d_k = q_row.shape[-1]
  cache.append(k_row, v_row)
  scores = np.matmul(q_row, cache.k_cache.T) * (1/np.sqrt(d_k))
  weights = softmax(scores)
  return np.matmul(weights, cache.v_cache)

if __name__ == "__main__":
  np.random.seed(42)

  seq_len = 8
  d_k = 64

  Q = np.random.randn(seq_len, d_k)
  K = np.random.randn(seq_len, d_k)
  V = np.random.randn(seq_len, d_k)

  prefill_output = prefill_attention(Q, K, V)

  cache = KV_Cache(d_k)
  decode_output_list = []

  for i in range(seq_len):
    # each row
    q_i = Q[i : i + 1,:]
    k_i = K[i : i + 1,:]
    v_i = V[i : i + 1,:]
    # output
    o_i = decode_attention_step(q_i, k_i, v_i, cache)
    decode_output_list.append(o_i)
  
  decode_output = np.vstack(decode_output_list)
  
  max_diff = np.max(np.abs(prefill_output - decode_output))
  print(f"Max abs diff = {max_diff: .2e}")
  assert np.allclose(prefill_output, decode_output, atol=1e-6), "VALUE NO MATCH OH NO"
  print("yay prefill and decode is de match")
