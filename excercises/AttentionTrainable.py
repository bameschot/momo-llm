import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x2 = inputs[1]
dimension_in = inputs.shape[1]
dimension_out = 2

torch.manual_seed(123)
wQuery  = torch.nn.Parameter(torch.rand(dimension_in,dimension_out),requires_grad=False)
wKey  = torch.nn.Parameter(torch.rand(dimension_in,dimension_out),requires_grad=False)
wValue  = torch.nn.Parameter(torch.rand(dimension_in,dimension_out),requires_grad=False)

query2 = x2 @ wQuery
key2 = x2 @ wKey
value2 = x2 @ wValue

print(query2)

query = inputs @ wQuery
keys = inputs @ wKey
values = inputs @ wValue

print(keys.shape)
print(values.shape)

attentionScore2 = query[1].dot(keys[1])
print(attentionScore2)

attentionScores2 = query[1] @ keys.T
print(attentionScores2)

key_dimension = keys.shape[-1]
print(key_dimension)

attentionWeights2 = torch.softmax(attentionScores2 / key_dimension ** 0.5,dim=-1)
print(attentionWeights2)
print(values)

contextVector2 = attentionWeights2 @ values
print(contextVector2)

