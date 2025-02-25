import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

#calculate attention score for the query
query = inputs[1]
queryAttentionScores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    queryAttentionScores[i] = torch.dot(x_i,query)
print(f"query      {queryAttentionScores}")

#normalize the attention scores for the query
#queryNormalizedAttentionWeights = queryAttentionScores / queryAttentionScores.sum()
queryNormalizedAttentionWeights = torch.softmax(queryAttentionScores, dim=0)

print(f"normalized {queryNormalizedAttentionWeights} = {queryNormalizedAttentionWeights.sum()}")

#calculate the context vector by multiplying each embedded inputtoken by the attention weight and summing the resulting  vectors
queryContextVector = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    queryContextVector += x_i * queryNormalizedAttentionWeights[i]
print(f"query context vector {queryContextVector}")

#All
attentionScores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attentionScores[i, j] = torch.dot(x_i,x_j)
#print(attentionScores)

#calculate all attention scores by multiplying inputs with inputs transposed
attentionScores = inputs @ inputs.T
#calculate all attention weights by sofmaxing each input row
attentionWeights = torch.softmax(attentionScores,dim=1)
#calculate all context vectors by multiplying the attention weights with the inputs
contextVectors = attentionWeights @ inputs 

print(contextVectors)
