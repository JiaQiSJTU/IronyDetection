# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)]) # clone the layer for N times
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class SoftMax(nn.Module):
    def __init__(self,n_input,n_out):
        super(SoftMax,self).__init__()
        self.fc = nn.Linear(n_input,n_out)
        self.softmax = nn.LogSoftmax(1)
        
    def forward(self, x):
        x = self.fc(x)
        y = self.softmax(x)
#         print(y)
        return y

# single task
# embeding --> encoder --> linear --> softmax
class SelfAttenClassifier(nn.Module):
    def __init__(self, encoder, classifier):
        super(SelfAttenClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        
    def forward(self, input_embeds, mask, addition_feats=None):
        batch_size = input_embeds.size(1)
        encoder_out = self.encoder(input_embeds, mask)
        feats = encoder_out.sum(dim=1)
        
        if addition_feats is not None:
            feats = torch.cat((feats, addition_feats),dim=1)
#         print(encoder_out.size(), feats.size())
        outputs = self.classifier(feats)
        return outputs,feats
