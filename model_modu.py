import torch
from torch import nn
from torch.nn import LSTM,AdaptiveAvgPool1d,Linear

class Model(nn.Module):
    def __init__(self, emb_output_dim, num_classes,max_length, hidden_size,dr):
        super().__init__()
        self.lstm = LSTM(emb_output_dim, hidden_size, num_layers=8, dropout=dr)
        self.adaptiveAveragePool = AdaptiveAvgPool1d(1)
        self.fc_layer = Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dr)

        self.globAveragePool = nn.AvgPool1d(kernel_size=emb_output_dim)
        self.linear1 = nn.Linear(max_length,hidden_size)
        self.linear2 = nn.Linear(hidden_size,num_classes)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, embedding, mask, lstm):
        if lstm:
            mask = torch.unsqueeze(mask, dim=-1) 
            llm_embedding_masked = embedding * mask
            out, last_hiddenState = self.lstm(llm_embedding_masked)  # batch, max, hidden 
            out = out.transpose(2,1) #  batch,hidden, max # torch.Size([10, 100, 200])
            average_poolOut = self.adaptiveAveragePool(out) # average on max_length to get a summary of the sequence 
            squeezeOut = torch.squeeze(average_poolOut, -1) 
            dropout_out1 = self.dropout(squeezeOut)
            fc1_out = self.fc_layer(dropout_out1) 
            return fc1_out
        else:
            mask = torch.unsqueeze(mask, dim=-1) 
            llm_embedding_masked = embedding * mask
            averagePool_out = self.globAveragePool(llm_embedding_masked)
            drop_out = self.dropout(averagePool_out).squeeze()
            linear_1 = self.linear1(drop_out)
            activation_out = self.softmax(linear_1)
            linear_2 = self.linear2(activation_out)
            return linear_2


