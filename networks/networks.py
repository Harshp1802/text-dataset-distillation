import torch.nn as nn
import torch.nn.functional as F
import logging
import itertools
from . import utils
import math
import torch
from transformers import BertModel
#from pytorch_pretrained_bert import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

# initialize multi lingual bert 
bert = BertModel.from_pretrained('bert-base-multilingual-cased').to('cuda')
# freeze bert
for param in bert.parameters():
    param.requires_grad = False
embeddings = bert.embeddings.word_embeddings.weight
bert.eval()

class LeNet(utils.ReparamModule):
    supported_dims = {28, 32}

    def __init__(self, state):
        if state.dropout:
            raise ValueError("LeNet doesn't support dropout")
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(state.nc, 6, 5, padding=2 if state.input_size == 28 else 0)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 if state.num_classes <= 2 else state.num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out), inplace=True)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out), inplace=True)
        out = F.relu(self.fc2(out), inplace=True)
        out = self.fc3(out)
        return out

class TextRNN1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextRNN1, self).__init__()
        output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        hidden_dim = 100
        n_layers = 2
        bidirectional=True
        dropout=0.5 if self.state.mode=="train" else 0
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = state.learnable_embedding

        self.rnn = nn.RNN(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           bias =True,
                           batch_first=True,
                           nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.sigm=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        if self.state.mode=="train":
            out = self.dropout(out)
        self.rnn.flatten_parameters()
        out, hidden = self.rnn(out)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)
class TextLSTM2(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextLSTM2, self).__init__()
        output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        hidden_dim = 100
        n_layers = 2
        bidirectional=True
        dropout=0.5 if self.state.mode=="train" else 0
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = state.learnable_embedding

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           bias =True,
                           batch_first=True,
                           )
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.sigm=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        if self.state.mode=="train":
            out = self.dropout(out)
        self.rnn.flatten_parameters()
        out, (hidden,cell) = self.rnn(out)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        return self.fc(hidden)

class TextRNN2(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextRNN2, self).__init__()
        output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        hidden_dim = 100
        n_layers = 1
        dropout=0
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = False

        self.rnn = nn.RNN(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=False,
                           dropout=dropout,
                           bias =True,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigm=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        #out = self.dropout(out)
        out, hidden = self.rnn(out)
        #assert torch.equal(out[:,-1,:], hidden.squeeze(0))

        return self.sigm(self.fc(hidden.squeeze(0)))
class TextLSTM1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextLSTM1, self).__init__()
        output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        hidden_dim = 10
        n_layers = 1
        dropout=0.7 if self.state.mode=="train" else 0
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = True

        self.rnn = nn.LSTM(int(state.ninp/4)-1,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=False,
                           dropout=dropout,
                           bias =True,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigm=nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.distilling_flag = False
        self.conv1 = nn.Conv1d(state.maxlen, 16, 5)
        self.relu=nn.ReLU()
        self.maxpool = nn.MaxPool1d(4)

    def forward(self, x):
        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        if self.state.mode=="train":
            out = self.dropout(out)
        out = self.relu(self.conv1(out))
        out = self.maxpool(out)
        self.rnn.flatten_parameters()
        out, (hidden, cell) = self.rnn(out)
        #print (out.size())
        #print (hidden.size())
        #assert torch.equal(out[:,-1,:], hidden[-1,:,:].squeeze(0))
        return self.fc(hidden[-1,:,:].squeeze(0))
        #return self.sigm(self.fc(hidden.squeeze(0)))

class Transformer1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(Transformer1, self).__init__()
        self.output_dim=1 if state.num_classes == 2 else state.num_classes
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        nhead=4
        hidden_dim = embedding_dim
        n_layers = 4
        dropout=0.1
        self.embed = nn.Embedding(ntoken, embedding_dim)
        self.embed.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.embed.weight.requires_grad = state.learnable_embedding
        self.decoder_layer = nn.TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward=hidden_dim, dropout=dropout, activation='relu')
        self.decoder = nn.TransformerDecoder(self.decoder_layer, n_layers)
        self.classifier_head = nn.Linear(hidden_dim, self.output_dim)
        #self.sigm=nn.Sigmoid()
        self.distilling_flag = False


    def forward(self, x):

        if self.state.textdata:
            if not self.distilling_flag:
                out = self.embed(x) #* math.sqrt(ninp)
            else:
                out=torch.squeeze(x)
        else:
            out = x
        #print(out.size())
        tgt_size=[i for i in out.size()]
        tgt_size[-2]=1
        #print(tgt_size)
        tgt=torch.rand(tgt_size).to(self.state.device)
        hidden = self.decoder(tgt, out).squeeze(1)
        return self.classifier_head(hidden)

class TextConvNet_BERT_INPUT_EMBEDDINGS(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextConvNet_BERT_INPUT_EMBEDDINGS, self).__init__()
        #if state.textdata:
        ntoken=state.ntoken
        n_filters = 100
        filter_sizes = [3,4,5]
        dropout=0.5
        output_dim=1 if state.num_classes == 2 else state.num_classes

        # frozen multi lingual bert
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.state.device)
        # embedding_dim = self.bert.config.hidden_size
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # self.bert.eval()
        embedding_dim = 768
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)
        self.fc4 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.newfc = nn.Linear(embedding_dim, output_dim)
        self.multibert = BertModel.from_pretrained('bert-base-multilingual-cased', torch_dtype=torch.float32)#.to(self.state.device)
        self.multibert.train()
        self.sigm=nn.Sigmoid()
        self.distilling_flag=False

    def forward(self, x):
        if self.state.textdata and not self.distilling_flag:
                # initialize multi lingual bert 
                # bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.state.device)
                # # freeze bert
                # for param in bert.parameters():
                #     param.requires_grad = False
                # bert.eval()
                # get bert embedding
                with torch.no_grad():
                    out = self.multibert(x)[0]
                out.unsqueeze_(1)
        else:
                # # take softmax across last dimensions
                # out=F.softmax(x, dim=-1)
                # # prepend cls token one hot vector
                # cls_token = torch.zeros(out.shape[-1]).to(self.state.device)
                # cls_token[101] = 1
                # out = torch.cat((cls_token.repeat(out.shape[0], out.shape[1], 1, 1), out), dim=2)
                # cls_token[101], cls_token[102] = 0, 1
                # out = torch.cat((out, cls_token.repeat(out.shape[0], out.shape[1], 1, 1)), dim=2)

                # out = torch.matmul(out, embeddings).squeeze(1)
                # out=torch.argmax(x,dim=-1).squeeze(1)
                # prepend cls token 
                # out=torch.cat((torch.tensor([101]).repeat(out.shape[0], 1).to(self.state.device), out), dim=-1)
                # # append separator token
                # out=torch.cat((out,torch.tensor([102]).repeat(out.shape[0], 1).to(self.state.device)),dim=-1)
                # get bert embedding with grad
                out = self.multibert(inputs_embeds = x.squeeze(1))[0]
                out.unsqueeze_(1) # batch size, 1, seq len, embedding dim
        
        # take average across sequence length
        out = torch.mean(out, dim=2)
        return self.newfc(out.squeeze(1))
        # out = self.dropout(out)
        conved = [F.relu(conv(out)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc4(self.fc3(self.fc2(self.fc(cat))))



class TextConvNet_BERT_NO_GUMBEL(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextConvNet_BERT_NO_GUMBEL, self).__init__()
        #if state.textdata:
        ntoken=state.ntoken
        n_filters = 100
        filter_sizes = [3,4,5]
        dropout=0.5
        output_dim=1 if state.num_classes == 2 else state.num_classes

        # frozen multi lingual bert
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.state.device)
        # embedding_dim = self.bert.config.hidden_size
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # self.bert.eval()
        embedding_dim = 768
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.fc3 = nn.Linear(output_dim, output_dim)
        self.fc4 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.newfc = nn.Linear(embedding_dim, output_dim)

        self.sigm=nn.Sigmoid()
        self.distilling_flag=False

    def forward(self, x):
        if self.state.textdata and not self.distilling_flag:
                # initialize multi lingual bert 
                # bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.state.device)
                # # freeze bert
                # for param in bert.parameters():
                #     param.requires_grad = False
                # bert.eval()
                # get bert embedding
                with torch.no_grad():
                    out = bert(x)[0]
                out.unsqueeze_(1)
        else:
                # take softmax across last dimensions
                out=F.softmax(x, dim=-1)
                # prepend cls token one hot vector
                cls_token = torch.zeros(out.shape[-1]).to(self.state.device)
                cls_token[101] = 1
                out = torch.cat((cls_token.repeat(out.shape[0], out.shape[1], 1, 1), out), dim=2)
                cls_token[101], cls_token[102] = 0, 1
                out = torch.cat((out, cls_token.repeat(out.shape[0], out.shape[1], 1, 1)), dim=2)

                out = torch.matmul(out, embeddings).squeeze(1)
                # out=torch.argmax(x,dim=-1).squeeze(1)
                # prepend cls token 
                # out=torch.cat((torch.tensor([101]).repeat(out.shape[0], 1).to(self.state.device), out), dim=-1)
                # # append separator token
                # out=torch.cat((out,torch.tensor([102]).repeat(out.shape[0], 1).to(self.state.device)),dim=-1)
                # get bert embedding with grad
                out = bert(inputs_embeds = out)[0]
                out.unsqueeze_(1)
        
        # take average across sequence length
        out = torch.mean(out, dim=2)
        return self.newfc(out.squeeze(1))
        # out = self.dropout(out)
        conved = [F.relu(conv(out)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        return self.fc4(self.fc3(self.fc2(self.fc(cat))))



class TextConvNet_BERT_MOD(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextConvNet_BERT_MOD, self).__init__()
        #if state.textdata:
        ntoken=state.ntoken
        n_filters = 100
        filter_sizes = [3,4,5]
        dropout=0.5
        output_dim=1 if state.num_classes == 2 else state.num_classes

        # frozen multi lingual bert
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.state.device)
        # embedding_dim = self.bert.config.hidden_size
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # self.bert.eval()
        embedding_dim = 768
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # self.fc2 = nn.Linear(output_dim, output_dim)
        # self.fc3 = nn.Linear(output_dim, output_dim)
        # self.fc4 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.newfc = nn.Linear(embedding_dim, output_dim)
        self.sigm=nn.Sigmoid()
        self.distilling_flag=False

    def forward(self, x):
        if self.state.textdata and not self.distilling_flag:
                # initialize multi lingual bert 
                # bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.state.device)
                # # freeze bert
                # for param in bert.parameters():
                #     param.requires_grad = False
                # bert.eval()
                # get bert embedding
                with torch.no_grad():
                    out = bert(x)[0]
                out.unsqueeze_(1)
        else:
                # take argmax across last dimensions
                out=F.gumbel_softmax(x, tau=1, hard=True)
                # prepend cls token one hot vector
                cls_token = torch.zeros(out.shape[-1]).to(self.state.device)
                cls_token[101] = 1
                out = torch.cat((cls_token.repeat(out.shape[0], out.shape[1], 1, 1), out), dim=2)
                cls_token[101], cls_token[102] = 0, 1
                out = torch.cat((out, cls_token.repeat(out.shape[0], out.shape[1], 1, 1)), dim=2)

                out = torch.matmul(out, embeddings).squeeze(1)
                # out=torch.argmax(x,dim=-1).squeeze(1)
                # prepend cls token 
                # out=torch.cat((torch.tensor([101]).repeat(out.shape[0], 1).to(self.state.device), out), dim=-1)
                # # append separator token
                # out=torch.cat((out,torch.tensor([102]).repeat(out.shape[0], 1).to(self.state.device)),dim=-1)
                # get bert embedding with grad
                out = bert(inputs_embeds = out)[0]
                out.unsqueeze_(1)

        # # take average across sequence length
        # out = torch.mean(out, dim=2)
        # return self.newfc(out.squeeze(1))
        # out = self.dropout(out)
        conved = [F.relu(conv(out)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        return self.fc(cat)


class TextConvNet_BERT(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextConvNet_BERT, self).__init__()
        #if state.textdata:
        ntoken=state.ntoken
        n_filters = 100
        filter_sizes = [3,4,5]
        dropout=0.5
        output_dim=1 if state.num_classes == 2 else state.num_classes

        # frozen multi lingual bert
        # self.bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.state.device)
        # embedding_dim = self.bert.config.hidden_size
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # self.bert.eval()

        embedding_dim = 768
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        # self.fc2 = nn.Linear(output_dim, output_dim)
        # self.fc3 = nn.Linear(output_dim, output_dim)
        # self.fc4 = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # self.new_fc = nn.Linear(embedding_dim, output_dim)
        
        self.sigm=nn.Sigmoid()
        self.distilling_flag=False

    def forward(self, x):
        if self.state.textdata and not self.distilling_flag:
                # initialize multi lingual bert 
                # bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(self.state.device)
                # # freeze bert
                # for param in bert.parameters():
                #     param.requires_grad = False
                # bert.eval()
                # get bert embedding
                with torch.no_grad():
                    out = bert(x)[0]
                out.unsqueeze_(1)
        else:
                out=x
        # out = self.dropout(out)

        conved = [F.relu(conv(out)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        return self.fc(cat)
        # out = torch.mean(out, dim=2)
        # return self.new_fc(out.squeeze(1))
class TextConvNet1(utils.ReparamModule):
    supported_dims = set(range(1,20000))
    def __init__(self, state):
        self.state=state
        super(TextConvNet1, self).__init__()
        #if state.textdata:
        embedding_dim=state.ninp #Maybe 32
        ntoken=state.ntoken
        n_filters = 100
        filter_sizes = [3,4,5]
        dropout=0.5
        output_dim=1 if state.num_classes == 2 else state.num_classes
        self.encoder = nn.Embedding(ntoken, embedding_dim)
        self.encoder.weight.data.copy_(state.pretrained_vec) # load pretrained vectors
        self.encoder.weight.requires_grad = state.learnable_embedding 
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.sigm=nn.Sigmoid()
        self.distilling_flag=False
    def forward(self, x):
        if self.state.textdata and not self.distilling_flag:
                out = self.encoder(x) #* math.sqrt(ninp)
                out.unsqueeze_(1)
                #out=x
                #print(out.size())
                #print(out.size())
        else:
                out=x
        #out = self.dropout(out)
        conved = [F.relu(conv(out)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        
        return self.fc(cat)     
    
class AlexCifarNet(utils.ReparamModule):
    supported_dims = {32}

    def __init__(self, state):
        super(AlexCifarNet, self).__init__()
        assert state.nc == 3
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 384),
            nn.ReLU(inplace=True),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.Linear(192, state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 4096)
        x = self.classifier(x)
        return x


# ImageNet
class AlexNet(utils.ReparamModule):
    supported_dims = {224}

    class Idt(nn.Module):
        def forward(self, x):
            return x

    def __init__(self, state):
        super(AlexNet, self).__init__()
        self.use_dropout = state.dropout
        assert state.nc == 3 or state.nc == 1, "AlexNet only supports nc = 1 or 3"
        self.features = nn.Sequential(
            nn.Conv2d(state.nc, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        if state.dropout:
            filler = nn.Dropout
        else:
            filler = AlexNet.Idt
        self.classifier = nn.Sequential(
            filler(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            filler(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1 if state.num_classes <= 2 else state.num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    

