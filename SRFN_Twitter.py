import numpy as np
import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import BertModel,BertTokenizer
import process_twitter as process_data
import copy
import random
import os
import tensorflow as tf


def set_seed(seed=42):
    """
        set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    tf.random.set_seed(seed)

class ReverseLayerF(Function):
    # @staticmethod
    def forward(self, x):
        self.lambd = 1#self.args.lambd
        return x.view_as(x)

    # @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF().forward(x)

# SRFN model
class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()
        self.args = args

        self.event_num = args.event_num
        self.class_num = args.class_num

        self.hidden_size = args.hidden_dim


        # bert
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        self.bertModel = bert_model

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        # hidden_size = args.hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False#屏蔽预训练模型的权重
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)

        # 共享空间和特定空间
        self.shared_zoom = shared_zoom(self.hidden_size)
        self.private_zoom = private_zoom(self.hidden_size)
        self.shared_linear=nn.Linear(self.hidden_size*2,self.hidden_size)

        # Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(3 * self.hidden_size, self.class_num))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # 拼接融合
        self.fusion_model = fusion_model(self.hidden_size, self.hidden_size,self.class_num)

        self.diff_loss = DiffLoss()
        self.shared_loss=MMD()


    def forward(self, text, image,mask=None):
        image = self.dropout(self.vgg(image))  # [N, 512]
        image = F.relu(self.image_fc1(image))

        text = self.dropout(self.bertModel(text).last_hidden_state[:,0,:])
        text = F.relu(self.fc2(text))

        shared_t,shared_v = self.shared_zoom(text, image)
        shared_t,shared_v = self.dropout(F.relu(shared_t)),self.dropout(F.relu(shared_v))
        shared_vetor=torch.cat([shared_t,shared_v], dim=1)
        shared_vetor=self.shared_linear(shared_vetor)

        private_t, private_v = self.private_zoom(text, image)
        private_t, private_v = self.dropout(F.relu(private_t)),self.dropout(F.relu(private_v))

        class_output = self.fusion_model(shared_vetor, private_t, private_v)

        diff_loss = self.diff_loss(private_t, shared_vetor)
        diff_loss += self.diff_loss(private_v, shared_vetor)
        diff_loss += self.diff_loss(private_v, private_t)
        shared_loss = self.shared_loss(shared_t,shared_v)

        return class_output,diff_loss,shared_loss,
        #return middle_h,class_output

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

#shared space
class shared_zoom(nn.Module):
    def __init__(self, hidden_size):
        super(shared_zoom, self).__init__()
        self.shared = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.activate = nn.Sigmoid()
        self.drop = nn.Dropout()

    def forward(self, text_features, image_features):
        shared_t = self.drop(self.bn(self.shared(text_features)))
        shared_v = self.drop(self.bn(self.shared(image_features)))
        return shared_t, shared_v
        # return shared_vector

# specific space
class private_zoom(nn.Module):
    def __init__(self, hidden_size):
        super(private_zoom, self).__init__()
        self.private_zoom_t = nn.Linear(hidden_size, hidden_size)
        self.private_zoom_v = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.drop = nn.Dropout()
        self.activate = nn.Sigmoid()

    def forward(self, text_features, image_features):
        private_t = self.drop(self.bn(self.private_zoom_t(text_features)))
        private_v = self.drop(self.bn(self.private_zoom_v(image_features)))

        return private_t, private_v

# Attention
class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SelfAttention, self).__init__()
        self.W = nn.Linear(input_size, hidden_size, True)
        self.u = nn.Linear(hidden_size, 1)

    def forward(self, x):
        u = torch.tanh(self.W(x))
        a = F.softmax(self.u(u), dim=1)
        x = a.mul(x)
        return x

# fusion network
class fusion_model(nn.Module):
    def __init__(self, start_size, hidden_size, out_dim):
        super(fusion_model, self).__init__()
        self.bn = nn.BatchNorm1d(start_size*3)
        self.linear_1 = nn.Linear(start_size * 3, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout()
        self.att = SelfAttention(hidden_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=start_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, shared_vetor,private_t,private_v):
        h = torch.stack((shared_vetor,private_t,private_v), dim=0)

        h = h.permute(1,0,2)
        h = h.split(1, dim=0)
        h = [self.att(e).squeeze(dim=1) for e in h]
        h = torch.cat(h, dim=0)
        h = h.view(h.shape[0],-1)

        h = F.relu(h)
        h= self.dropout(self.linear_1(h))
        h = torch.softmax(self.linear_2(h), dim=-1)
        return h


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
def to_np(x):
    return x.data.cpu().numpy()


def load_data(args):
    train, validate, test, event_num = process_data.get_data(args.text_only)
    args.event_num = event_num
    re_tokenize_sentence(train)
    re_tokenize_sentence(validate)
    re_tokenize_sentence(test)
    all_text = get_all_text(train, validate, test)
    tmp = np.array([len(x) for x in all_text])
    max_len = int(np.mean(tmp))+1
    print(max_len)
    args.sequence_len = max_len
    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    return train, validate, test

class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print('TEXT: %d, Image: %d, label: %d, Event: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]

#text processing
def re_tokenize_sentence(flag):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)
        tokenized_texts.append(tokenized_text)
    flag['post_text'] = tokenized_texts


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    return all_text


def align_data(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0

        if len(sentence)>=args.sequence_len:
            sen_embedding = sentence[:args.sequence_len]
        else:
            sen_embedding = sentence+[0 for x in range(args.sequence_len-len(sentence))]


        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text
    flag['mask'] = mask

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class MMD(nn.Module):

    def __init__(self):
        super(MMD, self).__init__()

    def forward(self, inital_1, inital_2):
        share_loss = self.mmd_rbf(inital_1, inital_2)
        return share_loss

    def guassian_kernel(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        Params:
    	    source: source domain data（n * len(x))
    	    target: target domain data（m * len(y))
    	    kernel_mul:
    	    kernel_num: The number of different Gaussian kernels used
    	    fix_sigma: The sigma values of difference Gaussian kernels
    	Return:
    		sum(kernel_val): The sum of multiple kernel matrices
        '''
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)  # /len(kernel_val)

    def mmd_rbf(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        Compute the MMD distance between the kernel of source domain data and target domain data.
        Params:
    	    source: source domain data（n * len(x))
    	    target: target domain data（m * len(y))
    	    kernel_mul:
    	    kernel_num: The number of different Gaussian kernels used
    	    fix_sigma: The sigma values of difference Gaussian kernels
    	Return:
    		loss: MMD loss
        '''
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss



