import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, AutoModel
from torch.autograd import Function
device = torch.device("cuda")

#SRFN model
class Text_Concat_Vision(torch.nn.Module):

    def __init__(self,
                 model_params
                 ):
        super(Text_Concat_Vision, self).__init__()

        self.text_encoder = TextEncoder(model_params['text_fc_out'],model_params['dropout_p'],model_params['data_type'])
        self.vision_encoder = VisionEncoder(model_params['img_fc_out'],model_params['dropout_p'])

        self.fusion = torch.nn.Linear(
            in_features=(model_params['text_fc_out'] + model_params['img_fc_out']),
            out_features=model_params['fusion_hidden_size']
        )
        self.fc = torch.nn.Linear(
            in_features=model_params['fusion_hidden_size'],
            out_features=model_params["num_class"]
        )
        self.fc_one = torch.nn.Linear(model_params['fusion_hidden_size'], model_params["num_class"])
        self.shared_linear = nn.Linear(model_params['fusion_hidden_size']*2, model_params['fusion_hidden_size'])
        self.dropout = torch.nn.Dropout(model_params['dropout_p'])

        self.shared_zoom = shared_zoom(model_params['shared_zoom_hidden_size'])
        self.private_zoom = private_zoom(model_params['private_zoom_hidden_size'])

        self.fusion_model = fusion_model(model_params['shared_zoom_hidden_size'], model_params['fusion_hidden_size'],
                                         model_params['num_class'])

        self.diff_loss = DiffLoss()
        self.shared_loss = MMD()


    def forward(self, text, image, label=None):
        ## text to Bert
        text_features = self.text_encoder(text[0], text[1])
        ## image to vgg
        image_features = self.vision_encoder(image)

        shared_t,shared_v=self.shared_zoom(text_features, image_features)
        private_t, private_v = self.private_zoom(text_features, image_features)

        shared_vector = torch.cat([shared_t,shared_v], dim=1)
        shared_vector=self.shared_linear(shared_vector)
        prediction = self.fusion_model(shared_vector, private_t, private_v)
        prediction = prediction.float()


        diff_loss = self.diff_loss(private_t, shared_vector)
        diff_loss += self.diff_loss(private_v, shared_vector)
        diff_loss += self.diff_loss(private_v, private_t)

        shared_loss = self.shared_loss(shared_t,shared_v)

        return prediction, diff_loss/3,shared_loss

# Bert
class TextEncoder(nn.Module):

    def __init__(self, text_fc_out, dropout_p=0.4, data_type=None):

        super(TextEncoder, self).__init__()

        # 实例化
        if data_type == 'twitter':
            pre_train_model = 'bert-base-uncased'
            self.bert = BertModel.from_pretrained(
                pre_train_model,
                # output_attentions = True,
                return_dict=True)

        elif data_type == 'weibo':
            pre_train_model = 'bert-base-chinese'
            self.bert = AutoModel.from_pretrained(
                pre_train_model,
                # output_attentions = True,
                return_dict=True)

        self.text_enc_fc2 = torch.nn.Linear(768, text_fc_out)
        self.bn = nn.BatchNorm1d(768, momentum=0.99)
        self.dropout = nn.Dropout(dropout_p)



    def forward(self, input_ids, attention_mask):
        # 输入BERT
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = out.last_hidden_state[:, 0, :]
        out = self.bn(out)
        # out = self.dropout(F.relu(self.text_enc_fc1(out)))
        x = self.dropout(F.relu(self.text_enc_fc2(out)))
        return x

# vgg19
class VisionEncoder(nn.Module):

    def __init__(self, img_fc1_out, dropout_p=0.4):
        super(VisionEncoder, self).__init__()

        # IMAGE bdann图像处理
        vgg_19 = models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features#1000
        self.vis_encoder = vgg_19
        self.vis_enc_fc1 = nn.Linear(num_ftrs, img_fc1_out)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, images):
        """
        :参数: images, tensor (batch_size, 3, image_size, image_size)
        :返回: encoded images
        """

        x = self.vis_encoder(images)

        x = self.dropout(
            torch.nn.functional.relu(
                self.vis_enc_fc1(x))
        )
        return x

# shared space
class shared_zoom(nn.Module):
    def __init__(self, hidden_size):
        super(shared_zoom, self).__init__()
        self.shared_t = nn.Linear(hidden_size, hidden_size)
        self.shared_v = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.activate = nn.Sigmoid()
        self.drop = nn.Dropout()

    def forward(self, text_features, image_features):
        # 共享空间
        shared_t = self.drop(self.bn(self.shared_t(text_features)))
        shared_v = self.drop(self.bn(self.shared_v(image_features)))
        return shared_t, shared_v

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
        # 特定空间self.activate(
        private_t = self.drop(self.bn(self.private_zoom_t(text_features)))
        private_v = self.drop(self.bn(self.private_zoom_v(image_features)))

        return private_t, private_v

# attention
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

# fusion layer
class fusion_model(nn.Module):
    def __init__(self, start_size, hidden_size, out_dim):
        super(fusion_model, self).__init__()
        self.bn = nn.BatchNorm1d(start_size*3)
        self.linear_1 = nn.Linear(start_size*3, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, out_dim)
        self.dropout = nn.Dropout()
        self.att = SelfAttention(hidden_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=start_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 是N个编码器层的堆叠

    def forward(self,shared_vetor, private_t, private_v):
        h = torch.stack((shared_vetor,private_t, private_v), dim=0)

        h = h.permute(1,0,2)
        h = h.split(1, dim=0)
        h = [self.att(e).squeeze(dim=1) for e in h]
        h = torch.cat(h, dim=0)
        h= h.view(h.shape[0],-1)

        h = self.bn(h)
        h = self.dropout(F.relu(self.linear_1(h)))
        h = torch.softmax(self.linear_2(h), dim=-1)
        return h

class ReverseLayerF(Function):

    # @staticmethod
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)

    # @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF().forward(x)

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)  # 统计个数
        mse = torch.sum(diffs.pow(2)) / n

        return mse

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

        # abs_tmp = torch.abs(input1_l2.pow(2) - input2_l2.pow(2))  # 距离
        # diff_ln = -torch.mean(torch.log(abs_tmp))  # ln函数注意负号

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
    	    source: Source domain data (n * len(x)).
            target: Target domain data (m * len(y)).
            kernel_mul:
            kernel_num: Number of different Gaussian kernels to take.
            fix_sigma: Sigma value for different Gaussian kernels.
    	Return:
    		sum(kernel_val): sum of multiple kernel matrices
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
        return sum(kernel_val)

    def mmd_rbf(self,source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        '''
        Calculate the MMD distance between source domain data and target domain data
        Params:
    	    source: Source domain data (n * len(x)).
            target: Target domain data (m * len(y)).
            kernel_mul:
            kernel_num: Number of different Gaussian kernels to take.
            fix_sigma: Sigma value for different Gaussian kernels.
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
