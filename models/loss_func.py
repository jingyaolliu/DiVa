
import torch
from arguments import args
#import nn
from torch import nn

bce_loss=torch.nn.BCELoss()

class FocalLossBCE(nn.Module):

    def __init__(self,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossBCE, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.crit = torch.nn.BCELoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossBCE()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        bce_losses=self.crit(logits, label)
        coeff = torch.abs(label - logits).pow(self.gamma)
        loss = bce_losses * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def crossentropy_loss(input):

    loss = -torch.log(input)
    return loss

# sigmoid_loss
def sigmoid_loss(input, reduction='elementwise_mean'):
    # y must be -1/+1
    # NOTE: torch.nn.functional.sigmoid is 1 / (1 + exp(-x)). BUT sigmoid loss should be 1 / (1 + exp(x))
    loss = torch.sigmoid(-input)
    
    return loss

def PULoss(y_pred, y_true, prior=args.positive_partition, gamma_p=args.gamma_p,gamma_n=args.gamma_n, eps=1e-7,alpha=args.alpha,use_focalloss=False):
    # y_true is -1/1
    #one_u = torch.ones(y_true.size()).cuda()
    one_u = torch.ones(y_true.size())
    positive = (y_true == 1).float().view(-1)
    unlabeled = (y_true == -1).float().view(-1)
    P_size = max(1., torch.sum(positive))
    u_size = max(1. ,torch.sum(unlabeled))
    # sigmoid loss
    y_positive = sigmoid_loss(y_pred).view(-1)
    # y_positive = crossentropy_loss(y_pred).view(-1)
    y_positive_pl=torch.sigmoid(y_pred).view(-1).clamp(eps, 1-eps)
    # sigmoid loss
    y_unlabeled = sigmoid_loss(-y_pred).view(-1)
    # y_unlabeled=crossentropy_loss(-y_pred).view(-1)
    y_unlabeled_pl=torch.sigmoid(-y_pred).view(-1).clamp(eps, 1-eps)
    #evaluate focal loss
    # focal_loss = - (1 - y_positive_pl) ** gamma 
    # positive_risk = (prior * y_positive * positive / P_size).sum()
    #todo with focalloss
    if use_focalloss:
        positive_risk=(alpha*prior*((1-y_positive_pl)**gamma_p)* y_positive * positive/P_size).sum()
        negative_risk_p=-((1-alpha)*prior * ((1-y_unlabeled_pl)**gamma_p)*y_unlabeled * positive / P_size).sum()
        negative_risk_u = ((1-alpha)*((1-y_unlabeled_pl)**gamma_n)*y_unlabeled * unlabeled / u_size).sum()
    else:
        positive_risk=(prior* y_positive * positive/P_size).sum()
        negative_risk_p=-(prior *y_unlabeled * positive / P_size).sum()
        negative_risk_u = (y_unlabeled * unlabeled / u_size).sum()
    # print('positive_risk_fl:{}'.format(positive_risk))
    # print('negative_risk_p_fl:{}, negative_risk_u_fl:{}'.format(negative_risk_p, negative_risk_u))
    # negative_risk = ((unlabeled / u_size - prior * positive / P_size) * y_unlabeled).sum()
    negative_risk=negative_risk_p+negative_risk_u
    print('positive_risk:{},negative_risk:{}------------------positive_num:{},unlabeled_num:{}---{}'.format(positive_risk,negative_risk,P_size,u_size,P_size/(P_size+u_size)))
    if negative_risk>=0:
        return positive_risk + negative_risk, positive_risk+negative_risk
    else:
        return positive_risk, -negative_risk
# test
# y_pred=torch.randn(10,1)
# print(y_pred)
# # sigmoid y_pred
# # y_pred=torch.sigmoid(y_pred)
# y_true=torch.ones(5,1)
# # shuffle y_true
# y_true=torch.tensor([1,-1,1,1,1,-1,1,-1,1,-1])
# y_true=torch.stack([y_true])
# pu_loss= PULoss(y_pred, y_true,0.5)
# print(pu_loss)

# negaitive sampling loss
ns_base_loss=args.ns_base_loss
#debug
# ns_base_loss='bceloss'
def NSLoss(y_pred, y_true,label_popularity_list,alpha_weight=args.ns_alpha_weight):
    '''
        pram :
        alpha_weight : (1+ alpha_weight * y_pred) is weight of y_pred
    '''
    # y_true is -1/1
    #one_u = torch.ones(y_true.size()).cuda()
    one_u = torch.ones(y_true.size())
    positive = (y_true == 1).float().view(-1)
    unlabeled = (y_true == 0).float().view(-1)
    # count non_zero item in label_popularity_list * unlabel
    hard_negative =(label_popularity_list!=0).float().view(-1)
    hard_negative_num_batch = torch.count_nonzero(hard_negative * unlabeled).item()
    hard_negative_pred_mean = ( hard_negative * unlabeled * y_pred.view(-1)).sum() / max(1.,hard_negative_num_batch)
    hard_negative_pred_max = (hard_negative * unlabeled * y_pred.view(-1)).max()
    print('\nhard_negative_num_batch:{},hard_negative_pred_mean:{},hard_negative_pred_max:{}'.format(hard_negative_num_batch,hard_negative_pred_mean,hard_negative_pred_max))
    
    P_size = max(1., torch.sum(positive))
    u_size = max(1. ,torch.sum(unlabeled))
    
    predict_loss_weight=alpha_weight * positive * y_true.view(-1) + 1
    #todo sigmoid loss    
    # sigmoid loss
    if ns_base_loss == 'sigmoid':
        y_positive = (1-y_pred.view(-1))*positive
        y_negative = y_pred.view(-1) * unlabeled
        y_positive_predict_loss = (predict_loss_weight * torch.pow(y_positive,2)).sum()
        y_negative_predict_loss = (predict_loss_weight * torch.pow(y_negative,2)).sum()
        # predict_loss= (predict_loss_weight * torch.pow(y_positive,2)).sum()
        predict_loss = y_positive_predict_loss + y_negative_predict_loss
        print('y_positive_predict_loss:{},y_negative_predict_loss:{}'.format(y_positive_predict_loss,y_negative_predict_loss))
        hard_negative_loss = (label_popularity_list * unlabeled * torch.pow(y_positive+y_negative,2)).sum()       
        
    #todo bce_loss
    
    # loss_val=bce_loss(y_pred.view(-1),y_true.view(-1),weight=predict_loss_weight,size_average=False) + (label_popularity_list * unlabeled * y_positive).sum()
    else:
        y_positive =- ( torch.log(1-y_pred.view(-1)) * unlabeled + torch.log(y_pred.view(-1)) * positive )
        bce_loss=nn.BCELoss(weight=predict_loss_weight,size_average=False)
        predict_loss = bce_loss(y_pred.view(-1),y_true.view(-1)) 
        hard_negative_loss =(label_popularity_list * unlabeled * y_positive).sum()
    
    print('predict_loss:{},hard_negative_loss:{}'.format(predict_loss,hard_negative_loss))
    if args.ns_loss_component =='with_hard_negative_loss':
        loss_val = predict_loss + hard_negative_loss
    else:
        loss_val = predict_loss
    
    return loss_val