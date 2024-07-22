import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import TensorDataset,DataLoader
import random
'''ANN classifier'''
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#ANN
class ANN(nn.Module):
    def __init__(self,indim,hiddim,outdim):
        super(ANN,self).__init__()
        self.ann=nn.Sequential(nn.BatchNorm1d(indim,momentum=0.5),
                               nn.Linear(indim,hiddim),
                               nn.LeakyReLU(),
                               nn.BatchNorm1d(hiddim,momentum=0.5),
                               nn.Linear(hiddim,hiddim),
                               nn.LeakyReLU(),
                               nn.BatchNorm1d(hiddim,momentum=0.5),
                               nn.Linear(hiddim,outdim)
                               )
    def forward(self,x):
        x = self.ann(x)
        return x
#train
class ML():
    def __init__(self,**params):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs=params['epochs']
        self.hs=params['hiddim']
        self.model=ANN(params['indim'],params['hiddim'],params['outdim']).to(device)
        self.loss=nn.CrossEntropyLoss().to(device)
        self.opt=AdamW(self.model.parameters(),lr=params['learning_rate'],weight_decay=params['weight_decay'])
        self.sch=LambdaLR(self.opt,lr_lambda=lambda epoch:1/(epoch/100+1))

        self.set_name=params['set_name']
        self.test_set=params['test_set']
        self.x_train=params['x_train']
        self.y_train=params['y_train']
        self.x_test=params['x_test']
        self.y_test=params['y_test']

        self.x_train_tensor=torch.from_numpy(self.x_train).float().to(device)
        self.y_train_tensor=torch.from_numpy(self.y_train).to(device)
        self.x_test_tensor=torch.from_numpy(self.x_test).float().to(device)
        self.y_test_tensor=torch.from_numpy(self.y_test).to(device)

        train_data=TensorDataset(self.x_train_tensor,self.y_train_tensor.long().squeeze())
        self.loader=DataLoader(dataset=train_data,batch_size=params['batch_size'],shuffle=True,num_workers=0)
        
    def set_path(self,path,name):
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path+'model'):
            os.makedirs(path+'model') 
        self.main_path=path
        self.name=name

    def load(self):
        self.model.load_state_dict(torch.load(self.main_path+'model\\'+self.name+'.pth'))
        self.opt.load_state_dict(torch.load(self.main_path+'model\\'+self.name+'_opt.pth'))

    def train(self,draw=False):
        self.best_acc=0
        epoch1,acc,error=[],[],[]
        for epoch in range(self.epochs):
            for step,(batch_x,batch_y) in enumerate(self.loader):
                self.model.train()
                pred=self.model(batch_x)
                loss=self.loss(pred,batch_y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.sch.step()
            if (epoch+1)%50==0:
                test_acc=self.score(self.x_test_tensor,self.y_test_tensor)   
                train_acc=self.score(self.x_train_tensor,self.y_train_tensor)  
                if test_acc>=self.best_acc:
                    self.best_acc=test_acc 
                    torch.save(self.model.state_dict(),self.main_path+'model\\'+self.name+'.pth')
                    torch.save(self.opt.state_dict(), self.main_path+'model\\'+self.name+'_opt.pth')
                    _,_,self.top3_acc=self.top_n(self.x_test_tensor,self.y_test_tensor,3)
                    _,_,self.top5_acc=self.top_n(self.x_test_tensor,self.y_test_tensor,5)
                    print('epoch:',epoch+1,'loss:',round(loss.item(),4),'train_acc:',round(train_acc,3),'top1_acc:',round(self.best_acc,3),'top3_acc:',round(self.top3_acc,3),'top5_acc:',round(self.top5_acc,3))
                else:
                    print('epoch:',epoch+1,'loss:',round(loss.item(),4),'train_acc:',round(train_acc,3),'test_acc:',round(test_acc,3))
                epoch1.append(epoch)
                acc.append(test_acc)
                error.append(loss.item())
            
                record=pd.DataFrame(np.array([epoch1,acc,error]).T)
                record.columns=['epoch','acc','error']
                record.to_csv(self.main_path+'train_record_'+self.name+'.csv',index=False)

        if draw:
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('epochs')
            ax1.set_ylabel('accuracy', color=color)
            ax1.plot(epoch1, acc, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            color = 'tab:blue'
            ax2 = ax1.twinx() 
            ax2.set_ylabel('loss', color=color)
            ax2.plot(epoch1, error, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout() 
            plt.title('best_acc:%.4f'%self.best_acc)
            plt.show()

    def run_model(self,x_tensor):
        self.model.eval()
        torch.no_grad()
        pred=self.model(x_tensor)
        return pred.detach()
    
    def score(self,x_tensor,label_tensor):
        y=self.run_model(x_tensor)
        y=torch.argmax(y,1).unsqueeze(1)
        score=torch.mean((y==label_tensor).to(float)).item()
        return score
    
    def top_n(self,x_tensor,label_tensor,n):#top-n accuracy
        y=self.run_model(x_tensor)
        y=torch.argsort(y,dim=1,descending=True)[:,:n]
        mask=torch.zeros(len(label_tensor))
        for i in range(len(mask)):
            if torch.isin(label_tensor[i],y[i]):
                mask[i]=1
        score=torch.mean(mask).item()
        duicuo=mask.cpu().numpy().reshape([-1,1])
        duicuo=pd.DataFrame(duicuo.astype(int))
        return y.cpu().numpy(),duicuo,score

    def output(self):#save prediction result
        test_acc=self.score(self.x_test_tensor,self.y_test_tensor)
        print('test_acc:',test_acc)
        print('top-n running')
        _,duicuo2,_=self.top_n(self.x_test_tensor,self.y_test_tensor, 2)
        _,duicuo3,_=self.top_n(self.x_test_tensor,self.y_test_tensor, 3)
        _,duicuo4,_=self.top_n(self.x_test_tensor,self.y_test_tensor, 4)
        top,duicuo5,_=self.top_n(self.x_test_tensor,self.y_test_tensor, 5)
        
        top_result=pd.DataFrame(top)
        top_name=['Top'+str(i) for i in range(1,6)]

        duicuo=np.ones([len(top[:,0]),1])
        duicuo[np.where(top[:,0].ravel()!=self.y_test.ravel())]=0
        duicuo=pd.DataFrame(duicuo.astype(int))

        self.out=pd.concat([self.test_set,top_result,duicuo,duicuo2,duicuo3,duicuo4,duicuo5],axis=1)
        self.out.columns=self.test_set.columns.to_list()+top_name+['Top1_Bool','Top2_Bool','Top3_Bool','Top4_Bool','Top5_Bool']
        self.out.to_csv(self.main_path+'ml\\result_'+self.name2+'_topn.csv',index=False)

        label_list=list(range(np.max(self.out['Label']+1)))
        temp=[]
        for l in label_list:
            ind=np.where(self.out['Label']==l)[0]
            if len(ind)>0:
                df=self.out.iloc[ind]
                temp.append([l,np.sum(df['Top1_Bool'])/len(df),np.sum(df['Top2_Bool'])/len(df),np.sum(df['Top3_Bool'])/len(df),np.sum(df['Top4_Bool'])/len(df),np.sum(df['Top5_Bool'])/len(df)])
            elif len(ind)==0:
                temp.append([l,-1,-1,-1,-1,-1,-1])
        df=pd.DataFrame(temp)
        df.columns=['Label','Top1_Acc','Top2_Acc','Top3_Acc','Top4_Acc','Top5_Acc']
        df.to_csv(self.main_path+'ml\\result_'+self.name2+'_topn_pipe.csv',index=False)
        print('output done!')
