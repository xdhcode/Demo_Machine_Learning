import os
import time
import json
import ctypes
import numpy as np
import pandas as pd
import multiprocessing as multi
from json_tool import JST
'''Simulation'''
class HS(JST):
    def __init__(self):
        super().__init__()
        self.thread=20

    def set_path(self,path,i,error,leak):
        self.main_path=path
        self.csv_path=self.main_path+'netcsv\\'
        self.data_path=self.main_path+'result_'+leak+'\\'+error+'\\pipe_'+str(i)+'\\'
        if not os.path.exists(self.main_path+'result_'+leak+'\\'):
                os.mkdir(self.main_path+'result_'+leak+'\\')
        if not os.path.exists(self.main_path+'result_'+leak+'\\'+error+'\\'):
                os.mkdir(self.main_path+'result_'+leak+'\\'+error+'\\')
        if not os.path.exists(self.data_path):
                os.mkdir(self.data_path)
        self.result='\\1.PipeDataResult.csv'

    def set_config(self,csv_list):#network file
        self.csv_list=csv_list
        self.csv_json=self.csv2json()
        # self.save_json(self.csv_json)
        
    def set_cons(self,name):
        self.cons=pd.read_csv(self.main_path+name).fillna('')
        self.cons=np.array(self.cons.astype(str))

    def single_run(self):
        time1=time.time()
        print('simu start')

        config=self.config_leak.copy()
        config["PipeLeakageList"]=[{
        "tp_code": 0,#int(cons[2]),#int
        "tp_id": '0',#cons[2],#str
        "leakage_m": 0.5,#float(cons[7]),#float
        "leakage_loc": 0.5,#float(cons[6])#float
        }]
        csv=self.csv_json.copy()
        #simulation
        simulator = ctypes.WinDLL(self.main_path+"Systemr32.dll")
        simulator.LeakageCalculate.restype = ctypes.c_char_p
        simu_info = json.loads((simulator.LeakageCalculate(json.dumps(csv).encode(),json.dumps(config).encode())).decode())

        print(simu_info["message"])
        self.json2csv(simu_info, self.data_path)
        # self.save_json(simu_info)

        time2=time.time()
        print("simu finished in:",round(time2-time1,3))

    def simu(self,cons):#one thread calculation
        data_path=self.data_path+cons[0]+'\\'
        config=self.config_leak.copy()
        config["PipeLeakageList"]=[{
        "tp_code": int(cons[2]),#int
        "tp_id": str(cons[3]),#str
        "leakage_m": float(cons[6]),#float
        "leakage_loc": float(cons[5])#float
        }]
        csv=self.csv_json.copy()
        scenario=cons[9:].tolist()
        csv['SinkList'] = list(map(lambda dict, value: {**dict, 'V': value}, csv['SinkList'], scenario))

        flag=1
        while flag<=3:
            simulator = ctypes.WinDLL(self.main_path+"Systemr32.dll")
            simulator.LeakageCalculate.restype = ctypes.c_char_p
            simu_info = json.loads((simulator.LeakageCalculate(json.dumps(csv).encode(),json.dumps(config).encode())).decode())
            if simu_info['state'] == 1:
                print('Pipe'+str(cons[2])+"Success: "+str(cons[0]))
                self.json2csv(simu_info, data_path)
                
                if os.path.exists(data_path+self.result):
                    break
                else:
                    print('Recall_'+str(flag)) 
                    time.sleep(0.5)
            else:
                print("Error: "+simu_info['message'])
            flag+=1

    def multi_simu(self,ind):
        print('pooling and ready')
        cons=self.cons[ind,:]
        try:
            pool = multi.Pool(processes=self.thread)
            pool.map(self.simu,cons)
            pool.close()
            pool.join()
        except Exception as e:
            print('Pool error:', e)
        finally:
            pass
    
    def check_pre(self):#check result
        ind=[]
        for i in range(len(self.cons)):
            if not os.path.exists(self.data_path+str(i)+self.result):
                ind.append(i)
        if len(ind)!=0:
            done=False
            print('Empty found')
            return done,ind
        elif len(ind)==0:
            done=True
            ind=list(range(len(self.cons)))
            print('No empty')
            return done,ind
        
    def run(self):
        done,ind=self.check_pre()
        if not done:
            self.multi_simu(ind)
        # while not done:
        #     self.multi_simu(ind)
        #     done,ind=self.check_pre()
            
    def check(self):
        done,ind=self.check_pre()
        if not done:
            return ind
        else:
            return []

if __name__=='__main__':
    name_list=['0100']
    for j in range(len(name_list)):
        error_name=name_list[j]
        for i in range(0,600):
            for leak_name in ['leak','noleak']:
                time1=time.time()
                a=HS()
                path='D:\\0-Data\\9-stcleak4\\'
                a.set_path(path,i,error_name,leak_name)
                a.set_config(csv_list = {
                                        'SourceList' : '0.SourceData',
                                        'PipeList' : '1.PipeData',
                                        'TwowaysList':'2.TwowaysData',
                                        'TeesList' : '3.TeesData',
                                        'CrossList' : '4.CrossData',
                                        'PlugList': '5.PlugData',
                                        'SinkList' : '6.SinkData',
                                        'ValveList': '7.ValveData',
                                        })
                a.set_cons('cons\\constraint_'+error_name+'_'+str(i)+'_'+leak_name+'.csv')
                a.run()
                time2=time.time()
                print(str(i)+" finished in:",round(time2-time1,3))
