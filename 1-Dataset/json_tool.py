import os
import json
import pandas as pd
'''Simulation setting'''
class JST():
    def __init__(self):
        self.config_leak={
                "config_id": "059462bd-a0fd-4274-9750-0943c1e5316311",
                "project_id": "059462bd-a0fd-4274-9750-0943c1e5316322",
                "topology_id": "059462bd-a0fd-4274-9750-0943c1e5316333",
                "CalcList": [{
                    "pipelenlimit": 1.0,
                    "timeStep": 0.1,
                    "minP": 0.001,
                    "threadCount": 1,
                    "errPLimit": 0.01,
                    "errMLimit": 0.001,
                    "loopCount": 10000000,
                    "pInit": 1000000,
                    "tInit": 300
                    }],
                "PipeLeakageList":[] ,  
                "TwowaysLeakageList": [],
                "TeesLeakageList": [],
                "CrossLeakageList": []
                }
    def csv2json(self):
        tp,kou=["tp_code","tp_id"],[["conn_code_"+str(i),"conn_port_"+str(i)] for i in [1,2,3,4]]
        norm=tp+kou[0]+kou[1]
        pump=["a2","a1","a0","rated_fre","current_fre"]
        name={
            "SourceList":tp+["calc_type"]+kou[0]+["P","V","T"],
            "SinkList":tp+["calc_type"]+kou[0]+["P","V","T"],
            "PipeList":tp+["inside","wall_thickness","length","roughness"]+kou[0]+["tp_height_1"]+kou[1]+["tp_height_2","heat_loss"],
            "PlugList":tp+kou[0],
            "TwowaysList":norm,
            "TeesList":norm+kou[2],
            "CrossList":norm+kou[2]+kou[3],
            "ValveList":norm+["on_off"],
            "ValvePRList":norm+["dp"],
            "ValveRLList":norm+["d100","cv","open_ness"],
            "ValveLineList":norm+["d100","kvs","rv","open_ness"],
            "ValveEPList":norm+["d100","kvs","rv","open_ness"],
            "PumpOLList":norm+pump,
            "PumpOLGList":norm+pump+["pump_num"],
            "PumpSPList":norm+pump+["cpp_p"],
            "PumpSPGList":norm+pump+["cpp_p","pump_num"]
            }
        InputJson = {"calc_id":"calc_id","project_id":"project_id","topology_id":"topology_id","data_time":"data_time",
                    "SourceList":[],"SinkList":[],"PipeList":[],
                    "PlugList":[],"TwowaysList":[],"TeesList":[],"CrossList":[],
                    "ValveList":[],"ValvePRList":[],"ValveRLList":[],"ValveLineList":[],"ValveEPList":[],
                    "PumpOLList":[],"PumpOLGList":[],"PumpSPList":[],"PumpSPGList":[]}
        for key in self.csv_list.keys():
            topoPDT=pd.read_csv(self.csv_path+self.csv_list[key]+'.csv')
            topoPDT.columns=name[key]
            if key in ["ValveList"]:
                topoPDT["on_off"]=topoPDT["on_off"].astype(int)
            InputJson[key] = topoPDT.apply(lambda row: dict(zip(topoPDT.columns, row)), axis=1).tolist()
        return  InputJson
    
    def save_json(self,j):
        with open(self.main_path+'csv.json','w') as f:
            json.dump(j,f,indent=4)
        print('csv.json saved')
        
    def json2csv(self, info0, path):
        if info0["state"] == 1:
            if not os.path.exists(path):
                os.mkdir(path)
            for key,value in self.csv_list.items():
                pd.DataFrame(info0[key]).to_csv(path+value+'Result.csv',index=False)
        else:
            print('state error code:',info0["state"])
    
    def json2csv_user(self, info0, path):
        if info0["state"] == 1:
            if not os.path.exists(path):
                os.mkdir(path)
            pd.DataFrame(info0['SinkList']).to_csv(path+'319.SourceSinkResult.csv',index=False)
        else:
            print('state error code:',info0["state"])