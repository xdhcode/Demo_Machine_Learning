'''delete node and merge pipelines connecting to node'''
import numpy as np
import pandas as pd

def delete_node(path,node_to_remove,new_pipe_code):
    encoding='GBK'
    source=pd.read_csv(path+'netcsv\\0.SourceData.csv',header=0,encoding=encoding)
    source1=source.copy()
    source1['attr']='source'
    pipe=pd.read_csv(path+'netcsv\\1.PipeData.csv',header=0,encoding=encoding)
    pipe1=pipe.copy()
    pipe1['attr']='pipe'
    two=pd.read_csv(path+'netcsv\\2.TwowaysData.csv',header=0,encoding=encoding)
    two1=two.copy()
    two1['attr']='two'
    tee=pd.read_csv(path+'netcsv\\3.TeesData.csv',header=0,encoding=encoding)
    tee1=tee.copy()
    tee1['attr']='tee'
    cross=pd.read_csv(path+'netcsv\\4.CrossData.csv',header=0,encoding=encoding)
    cross1=cross.copy()
    cross1['attr']='cross'
    plug=pd.read_csv(path+'netcsv\\5.PlugData.csv',header=0,encoding=encoding)
    plug1=plug.copy()
    plug1['attr']='plug'
    sink=pd.read_csv(path+'netcsv\\6.SinkData.csv',header=0,encoding=encoding)
    sink1=sink.copy()
    sink1['attr']='sink'
    valve=pd.read_csv(path+'netcsv\\7.ValveData.csv',header=0,encoding=encoding)
    valve1=valve.copy()
    valve1['attr']='valve'
    #files with 1 for searching, origin files for modifying
    nodes=pd.concat([source1,sink1,valve1,plug1,two1,tee1,cross1],axis=0).reset_index(drop=True)
    files={'source':source,'pipe':pipe,'two':two,'tee':tee,'cross':cross,'plug':plug,'sink':sink,'valve':valve}
    files_name={'source':'0.SourceData','pipe':'1.PipeData','two':'2.TwowaysData','tee':'3.TeesData','cross':'4.CrossData','plug':'5.PlugData','sink':'6.SinkData','valve':'7.ValveData'}
    #find node code
    node_index=np.where(nodes['编号']==node_to_remove)[0][0]
    node_attr=nodes.loc[node_index,'attr']
    node_file_index=np.where(files[node_attr]['编号']==node_to_remove)[0][0]
    #find pipelines to delete
    pipe1,pipe2=nodes.loc[node_index,'口1连接元件编号'],nodes.loc[node_index,'口2连接元件编号']
    pipe1_index,pipe2_index=np.where(files['pipe']['编号']==pipe1)[0][0],np.where(files['pipe']['编号']==pipe2)[0][0]
    pipe1_len,pipe2_len=files['pipe'].loc[pipe1_index,'长度'],files['pipe'].loc[pipe2_index,'长度']
    pipe1_inside,pipe2_inside=files['pipe'].loc[pipe1_index,'内径'],files['pipe'].loc[pipe2_index,'内径']
    new_pipe_len=pipe1_len+pipe2_len
    new_pipe_inside=(pipe1_inside+pipe2_inside)/2
    #find nodes on the other side of pipelines
    node1_pipe_kou=np.where(files['pipe'].loc[pipe1_index,['口1连接元件编号','口2连接元件编号']]!=node_to_remove)[0][0]+1
    node2_pipe_kou=np.where(files['pipe'].loc[pipe2_index,['口1连接元件编号','口2连接元件编号']]!=node_to_remove)[0][0]+1
    #find node code
    node1=files['pipe'].loc[pipe1_index,'口'+str(node1_pipe_kou)+'连接元件编号']
    node2=files['pipe'].loc[pipe2_index,'口'+str(node2_pipe_kou)+'连接元件编号']
    node1_kou=files['pipe'].loc[pipe1_index,'口'+str(node1_pipe_kou)+'连接元件接口编号']
    node2_kou=files['pipe'].loc[pipe2_index,'口'+str(node2_pipe_kou)+'连接元件接口编号']
    #find node index in files
    node1_index,node2_index=np.where(nodes['编号']==node1)[0][0],np.where(nodes['编号']==node2)[0][0]
    node1_attr,node2_attr=nodes.loc[node1_index,'attr'],nodes.loc[node2_index,'attr']
    node1_file_index,node2_file_index=np.where(files[node1_attr]['编号']==node1)[0][0],np.where(files[node2_attr]['编号']==node2)[0][0]
    #change connected pipelines of nodes to new pipelines
    files[node1_attr].loc[node1_file_index,'口'+str(node1_kou)+'连接元件编号']=new_pipe_code
    files[node2_attr].loc[node2_file_index,'口'+str(node2_kou)+'连接元件编号']=new_pipe_code
    new_pipe_kou1=files[node1_attr].loc[node1_file_index,'口'+str(node1_kou)+'连接元件接口编号']
    new_pipe_kou2=files[node2_attr].loc[node2_file_index,'口'+str(node2_kou)+'连接元件接口编号']
    if node1_attr=='sink' or node2_attr=='source':
        new_pipe_kou1,new_pipe_kou2=new_pipe_kou2,new_pipe_kou1
        node1,node2=node2,node1
        node1_kou,node2_kou=node2_kou,node1_kou 
        node1_attr,node2_attr=node2_attr,node1_attr
        node1_file_index,node2_file_index=node2_file_index,node1_file_index
    if new_pipe_kou1==new_pipe_kou2:
        files[node1_attr].loc[node1_file_index,'口'+str(node1_kou)+'连接元件接口编号']=1
        files[node2_attr].loc[node2_file_index,'口'+str(node2_kou)+'连接元件接口编号']=2
    if new_pipe_kou1>new_pipe_kou2:
        new_pipe_kou1,new_pipe_kou2=new_pipe_kou2,new_pipe_kou1
        node1,node2=node2,node1
        node1_kou,node2_kou=node2_kou,node1_kou

    #add merged pipelines
    files['pipe']=files['pipe'].append({
    '编号':new_pipe_code,
    '名称':str(new_pipe_code),
    '内径':new_pipe_inside,
    '壁厚':0.01,
    '长度':new_pipe_len,
    '粗糙度':0.0001,
    '口1连接元件编号':node1,
    '口1连接元件接口编号':node1_kou,
    '口1高度':0,
    '口2连接元件编号':node2,
    '口2连接元件接口编号':node2_kou,
    '口2高度':0,
    '热损':0},ignore_index=True)
    #delete original node&pipe
    files['pipe']=files['pipe'].drop([pipe1_index,pipe2_index])
    files[node_attr]=files[node_attr].drop(node_file_index)

    for name in files.keys():
        files[name].to_csv(path+'netcsv\\'+files_name[name]+'.csv',index=False,encoding=encoding)
    print('removed node code:',node_to_remove)

path='E:\\0-Work\\Data\\9-stcleak2\\'

new_pipe_code=10000
while True:
    encoding='GBK'
    two=pd.read_csv(path+'netcsv\\2.TwowaysData.csv',header=0,encoding=encoding)
    valve=pd.read_csv(path+'netcsv\\7.ValveData.csv',header=0,encoding=encoding)
    nodes=pd.concat([two,valve],axis=0).reset_index(drop=True)#node list
    if len(nodes)==0:
        break
    else:
        node_remove_list=nodes['编号'].tolist()
        for node_to_remove in node_remove_list:
            delete_node(path=path,node_to_remove=node_to_remove,new_pipe_code=new_pipe_code)
            new_pipe_code+=1
