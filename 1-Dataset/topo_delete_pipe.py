'''delete pipeline between two nodes and merge nodes'''
import numpy as np
import pandas as pd

def delete_pipe(path,pipe_to_remove,new_node_code,delete_source_sink):
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
    #find nodes of pipeline to delte
    pipe_index=np.where(files['pipe']['编号']==pipe_to_remove)[0][0]
    node1,node2=files['pipe'].loc[pipe_index,'口1连接元件编号'],files['pipe'].loc[pipe_index,'口2连接元件编号']
    #find node index
    node1_index,node2_index=np.where(nodes['编号']==node1)[0][0],np.where(nodes['编号']==node2)[0][0]
    node1_attr,node2_attr=nodes.loc[node1_index,'attr'],nodes.loc[node2_index,'attr']
    node1_file_index,node2_file_index=np.where(files[node1_attr]['编号']==node1)[0][0],np.where(files[node2_attr]['编号']==node2)[0][0]
    if (node1_attr=='source' or node1_attr=='sink' or node2_attr=='source' or node2_attr=='sink') and delete_source_sink==False:
        print('source&sink are not allowed to remove, declined pipe code:',pipe_to_remove)
    else:
        #find pipe code of nodes
        pipe_list=[]
        for i in range(1,5):
            for node_attr,node_file_index in zip([node1_attr,node2_attr],[node1_file_index,node2_file_index]):
                try:
                    pipe_connected=files[node_attr].loc[node_file_index,'口'+str(i)+'连接元件编号']
                    pipe_connected_kou=files[node_attr].loc[node_file_index,'口'+str(i)+'连接元件接口编号']
                    pipe_connected_index=np.where(files['pipe']['编号']==pipe_connected)[0][0]
                    if pipe_connected!=pipe_to_remove:
                        pipe_list.append([pipe_connected,pipe_connected_kou,pipe_connected_index])
                except Exception as e:
                    pass
                finally:
                    pass
        #merge pipelines up to four
        if len(pipe_list)<=4:
            if len(pipe_list)==1:#two-way&valve+source&user&plug:plug
                files['plug']=files['plug'].append({
                '编号':new_node_code,
                '名称':str(new_node_code),
                '口1连接元件编号':pipe_list[0][0],
                '口1连接元件接口编号':pipe_list[0][1]},ignore_index=True)
            if len(pipe_list)==2:#two-way&valves+two-way&valve,tee+source&user&plug:twoway
                files['two']=files['two'].append({
                '编号':new_node_code,
                '名称':str(new_node_code),
                '口1连接元件编号':pipe_list[0][0],
                '口1连接元件接口编号':pipe_list[0][1],
                '口2连接元件编号':pipe_list[1][0],
                '口2连接元件接口编号':pipe_list[1][1]},ignore_index=True)
            if len(pipe_list)==3:#source&user&plug+cross,two-way+tee:tee 
                files['tee']=files['tee'].append({
                '编号':new_node_code,
                '名称':str(new_node_code),
                '口1连接元件编号':pipe_list[0][0],
                '口1连接元件接口编号':pipe_list[0][1],
                '口2连接元件编号':pipe_list[1][0],
                '口2连接元件接口编号':pipe_list[1][1],
                '口3连接元件编号':pipe_list[2][0],
                '口3连接元件接口编号':pipe_list[2][1]},ignore_index=True)
            if len(pipe_list)==4:#two-way+cross,tee+tee:cross
                files['cross']=files['cross'].append({
                '编号':new_node_code,
                '名称':str(new_node_code),
                '口1连接元件编号':pipe_list[0][0],
                '口1连接元件接口编号':pipe_list[0][1],
                '口2连接元件编号':pipe_list[1][0],
                '口2连接元件接口编号':pipe_list[1][1],
                '口3连接元件编号':pipe_list[2][0],
                '口3连接元件接口编号':pipe_list[2][1],
                '口4连接元件编号':pipe_list[3][0],
                '口4连接元件接口编号':pipe_list[3][1]},ignore_index=True)
            for i in range(len(pipe_list)):#update nodes
                files['pipe'].loc[pipe_list[i][2],'口'+str(pipe_list[i][1])+'连接元件编号']=new_node_code
                files['pipe'].loc[pipe_list[i][2],'口'+str(pipe_list[i][1])+'连接元件接口编号']=i+1
            #delete original pipelines and nodes
            files['pipe']=files['pipe'].drop(pipe_index)
            if node1_attr==node2_attr:
                files[node1_attr]=files[node1_attr].drop([node1_file_index,node2_file_index])
            else:
                files[node1_attr],files[node2_attr]=files[node1_attr].drop(node1_file_index),files[node2_attr].drop(node2_file_index)

            for name in files.keys():
                files[name].to_csv(path+'netcsv\\'+files_name[name]+'.csv',index=False,encoding=encoding)
            print('removed pipe code:',pipe_to_remove)
        else:
            print('merging pipes overnumbered 4, declined pipe code:',pipe_to_remove)

path='E:\\0-Work\\Data\\9-stcleak2\\'


new_node_code=10000
for _ in range(2):
    encoding='GBK'
    pipe=pd.read_csv(path+'netcsv\\1.PipeData.csv',header=0,encoding=encoding)
    pipe_index=np.where(pipe['长度']<=25)[0]#pipe list
    pipe_remove_list=pipe.loc[pipe_index,'编号'].tolist()
    for pipe_to_remove in pipe_remove_list:
        delete_pipe(path=path,pipe_to_remove=pipe_to_remove,new_node_code=new_node_code,delete_source_sink=False)
        new_node_code+=1