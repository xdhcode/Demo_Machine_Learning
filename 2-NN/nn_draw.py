'''draw prediction results based on labels'''
import pandas as pd
import matplotlib.pyplot as plt

path='D:\\0-Data\\9-stcleak3\\'

label_info=pd.read_csv(path+'pipe_label_xy.csv')
x1,y1,x2,y2=label_info['Start_x'].tolist(),label_info['Start_y'].tolist(),label_info['End_x'].tolist(),label_info['End_y'].tolist()

acc_file=pd.read_csv(path+'ml\\result_hid500_stc878_l878_cv0of5_all_pipe.csv')

name_list=['Top1_Acc','Top2_Acc','Top3_Acc','Top4_Acc','Top5_Acc']
for acc_name in name_list:
    acc=acc_file[acc_name].to_list()
    fig, ax = plt.subplots()
    for i in range(len(acc_file)):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color=plt.cm.viridis(acc[i]))
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label=acc_name)
    plt.title('net_'+acc_name)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(path+'pic\\net_'+acc_name+'.png')
    plt.show()