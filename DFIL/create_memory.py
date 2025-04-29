import pandas as pd
import math
import numpy as np

if __name__ == '__main__':
    csv_path = ''
    df = pd.read_csv("29042025_Task3_DFD_by_all_img_info.csv")

#------------------------------------------------------------------------------------------------ All Center

    # df1 = df[df['image_label']==1]
    # df0 = df[df['image_label']==0]
    # df1 = df1.sort_values('dis2mean',ascending=True)[['image_label','image_info']]   
    # df1[:250].to_csv('20230502_DFD_all_center_1(250)_select-samples.txt',sep=' ',index=0,header=0)
    
    # df0 = df0.sort_values('dis2mean',ascending=True)[['image_label','image_info']]
    # df0[:250].to_csv('20230502_DFD_all_center_0(250)_select-samples.txt',sep=' ',index=0,header=0)
    

    # exit()



#------------------------------------------------------------------------------------------------ All Margin

    # df1 = df[df['image_label']==1]
    # df0 = df[df['image_label']==0]
    # df1 = df1.sort_values('dis2mean',ascending=False)[['image_label','image_info']]   
    # df1[:250].to_csv('20230502_DFD_all_margin_1(250)_select-samples.txt',sep=' ',index=0,header=0)
    
    # df0 = df0.sort_values('dis2mean',ascending=False)[['image_label','image_info']]
    # df0[:250].to_csv('20230502_DFD_all_margin_0(250)_select-samples.txt',sep=' ',index=0,header=0)
    

    # exit()



#------------------------------------------------------------------------------------------------ All Easy
    # df1 = df[df['image_label']==1]
    # df0 = df[df['image_label']==0]
    # df1 = df1.sort_values('image_entropy',ascending=True)[['image_label','image_info']]   
    # df1[:250].to_csv('20230502_DFD_all_easy_1(250)_select-samples.txt',sep=' ',index=0,header=0)
    
    # df0 = df0.sort_values('image_entropy',ascending=True)[['image_label','image_info']]
    # df0[:250].to_csv('20230502_DFD_all_easy_0(250)_select-samples.txt',sep=' ',index=0,header=0)
    

    # exit()


#------------------------------------------------------------------------------------------------ All hard

    # df1 = df[df['image_label']==1]
    # df0 = df[df['image_label']==0]
    # df1 = df1.sort_values('image_entropy',ascending=False)[['image_label','image_info']]   
    # df1[:250].to_csv('20230502_DFD_all_hard_1(250)_select-samples.txt',sep=' ',index=0,header=0)
    
    # df0 = df0.sort_values('image_entropy',ascending=False)[['image_label','image_info']]
    # df0[:250].to_csv('20230502_DFD_all_hard_0(250)_select-samples.txt',sep=' ',index=0,header=0)
    

    # exit()
#------------------------------------------------------------------------------------------------ random
    # np.random.seed(10) #若不设置随机种子，则每次抽样的结果都不一样
    # #按个数抽样，不放回
    # df = df.sample(n=500)
    # df = df[['image_label','image_info']]
    # df.to_csv('20230419_DFD_random_select-samples.txt',sep=' ',index=0,header=0)
    # exit()


#------------------------------------------------------------------------------------------------ 500 hard + 500 center

    df1 = df[df['image_label']==1]
    df0 = df[df['image_label']==0]
    df1 = df1.sort_values('image_entropy',ascending=False)[['image_label','image_info']]
    df1[:250].to_csv('Memory_Set_New/29042025_DFD_1(250)_sorted_by_entropy.txt',sep=',',index=0,header=0)
    
    df0 = df0.sort_values('image_entropy',ascending=False)[['image_label','image_info']]
    df0[:250].to_csv('Memory_Set_New/29042025_DFD_0(250)_sorted_by_entropy.txt',sep=',',index=0,header=0)
    
    df1 = df[df['image_label']==1]
    df0 = df[df['image_label']==0]
    df1 = df1.sort_values('dis2mean',ascending=True)[['image_label','image_info']]
    df1[:250].to_csv('Memory_Set_New/29042025_DFD_1(250)_sorted_by_dis2mean.txt',sep=',',index=0,header=0)
    
    df0 = df0.sort_values('dis2mean',ascending=True)[['image_label','image_info']]
    df0[:250].to_csv('Memory_Set_New/29042025_DFD_0(250)_sorted_by_dis2mean.txt',sep=',',index=0,header=0)
    
    print("Memory set Created successfully.")

    exit()

#------------------------------------------------------------------------------------------------ 
