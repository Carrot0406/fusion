# 验证模型的泛化性，处理其他条件下的文件路径

## 雨天 使用RSF6
## 雪天 使用RSF3
## 晴天高车 使用RSF5
## 晴天sub-6 GHz 使用RSF5

import os

base_roots = {
    'base_root1': '/home/bailu/dataset/Depth_map',
    'base_root2': '/home/bailu/dataset/RGB',
    'base_root3': '/home/bailu/dataset/Lidar'
}

sub_dir = ['rainy_mmWave_mediumVTD','snowy_mmWave_mediumVTD','sunny_mmWave_highVTD','sunny_sub6_mediumVTD']

device = {
    'rainy_mmWave_mediumVTD':'RSF6_train',
    'snowy_mmWave_mediumVTD':'RSF3_train',
    'sunny_mmWave_highVTD':'RSF5_train',
    'sunny_sub6_mediumVTD':'RSF5_train'
}

last_name = {
    'base_root1': 'Depth',
    'base_root2': 'RGB',
    'base_root3': 'Lidar'
}
first_name = {
    'rainy_mmWave_mediumVTD':'rainy',
    'snowy_mmWave_mediumVTD':'snowy',
    'sunny_mmWave_highVTD':'highVTD',
    'sunny_sub6_mediumVTD':'sub6'
}

out_root = '/home/bailu/fusion/dataset/28GHz/split/test'
for directory in sub_dir:
    for i in range(1, 4):
        root = base_roots['base_root{}'.format(i)]  # 根据循环索引获取对应的 base_root 变量
        full_path = '{}/{}'.format(root, directory)  # 拼接完整路径
        txt_name = first_name[directory]+'_'+last_name['base_root{}'.format(i)]+'_test.txt'
        # print(txt_name)
        if i==1:
            full_path = full_path+'/DepthPer'
        # print(full_path)  # 打印结果，你可以在这里进行你想要的操作，比如处理路径或执行某些操作
        path = full_path+'/'+device[directory]
        # print(path)
        files = os.listdir(path)
        
        out_path = os.path.join(out_root,txt_name)
        with open (out_path,'w') as f:
            for j in range(300):
                f.write(os.path.join(path,files[j])+'\n')
        

