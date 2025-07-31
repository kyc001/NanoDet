git init
git add README.md
git commit -m "first commit"
git branch -M master
git remote add origin git@github.com:kyc001/NanoDet.git
git push -u origin master


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 


# 安装兼容NumPy 1.x的版本（示例版本，可根据实际情况调整）
pip install "numpy<2.0" "opencv-python<4.9" "Pillow<10.1" -i https://pypi.tuna.tsinghua.edu.cn/simple 

pip install "numpy<2.0" "opencv-python<4.9" "Pillow<10.1" -i https://pypi.tuna.tsinghua.edu.cn/simple 


pip install lxml  -i https://pypi.tuna.tsinghua.edu.cn/simple 


模型: NanoDetPlus，4.2M参数 (8.4M包含EMA)
数据集: VOC2007，5011训练图片，4952验证图片
分辨率: 320×320
批次大小: 32 (显存使用2.66GB)
混合精度: 16bit AMP
总轮数: 20 epochs


所以我们现在是在进行pytorch版本模型对于VOC数据集的微调是吗（如果不是的话epoches太少了），接下来是用jittor加载模型初始权重用一样的训练参数也进行20轮微调？


[NanoDet][07-31 21:18:05]INFO:
| class       | AP50   | mAP   | class       | AP50   | mAP   |
|:------------|:-------|:------|:------------|:-------|:------|
| aeroplane   | 55.9   | 33.2  | bicycle     | 59.7   | 35.8  |
| bird        | 40.3   | 19.5  | boat        | 27.3   | 13.8  |
| bottle      | 10.7   | 4.7   | bus         | 54.9   | 40.1  |
| car         | 60.9   | 40.6  | cat         | 64.7   | 40.2  |
| chair       | 21.1   | 9.5   | cow         | 46.7   | 25.0  |
| diningtable | 48.7   | 29.6  | dog         | 54.4   | 32.2  |
| horse       | 64.6   | 35.6  | motorbike   | 59.0   | 35.2  |
| person      | 60.7   | 31.2  | pottedplant | 24.7   | 10.6  |
| sheep       | 37.7   | 20.3  | sofa        | 39.3   | 24.6  |
| train       | 68.7   | 44.4  | tvmonitor   | 50.1   | 27.9  |
[NanoDet][07-31 21:18:06]INFO:Saving model to workspace/nanodet-plus-m_320_voc/model_best/nanodet_model_best.pth
[NanoDet][07-31 21:18:06]INFO:Val_metrics: {'mAP': 0.2769359509229967, 'AP_50': 0.47515081898881684, 'AP_75': 0.2774884195963169, 'AP_small': 0.02392784434558522, 'AP_m': 0.112777037179842, 'AP_l': 0.37140879055156667}
`Trainer.fit` stopped: `max_epochs=20` reached.



有几个地方确认了吗：
[w 0731 21:30:44.209451 72 grad.cc:81] grads[375] 'head.distribution_project.project' doesn't have gradient. It will be set to zero: Var(9525:1:1:1:i0:o0:s1:n0:g1,float32,head.distribution_project.project,706f1a000)[8,]
jittor加载的权重应该是imageNet预训练权重，不是pytorch训练结束后的权重
jittor文件结构是否与pytorch必要的对齐，是否也能像pytorch一样导入nanodet包实现模块化！
jittor模型架构细节是否与pytorch版本严格一致！
jittor真的能实现100%加载预训练权重了吗？？
jittor训练使用的训练参数和pytorch微调训练使用的100%一致了吗，训练使用的方法也一样吗？！！
jittor训练两轮后 Best mAP: 0.0000，再怎么说也有一点点吧？
修复以上所有问题，实现100%对齐，训练前再调用mcp-feedback-enhanced mcp工具征求我的意见






没有解决问题就不要擅自开始训练，必须征求我同意！！
为什么pytorch版本模型会有[NanoDet][07-31 21:18:05]前缀！
转换权重是为了测试是否模型架构严格对齐，能否实现pytorch和jittor两者权重自由转换！
另外，该实验设计应该有几种测评角度：
1是直接用预训练权重进行测评
2是用pytorch微调后的模型进行测评，如下：
[NanoDet][07-31 21:18:05]INFO:
| class       | AP50   | mAP   | class       | AP50   | mAP   |
|:------------|:-------|:------|:------------|:-------|:------|
| aeroplane   | 55.9   | 33.2  | bicycle     | 59.7   | 35.8  |
| bird        | 40.3   | 19.5  | boat        | 27.3   | 13.8  |
| bottle      | 10.7   | 4.7   | bus         | 54.9   | 40.1  |
| car         | 60.9   | 40.6  | cat         | 64.7   | 40.2  |
| chair       | 21.1   | 9.5   | cow         | 46.7   | 25.0  |
| diningtable | 48.7   | 29.6  | dog         | 54.4   | 32.2  |
| horse       | 64.6   | 35.6  | motorbike   | 59.0   | 35.2  |
| person      | 60.7   | 31.2  | pottedplant | 24.7   | 10.6  |
| sheep       | 37.7   | 20.3  | sofa        | 39.3   | 24.6  |
| train       | 68.7   | 44.4  | tvmonitor   | 50.1   | 27.9  |
[NanoDet][07-31 21:18:06]INFO:Saving model to workspace/nanodet-plus-m_320_voc/model_best/nanodet_model_best.pth
[NanoDet][07-31 21:18:06]INFO:Val_metrics: {'mAP': 0.2769359509229967, 'AP_50': 0.47515081898881684, 'AP_75': 0.2774884195963169, 'AP_small': 0.02392784434558522, 'AP_m': 0.112777037179842, 'AP_l': 0.37140879055156667}
`Trainer.fit` stopped: `max_epochs=20` reached.
3是用jittor微调后的模型进行测评


还有一个验证模型是否对齐的方法就是能不能直接加载imagenet预训练权重进行测评！






[w 0731 21:49:45.892596 44 __init__.py:1645] load parameter conv5.0.weight failed ...
[w 0731 21:49:45.892668 44 __init__.py:1645] load parameter conv5.1.weight failed ...
[w 0731 21:49:45.892703 44 __init__.py:1645] load parameter conv5.1.bias failed ...
[w 0731 21:49:45.892736 44 __init__.py:1645] load parameter conv5.1.running_mean failed ...
[w 0731 21:49:45.892768 44 __init__.py:1645] load parameter conv5.1.running_var failed ...
[w 0731 21:49:45.892800 44 __init__.py:1645] load parameter fc.weight failed ...
[w 0731 21:49:45.892835 44 __init__.py:1645] load parameter fc.bias failed ...
[w 0731 21:49:45.892867 44 __init__.py:1664] load total 282 params, 7 failed

光是测试没有意义啊，我要看到像如下格式指标！
[NanoDet][07-31 21:18:05]INFO:
| class       | AP50   | mAP   | class       | AP50   | mAP   |
|:------------|:-------|:------|:------------|:-------|:------|
| aeroplane   | 55.9   | 33.2  | bicycle     | 59.7   | 35.8  |
| bird        | 40.3   | 19.5  | boat        | 27.3   | 13.8  |
| bottle      | 10.7   | 4.7   | bus         | 54.9   | 40.1  |
| car         | 60.9   | 40.6  | cat         | 64.7   | 40.2  |
| chair       | 21.1   | 9.5   | cow         | 46.7   | 25.0  |
| diningtable | 48.7   | 29.6  | dog         | 54.4   | 32.2  |
| horse       | 64.6   | 35.6  | motorbike   | 59.0   | 35.2  |
| person      | 60.7   | 31.2  | pottedplant | 24.7   | 10.6  |
| sheep       | 37.7   | 20.3  | sofa        | 39.3   | 24.6  |
| train       | 68.7   | 44.4  | tvmonitor   | 50.1   | 27.9  |
现在应该是四个角度
pytorch：imagenet与微调后的性能对比
jittor：imagenet与微调后的性能对比
已实现：pytorch微调后的性能数据！
为什么pytorch能够实现这么整齐的日志系统，是自动实现测评了吗？？，能不能jittor也模仿功能！：
[NanoDet][07-31 21:18:05]INFO:
| class       | AP50   | mAP   | class       | AP50   | mAP   |
|:------------|:-------|:------|:------------|:-------|:------|
| aeroplane   | 55.9   | 33.2  | bicycle     | 59.7   | 35.8  |
| bird        | 40.3   | 19.5  | boat        | 27.3   | 13.8  |
| bottle      | 10.7   | 4.7   | bus         | 54.9   | 40.1  |
| car         | 60.9   | 40.6  | cat         | 64.7   | 40.2  |
| chair       | 21.1   | 9.5   | cow         | 46.7   | 25.0  |
| diningtable | 48.7   | 29.6  | dog         | 54.4   | 32.2  |
| horse       | 64.6   | 35.6  | motorbike   | 59.0   | 35.2  |
| person      | 60.7   | 31.2  | pottedplant | 24.7   | 10.6  |
| sheep       | 37.7   | 20.3  | sofa        | 39.3   | 24.6  |
| train       | 68.7   | 44.4  | tvmonitor   | 50.1   | 27.9  |
jittor能不能也实现！



jittor loading params warnings:
[w 0731 22:11:56.416725 68 __init__.py:1645] load parameter conv5.0.weight failed ...
[w 0731 22:11:56.416798 68 __init__.py:1645] load parameter conv5.1.weight failed ...
[w 0731 22:11:56.416834 68 __init__.py:1645] load parameter conv5.1.bias failed ...
[w 0731 22:11:56.416868 68 __init__.py:1645] load parameter conv5.1.running_mean failed ...
[w 0731 22:11:56.416903 68 __init__.py:1645] load parameter conv5.1.running_var failed ...
[w 0731 22:11:56.416931 68 __init__.py:1645] load parameter fc.weight failed ...
[w 0731 22:11:56.416964 68 __init__.py:1645] load parameter fc.bias failed ...
[w 0731 22:11:56.416998 68 __init__.py:1664] load total 282 params, 7 failed