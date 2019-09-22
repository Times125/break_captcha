# Break Captcha

## 项目介绍
验证码识别 - 该项目是基于 `CNN5/ResNet+BiLSTM/BiGRU+CTC` 来实现验证码识别。

## 注意事项

1. **如何使用CPU训练：**

	本项目默认安装`TensorFlow-GPU`版，建议使用`GPU`进行训练，如需换用CPU训练请替换 `requirements.txt` 文件中的`tensorflow-gpu==2.0.0b1` 为`tensorflow==2.0.0b1`，其他无需改动。
	需要注意的是，本项目只在`tensorflow-gpu 2.0b1`下经过完整测试，由于`tf2`当前还处于测试阶段，每个版本都有一些变更，比如目前`tensorflow-gpu 2.0rc0` 以及`rc1` 需要修改源码中`tensorflow`的导入方式，详情阅读`tensorflow` 升级文档和`issues`.

2. **关于BiLSTM/BiGRU网络**:

	保证`CNN`得到的`featuremap`输入到`Bi-LSTM`/`Bi-GRU`时的宽度大于等于最大字符数，即`time_step`大于等于最大字符数。最好的情况是保证`time_step` 大于等于最大字符数的1倍，比如你的验证码中最大的字符数位6位，那么需要保证`time_step >= 6`, 最好是`time_step >= 12`

3. **No valid path found 问题解决**：

	在`config.yaml`中修改`model -> resize`的参数，自行调整为合适的值，可以尝试这个较为通用的值：`resize: [150, 50]`。如果你使用`resize: [150, 50]`还是遇到了`No valid path found`问题，可以考虑把图像`resize`到更大的尺寸或者在`config.yaml`中修改`model -> preprocess_collapse_repeated`的参数为`True`。

4. **参数修改：**

	切记，如果修改了训练参数如：`ImageWidth，ImageHeight，Resize，CharSet，CNNNetwork，RecurrentNetwork，HiddenNum` 这类影响计算图的参数，需要删除`checkpoint`, `tensorboard`路径下的旧文件，重新训练。本项目默认支持断点续练。

## 准备工作
如果你准备使用`GPU`训练，请先安装`CUDA`和`cuDNN`。需要注意的是，这里使用的是`tensorflow 2.0`, 需要使用较新的`CUDA`和`cuDNN`。我的环境使用的是`cuda 10` 以及`cudnn 7.6.0.64 `。供大家参考。

## 如何使用
1.	首先下载安装`Anoconda`;
2.	然后使用`conda` 创建一个名为`captcha`的新虚拟环境`conda create -n captcha python=3.6.8`；
3.	激活此虚拟环境`conda activate captcha`. (Windows版本激活环境命令是`activate captcha`)；
4.	安装依赖的包`pip install -r requirements.txt`；
5.	在`congig.yaml`文件中配置好相关参数，包括数据集名称、路径，所需的数据预处理操作以及训练使用的模型；
6.	运行脚本`make_dataset.py`创建所需要的`tfrecord`文件；
7.	将所有图片数据转换为`tfrecord`文件后，就可以开始训练了，运行脚本`training.py`开始训练。
8.	模型的训练过程结果都记录在`tensorboard logfile`中，通过运行`tensorboard --logdir=tensorboard/your_dataset_name --host=127.0.0.1`即可可视化训练结果；
9.  训练要有耐心，如果图片较多，一般要训练几百个epoch才可能看到明显的效果提升。
10. 建议先使用较大的学习率进行初步训练，等准确率开始提升，模型收敛到一定程度，然后再使用较小的学习率，这样能加速得到结果。
