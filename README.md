# Break Captcha

## 如何使用
- 首先安装依赖的包`pip install -r requirements.txt` 需要注意的是，这里使用的是`tensorflow 2.0`, 需要使用较新的`cuda`和`cudnn`。我的环境使用的是`cuda 10` 以及`cudnn 7.5 `。供大家参考。

- 需要在`congig.yaml`文件中配置好相关参数，包括数据集名称、路径，所需的数据预处理操作以及训练使用的模型。

- `python make_dataset.py`创建tfrecord文件，然后执行`python training.py`开始训练。

