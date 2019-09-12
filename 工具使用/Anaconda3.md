# Anaconda3

## 安装

用默认配置即可

## 设置环境变量

D:\devtools\Anaconda3 ： 使得在cmd可以进入python环境

D:\devtools\Anaconda3\Scripts ： 使在cmd中conda可用

## python for vscode

1. 安装扩展Python extension for Visual Studio Code

2. Python: Select Interpreter ：选择Anaconda3

## debug

调试处添加配置即可

## 更新源

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
# Conda Forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/

# msys2
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/

# bioconda
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/

#menpo
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/

# pytorch
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
```

其余参考[Anaconda 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)

## 安装pytorch

首先需要安装cuda，下载后自定义安装，只选中cuda

然后用conda安装

```shell
conda install pytorch torchvision cudatoolkit=10.0
```

## 清理包

```shell
conda clean -p      //删除没有用的包
conda clean -y -all //删除所有的安装包及cache
```

