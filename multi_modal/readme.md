## 介绍
TransGPT-MM-V0是基于开源的清华[VisualGLM-6B](https://github.com/THUDM/VisualGLM-6B)为backbone，在交通领域数据集上进行微调。

由[SwissArmyTransformer](https://github.com/THUDM/SwissArmyTransformer)(简称sat) 库训练，这是一个支持Transformer灵活修改、训练的工具库，支持Lora、P-tuning等参数高效微调方法。

## 环境配置
```
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
# 最好再安装最新版的sat
git clone https://github.com/THUDM/SwissArmyTransformer
cd SwissArmyTransformer
pip install .
```

## 微调
```
bash finetune/finetune_visualglm.sh
```

目前支持三种方式的微调：

LoRA：样例中为全部28层加入了rank=32的LoRA微调，可以根据具体情景和数据量调整--layer_range和--lora_rank参数。
QLoRA：如果资源有限，可以考虑使用bash finetune/finetune_visualglm_qlora.sh，QLoRA将ChatGLM的线性层进行了4-bit量化，只需要9.8GB显存即可微调。
P-tuning：可以将--use_lora替换为--use_ptuning，不过不推荐使用，除非模型应用场景非常固定。
