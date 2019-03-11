# 说明
本文件说明使用Mxnet中accnn工具加速模型遇到得问题。</br>
1.'params'改为attrs;</br>
2.加速得模型中层名字中结尾'_fwd'去掉，因为层名加'_weight'或'_bias'才是存放参数对应得key（在模型的.json中修改）;<br>
3.先由rank_selection生成config.json，在读取此文件加速模型（中间要根据情况修改此json文件;</br>
4.根据模型的.json文件，将stride大于1的卷积层从config.json中删除（目前没细看加速方法，先这样不去加速这些卷积层）;</br>
5.针对elementwise_add等多操作数op单独处理（默认都按单输入自动生成symbol）;</br>
6.添加了对no-bias卷积层的处理
7.将aux_param中的key值中running改为moving

