本文件中代码为
2020年数学建模竞赛B题中第二题，第三关的强化学习+深度强化学习的解

'No3_map.py'
是地图文件，题设被当作有向图，地图的邻接矩阵被存入map中


'No3_main_off_policy_life_considering.py'
比较完善的训练版本
训练中天气是每一个episode里都是随机产生一个概率值然后计算的
决策基于【当前位置，当前天气，第几天，剩余水，剩余食物】


'No3_main_off_policy_life_considering_version_2.py'
上述训练的进阶版本：改进了状态空间
在矿山选择的动作有三种：【行进，停留，挖矿】（之前版本在矿山停留==挖矿）
从题设看，理论上讲在矿山里无论天气如何，挖矿挖到最后都是最好的选择，除非自己有生命危险
目前最新的版本训练了100w次，挖矿的选择相比上个版本并没有多很多，但是因为空间增大所以收敛变慢了很多
总的说来训练结果的100w次没有效果非常好


'No3_main_off_policy_life_considering_version_2_plus.py'
针对version_2改进：可以对已经保存的训练数据进行重新加载和重新训练
但是由于超参数设置和epsilon的原因，追加训练效果并不好


'No3_main_off_policy_life_considering_version_3.py'

