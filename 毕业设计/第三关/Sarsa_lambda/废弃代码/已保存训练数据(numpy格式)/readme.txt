已经存储的Numpy格式的数据以如下的格式保存

Q[ position, weather, day, next_postion]

读取时注意：
0.X    字样代表“好天气的概率”，即该模型是在好天气的概率为0.X的环境下运行的

Training_    后的数字表示训练的总次数

需要展示数据用：No3_reload.py 文件打开