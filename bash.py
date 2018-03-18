# coding: utf-8

suffics = ['cha', 'kar' , 'kar_x', 'kar_xy']
f = open('bash.txt', 'wt')

for i in range(1, 3):
    for j in range(1, 5):
        for suf in suffics:
            f.write('cd ~/NN_bot/' + str(i) + 'l/1\*10-' + str(j) + '/' +
                    ' && python pong_bot_' + suf + '.py &' + '\n')
