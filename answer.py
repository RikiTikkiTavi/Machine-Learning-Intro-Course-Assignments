def answer(list, name):
    f = open(name, 'w')
    space = ''
    for i in list:
        f.write(space + str(i))
        space = ' '