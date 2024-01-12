

def read_tsp(dir):
    f = open(dir, "r")
    for i in range(3):
        f.readline()
    dim = int(f.readline().split(":")[1])
    for i in range(2):
        f.readline()

    data = []
    for i in range(dim):
        temp = f.readline().lstrip().split(" ")
        temp2 = []
        temp2.append(float(temp[1]))
        temp2.append(float(temp[2]))
        data.append(temp2)
    return data, dim