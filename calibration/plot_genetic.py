


from collections import defaultdict


def main():
    f = open("logs/gen2_4294967294.out", "r")
    lines = f.readlines()
    begs = [f't{i}y{j}' for i in range(2) for j in range(2)]
    itervals = defaultdict(lambda : [])
    for line in lines:
        if line[:4] not in begs:
            continue
        line = line.split(' ')
        tag = line[0]
        rsq = line[2].replace(',','')
        nnz = line[5].replace(',','')
        reg = line[8]
        print(line)
        print(tag,rsq,nnz,reg)   
        itervals[tag]     
        return
    




if __name__ == '__main__':
    main()