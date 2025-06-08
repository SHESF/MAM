import networkx as nx
import math
VE=1;
V=1;
def Read(name):
    global VE,V
    G = nx.Graph()
    f = open(name, "r")
    firstLine=f.readline()
    V=int(firstLine.split(" ")[0])
    #print(VE)
    for i in range(0,int((firstLine.split(" ")[1].split("\n"))[0])):
        #print(f.readline())
        NewEdge=f.readline().split(" ")
        v1=int(NewEdge[0])
        v2=int((NewEdge[1].split("\n")[0]))
        G.add_edge(v1,v2)
    f.close()
    return G

def MAM(DataSet):
    fileName=DataSet+".txt"
    G = Read(fileName)
    G.remove_edges_from(nx.selfloop_edges(G))
    DicKcore = nx.core_number(G)
    SortedKShell = sorted(DicKcore.items(), key=lambda x: x[1], reverse=True)
    MaxShell = SortedKShell[0][1]
    #DicEigen = nx.eigenvector_centrality(G)

    SortedDegree = sorted(G.degree(), key=lambda tup: tup[1], reverse=True)
    MaxDegree = SortedDegree[0][1]
    deg=dict(G.degree())
    sumDegreesOfNeighbors=dict()
    for vertex in deg.keys():
        sum=0
        neighbors=G.neighbors(vertex)
        for u in neighbors:
            sum +=deg[u]
        sumDegreesOfNeighbors[vertex]=sum
    print(sumDegreesOfNeighbors)
    AllMultiIndex=dict()


    dicAtrb1=dict()
    dicAtrb2=dict()
    maxAtrb1=0
    maxAtrb2=0
    for vertex in deg.keys():
        if(DicKcore[vertex]==0):
            Atrb1=0
            Atrb2=0
        else:
            Atrb1 = (sumDegreesOfNeighbors[vertex]) / (DicKcore[vertex] ** 2)
            Atrb2 = deg[vertex] / DicKcore[vertex]

        if maxAtrb1<Atrb1:
            maxAtrb1=Atrb1
        if maxAtrb2<Atrb2:
            maxAtrb2=Atrb2

        dicAtrb1[vertex]=Atrb1
        dicAtrb2[vertex]=Atrb2
    d = dict(nx.all_pairs_shortest_path_length(G, cutoff=2))
    for vertex in d.keys():
        neighborsUntil2 = d[vertex]
        
        neighborsUntil2.pop(vertex, None)
        sum=0
        for neighbor in neighborsUntil2.keys():
            
            sum += ((dicAtrb1[neighbor]/maxAtrb1)+(dicAtrb2[neighbor]/maxAtrb2)+(DicKcore[neighbor]/MaxShell))/(neighborsUntil2[neighbor]**2)
        AllMultiIndex[vertex] = (DicKcore[vertex])*sum
       
    res=dict()
    for vertex in deg.keys():
        neigbors = G.neighbors(vertex)
        c = 0
        for node in neigbors:
            c += AllMultiIndex[node]
        res[vertex] = c
    #sort AllGindex by Gravity value
    all_sort=sorted(res.items(), key=lambda x: x[1], reverse=True)
    top=[]
    global VE,V
    return AllMultiIndex