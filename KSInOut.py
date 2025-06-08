import networkx as nx
import random
def Read_Graph(name):
    G = nx.DiGraph()
    f = open(name, "r")
    firstLine = f.readline()
    VE = firstLine.split(",")
    # print(VE)
    size = int(VE[0])
    Edge = int((VE[1].split("\n"))[0])
    for i in range(0, Edge):
        # print(f.readline())
        NewEdge = f.readline().split(",")
        #print(NewEdge[0])
        v1 = int(NewEdge[0])
        v2 = int((NewEdge[1].split("\n")[0]))
        G.add_edge(v1, v2) # for weighted network: G.add_edge(1, 2, weight=4.7)
    f.close()
    return G

def KS_directed(graph):
    """Computes the k-shell index based on the sum of in-degree and out-degree for each node in a directed graph."""
    if not graph.is_directed():
        raise ValueError("The input graph must be directed.")

    G = graph.copy()  # ایجاد یک کپی از گراف برای جلوگیری از تغییر نسخه اصلی
    k_shell_index = {}  # دیکشنری برای ذخیره مقدار k-shell هر گره
    k = 1  # مقدار اولیه k-shell

    while G.number_of_nodes() > 0:
        # پیدا کردن گره‌هایی که مجموع درجه ورودی و خروجی‌شان کمتر یا مساوی k است
        low_degree_nodes = {node for node in G.nodes() if (G.in_degree(node) + G.out_degree(node)) <= k}

        while low_degree_nodes:
            # اختصاص مقدار k-shell به این گره‌ها
            k_shell_index.update({node: k for node in low_degree_nodes})
            G.remove_nodes_from(low_degree_nodes)
            low_degree_nodes = {node for node in G.nodes() if (G.in_degree(node) + G.out_degree(node)) <= k}

        k += 1  # افزایش مقدار k-shell

    return k_shell_index


def KS_In(graph):
    """Computes the k-shell index based on in-degree for each node in a directed graph."""
    if not graph.is_directed():
        raise ValueError("The input graph must be directed.")

    G = graph.copy()  # ایجاد یک کپی از گراف برای جلوگیری از تغییر نسخه اصلی
    k_shell_index = {}  # دیکشنری برای ذخیره مقدار k-shell هر گره
    k = 1  # مقدار اولیه k-shell

    while G.number_of_nodes() > 0:
        # پیدا کردن گره‌هایی که درجه ورودی آن‌ها کمتر یا مساوی k است
        low_in_degree_nodes = {node for node, degree in G.in_degree() if degree <= k}

        while low_in_degree_nodes:
            # اختصاص مقدار k-shell به این گره‌ها
            k_shell_index.update({node: k for node in low_in_degree_nodes})
            G.remove_nodes_from(low_in_degree_nodes)
            low_in_degree_nodes = {node for node, degree in G.in_degree() if degree <= k}

        k += 1  # افزایش مقدار k-shell

    return k_shell_index

def KS_Out(graph):
    """Computes the k-shell index based on in-degree for each node in a directed graph."""
    if not graph.is_directed():
        raise ValueError("The input graph must be directed.")

    G = graph.copy()  # ایجاد یک کپی از گراف برای جلوگیری از تغییر نسخه اصلی
    k_shell_index = {}  # دیکشنری برای ذخیره مقدار k-shell هر گره
    k = 1  # مقدار اولیه k-shell

    while G.number_of_nodes() > 0:
        # پیدا کردن گره‌هایی که درجه خروجی آن‌ها کمتر یا مساوی k است
        low_out_degree_nodes = {node for node, degree in G.out_degree() if degree <= k}

        while low_out_degree_nodes:
            # اختصاص مقدار k-shell به این گره‌ها
            k_shell_index.update({node: k for node in low_out_degree_nodes})
            G.remove_nodes_from(low_out_degree_nodes)
            low_out_degree_nodes = {node for node, degree in G.out_degree() if degree <= k}

        k += 1  # افزایش مقدار k-shell

    return k_shell_index


import networkx as nx


def KS_directed_both(graph):
    """Computes the k-shell index for a directed graph based on in-degree < l and out-degree < k conditions."""
    if not graph.is_directed():
        raise ValueError("The input graph must be directed.")

    G = graph.copy()  # ایجاد یک کپی از گراف برای جلوگیری از تغییر نسخه اصلی
    k_shell_index = {}  # دیکشنری برای ذخیره مقدار k-shell هر گره
    max_in_degree = max(dict(G.in_degree()).values(), default=0)
    max_out_degree = max(dict(G.out_degree()).values(), default=0)

    # مقداردهی اولیه برای l و k
    for l in range(1, max_in_degree + 1):
        for k in range(1, max_out_degree + 1):
            remaining_nodes = set(G.nodes())  # مجموعه‌ای از گره‌های باقی‌مانده

            while remaining_nodes:
                # پیدا کردن گره‌هایی که شرایط را دارند
                removable_nodes = {node for node in remaining_nodes
                                   if G.in_degree(node) <= l and G.out_degree(node) <= k}

                if not removable_nodes:
                    break

                # حذف گره‌ها و ثبت مقدار k-shell
                k_shell_index.update({node: [k, l] for node in removable_nodes})
                G.remove_nodes_from(removable_nodes)
                remaining_nodes -= removable_nodes

    return k_shell_index


def main(dataset):
    G = Read_Graph(dataset+".txt")
    G.remove_edges_from(nx.selfloop_edges(G))
    V=KS_directed_both(G)
    all_sort= sorted(V.items(), key=lambda x: x[1], reverse=True)
    outlst= {}
    inlst={}
    for i in range(0,len(all_sort)):
        outlst[all_sort[i][0]]=all_sort[i][1][0]
        inlst[all_sort[i][0]]=all_sort[i][1][1]
    out_sort = sorted(outlst.items(), key=lambda x: x[1], reverse=True)
    in_sort = sorted(inlst.items(), key=lambda x: x[1], reverse=True)
    f = open(dataset + "/KSInOut_Out_" + dataset+".txt", "w")
    f.write("id of nodes : value of method\n")
    for i in range(0, len(outlst)):
        f.write(str(out_sort[i][0]) + ":" + str(out_sort[i][1]) + "\n")
    f.close()
    f = open(dataset + "/KSInOut_In_" + dataset + ".txt", "w")
    f.write("id of nodes : value of method\n")
    for i in range(0, len(in_sort)):
        f.write(str(in_sort[i][0]) + ":" + str(in_sort[i][1]) + "\n")
    f.close()

    V = KS_In(G)
    all_sort = sorted(V.items(), key=lambda x: x[1], reverse=True)
    f = open(dataset + "/KSIn_" + dataset + ".txt", "w")
    f.write("id of nodes : value of method\n")
    for i in range(0, len(all_sort)):
        f.write(str(all_sort[i][0]) + ":" + str(all_sort[i][1]) + "\n")
    f.close()

    V = KS_Out(G)
    all_sort = sorted(V.items(), key=lambda x: x[1], reverse=True)
    f = open(dataset + "/KSOut_" + dataset + ".txt", "w")
    f.write("id of nodes : value of method\n")
    for i in range(0, len(all_sort)):
        f.write(str(all_sort[i][0]) + ":" + str(all_sort[i][1]) + "\n")
    f.close()

    V = KS_directed(G)
    all_sort = sorted(V.items(), key=lambda x: x[1], reverse=True)
    f = open(dataset + "/KSFull_" + dataset + ".txt", "w")
    f.write("id of nodes : value of method\n")
    for i in range(0, len(all_sort)):
        f.write(str(all_sort[i][0]) + ":" + str(all_sort[i][1]) + "\n")
    f.close()

main('bison')
main('openflight')
main('domStrauss')
#main('HighschoolWeightLess')
#main('physician')
#main('Cintestinalis')
#main('Faa_routes')
#main('Stelzl')
#main('Uni_email')