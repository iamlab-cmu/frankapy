import numpy as np
import math

from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    X = np.load('/home/student/Prog/frankapy/examples/data/trial3/final_X.npy', allow_pickle=True)
    Y = np.load('/home/student/Prog/frankapy/examples/data/trial3/final_Y.npy', allow_pickle=True)
    F = np.load('/home/student/Prog/frankapy/examples/data/trial3/final_F.npy', allow_pickle=True)
    edge_def = np.load('/home/student/Prog/frankapy/examples/data/trial3/X_edge_def3.npy', allow_pickle=True) 

    # print(f"-==================== FXY {F.shape}, {X.shape, Y.shape} -==================== ")


    x = []
    y = []
    z = []
    c = []

    data1 = np.zeros((4,360*10))

    nodes_position_list = []
    count = 0

    for i in range (10):

        for pushes in Y:
            
            node = i
            # pushes = nodes x pose [N,7]
            data1[0,count] = pushes[node][0]
            data1[1,count] = pushes[node][1]
            data1[2,count] = pushes[node][2]
            data1[3,count] = node
            # print(f"x,y,z, {data1[0,count], data1[1,count], data1[2,count]} ")
            count = count + 1

    x = data1[0]     
    y = data1[1] 
    z = data1[2] 
    c = data1[3]

    print(f"size of xyz is {x.shape, y.shape, z.shape}")

    # plotting
    fig = plt.figure()
    
    # syntax for 3-D projection
    ax = plt.axes(projection ='3d')
 

    scatter = ax.scatter(x, y, z, c = c, s=2)
    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="branch")

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    # plt.zlabel("z (m)")

    ax.set_title('Tree Displacement in 3D  ')

    #= init pt plot ===
    print(f" size of X {X.shape} and {X[0].shape}")


    initx_array = np.zeros((3,10))


    for n in range(10):
        initx_array[0,n] = X[0][n][0]
        initx_array[1,n] = X[0][n][1]
        initx_array[2,n] = X[0][n][2]
    
    scatter2 = ax.scatter(initx_array[0], initx_array[1],initx_array[2], c='r', s = 50)
    print(f" size {initx_array.shape} {initx_array}")



    #======draw lines between tree============================================================
    # print(f"{edge_def}")
#     [[0 1]
#  [1 2]
#  [1 7]
#  [2 3]
#  [2 6]
#  [3 4]
#  [3 5]
#  [7 8]
#  [7 9]]


    xtreelist = []
    ytreelist = []
    ztreelist = []

    line_3D_list = []

    for idx,edge in enumerate(edge_def):
        edge_a = edge[0]
        edge_b = edge[1]
        print(f"idx {idx} with {edge_a,edge_b}")

        line_3D_list.append([ initx_array[:,edge_a] , initx_array[:,edge_b]])


    x0_lc = Line3DCollection(line_3D_list, colors=[1,0,0,1], linewidths=1)

    # xtreelist.append(initx_array[0][0])
    # ytreelist.append(initx_array[1][0])
    # ztreelist.append(initx_array[2][0])

    # xtreelist.append(initx_array[0][1])
    # ytreelist.append(initx_array[1][1])
    # ztreelist.append(initx_array[2][1])

    ax.add_collection(x0_lc)

    plt.show()

