from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
import concurrent.futures


class Map:

    # Default constructor
    def __init__(self):
        self.width = 0
        self.height = 0
        self.cells = []
    
    # Converting a string (with '#' representing obstacles and '.' representing free cells) to a grid
    def ReadFromString(self, cellStr, width, height):
        self.width = width
        self.height = height
        self.cells = [[0 for _ in range(width)] for _ in range(height)]
        cellLines = cellStr.split("\n")
        i = 0
        j = 0
        for l in cellLines:
            if len(l) != 0:
                j = 0
                for c in l:
                    if c == '.':
                        self.cells[i][j] = 0
                    elif c == '@':
                        self.cells[i][j] = 1
                    else:
                        continue
                    j += 1
                # TODO
                if j != width:
                    raise Exception("Size Error. Map width = ", j, ", but must be", width )
                
                i += 1

        if i != height:
            raise Exception("Size Error. Map height = ", i, ", but must be", height )
    
    # Initialization of map by list of cells.
    def SetGridCells(self, width, height, gridCells):
        self.width = width
        self.height = height
        self.cells = gridCells

    # Check if the cell is on a grid.
    def inBounds(self, i, j):
        return (0 <= j < self.width) and (0 <= i < self.height)
    
    # Check if thec cell is not an obstacle.
    def Traversable(self, i, j):
        return not self.cells[i][j]

    # Get a list of neighbouring cells as (i,j) tuples.
    # It's assumed that grid is 4-connected (i.e. only moves into cardinal directions are allowed)
    def GetNeighbors(self, i, j):
        neighbors = []
        delta = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        for d in delta:
            if self.inBounds(i + d[0], j + d[1]) and self.Traversable(i + d[0], j + d[1]):
                neighbors.append((i + d[0], j + d[1]))

        return neighbors
    
    def get_item(self, i, j):
        
        return self.cells[i][j]



def CalculateCost(i1, j1, i2, j2):
    return math.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)


class Node:
    def __init__(self, i, j, g = math.inf, h = math.inf, F = None, parent = None, alternating=None):
        self.i = i
        self.j = j
        self.g = g
        if F is None:
            self.F = self.g + h
        else:
            self.F = F 
        # Calculation for avoiding of reexpansions
        if alternating is not None:
            self.F = max(self.F, 2 * self.g)
        self.parent = parent
    
    def __eq__(self, other):
        return (self.i == other.i) and (self.j == other.j)
    
    def __hash__(self):
        
        return self.i << 16 | self.j
        #return self.i + self.j
        


class OpenList():
    def __init__(self):
        self.elements = []
    
    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def isEmpty(self):
        if len(self.elements) != 0:
            return False
        return True
    

    def GetBestNode(self):
        bestF = math.inf
        bestCoord = 0
        for i in range(len(self.elements)):
            if self.elements[i].F - bestF < -0.000005:
                bestCoord = i
                bestF = self.elements[i].F
                
        best = self.elements.pop(bestCoord)
        return best
    #Create your own modification of CLOSED#
    

    def AddNode(self, item : Node):
        for coord in range(len(self.elements)):
            if self.elements[coord].i == item.i and self.elements[coord].j == item.j:
                if self.elements[coord].g - item.g > 0.000005:
                    self.elements[coord].F = item.F
                    self.elements[coord].g = item.g
                    self.elements[coord].parent = item.parent
                    return
                else:
                    return
        self.elements.append(item)
        return
    
    def WasOpened(self, item : Node):
        return item in self.elements
    
    def GetNode(self, item : Node):
        i = self.elements.index(item)
        return self.elements[i]
    
    def GetBestg(self):
        bestF = math.inf
        for i in range(len(self.elements)):
            if self.elements[i].F - bestF < -0.000005:
                bestF = self.elements[i].F
                
        return bestF
    
    def Connect(self, item):
        self.elements += item.elements
        return self


class YourClosed():

    def __init__(self):
        self.elements = set()


    def __iter__(self):
        return iter(self.elements)
    
    def __len__(self):
        return len(self.elements)
    
    # AddNode is the method that inserts the node to CLOSED
    def AddNode(self, item : Node, *args):
        self.elements.add(item)

    # WasExpanded is the method that checks if a node has been expanded
    def WasExpanded(self, item : Node, *args):
        return item in self.elements
    
    
    def Connect(self, item):
        self.elements = self.elements.union(item.elements)
        return self
    
    def GetNode(self, item : Node):
        #return self.elements.intersection({item}).pop()
        #q = {item}.intersection(self.elements).pop()
        #print('fdfd', q is item)
        for q in self.elements:
            if item == q:
                #print('eq', item is q)
                return q
        #return {item}.intersection(self.elements).pop()
        


from random import randint

def write_file(target_map, save_path):
    
    with open(save_path, 'w') as fopen:
        for line in target_map:
            str_line = ''.join([str(element) + ' ' for element in line]) + '\n'
            fopen.write(str_line)
            
    return

def generate_one_point(taskMap, Goals):
    
    
    height = taskMap.height
    width = taskMap.width
        
    iGoal = randint(0, height - 1)
    jGoal = randint(0, width - 1)

    while (iGoal, jGoal) in Goals:
        iGoal = randint(0, height - 1)
        jGoal = randint(0, width - 1)
    
    Goals.add((iGoal, jGoal))
    
    return iGoal, jGoal

from copy import deepcopy

def get_target_map(taskMap, CLOSED):
    
    height = taskMap.height
    width = taskMap.width
    
    target_map = [[0 for j in range(width)] for i in range(height)]
    
    #for i in range(height):
        #for j in range(width):
            #if taskMap.cells[i][j] == 1:
                #target_map[i][j] = 10 ** 6
                
    for node in CLOSED:
        i = node.i
        j = node.j
        target_map[i][j] = node.g
        
    return target_map

def get_Goal_map(taskMap, iGoal, jGoal):
    
    height = taskMap.height
    width = taskMap.width
    
    Goal_map = [[0 for j in range(width)] for i in range(height)]
    Goal_map[iGoal][jGoal] = 1
    
    return Goal_map

    

def generate(args):
    
    #print('Start generating')
    #global quantity
    quantity = 0
    
    taskMap, input_directory, output_directory, map_file, SearchFunction, quantity_Goals = args
    #print(map_file)
    
    input_path1 = input_directory + 'input_map/' + map_file
    write_file(taskMap.cells, input_path1)
        
    Goals = set()
    limit = taskMap.height * taskMap.width / 4
    
    for i in range(quantity_Goals):

        closed_size = 0
        
        start = time.time()
        j = 0
        while closed_size < limit:
            #print(i, j, end='\r')
            
            iGoal, jGoal = generate_one_point(taskMap, Goals)
            CLOSED = SearchFunction(taskMap, iGoal, jGoal)
            closed_size = len(CLOSED)
            #_ = input()
            
            j += 1
         
        quantity += 1
        
        input_path2 = input_directory + 'Goal_map/' + str(iGoal) + '_' + str(jGoal) + '_' + map_file
        Goal_map = get_Goal_map(taskMap, iGoal, jGoal)
        write_file(Goal_map, input_path2)
        
        target_map = get_target_map(taskMap, CLOSED)
        output_path = output_directory + str(iGoal) + '_' + str(jGoal) + '_' + map_file
        write_file(target_map, output_path)
        
        #o_map = np.array(target_map)
        #img = Image.fromarray(o_map)
        #img.show()
        
        print(' ' * 60, end='\r')
        print('       || {} || {}  || {}'.format(time.ctime(), map_file, quantity), end='\r')

    print()        
        
    return
        


def ReadTaskFromFile(path):
    tasksFile = open(path)
    count = 0
    _ = tasksFile.readline()
    height = int(tasksFile.readline().split()[1])
    width = int(tasksFile.readline().split()[1])
    _ = tasksFile.readline()
    cells = [[0 for _ in range(width)] for _ in range(height)]
    i = 0
    j = 0

    allowed_quantity = 0
    not_allowed_quantity = 0
    
    for l in tasksFile:
        j = 0
        for c in l:
#             print(c)
#             _ = input()
            if c == '.':
                cells[i][j] = 0
                allowed_quantity += 1
            elif c == '@':
                cells[i][j] = 1
                not_allowed_quantity += 1
            else:
                continue
            
            j += 1
            
        if j != width:
            raise Exception("Size Error. Map width = ", j, ", but must be", width, "(map line: ", i, ")")
                
        i += 1
        if(i == height):
            break
    
#     iStart = int(tasksFile.readline())
#     jStart = int(tasksFile.readline())
#     iGoal = int(tasksFile.readline())
#     jGoal = int(tasksFile.readline())
#     length = float(tasksFile.readline())
#     print('allowed = {} not_allowed = {}'.format(allowed_quantity /  height ** 2, \
#                                                  not_allowed_quantity / height ** 2))
    
    return (width, height, cells)

import time
import os

def arg_generator(SearchFunction, train_or_test=False, quantity_Goals=1, map_size=256):
    maps_directory = 'data/files/maps/' + str(map_size) + '/'
    if train_or_test:
        input_directory = 'data/files/dataset/' + str(map_size) + '/test/input/'
        output_directory = 'data/files/dataset/' + str(map_size) + '/test/output/'
    else:
        input_directory = 'data/files/dataset/' + str(map_size) + '/train/input/'
        output_directory = 'data/files/dataset/' + str(map_size) + '/train/output/'
    
    for map_file in os.listdir(maps_directory):
        #print(map_file)
        taskFileName = maps_directory + map_file
        #print(taskFileName)
        width, height, cells = ReadTaskFromFile(taskFileName)
        taskMap = Map()
        taskMap.SetGridCells(width,height,cells)
        yield (taskMap, input_directory, output_directory, map_file, SearchFunction, quantity_Goals)
    

def MassiveTest(SearchFunction, train_or_test=False, quantity_Goals=1, map_size=256):
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(generate, arg_generator(SearchFunction, train_or_test=train_or_test, quantity_Goals=quantity_Goals, map_size=map_size))
        

def Dijkstra(gridMap : Map, iGoal : int, jGoal : int, openType = OpenList, closedType = YourClosed):
    
    Goal = Node(iGoal, jGoal, 0)
    
    OPEN = openType()
    OPEN.AddNode(Goal)
    CLOSED = closedType()
    
    #print('part1')
    while not OPEN.isEmpty():
        #print('\nwhile')
        print(len(CLOSED), end='\r')
        
        s = OPEN.GetBestNode()
        #print('got best s')
        CLOSED.AddNode(s)
        #print('addNode')
    
        
        #print('get neighbors')
        neighbors = gridMap.GetNeighbors(s.i, s.j)
        #print('got neighbors')
        for new_s in neighbors:
            
            new_i, new_j = new_s
            new_g = s.g + CalculateCost(s.i, s.j, new_i, new_j)
            new_node = Node(new_i, new_j, new_g, h=0, parent=s)
            
            if not CLOSED.WasExpanded(new_node):
                
                OPEN.AddNode(new_node)
            
    

    return CLOSED

# map_size = 256
#print(time.ctime())
#MassiveTest(Dijkstra, quantity_Goals=1250, map_size=256)
#print(time.ctime())
#MassiveTest(Dijkstra, train_or_test=True, quantity_Goals=125, map_size=256)
#print(time.ctime())

# map_size = 128
print(time.ctime())
MassiveTest(Dijkstra, quantity_Goals=1250, map_size=128)
print(time.ctime())
MassiveTest(Dijkstra, train_or_test=True, quantity_Goals=125, map_size=128)
print(time.ctime())

# map_size = 64
print(time.ctime())
MassiveTest(Dijkstra, quantity_Goals=1250, map_size=64)
print(time.ctime())
MassiveTest(Dijkstra, train_or_test=True, quantity_Goals=125, map_size=64)
print(time.ctime())


