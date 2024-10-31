import pandas as pd
import numpy as np
from math import floor
from timeit import default_timer as timer
from sklearn.feature_extraction.text import TfidfVectorizer

pd.options.display.max_rows = 9999

df = pd.read_csv('DatasetTest.csv',encoding= 'unicode_escape') #Dhmiourgia dataframe me vasei to csv poy exei Surname, #ofAwards
                                                               #kai education gia tous computer scientists..
names_awards_list = df[[df.columns[0],df.columns[1]]].values.tolist() #Antigrafh twn dyo stylwn surname kai #ofawards se mia lista
                                                                    #poy tha apotelesei eisodo sth dimiourgia tou kdtree.

names_education_list = df[[df.columns[0],df.columns[2]]].values.tolist() #Antigrafh twn dyo stylwn surname kai #education se mia lista
                                                                    #poy xrhsimopoieite kata to range search.

# Function to create hash functions for LSH
def create_hash_functions(num_functions, num_buckets):
        hash_functions = []
        for _ in range(num_functions):
            # Generate random coefficients for the hash function
            a = np.random.randint(1, num_buckets)
            b = np.random.randint(0, num_buckets)
            hash_functions.append((a, b))
        return hash_functions

    # Function to hash a value using a hash function
def hash_value(value, a, b, num_buckets):
        hash_code = hash(value)
        return (a * hash_code + b) % num_buckets

    # Function to create LSH buckets
def create_lsh_buckets(num_buckets):
        buckets = {}
        for i in range(num_buckets):
            buckets[i] = []
        return buckets

    # Function to compute cosine similarity between two text strings
def cosine_similarity(text1, text2):
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform([text1, text2])
        cosine_sim = np.dot(tfidf_matrix, tfidf_matrix.T).toarray()[0, 1]
        return cosine_sim

    # Function to perform LSH on the education paragraphs
def lsh_education(dataframe, num_hash_functions, num_buckets, similarity_threshold,hash_value):

        hash_functions = create_hash_functions(num_hash_functions, num_buckets)
        buckets = create_lsh_buckets(num_buckets)

        # Hash each scientist's education paragraph into buckets

        for i, scientist in dataframe.iterrows():
            education = scientist["Education"]
            hash_values = []
            for hash_function in hash_functions:
                a, b = hash_function
                hash_val = hash_value(education,a, b, num_buckets)
                hash_values.append(hash_val)

            # Add the scientist's index to the corresponding bucket(s)
            for hash_val in hash_values:
                buckets[hash_val].append(i)

        # Find similar education paragraphs
        similar_scientists = set()
        for hash_function in hash_functions:
            h_val = hash_value(education,hash_function[0], hash_function[1], num_buckets)
            similar_scientists.update(buckets[h_val])

        similar_scientists = list(similar_scientists)
        results = []
        similar_scientists_sorted = sorted(similar_scientists)
        for i  in range(len(similar_scientists_sorted)-1) :
          scientist_index_1 = similar_scientists[i]
          scientist_index_2 = similar_scientists_sorted[i + 1]
          education1 = dataframe.loc[scientist_index_1]["Education"]
          education2 = dataframe.loc[scientist_index_2]["Education"]
        
          similarity = cosine_similarity(education1, education2)
          if similarity >= similarity_threshold:
            results.append(scientist_index_1)
        if len(similar_scientists_sorted) > 0:
            results.append(similar_scientists_sorted[-1])
        return results

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.left = None
        self.right = None
        self.minr = None  
        self.maxr = None
        self.assoc = None

def makeRange2dTree(points_list, xtree=True, ytree=False): #Sinartisi gia dhmiourgia dentrou. An xtree = True ftiaxnei to xtree alliws an ytree = True ftiaxnei to ytree
    
    if len(points_list) == 0:
        return None
    
    if len(points_list) % 2 == 0:   #Ypologismos median analoga an einai artios h perittos o arithmos twn stoixeiwn sth lista.
        median = int(len(points_list)/2)
    else:
        median = floor(len(points_list)/2)
    
    if xtree:
        sorted_list = sorted(points_list, key = lambda x: x[0])
        root = Node(sorted_list[median][0],sorted_list[median][1])
        root.minr = sorted_list[0][0]
        root.maxr = sorted_list[-1][0]
        root.left = makeRange2dTree(sorted_list[:median], xtree, ytree)
        root.right = makeRange2dTree(sorted_list[median+1:], xtree, ytree)
    elif ytree:
        sorted_list = sorted(points_list, key = lambda x: x[1])
        root = Node(sorted_list[median][0],sorted_list[median][1])
        root.minr = sorted_list[0][1]
        root.maxr = sorted_list[-1][1]
        root.left = makeRange2dTree(sorted_list[:median], xtree, ytree)
        root.right = makeRange2dTree(sorted_list[median+1:], xtree, ytree)
    if not ytree:
        root.assoc = makeRange2dTree(sorted(points_list, key=lambda x: x[1]), xtree = False, ytree = True)
       
    return root
    

# Print the 2D range tree
def printTree(root,string=" "): #Sinartisi gia emfanisi tou dentrou pou emfanizei kai ta paidia kathe komvou.
    if root:
        print(string + "X:" + str(root.x) + " " + "Y:" + str(root.y) + " " + "Range: " + str(root.minr) + " - " + str(root.maxr))
        printTree(root.left,  string + "-leftchild-")
        printTree(root.right,  string + "-rightchild-")
        printTree(root.assoc, string + "-Ytree-")
                
def querySearchXtree(root, x_min, x_max, y_min, y_max, s_list):
    if root is None:
        return
    
    if x_min > x_max: 
        x_min, x_max = x_max, x_min
        
    if y_min > y_max:
        y_min, y_max = y_max, y_min
        
    if root.minr[0] >= x_min and root.maxr[0] <= x_max:
        querySearchYtree(root.assoc, y_min, y_max, s_list)
    else:
        if root.x[0] >= x_min and root.x[0] <= x_max and root.y >= y_min and root.y <= y_max:
            for i in range(len(names_education_list)):
                if(root.x == names_education_list[i][0]):
                    educ = names_education_list[i][1]
            s_list.append([root.x, root.y, educ])

        querySearchXtree(root.left, x_min, x_max, y_min, y_max, s_list)
        querySearchXtree(root.right, x_min, x_max, y_min, y_max, s_list)
        
def querySearchYtree(root, y_min, y_max, s_list):
    if root is None:
        return
        
    if y_min > y_max:
        y_min, y_max = y_max, y_min
        
    if root.y >= y_min and root.y <= y_max:
        for i in range(len(names_education_list)):
                if(root.x == names_education_list[i][0]):
                    educ = names_education_list[i][1]
        s_list.append([root.x, root.y, educ])
        
        
    querySearchYtree(root.left, y_min, y_max, s_list)
    querySearchYtree(root.right, y_min, y_max, s_list)

        
def insert(root, node, xtree = True, ytree = False):
    if root is None:
        return None
    
    if xtree:
        if node.x > root.x:
            if root.right is None:
                root.right=node
            else:
                insert(root.right, node, xtree, ytree)
        elif node.x < root.x:
            if root.left is None:
                root.left=node
            else:
                insert(root.left, node, xtree, ytree)
        else:
            return None
    elif ytree:
        if node.y > root.y:
            if root.right is None:
                root.right=node
            else:
                insert(root.right, node, xtree, ytree)
        elif node.y < root.y:
            if root.left is None:
                root.left=node
            else:
                insert(root.left, node, xtree, ytree)
        else:
            return None
        
    if not ytree:
        insert(root.assoc, node, xtree = False, ytree = True)

        
def delete(root, node, xtree=True, ytree=False):
    if root is None:
        return None

    # Search for the node to delete
    if xtree:
        if node.x > root.x:
            root.right = delete(root.right, node, xtree, ytree)
        elif node.x < root.x:
            root.left = delete(root.left, node, xtree, ytree)
        else:
            # Found the node in the x-tree, switch to y-tree for deletion
            root.assoc = delete(root.assoc, node, xtree=False, ytree=True)
    elif ytree:
        if node.y > root.y:
            root.right = delete(root.right, node, xtree, ytree)
        elif node.y < root.y:
            root.left = delete(root.left, node, xtree, ytree)
        else:
            # Found the node in the y-tree, switch to x-tree for deletion
            root.assoc = delete(root.assoc, node, xtree=True, ytree=False)
    else:
        # Node to delete has been found in both x-tree and y-tree, delete it
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left

        # If the node has two children, find the minimum node in the right subtree
        min_right = find_min(root.right)

        # Copy the values from the minimum node to the current node
        root.x = min_right.x
        root.y = min_right.y

        # Delete the minimum node from the right subtree
        root.right = delete(root.right, min_right, xtree, ytree)

    return root

def find_min(node):
    while node.left is not None:
        node = node.left
    return node


def update_range(root, points_list, xtree = True, ytree = False):
    if root is None:
        return None
    
    if len(points_list) == 0:
        return None
    
    if len(points_list) % 2 == 0:  #Ypologismos median
        median = int(len(points_list)/2)
    else:
        median = floor(len(points_list)/2)
    
    if xtree:
        sorted_list = sorted(points_list, key = lambda x: x[0])
        root.minr = sorted_list[0][0]
        root.maxr = sorted_list[-1][0]
        update_range(root.right,sorted_list[median+1:], xtree, ytree)
        update_range(root.left,sorted_list[:median],xtree, ytree)
    elif ytree:
        sorted_list = sorted(points_list, key = lambda x: x[1])
        root.minr = sorted_list[0][1]
        root.maxr = sorted_list[-1][1]
        update_range(root.right,sorted_list[median+1:], xtree, ytree)
        update_range(root.left,sorted_list[:median], xtree, ytree)
    if not ytree:
        update_range(root.assoc, sorted(points_list, key=lambda x: x[1]), xtree = False, ytree = True)


def searchNode(root, x_coord, y_coord):
    if root is None:
        return False
    
    if root.x == x_coord and root.y == y_coord:
        return True
    
    if root.x >= x_coord:  # Traverse left or right based on x-coordinate
        return searchNode(root.left, x_coord, y_coord)
    else:
        return searchNode(root.right, x_coord, y_coord)


def updateNode(root, old_x, old_y, new_x, new_y, xtree=True , ytree=False):
    if root is None:
        return None
    
    if xtree:
        if root.x == old_x and root.y == old_y:
            root.x = new_x
            root.y = new_y
        elif root.x < old_x:
            updateNode(root.right, old_x, old_y, new_x, new_y, xtree, ytree)
        elif root.x > old_x:
            updateNode(root.left, old_x, old_y, new_x, new_y, xtree, ytree)
    elif ytree:
        if root.x == old_x and root.y == old_y:
            root.x = new_x
            root.y = new_y
        elif root.y < old_y:
            updateNode(root.right, old_x, old_y, new_x, new_y, xtree, ytree)
        elif root.y > old_y:
            updateNode(root.left, old_x, old_y, new_x, new_y, xtree, ytree)
            
    if not ytree:
        updateNode(root.assoc, old_x, old_y, new_x, new_y, xtree=False , ytree=True)
    

################################################################################################
#MAIN

num_hash_functions = 50  # Number of hash functions
num_buckets = 1000  # Number of buckets
root = None
choice=-1

while choice!=0:
    print("Give a number for each choice below: ")
    print("0) End the program")
    print("1) Create the Range tree")
    print("2) Print the Range tree")
    print("3) Insert a node")
    print("4) Delete a node")
    print("5) Search for a node")
    print("6) Update a node")
    print("7) Query search")
    choice=int(input())
    if choice==1:#CREATE THE RANGE TREE
        root = None
        start_time = timer()
        root = makeRange2dTree(names_awards_list, xtree=True, ytree=False)
        end_time = timer()
        execution_time = end_time - start_time
        print("The Range Tree: ")
        printTree(root)
        print("\nCreate Range Tree executed in: ", execution_time, " seconds")

    elif choice==2:#PRINT THE RANGE TREE
        print("The Range Tree: ")
        printTree(root)
    
    elif choice==3:#INSERT A NODE
        print("Give the coordinates for the node to insert: ")
        x_coord_to_insert = str(input("Enter X coordinate: "))
        y_coord_to_insert = int(input("Enter Y coordinate: "))
        start_time = timer()
        insert_node=Node(x_coord_to_insert, y_coord_to_insert)
        insert(root,insert_node)
        end_time = timer()
        execution_time = end_time - start_time
        Dict = {'Name': [insert_node.x], 'Awards': [insert_node.y]}
        fd = pd.DataFrame(Dict)
        point = fd.values.tolist()
        names_awards_list += point
        update_range(root, names_awards_list)
        print("Tree after inserting the node:")
        printTree(root)
        print("\nInsert node executed in: ", execution_time, " seconds")
        

    elif choice==4:#DELETE A NODE
        print("Give the coordinates for the node to delete: ")
        x_coord_to_delete = str(input("Enter X coordinate to delete: "))
        y_coord_to_delete = int(input("Enter Y coordinate to delete: "))
        start_time = timer()
        delete_node = Node(x_coord_to_delete, y_coord_to_delete)
        if [delete_node.x, delete_node.y] in names_awards_list:
            names_awards_list.remove([delete_node.x, delete_node.y])
            root = delete(root, delete_node)
            end_time = timer()
            execution_time = end_time - start_time
            update_range(root, names_awards_list)
            root = makeRange2dTree(names_awards_list, xtree=True, ytree=False)
            print("Tree after deleting the node:")
            printTree(root)
        else:
            print("Node not found in the list.")
        print("\nDelete node executed in: ", execution_time, " seconds")

    elif choice==5:#SEARCH FOR A SPECIFIC NODE
        x_coord_to_search = str(input("Enter X coordinate to search: "))
        y_coord_to_search = int(input("Enter Y coordinate to search: "))
        start_time = timer()
        result = searchNode(root, x_coord_to_search, y_coord_to_search)
        end_time = timer()
        execution_time = end_time - start_time
        if result:
            print("Node exists in the tree.")
        else:
            print("Node does not exist in the tree.")
        print("\nSearch node executed in: ", execution_time, " seconds")

    elif choice==6:#UPDATE A NODE IN THE TREE
        print("Update a node:")
        old_x = str(input("Enter old X coordinate: "))
        old_y = int(input("Enter old Y coordinate: "))
        new_x = str(input("Enter new X coordinate: "))
        new_y = int(input("Enter new Y coordinate: "))
        start_time = timer()
        updateNode(root, old_x, old_y, new_x, new_y)
        end_time = timer()
        execution_time = end_time - start_time
        print("Tree after updating the node:")
        printTree(root)
        print("\nUpdate node executed in: ", execution_time, " seconds")
        
    elif choice==7:#QUERY SEARCH->GIVES SEARCH LIST
        root = makeRange2dTree(names_awards_list, xtree=True, ytree=False)
        x_min = input("Enter Xmin coordinate: ")
        x_max = input("Enter Xmax coordinate: ")
        y_min = int(input("Enter Ymin coordinate: "))
        y_max = int(input("Enter Ymax coordinate: "))
        similarity_percentage = float(input("Enter the similarity percentage (0.0-100.0): ")) / 100.0
        rows, columns = (0,3)
        search_list = [[0 for i in range(columns)] for j in range(rows)]
        start_time = timer()
        querySearchXtree(root, x_min, x_max, y_min, y_max, search_list)
        data = pd.DataFrame(search_list, columns=['Surname','#ofAwards','Education'] )
        similar_education_results = lsh_education(data, num_hash_functions, num_buckets, similarity_percentage,hash_value)
        print("Scientists with similar education paragraphs:")
        for scientist_index in similar_education_results:
            scientist = data.loc[scientist_index]
            print(scientist[0], scientist[1])
        end_time = timer()
        execution_time = end_time - start_time
        print("\nQuery search executed in: ", execution_time, " seconds")
