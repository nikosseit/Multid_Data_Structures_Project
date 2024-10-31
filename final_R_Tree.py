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
    
class MinimumBoundingObject:
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.child = []

class RTree:
    def __init__(self):
        self.names_awards_list = []  # Initialize an empty list to store the data points
        self.root = None
    def create_rtree(self, points, dimensions):
        M = 4
        upper_level_items = []

        points.sort(key=lambda point: point[0])
        
        while len(points) > M:
            group_length = min(M, len(points))
            new_minimum_bounding_object = self.minimum_bounding_object_calculator(points[:group_length], dimensions)
            new_minimum_bounding_object.child = points[:group_length]
            upper_level_items.append(new_minimum_bounding_object)
            points = points[group_length:]

        if points:
            new_minimum_bounding_object = self.minimum_bounding_object_calculator(points, dimensions)
            new_minimum_bounding_object.child = points
            upper_level_items.append(new_minimum_bounding_object)

        if len(upper_level_items) <= M:
            return upper_level_items
        else:
            return self.create_rtree(upper_level_items, dimensions)

    def minimum_bounding_object_calculator(self, points, dimensions):
        lower = ['zzzzzzz'] * dimensions
        upper = [''] * dimensions

        for point in points:
            x_coord = point[0]
            y_coord = point[1]
            lower[0] = min(lower[0], x_coord)
            upper[0] = max(upper[0], x_coord)
            lower[1] = str(min(int(lower[1]) if lower[1] != 'zzzzzzz' else int(y_coord), int(y_coord)))
            if upper[1] == '':
                upper[1] = str(y_coord)
            else:
                upper[1] = str(max(int(upper[1]), int(y_coord)))

        return MinimumBoundingObject(lower, upper)
    
    def query(self, points, x_min, x_max, y_min, y_max, s_list):
        
        for point in points:
            x_coord = point[0]
            y_coord = point[1]

            if x_coord[0] <= x_max and x_coord[0] >= x_min and y_coord <= y_max and y_coord >= y_min:
                for i in range(len(names_education_list)):
                    if x_coord == names_education_list[i][0]:
                         educ = names_education_list[i][1]
                s_list.append([x_coord, y_coord,educ])

    def insert_recursive(self, node, x_coord, y_coord):
        if isinstance(node, MinimumBoundingObject):
            for item in node.child:
                if y_coord >= int(item.low[1]) and y_coord <= int(item.high[1]):
                    new_children = self.insert_recursive(item.child, x_coord, y_coord)
                    if new_children:
                        item.child = new_children
                        item.low[0] = min(item.low[0], x_coord)
                        item.high[0] = max(item.high[0], x_coord)
                        item.low[1] = str(min(int(item.low[1]), y_coord))
                        item.high[1] = str(max(int(item.high[1]), y_coord))
            if len(node.child) < 4:
                node.child.append([x_coord, str(y_coord)])
                # Update MBO coordinates if necessary
                node.low[0] = min(node.low[0], x_coord)
                node.high[0] = max(node.high[0], x_coord)
                node.low[1] = str(min(int(node.low[1]), y_coord))
                node.high[1] = str(max(int(node.high[1]), y_coord))
            else:
                # Split MBO and create new MBOs if capacity is exceeded
                new_min_bounding_objs = []
                for item in node.child:
                    new_min_bounding_obj = self.minimum_bounding_object_calculator([item], dimensions=2)
                    new_min_bounding_obj.child = [item]
                    new_min_bounding_objs.append(new_min_bounding_obj)
                new_min_bounding_obj = self.minimum_bounding_object_calculator([[x_coord, str(y_coord)]], dimensions=2)
                new_min_bounding_obj.child = [[x_coord, str(y_coord)]]
                new_min_bounding_objs.append(new_min_bounding_obj)
                return new_min_bounding_objs
        else:
            if len(node) < 4:
                node.append([x_coord, str(y_coord)])
            else:
                # Convert node into an MBO and create new MBOs if capacity is exceeded
                new_min_bounding_objs = []
                for item in node:
                    new_min_bounding_obj = self.minimum_bounding_object_calculator([item], dimensions=2)
                    new_min_bounding_obj.child = [item]
                    new_min_bounding_objs.append(new_min_bounding_obj)
                new_min_bounding_obj = self.minimum_bounding_object_calculator([[x_coord, str(y_coord)]], dimensions=2)
                new_min_bounding_obj.child = [[x_coord, str(y_coord)]]
                new_min_bounding_objs.append(new_min_bounding_obj)
                return new_min_bounding_objs
            
    def delete_recursive(self, node, x_coord, y_coord):
            if isinstance(node, MinimumBoundingObject):
                new_children = []
                for item in node.child:
                    if y_coord >= int(item.low[1]) and y_coord <= int(item.high[1]):
                        new_children = self.delete_recursive(item.child, x_coord, y_coord)
                    else:
                        new_children.append(item)
                
                node.child = new_children
                if not node.child:
                    return None  # Remove this MBO since it has no children left
                
                node.low = ['zzzzzzz'] * 2
                node.high = [''] * 2
                for item in node.child:
                    node.low[0] = min(node.low[0], item[0])
                    node.high[0] = max(node.high[0], item[0])
                    node.low[1] = str(min(int(node.low[1]), int(item[1])))
                    node.high[1] = str(max(int(node.high[1]), int(item[1])))

                return node
            
            else:
                new_children = []
                for item in node:
                    if item[0] != x_coord or int(item[1]) != y_coord:
                        new_children.append(item)
                
                if not new_children:
                    return None  # Remove this node since it's the only one left
                
                return new_children
            
    def delete(self, x_coord, y_coord):
         self.names_awards_list = self.delete_recursive(self.names_awards_list, x_coord, y_coord)
         self.root = self.create_rtree(self.names_awards_list, dimensions=2)
    

def search_point_in_rtree(root, x_coord, y_coord):
    def search_recursive(node):
        found = False
        for item in node:
            if isinstance(item, MinimumBoundingObject):
                if y_coord >= int(item.low[1]) and y_coord <= int(item.high[1]):
                    found = search_recursive(item.child) or found
            else:
                if item[0] == x_coord and int(item[1]) == y_coord:
                    found = True
                    break
        return found
    
    if search_recursive(root):
        print(f"Point ({x_coord}, {y_coord}) exists in the R-tree.")
    else:
        print(f"Point ({x_coord}, {y_coord}) does not exist in the R-tree.")

def print_tree(root, indent=0):
    if isinstance(root, MinimumBoundingObject):
        print("  " * indent + f"Minimum Bounding Object: {root.low} - {root.high}")
        if root.child:
            print_tree(root.child, indent + 1)
    else:
        for item in root:
            if isinstance(item, MinimumBoundingObject):
                print("  " * indent + f"Minimum Bounding Object: {item.low} - {item.high}")
                print_tree(item.child, indent + 1)
            else:
                print("  " * indent + f"Point: ({item[0]}, {item[1]})")

def update_node_in_rtree(rtree_instance, old_x_coord, old_y_coord, new_x_coord, new_y_coord):
    # Delete the old point
    rtree_instance.delete(old_x_coord, old_y_coord)
    
    # Insert the new point
    rtree_instance.root = rtree_instance.insert_recursive(rtree_instance.root, insert_x_coord, insert_y_coord)
    rtree_instance.names_awards_list.append([new_x_coord, new_y_coord])
    update_rtree_and_list(rtree_instance, rtree_instance.names_awards_list)
    
    print(f"Point ({old_x_coord}, {old_y_coord}) has been updated to ({new_x_coord}, {new_y_coord}).")

def update_rtree_and_list(rtree_instance, names_awards_list):
    rtree_instance.names_awards_list = names_awards_list
    rtree_instance.root = rtree_instance.create_rtree(names_awards_list, dimensions=2)

################################################################################################
#MAIN
    
num_hash_functions = 50  # Number of hash functions
num_buckets = 1000  # Number of buckets
root = None
choice=-1
rtree_instance = RTree()

while choice!=0:
    print("Give a number for each choice below: ")
    print("0) End the program")
    print("1) Create the R tree")
    print("2) Print the R tree")
    print("3) Insert a node")
    print("4) Delete a node")
    print("5) Search for a node")
    print("6) Update a node")
    print("7) Query search")
    choice=int(input())

    if choice==1:# CREATE THE TREE
        start_time = timer()
        rtree_instance.names_awards_list = names_awards_list
        root = rtree_instance.create_rtree(names_awards_list, dimensions=2)
        rtree_instance.root = rtree_instance.create_rtree(names_awards_list, dimensions=2)
        end_time = timer()
        execution_time = end_time - start_time
        print_tree(rtree_instance.root)
        print("\nCreate R Tree executed in: ", execution_time, " seconds")

    elif choice==2:#PRINT THE TREE
        update_rtree_and_list(rtree_instance, rtree_instance.names_awards_list)
        print_tree(rtree_instance.root)

    elif choice==3:#INSERT A NODE
        print("Give the coordinates to insert:")
        insert_x_coord = input("Enter X coordinate: ")
        insert_y_coord = int(input("Enter Y coordinate: "))
        start_time = timer()
        rtree_instance.root = rtree_instance.insert_recursive(rtree_instance.root, insert_x_coord, insert_y_coord)
        rtree_instance.names_awards_list.append([insert_x_coord, insert_y_coord])  # Append the new point        
        update_rtree_and_list(rtree_instance, rtree_instance.names_awards_list)
        end_time = timer()
        execution_time = end_time - start_time
        print("The updated R-tree after insertion: ")
        print_tree(rtree_instance.root)
        print("\nInsert node executed in: ", execution_time, " seconds")
        

    elif choice==4:#DELETE A NODE
        print("Give the coordinates to delete:")
        delete_x_coord = input("Enter X coordinate: ")
        delete_y_coord = int(input("Enter Y coordinate: "))
        start_time = timer()
        rtree_instance.delete(delete_x_coord, delete_y_coord)
        update_rtree_and_list(rtree_instance, rtree_instance.names_awards_list)
        end_time = timer()
        execution_time = end_time - start_time
        print("The updated R-tree after deletion: ")
        print_tree(rtree_instance.root)
        print("\nDelete node executed in: ", execution_time, " seconds")
        


    elif choice==5:#SEARCH FOR A SPECIFIC NODE
        print("Give the coordinates to search for:")
        search_x_coord = input("Enter X coordinate: ")
        start_time = timer()
        search_y_coord = int(input("Enter Y coordinate: "))
        search_point_in_rtree(rtree_instance.root, search_x_coord, search_y_coord)
        end_time = timer()
        execution_time = end_time - start_time
        print("\nSearch node executed in: ", execution_time, " seconds")
        

    elif choice==6:#UPDATE A NODE IN THE TREE
        print("Give the coordinates of the existing point to update:")
        existing_x_coord = input("Enter X coordinate: ")
        existing_y_coord = int(input("Enter Y coordinate: "))
        new_x_coord = input("Enter the new X coordinate: ")
        new_y_coord = int(input("Enter the new Y coordinate: "))
        start_time = timer()
        update_node_in_rtree(rtree_instance, existing_x_coord, existing_y_coord, new_x_coord, new_y_coord)
        end_time = timer()
        execution_time = end_time - start_time
        print("The updated R-tree after updating a point: ")
        print_tree(rtree_instance.root)
        print("\nUpdate node executed in: ", execution_time, " seconds")
        
    elif choice==7:#QUERY SEARCH->GIVES SEARCH LIST
        print("Give the boundary points: ")
        x_min = input("Enter Xmin coordinate: ")
        x_max = input("Enter Xmax coordinate: ")
        y_min = int(input("Enter Ymin coordinate: "))
        y_max = int(input("Enter Ymax coordinate: "))
        boundaries = [x_min, x_max, y_min, y_max]
        similarity_percentage = float(input("Enter the similarity percentage (0.0-100.0): ")) / 100.0
        rows, columns = (0,3)
        search_list = [[0 for i in range(columns)] for j in range(rows)]
        start_time = timer()
        rtree_instance.query(rtree_instance.names_awards_list, *boundaries, search_list)
        data = pd.DataFrame(search_list, columns=['Surname','#ofAwards','Education'] )
        similar_education_results = lsh_education(data, num_hash_functions, num_buckets, similarity_percentage,hash_value)
        print("Scientists with similar education paragraphs:")
        for scientist_index in similar_education_results:
            scientist = data.loc[scientist_index]
            print(scientist[0], scientist[1])
        end_time = timer()
        execution_time = end_time - start_time
        print("\nQuery search executed in: ", execution_time, " seconds")


