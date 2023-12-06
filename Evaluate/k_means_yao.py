from collections import defaultdict
from random import uniform
from math import sqrt
import numpy as np
from scipy import spatial
import random
import json

random.seed(1)
vec_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/golden_dict_1536_300_pca512.npy'
# golden_dict_xlnet.npy

vec_dict=np.load(vec_path,allow_pickle=True).item()
# 所有golden labels 
# todo 用来查看扩增的golden labels
golden_labels=list(vec_dict.keys())

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    
    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest = float('inf') # positive infinity
        shortest_index = 0
        # 比较该点距离哪个center最近
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    cosine distance
    """
    a=np.array([a])
    b=np.array([b])
    return spatial.distance.cdist(a, b, 'cosine')
    # dimensions = len(a)
    
    # _sum = 0
    # for dimension in range(dimensions):
    #     difference_sq = (a[dimension] - b[dimension]) ** 2
    #     _sum += difference_sq
    # return sqrt(_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            
            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    # 根据每维度的最大值和最小值，随机生成k个点
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    i=0
    while assignments != old_assignments:
        i+=1
        if i%10==0:
            print('iter:{}'.format(i))
        if i==300:
            break
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    return assignments ,new_centers,i

def M_N_clustring(centers_save_path,datas,iter,clusters_num,clustering_times,init=False):
    """
    10 次 15(clusters num) 分类 pca512
    M 次 N 聚类 ： 模拟M 个人同时进行分类，若多个人都认为属于某一类，则属于该类 
    input: centers_save_path: 保存每次聚类的中心点的路径
           datas: 用于聚类的数据集 {labesl:vec}
           cluters_num: 初始聚类的簇数
           clustering_times: 每次迭代聚类的次数
    output: None
    """
    times=clustering_times
    k=clusters_num
    centers_info=[]
    if iter==0:
        for i in range(times):
            print('iter:{}'.format(i))
            random.seed(i)
            _,new_centers,_= k_means(datas, k)
            centers_info.append(new_centers)
    else:
        #todo 不是第一次迭代，未知出现多少新簇
        for i in range(times):
            print('iter:{}'.format(i))
            if i==0:
                while(True):
                    _,new_centers,iter_times= k_means(datas, k)
                    if iter_times == 300:
                        k+=2
                    else:
                        break
            else:
                _,new_centers,_= k_means(datas, k)
            centers_info.append(new_centers)
    print('recent iter:{}  cluster num:{}'.format(iter,k))
    centers_info=np.array(centers_info)
    # save centers_info
    # if not init:
    #     np.save(centers_save_path,centers_info)
    return centers_info

# 将新词添加进原有cluster，并适当增加cluster的数量
def cluster_filtering_with_addition(new_labels,last_centers_num):
    """
    input: centers: 用于聚类的中心点
           new_labels: 新的labels(包含原来的)
    output: new_centers: 新的中心点
    """
    song_candidate_words_dict = np.load('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/candidate_words_context_embedding_dict_1536_300_pca512.npy', allow_pickle=True).item()
    # get datas [list of vec]
    datas=[]
    for label in new_labels:
        datas.append(song_candidate_words_dict[label])
    k=last_centers_num+2
    while(True):
        random.seed(1)
        _,new_centers,i= k_means(datas, k)
        if i==300:
            print('xxxxxxx-{}'.format(k))
            k+=1
        else:
            break
# iter=0
# new_labels=json.load(open('/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/yao_preprocess_data/iter_labels/all_labels_iter{}_2.json'.format(iter),'r'))
# last_centers_num=15
# cluster_filtering_with_addition(new_labels,last_centers_num)

    

# if __name__ == "__main__":
#     # test data
#     # data = [[1, 2], [1, 4], [1, 0],
#     #         [10, 2], [10, 4], [10, 0]]
#     # k = 4
#     # print(k_means(data, k))

#     # use

#     # 采用m个n分类 将每次的center保存下来
#     iter=1
#     times=15
#     k=10
#     save_folder_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Evaluate/yao_check_datas/k-means/'
#     txt_name='{}-means(pca512).txt'.format(k)
#     vec_path='/home/yaoyao123/yaoyao123_data/keyphrasegenration/keyphrase-classification-New-Architecture-New-Data-Split/Preprocess/tme_big_data/golden_dict_1536_300_pca512.npy'
#     vec_dict=np.load(vec_path,allow_pickle=True).item()
#     labels=list(vec_dict.keys())
#     dataset=list(vec_dict.values())
#     centers_info=[]
#     for time in range(times):
#         random.seed(time)
#         _,new_centers= k_means(dataset, k)
#         centers_info.append(new_centers)
    
#     centers_info=np.array(centers_info)
#     # save centers_info
#     np.save(save_folder_path+'centers_vecs_iter{}.npy'.format(iter),centers_info)
#     # 计算各-center间的cosine距离
#     # center_dist_dict{1:{2:0.1,3:0.2,4:0.3},2:{3:0.1,4:0.2},3:{4:0.1}}
#     # with open(save_folder_path+txt_name,'w') as f:
#     #     i=0
#     #     for center in new_centers:
#     #         i+=1
#     #         j=0
#     #         for center2 in new_centers:
#     #             j+=1
#     #             dist=distance(center,center2)
#     #             dist_str='{}-{}:{}\n'.format(i,j,dist)
#     #             f.write(dist_str)
#     # # write in txt
    
#     #     for i in range(k):
#     #         for j in range(len(assign_points)):
#     #             if assign_points[j]==i:
#     #                 f.write(labels[j])
#     #                 f.write(' ')
#     #                 # print(labels[j])
#     #         f.write('\n--------------------------------------------------------------\n')
#     #         # print('----------------------------------------------------------------\n')
    
#     # f.close()