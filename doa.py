import numpy as np
import multiprocessing
from tqdm import tqdm
from joblib import Parallel, delayed
import time
def calculate_doa_k_block(mas_level, q_matrix, r_matrix, k,k_position, q_diff, block_size=50):
    n_students = mas_level.shape[0]
    n_questions, _ = q_matrix.shape
    n_attempts = r_matrix.shape[1]
    DOA_k = 0.0
    numerator = 0
    denominator = 0
    question_hask = np.where(q_matrix[:, k] != 0)[0].tolist() 
    for start in range(0, n_students, block_size):
        end = min(start + block_size, n_students)
        mas_level_block = mas_level[start:end, :]

        r_matrix_block = r_matrix[start:end, :]
        for j in question_hask:
            row_vector = (r_matrix_block[:, j].reshape(1, -1) != -1).astype(int)
            columen_vector = (r_matrix_block[:, j].reshape(-1, 1) != -1).astype(int)
            mask = row_vector * columen_vector
            
            delta_r_matrix = r_matrix_block[:, j].reshape(-1, 1) > r_matrix_block[:, j].reshape(1, -1)
            
            I_matrix = r_matrix_block[:, j].reshape(-1, 1) != r_matrix_block[:, j].reshape(1, -1)
            
            numerator_ = np.logical_and(mask, delta_r_matrix)
            denominator_ = np.logical_and(mask, I_matrix)
 
            delta_matrix_block = mas_level[start:end, 0 ,k_position[k], q_diff[j]].reshape(-1, 1) > mas_level[start:end,0 ,k_position[k], q_diff[j]].reshape(1, -1)
            numerator += np.sum(delta_matrix_block * numerator_)
            denominator += np.sum(delta_matrix_block * denominator_)
    if denominator == 0:
        DOA_k = 0
    else:
        DOA_k = numerator / denominator
    return DOA_k
def DOA(mastery_level,k_position, q_diff, q_matrix, r_matrix, data_name):
    if data_name in ["a0405"]:
        concepts = [9, 3, 1, 14, 8, 4, 7, 5, 0, 80]
        block_size = 600
    elif data_name == "a2012":
        concepts = [17, 30, 46, 51, 5, 89, 16, 60, 15, 4]
        block_size = 2048
    elif data_name == "ednet":
        concepts = [11, 10, 18, 14, 57, 0, 24, 52, 153, 111]
        block_size = 1024
    elif data_name == "XES3G5M":
        concepts = [78, 57, 18, 138, 55, 208, 73, 155, 56, 256]
        block_size = 2048
    elif data_name in ["a2017","a2017_d25"]:
        concepts = [21, 14, 58, 37, 60, 34, 4, 5, 7, 33]
        block_size = 1024
    elif data_name == "ednet_raw":
        concepts = [13, 2, 90, 46, 0, 105, 141, 136, 61, 144]
        block_size = 1024
    else:
        concepts = []
    know_n = q_matrix.shape[1]
    doa_k_list = Parallel(n_jobs=-1)(
        delayed(calculate_doa_k_block)(mastery_level, q_matrix, r_matrix, k,k_position, q_diff,  block_size) for k in concepts)

    return np.mean(doa_k_list)