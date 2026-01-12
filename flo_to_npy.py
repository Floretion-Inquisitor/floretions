import numpy as np
import os
from multiprocessing import Pool, cpu_count
from floretion import Floretion
# flo_to_npy.py (in cima al file)



"""
This module computes and saves indices and signs matrices for floretion multiplication,
facilitating efficient computation of high-order floretion products.
"""


def compute_table_segment(args):
    """
    Computes indices and signs matrices for a segment of base vectors and stores as numpy binary files npy

    Args:
        args (tuple): Tuple containing base vectors, index range, floretion order, and segment number.

    Returns:
        str: Success message indicating that the segment has been processed and saved.
    """
    base_vecs, index_range, flo_order, segment, filedir = args
    unit_flo = Floretion.from_string(f'1{"e" * flo_order}')
    all_flo = Floretion(np.ones(4 ** flo_order), unit_flo.base_vec_dec_all, format_type="dec")
    num_base_vecs = 4 ** flo_order

    results_ind = np.zeros((len(index_range), num_base_vecs), dtype="int32")
    results_sgn = np.zeros((len(index_range), num_base_vecs), dtype="int32")

    for i, index_main in enumerate(index_range):
        z = base_vecs[index_main]
        result_ind_array = []
        result_sgn_array = []

        for index_y, y in enumerate(unit_flo.base_vec_dec_all):
            x = Floretion.mult_flo_base_absolute_value(z, y, flo_order)
            index_x = all_flo.base_to_grid_index[x]
            sign_xy = Floretion.mult_flo_sign_only(x, y, flo_order)

            result_ind_array.append(index_x)
            result_sgn_array.append(sign_xy)

        results_ind[i] = result_ind_array
        results_sgn[i] = result_sgn_array

    # Save directly to disk
    segment_string = str(segment).zfill(3)
    file_name_ind = f'{filedir}/floretion_order_{flo_order}_segment_{segment_string}_indices.npy'
    file_name_sgn = f'{filedir}/floretion_order_{flo_order}_segment_{segment_string}_signs.npy'
    np.save(file_name_ind, results_ind)
    np.save(file_name_sgn, results_sgn)

    return f"Segment {segment_string} processed and saved."

 


def save_table_segments(flo_order, total_segments, filedir, cores_per_batch=cpu_count()):
    """
    Saves indices and signs matrices for floretion of a given order across multiple segments.

    Args:
        flo_order (int): The order of floretion.
        total_segments (int): Total number of segments to divide the task into.
        filedir (str): Directory path to save the segment files.
        cores_per_batch (int): Number of cores per batch to use in parallel processing. Defaults to all available cores.
    """
    num_base_vecs = 4 ** flo_order
    base_vecs = Floretion.from_string(f'1{"e" * flo_order}').base_vec_dec_all
    os.makedirs(filedir, exist_ok=True)

    for batch_start in range(0, total_segments, cores_per_batch):
        with Pool(processes=cores_per_batch) as pool:
            tasks = [(base_vecs, range(segment * (num_base_vecs // total_segments),
                                       (segment + 1) * (num_base_vecs // total_segments)),
                      flo_order, segment, filedir)
                     for segment in range(batch_start, min(batch_start + cores_per_batch, total_segments))]

            results = pool.map(compute_table_segment, tasks)

        for result in results:
            print(result)

    print("All computations and storage completed for order", flo_order)


if __name__ == "__main__":

    from multiprocessing import freeze_support
    freeze_support()

    # esempio: 64 segmenti per ordine 7, NPY consigliato
    #save_centers_segmented(7, parity="both", total_segments=64, format_type="npy", cores_per_batch=8)
    #save_centers_segmented(7, parity="pos",  total_segments=64, format_type="npy", cores_per_batch=8)
    #save_centers_segmented(7, parity="neg",  total_segments=64, format_type="npy", cores_per_batch=8)
    #exit(-1)

    # in flo_to_npy.py
    filedir = "./data/npy/order_8.segments"
    save_table_segments(flo_order=8, total_segments=512, filedir=filedir, cores_per_batch=8)
    exit(-1)

    filedir = "./data/npy/order_6.segments"
    save_table_segments(flo_order=6, total_segments=64, filedir=filedir, cores_per_batch=8)
    exit(-1)

    flo_order = 7
    filedir = f"./data/npy/order_{flo_order}"
    total_segments = 1
    cores_per_batch = 8

    #save_table_segments(flo_order, total_segments, filedir, cores_per_batch)
    #exit(-1)
    #flo_order  = 8  # Example, change as needed
    #format_type = 'npy'  # Change to 'npy' or 'json' as needed
    flo_order = 7

    save_centers(flo_order, parity="both", format_type="json")
    save_centers(flo_order, parity="pos", format_type="json")
    save_centers(flo_order, parity="neg", format_type="json")

    #save_centers(flo_order, parity="both", format_type="npy")
    #save_centers(flo_order, parity="pos", format_type="npy")
    #save_centers(flo_order, parity="neg", format_type="npy")

