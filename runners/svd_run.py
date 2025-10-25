import matlab.engine
import time
import os
import numpy as np

def run_svd(spikes, coords_foo, pars_matlab, relative_folder_path = 'analysis/SVD', include_answer = True):
    """
    Initializes and runs the MATLAB engine to execute the SVD algorithm on neural activity data in parallel. 
    This function also handles MATLAB path setup, updates parameter values in the GUI, and plots the results.

    :param spikes: Matrix of neural activity data to be processed.
    :type spikes: matlab.double
    :param coords_foo: Matrix of dummy coordinates used as input for the SVD function.
    :type coords_foo: matlab.double
    :param pars_matlab: MATLAB structure of parameters for the SVD algorithm.
    :type pars_matlab: dict
    :return: List of times taken for MATLAB engine setup, SVD execution, and plotting.
    :rtype: list[float]
    """
    log_flag = "GUI SVD:"
    success = True
    spikes_mat = matlab.double(spikes.tolist())

    print(f"{log_flag} Starting MATLAB engine...")
    start_time = time.time()
    eng_svd = matlab.engine.start_matlab()
    # Adding to path
    folder_path = os.path.abspath(relative_folder_path)
    folder_path_with_subfolders = eng_svd.genpath(folder_path)
    eng_svd.addpath(folder_path_with_subfolders, nargout=0)
    end_time = time.time()
    engine_time = end_time - start_time
    print(f"{log_flag} Loaded MATLAB engine.")

    start_time = time.time()
    try:
        answer = eng_svd.Stoixeion(spikes_mat, coords_foo, pars_matlab)
    except:
        print(f"{log_flag} An error occurred while executing the algorithm. Check console logs for more info.")
        answer = None
    end_time = time.time()
    algorithm_time = end_time - start_time
    print(f"{log_flag} Done.")
    print(f"{log_flag} Terminating MATLAB engine...")
    eng_svd.quit()
    print(f"{log_flag} Done.")

    ensgui_results = {}

    if answer != None:
        cant_neurons = spikes.shape[0]
        cant_timepoints = spikes.shape[1]
        num_state = int(answer['num_state'])

        # Plot the ensembles timecourse
        Pks_Frame = np.array(answer['Pks_Frame'])
        sec_Pk_Frame = np.array(answer['sec_Pk_Frame'])
        ensembles_timecourse = np.zeros((num_state, cant_timepoints))
        framesActiv = Pks_Frame.shape[1]
        for it in range(framesActiv):
            currentFrame = int(Pks_Frame[0, it])
            currentEns = int(sec_Pk_Frame[it, 0])
            if currentEns != 0: 
                ensembles_timecourse[currentEns-1, currentFrame-1] = 1

        # Identify the neurons that belongs to each ensamble
        Pools_coords = np.array(answer['Pools_coords'])
        neurons_in_ensembles = np.zeros((num_state, cant_neurons))
        for ens in range(num_state):
            cells_in_ens = Pools_coords[:, :, ens]
            for neu in range(cant_neurons):
                cell_id = int(cells_in_ens[neu][2])
                if cell_id == 0:
                    break
                else:
                    neurons_in_ensembles[ens, cell_id-1] = 1

        # Save the results
        ensgui_results['timecourse'] = ensembles_timecourse
        ensgui_results['ensembles_cant'] = num_state
        ensgui_results['neus_in_ens'] = neurons_in_ensembles
    else:
        success = False

    results = {
        "results": ensgui_results,
        "engine_time": engine_time,
        "algorithm_time": algorithm_time,
        "success": success
    }

    if include_answer: 
        results["answer"] = answer

    return results