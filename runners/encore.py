import matlab.engine
import time
import os
import numpy as np
import scipy.stats as stats
import utils.data_converters as converters

def run_svd(input_data, pars_validated, relative_folder_path = 'analysis/SVD', include_answer = True, logger = None):
    """
    Initializes and runs the MATLAB engine to execute the SVD algorithm on neural activity data. 
    This function also handles MATLAB path setup and data conversion to MATLAB.

    :param raster: Matrix of neural activity data to be processed.
    :type raster: numpy.ndarray
    :param pars_validated: Dictionary with parameters for the SVD algorithm.
    :type pars_validated: dict
    :param relative_folder_path: String with the path to the analysis code. Defaults to 'analysis/SVD'
    :type relative_folder_path: str, optional
    :param include_answer: Flag to indicate wether or not the original full answer of the algorithm. Defaults to True
    :type include_answer: bool, optional
    :param logger: Function to show the log of the function execution. 
        This function should receive two strings as parameters, just like MainWindow.update_console_log.
        Defaults to None.
    :type logger: function, optional
    :return: Dictionary with the result of the algorithm. 
    :rtype: dict
    """
    log_flag = "SVD:"
    success = True
    
    if logger:
        logger(f"{log_flag} Converting Python data to MATLAB data...", "log")
    
    # The raster is at the first position
    raster = input_data[0]
    # Convert the raster to a MATLAB matrix
    raster_mat = matlab.double(raster.tolist())
    #Prepare dummy data
    data = np.zeros((raster.shape[1],2))
    coords_foo = matlab.double(data.tolist())
    # Prepare MATLAB parameters
    pars_matlab = converters.dict_to_matlab_struct(pars_validated)
    if logger:
        logger(f"{log_flag} Done converting.", "complete")

    if logger:
        logger(f"{log_flag} Starting MATLAB engine...", "log")
    start_time = time.time()
    eng_svd = matlab.engine.start_matlab()
    # Adding to path
    folder_path = os.path.abspath(relative_folder_path)
    folder_path_with_subfolders = eng_svd.genpath(folder_path)
    eng_svd.addpath(folder_path_with_subfolders, nargout=0)
    end_time = time.time()
    engine_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Loaded MATLAB engine.", "complete")

    if logger:
        logger(f"{log_flag} Running SVD algorithm...", "log")
    start_time = time.time()
    try:
        answer = eng_svd.Stoixeion(raster_mat, coords_foo, pars_matlab)
    except:
        if logger:
            logger(f"{log_flag} An error occurred while executing the algorithm. Check console logs for more info.", "error")
        answer = None
    end_time = time.time()
    algorithm_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Done.", "complete")
        logger(f"{log_flag} Terminating MATLAB engine...", "log")
    eng_svd.quit()
    if logger:
        logger(f"{log_flag} Done.", "complete")

    ensgui_results = {}

    if answer != None:
        cant_neurons = raster.shape[0]
        cant_timepoints = raster.shape[1]
        num_state = int(answer['num_state'])

        # Get the ensembles timecourse
        Pks_Frame = np.array(answer['Pks_Frame'])
        sec_Pk_Frame = np.array(answer['sec_Pk_Frame'])
        ensembles_timecourse = np.zeros((num_state, cant_timepoints))
        framesActiv = Pks_Frame.shape[1]
        for it in range(framesActiv):
            currentFrame = int(Pks_Frame[0, it])
            currentEns = int(sec_Pk_Frame[it, 0])
            if currentEns != 0: 
                ensembles_timecourse[currentEns-1, currentFrame-1] = 1

        # Identify the neurons that belongs to each ensemble
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
        "success": success,
        "update_params": {
            'pks': int(answer['pks']),
            'scut': answer['scut']
        }
    }

    if include_answer: 
        results["answer"] = answer
        results["answer"]['timecourse'] = ensembles_timecourse
        results["answer"]['neus_in_ens'] = neurons_in_ensembles

    return results

def run_pca(input_data, pars_validated, relative_folder_path = 'analysis/NeuralEnsembles', include_answer = True, logger = None):
    """
    Initializes and runs the MATLAB engine to execute the PCA algorithm on neural activity data. 
    This function also handles MATLAB path setup and data conversion to MATLAB.

    :param spikes: Matrix of neural activity data to be processed.
    :type spikes: numpy.ndarray
    :param pars_validated: Dictionary with parameters for the SVD algorithm.
    :type pars_validated: dict
    :param relative_folder_path: String with the path to the analysis code. Defaults to 'analysis/NeuralEnsembles'
    :type relative_folder_path: str, optional
    :param include_answer: Flag to indicate wether or not the original full answer of the algorithm. Defaults to True
    :type include_answer: bool, optional
    :param logger: Function to show the log of the function execution. 
        This function should receive two strings as parameters, just like MainWindow.update_console_log.
        Defaults to None.
    :type logger: function, optional
    :return: Dictionary with the result of the algorithm. 
    :rtype: dict
    """
    log_flag = "PCA:"
    success = True
    
    if logger:
        logger(f"{log_flag} Converting Python data to MATLAB data...", "log")
    # Prepare the parameters
    pars_matlab = converters.dict_to_matlab_struct(pars_validated)
    # Prepare the raster
    raster = input_data[0]
    raster_mat = matlab.double(raster.tolist())
    if logger:
        logger(f"{log_flag} Done converting.", "complete")
    
    if logger:
        logger(f"{log_flag} Starting MATLAB engine...", "log")
    start_time = time.time()
    eng_pca = matlab.engine.start_matlab()
    # Adding to path
    folder_path = os.path.abspath(relative_folder_path)
    folder_path_with_subfolders = eng_pca.genpath(folder_path)
    eng_pca.addpath(folder_path_with_subfolders, nargout=0)
    end_time = time.time()
    engine_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Loaded MATLAB engine.", "complete")
    
    if logger:
        logger(f"{log_flag} Running PCA algorithm...", "log")
    start_time = time.time()
    try:
        answer = eng_pca.raster2ens_by_density(raster_mat, pars_matlab)
    except:
        if logger:
            logger(f"{log_flag} An error occurred while executing the algorithm. Check the Python console for more info.", "error")
        answer = None
    end_time = time.time()
    algorithm_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Done.", "complete")
        logger(f"{log_flag} Terminating MATLAB engine...", "log")
    eng_pca.quit()
    if logger:
        logger(f"{log_flag} Done.", "complete")
    
    ensgui_results = {}
    
    if answer != None:
        # Extract the eigen values parameter from the parameters object
        # so it is easier to use in the plot function
        answer['seleig'] = int(np.array(pars_matlab['npcs'])[0][0])
        
        # Save the results
        ensgui_results['timecourse'] = np.array(answer["sel_ensmat_out"]).astype(int)
        ensgui_results['ensembles_cant'] = ensgui_results['timecourse'].shape[0]
        ensgui_results['neus_in_ens'] = np.array(answer["sel_core_cells"]).T.astype(float)
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

def run_ica(input_data, pars_validated, relative_folder_path = 'analysis/Cell-Assembly-Detection', include_answer = True, logger = None):
    """
    Initializes and runs the MATLAB engine to execute the ICA algorithm on neural activity data. 
    This function also handles MATLAB path setup and data conversion to MATLAB.

    :param raster: Matrix of neural activity data to be processed.
    :type raster: numpy.ndarray
    :param pars_validated: Dictionary with parameters for the SVD algorithm.
    :type pars_validated: dict
    :param relative_folder_path: String with the path to the analysis code. Defaults to 'analysis/Cell-Assembly-Detection'
    :type relative_folder_path: str, optional
    :param include_answer: Flag to indicate wether or not the original full answer of the algorithm. Defaults to True
    :type include_answer: bool, optional
    :param logger: Function to show the log of the function execution. 
        This function should receive two strings as parameters, just like MainWindow.update_console_log.
        Defaults to None.
    :type logger: function, optional
    :return: Dictionary with the result of the algorithm. 
    :rtype: dict
    """
    log_flag = "ICA:"
    success = True
    
    if logger:
        logger(f"{log_flag} Converting Python data to MATLAB data...", "log")
    # Convert the raster
    raster = input_data[0]
    raster_mat = matlab.double(raster.tolist())
    # Convert the parameters
    
    parameters_reformatted = {
        "threshold": {
            "method": pars_validated['threshold_method'],
            "permutations_percentile": pars_validated['permutations_percentile'],
            "number_of_permutations": pars_validated['number_of_permutations'],
        },
        "Patterns": {
            "method": pars_validated['patterns_method'],
            "number_of_iterations": pars_validated['number_of_iterations']
        }
    }
    pars_matlab = converters.dict_to_matlab_struct(parameters_reformatted)
    if logger:
        logger(f"{log_flag} Done converting.", "complete")
    
    if logger:
        logger(f"{log_flag} Starting MATLAB engine...", "log")
    start_time = time.time()
    eng_ica = matlab.engine.start_matlab()
    # Adding to path
    folder_path = os.path.abspath(relative_folder_path)
    folder_path_with_subfolders = eng_ica.genpath(folder_path)
    eng_ica.addpath(folder_path_with_subfolders, nargout=0)
    end_time = time.time()
    engine_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Loaded MATLAB engine.", "complete")
    
    if logger:
        logger(f"{log_flag} Looking for patterns...", "log")
    start_time = time.time()
    try:
        answer = eng_ica.assembly_patterns(raster_mat, pars_matlab)
    except:
        if logger:
            logger(f"{log_flag} An error occurred while executing the algorithm, while looking for patterns. Check the Python console for more info.", "error")
        answer = None
    if logger:
        logger(f"{log_flag} Done looking for patterns.", "complete")
    
    ensgui_results = {}
    original_results = {}
    if include_answer: 
        original_results["original_answer"] = {}

    if answer != None:
        if include_answer: 
            original_results["original_answer"]['patterns'] = answer
        assembly_templates = np.array(answer['AssemblyTemplates']).T
        if logger:
            logger(f"{log_flag} Looking for assembly activity...", "log")
        try:
            answer = eng_ica.assembly_activity(answer['AssemblyTemplates'],raster_mat)
        except:
            if logger:
                logger(f"{log_flag} An error occurred while executing the algorithm, looking for assembly activity. Check the Python console for more info.", "error")
            answer = None
        if logger:
            logger(f"{log_flag} Done looking for assembly activity.", "complete")
    end_time = time.time()
    algorithm_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Done.", "complete")
    if logger:
        logger(f"{log_flag} Terminating MATLAB engine...", "log")
    eng_ica.quit()
    if logger:
        logger(f"{log_flag} Done.", "complete")
    
    if answer != None:
        if include_answer:
            original_results["original_answer"]['assembly_activity'] = answer
        
        time_projection = np.array(answer["time_projection"])
        ## Identify the significative values to binarize the matrix
        threshold = 1.96    # p < 0.05 for the z-score
        binary_assembly_templates = np.zeros(assembly_templates.shape)
        for a_idx, assembly in enumerate(assembly_templates):
            z_scores = stats.zscore(assembly)
            tmp = np.abs(z_scores) > threshold
            binary_assembly_templates[a_idx,:] = [int(v) for v in tmp]

        binary_time_projection = np.zeros(time_projection.shape)
        for a_idx, assembly in enumerate(time_projection):
            z_scores = stats.zscore(assembly)
            tmp = np.abs(z_scores) > threshold
            binary_time_projection[a_idx,:] = [int(v) for v in tmp]

        answer = {
            'assembly_templates': assembly_templates,
            'time_projection': time_projection,
            'binary_assembly_templates': binary_assembly_templates,
            'binary_time_projection': binary_time_projection
        }

        # Save the results
        ensgui_results['timecourse'] = binary_time_projection
        ensgui_results['ensembles_cant'] = binary_time_projection.shape[0]
        ensgui_results['neus_in_ens'] = binary_assembly_templates
    else:
        success = False
    
    results = {
        "results": ensgui_results,
        "engine_time": engine_time,
        "algorithm_time": algorithm_time,
        "success": success
    }

    if include_answer: 
        results["original_answer"] = original_results
        results["answer"] = answer

    return results

def run_x2p(input_data, pars_validated, relative_folder_path = 'analysis/Xsembles2P', include_answer = True, logger = None):
    """
    Initializes and runs the MATLAB engine to execute the Xsembles2P algorithm on neural activity data. 
    This function also handles MATLAB path setup and data conversion to MATLAB.

    :param raster: Matrix of neural activity data to be processed.
    :type raster: numpy.ndarray
    :param pars_validated: Dictionary with parameters for the SVD algorithm.
    :type pars_validated: dict
    :param relative_folder_path: String with the path to the analysis code. Defaults to 'analysis/Xsembles2P'
    :type relative_folder_path: str, optional
    :param include_answer: Flag to indicate wether or not the original full answer of the algorithm. Defaults to True
    :type include_answer: bool, optional
    :param logger: Function to show the log of the function execution. 
        This function should receive two strings as parameters, just like MainWindow.update_console_log.
        Defaults to None.
    :type logger: function, optional
    :return: Dictionary with the result of the algorithm. 
    :rtype: dict
    """
    log_flag = "X2P:"
    success = True
    
    if logger:
        logger(f"{log_flag} Converting Python data to MATLAB data...", "log")
    # Convert the raster to logical MATLAB
    raster = input_data[0]
    raster_mat = matlab.logical(raster.tolist())
    # Convert the parameters
    pars_validated['FileLog'] = ''
    pars_matlab = converters.dict_to_matlab_struct(pars_validated)
    if logger:
        logger(f"{log_flag} Done converting.", "complete")
    
    if logger:
        logger(f"{log_flag} Starting MATLAB engine...", "log")
    start_time = time.time()
    eng_x2p = matlab.engine.start_matlab()
    # Adding to path
    folder_path = os.path.abspath(relative_folder_path)
    folder_path_with_subfolders = eng_x2p.genpath(folder_path)
    eng_x2p.addpath(folder_path_with_subfolders, nargout=0)
    end_time = time.time()
    engine_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Loaded MATLAB engine.", "complete")
    
    if logger:
        logger(f"{log_flag} Running X2P algorithm...", "log")
    start_time = time.time()
    try:
        answer = eng_x2p.Get_Xsembles(raster_mat, pars_matlab)
    except:
        if logger:
            logger(f"{log_flag} An error occurred while executing the algorithm. Check the Python console for more info.", "error")
        answer = None
    end_time = time.time()
    algorithm_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Done.", "complete")
        logger(f"{log_flag} Terminating MATLAB engine...", "log")
    eng_x2p.quit()
    if logger:
        logger(f"{log_flag} Done.", "complete")
        
    ensgui_results = {}
    
    if answer != None:
        start_time = time.time()
        clean_answer = {}
        clean_answer['similarity'] = np.array(answer['Clustering']['Similarity'])
        clean_answer['EPI'] = np.array(answer['Ensembles']['EPI'])
        clean_answer['OnsembleActivity'] = np.array(answer['Ensembles']['OnsembleActivity'])
        clean_answer['OffsembleActivity'] = np.array(answer['Ensembles']['OffsembleActivity'])
        clean_answer['Activity'] = np.array(answer['Ensembles']['Activity'])
        cant_ens = int(answer['Ensembles']['Count'])
        clean_answer['Count'] = cant_ens
        ## Format the onsemble and offsemble neurons
        clean_answer['OnsembleNeurons'] = np.zeros((cant_ens, int(answer['Neurons'])))
        for ens_it in range(cant_ens):
            members = np.array(answer['Ensembles']['OnsembleNeurons'][ens_it]) - 1
            members = members.astype(int)
            clean_answer['OnsembleNeurons'][ens_it, members] = 1
        answer['Ensembles']['OnsembleNeurons'] = clean_answer['OnsembleNeurons']
        clean_answer['OffsembleNeurons'] = np.zeros((cant_ens, int(answer['Neurons'])))
        for ens_it in range(cant_ens):
            members = np.array(answer['Ensembles']['OffsembleNeurons'][ens_it]) - 1
            members = members.astype(int)
            clean_answer['OffsembleNeurons'][ens_it, members] = 1
        answer['Ensembles']['OffsembleNeurons'] = clean_answer['OffsembleNeurons']
        
        # Clean other variables for the h5 save file
        new_clean = {}
        new_clean['Durations'] = {}
        new_clean['Indices'] = {}
        new_clean['Vectors'] = {}
        for ens_it in range(cant_ens):
            new_clean['Durations'][f"{ens_it}"] = np.array(answer['Ensembles']['Durations'][ens_it])
            new_clean['Indices'][f"{ens_it}"] = np.array(answer['Ensembles']['Indices'][ens_it])
            new_clean['Vectors'][f"{ens_it}"] = np.array(answer['Ensembles']['Vectors'][ens_it])
        answer['Ensembles']['Vectors'] = new_clean['Vectors']
        answer['Ensembles']['Indices'] = new_clean['Indices']
        answer['Ensembles']['Durations'] = new_clean['Durations']

        ensgui_results['timecourse'] = clean_answer['Activity']
        ensgui_results['ensembles_cant'] = cant_ens
        ensgui_results['neus_in_ens'] = clean_answer['OnsembleNeurons']
        
    else:
        success = False

    results = {
        "results": ensgui_results,
        "engine_time": engine_time,
        "algorithm_time": algorithm_time,
        "success": success
    }

    if include_answer: 
        results["original_answer"] = answer
        results['answer'] = clean_answer

    return results

def run_sgc(input_data, pars_validated, relative_folder_path = 'analysis/SGC_neural_assembly_detection', include_answer = True, logger = None):
    """
    Initializes and runs the MATLAB engine to execute the SGC algorithm on neural activity data. 
    This function also handles MATLAB path setup and data conversion to MATLAB.

    :param raster: Matrix of neural activity data to be processed.
    :type raster: numpy.ndarray
    :param pars_validated: Dictionary with parameters for the SVD algorithm.
    :type pars_validated: dict
    :param relative_folder_path: String with the path to the analysis code. Defaults to 'analysis/SGC_neural_assembly_detection'
    :type relative_folder_path: str, optional
    :param include_answer: Flag to indicate wether or not the original full answer of the algorithm. Defaults to True
    :type include_answer: bool, optional
    :param logger: Function to show the log of the function execution. 
        This function should receive two strings as parameters, just like MainWindow.update_console_log.
        Defaults to None.
    :type logger: function, optional
    :return: Dictionary with the result of the algorithm. 
    :rtype: dict
    """
    log_flag = "SGC:"
    success = True
    
    if logger:
        logger(f"{log_flag} Converting Python data to MATLAB data...", "log")
    
    # Check for the first derivative flag
    dFFo = input_data[0]
    if pars_validated['use_first_derivative']:
        dx = np.gradient(dFFo, axis=1) # Axis 1 to get the derivative of the signal of every neuron
        dFFo_mat = matlab.double(dx.tolist())
    else:
        # Convert the dFFo matrix
        dFFo_mat = matlab.double(dFFo.tolist())
    # Prepare MATLAB parameters
    pars_matlab = converters.dict_to_matlab_struct(pars_validated)
    if logger:
        logger(f"{log_flag} Done converting.", "complete")
    
    if logger:
        logger(f"{log_flag} Starting MATLAB engine...", "log")
    start_time = time.time()
    eng_sgc = matlab.engine.start_matlab()
    # Adding to path
    folder_path = os.path.abspath(relative_folder_path)
    folder_path_with_subfolders = eng_sgc.genpath(folder_path)
    eng_sgc.addpath(folder_path_with_subfolders, nargout=0)
    end_time = time.time()
    engine_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Loaded MATLAB engine.", "complete")
    
    if logger:
        logger(f"{log_flag} Running SGC algorithm...", "log")
    start_time = time.time()
    try:
        answer = eng_sgc.EnsemblesGUI_linker_SGC(dFFo_mat, pars_matlab)
    except:
        if logger:
            logger(f"{log_flag} An error occurred while executing the algorithm. Check console logs for more info.", "error")
        answer = None
    end_time = time.time()
    algorithm_time = end_time - start_time
    if logger:
        logger(f"{log_flag} Done.", "complete")
        logger(f"{log_flag} Terminating MATLAB engine...", "log")
    eng_sgc.quit()
    if logger:
        logger(f"{log_flag} Done.", "complete")
        
    ensgui_results = {}
    
    if answer != None:
        cant_neurons, cant_timepoints = dFFo.shape
        
        # Extracting neurons in ensembles
        assemblies_raw = answer['assemblies']
        assemblies = [np.array(list(assembly[0])) for assembly in assemblies_raw]
        num_states = len(assemblies)
        neurons_in_ensembles = np.zeros((num_states, cant_neurons))
        for ens_idx, ensmeble in enumerate(assemblies):
            ensemble_fixed = [cell-1 for cell in ensmeble]
            neurons_in_ensembles[ens_idx, ensemble_fixed] = 1
        # Extracting timepoints of activations
        activity_raster_peaks_raw = answer['activity_raster_peaks']
        activity_raster_peaks = np.array([int(np.array(raster_peak)[0][0])-1 for raster_peak in activity_raster_peaks_raw])
        i_assembly_patterns_raw = answer['assembly_pattern_detection']['assemblyIActivityPatterns']
        i_assembly_patterns = [np.array(list(assembly[0])).astype(int) for assembly in i_assembly_patterns_raw]
        # Formatting ensmbles timecourse 
        activations = [activity_raster_peaks[I-1] for I in i_assembly_patterns]
        ensembles_timecourse = np.zeros((num_states, cant_timepoints))
        for ens_idx, ensmeble_activations in enumerate(activations):
            ensembles_timecourse[ens_idx, ensmeble_activations] = 1
        
        # Saving results
        ensgui_results['timecourse'] = ensembles_timecourse
        ensgui_results['ensembles_cant'] = ensembles_timecourse.shape[0]
        ensgui_results['neus_in_ens'] = neurons_in_ensembles
        
        # Correct the answer saved
        answer['assemblies'] = {}
        for idx, assembly in enumerate(assemblies):
            answer['assemblies'][f"{idx}"] = assembly
        answer['assembly_pattern_detection']['assemblyIActivityPatterns'] = {}
        for idx, act_patt in enumerate(i_assembly_patterns):
            answer['assembly_pattern_detection']['assemblyIActivityPatterns'][f"{idx}"] = act_patt

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
        results["answer"]['timecourse'] = ensembles_timecourse
        results["answer"]['neus_in_ens'] = neurons_in_ensembles

    return results

def run_example(input_data: list, params: dict, relative_folder_path='', include_answer=True, logger=None):
    logger(f"Input data len: {len(input_data)}", "log")
    for idx, element in enumerate(input_data):
        logger(f" - Element at idx {idx} has shape {element.shape}", "log")
    
    logger("Parameters passed:", "log")
    logger(f"{params}")
    
    # This is to measure the algorithm running time
    start_time = time.time()
    
    num_neurons, time_points = input_data[1].shape
    
    ## Calculate the data used to update fields in the GUI
    value_A = params['int_parameter_A']
    value_B = params['int_parameter_B']
    variable_to_update_name = 'float_parameter'
    if params['selection_parameter'] == "SUM":  
        new_value = value_A + value_B
    elif params['selection_parameter'] == "MEAN":
        new_value = (value_A + value_B)/2

    ## Generate the dummy results data
    number_of_ensembles = int(params['int_parameter_ensembles'])
    threshold = params['threshold_parameter'] # To binarize the data
    # Generate dummy ensembles activation data
    timecourse = np.random.rand(number_of_ensembles, time_points)
    timecourse_thresholded = (timecourse > threshold).astype(np.int_)
    # Generate dummy neurons in each ensemble
    neurons_in_ensembles = np.random.rand(number_of_ensembles, num_neurons)
    neurons_in_ensembles_thresholded = (neurons_in_ensembles > threshold).astype(np.int_)
    
    ## Create data for the 'answer' field
    data_for_raster = timecourse * 2
    dFFo_data = input_data[0]
    data_for_neuron_dFFo = dFFo_data[0,:]   # dFFo signal for neuron 0
    secondary_dFFo = dFFo_data[1,:]
    many_dFFo = dFFo_data[0:4,:]
    
    ## Finish taking the time for the algorithm
    end_time = time.time()
    algorithm_time = end_time - start_time
    
    # Pack the results
    results = {
        "results": {
            'timecourse': timecourse_thresholded,
            'ensembles_cant': number_of_ensembles,
            'neus_in_ens': neurons_in_ensembles_thresholded
        },
        "answer": {
            "raster": data_for_raster,
            "neuron_dFFo": data_for_neuron_dFFo,
            "other_dFFo": secondary_dFFo,
            "many_neurons_dFFo": many_dFFo
        },
        "update_params":{
            variable_to_update_name: new_value
        },
        "algorithm_time": algorithm_time,
        "success": True
    }
    
    return results