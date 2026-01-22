function [answer] = EnsemblesGUI_linker_SGC(ffo, pars)
    disp(" - MAT SGC: Finding significant dFF coactivity")
    [activity_raster, activity_raster_threshold, activity_raster_peaks] = findSignificantDF_FCoactivity(ffo', pars);

    % From SGC_ASSEMBLY_DETECTION
    disp(" - MAT SGC: Getting activity patterns")
    activity_patterns = num2cell(activity_raster(activity_raster_peaks, :), 2);
    disp(" - MAT SGC: Finding assembly patterns")
    assembly_pattern_detection = findAssemblyPatterns(activity_patterns, pars);
    disp(" - MAT SGC: Formatting assemblies")
    % Initialize the output cell array to match the input size
    numPatterns = numel(assembly_pattern_detection.assemblyActivityPatterns);
    assemblies = cell(numPatterns, 1);

    % Loop through each cell in assemblyActivityPatterns
    for i = 1:numPatterns
        % Extract the current pattern
        currentPattern = assembly_pattern_detection.assemblyActivityPatterns{i};
        % Find indices of non-zero elements
        nonZeroIndices = find(currentPattern ~= 0);
        % Convert indices to uint16 format
        assemblies{i} = uint16(nonZeroIndices);
        % Optional: Display for debugging purposes
        fprintf('   - MAT SGC: Pattern %d: Found %d active elements\n', i, numel(nonZeroIndices));
    end

    bkup = assembly_pattern_detection.patternSimilarityAnalysis.graph;
    edges = table2array(bkup.Edges);
    edges_labels = bkup.Edges.Properties.VariableNames;
    nodes = table2array(bkup.Nodes);
    nodes_labels = bkup.Nodes.Properties.VariableNames;
    assembly_pattern_detection.patternSimilarityAnalysis.graph = {};
    assembly_pattern_detection.patternSimilarityAnalysis.graph.edges = edges;
    assembly_pattern_detection.patternSimilarityAnalysis.graph.edges_labels = edges_labels;
    assembly_pattern_detection.patternSimilarityAnalysis.graph.nodes = nodes;
    assembly_pattern_detection.patternSimilarityAnalysis.graph.nodes_labels = nodes_labels;

    disp(" - MAT SGC: Packing results...")
    % Return answer to EnseemblesGUI
    answer.activity_raster = activity_raster;
    answer.activity_raster_threshold = activity_raster_threshold;
    answer.activity_raster_peaks = activity_raster_peaks;
    answer.activity_patterns = activity_patterns;
    assembly_pattern_detection.patternSimilarityAnalysis.communityStructure = rmfield(assembly_pattern_detection.patternSimilarityAnalysis.communityStructure, 'markovChainMonteCarloSamples');
    answer.assembly_pattern_detection = assembly_pattern_detection;
    answer.assemblies = assemblies;
end