function [smm_bs, smm_ds] = glm_searchlightMapping_MEG_source(singleMesh, Models, userOptions)

%  [smm_rs, smm_ps, n, searchlightRDMs] = searchlightMapping_MEG_source(singleMesh, Models, mask, userOptions, localOptions)
%  based on Li Su's script
% CW 5-2010, last updated by Li Su 3-2012

%% Get parameters

% Prepare Models
nModels = size(Models, 2);

if nModels>1
    modelRDMs_utv = squeeze(unwrapRDMs(vectorizeRDMs(Models)))'; % note by IZ: if Models has 1 model, this is a column vector, else models are put into rows
else
    modelRDMs_utv = squeeze(unwrapRDMs(vectorizeRDMs(Models)));
end

nVertices = userOptions.nVertices;
nConditions = userOptions.nConditions;
%nSessions = userOptions.nSessions;

% How long is the stimulus (in data points)?
epochLength = size(singleMesh, 2); % (vertex, TIME, condition, session)

% Number of DATA points to loop with given the width and time step of
% searchlight updated by IZ 09-12
nTimePoints = floor((epochLength - (userOptions.temporalSearchlightWidth * userOptions.toDataPoints)) / ...
    (userOptions.temporalSearchlightResolution * userOptions.toDataPoints * userOptions.temporalDownsampleRate));


%% similarity-graph-map the volume with the searchlight

%n = nan(nVertices);
%searchlightRDMs = nan([nConditions, nConditions, nVertices, nTimePoints]);

% vertices change on the basis of maskng flag's value IZ 11-12
% updated: all searchlight run as masks IZ 03/12
vertices = userOptions.maskIndices.(userOptions.chi);


% Preallocate looped matrices for speed
smm_bs = zeros([nVertices, nTimePoints, nModels]);
smm_ds = zeros([nVertices, nTimePoints]);

% if strcmp(userOptions.groupStats,'FFX')
%     searchlightRDMs = single(zeros(nConditions,nConditions, nVertices, nTimePoints));
% end

for k = 1:length(vertices)
    vertex = vertices(k);
    % Determine which vertexes are within the radius of the currently-picked vertex
    
    verticesCurrentlyWithinRadius = userOptions.adjacencyMatrix(vertex,:);
    
    % remove nans
    verticesCurrentlyWithinRadius = verticesCurrentlyWithinRadius(~isnan(verticesCurrentlyWithinRadius));
    
    % add current vertex
    verticesCurrentlyWithinRadius = [vertex, verticesCurrentlyWithinRadius]; % add by Li Su 1-02-201
    
    % If masks are used, finding corresponding mask indices - update IZ 11-12
    if userOptions.maskingFlag
        location = ismember(verticesCurrentlyWithinRadius, vertices);
        verticesCurrentlyWithinRadius= verticesCurrentlyWithinRadius(location);
    end
    
    for t = 1:nTimePoints+1
        
        % Work out the current time window
        % converted to data points - updated by IZ 09-12
        currentTimeStart = (t - 1) * ...
            (userOptions.temporalSearchlightResolution * userOptions.temporalDownsampleRate * userOptions.toDataPoints) + 1;
        currentTimeWindow = ceil((currentTimeStart : currentTimeStart + ...
            (userOptions.temporalSearchlightWidth * userOptions.toDataPoints) - 1));
        
        currentData = singleMesh(verticesCurrentlyWithinRadius, currentTimeWindow, :, :); % (vertices, time, condition, session)
        
        searchlightRDM = zeros(nConditions, nConditions);
        
        % Average across sessions
            
        % Median over the time window
        switch lower(userOptions.searchlightPatterns)
            case 'spatial'
                % Spatial patterns: median over time window
                currentData = median(currentData, 2); % (vertices, 1, conditions, sessions)
                currentData = squeeze(currentData); % (vertices, conditions, sessions);
            case 'temporal'
                % Temporal patterns: mean over vertices within searchlight
                currentData = mean(currentData, 1); % (1, timePoints, conditions, sessions)
                currentData = squeeze(currentData); % (timePionts, conditions, sessions)
            case 'spatiotemporal'
                % Spatiotemporal patterns: all the data concatenated
                currentData = reshape(currentData, [], size(currentData, 3), size(currentData, 4)); % (dataPoints, conditions, sessions)
        end%switch:userOptions.sensorSearchlightPatterns

        for session = 1:userOptions.nSessions

            searchlightRDM = searchlightRDM + squareform(pdist(squeeze(currentData(:,:,session))',userOptions.distance));

        end%for:sessions
        
        searchlightRDM = searchlightRDM / userOptions.nSessions;
        
        searchlightRDM = vectorizeRDM(searchlightRDM);
        
        [bs, d] = glmfit(modelRDMs_utv', searchlightRDM', 'normal');
        
        % ignoring that all-1s predictor
        smm_bs(vertex, t, :) = bs(2:end);
        smm_ds(vertex, t) = d;
        
    end%for:t
    
    % Indicate progress every once in a while...
    if mod(vertex, floor(length(vertices) / 20)) == 0, fprintf('.'); end%if
    
end%for:vertices

end%function
