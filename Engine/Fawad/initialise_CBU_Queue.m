%FJ 10-2014
% Jana updated 10-2014

function initialise_CBU_Queue(userOptions)

    if userOptions.run_in_parallel || run_in_parallel_in_cluster
        try 
            matlabpool close;
        catch
            disp('Matlabpool initialising ...');
        end
        if userOptions.run_in_parallel_in_cluster
            P=cbupool(userOptions.nWorkers);
            P.ResourceTemplate = ['-l walltime=' num2str(userOptions.wallTime), ',mem=' num2str(userOptions.memReq), 'gb', ',nodes=^N^'];    
            matlabpool(P);
        else
            matlabpool open; 
        end  
    end
end