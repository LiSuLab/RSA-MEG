function hirf=boyntonClassHRmodel(res,delta_tau_n)

%returns a vector containing the hemodynamic impulse response function
%as estimated by boynton, engel, glover and heeger (1996) for two subjects' V1.
%the parameter res controls the temporal resolution (time bin width is 1 ms).


if ~exist('delta_tau_n','var')
    % default to Boynton (1996) model
    delta=2500; %as in BV (averages of the subjects?)
    tau=1250;
    n=3;
else
    delta=delta_tau_n(1);
    tau=delta_tau_n(2);
    n=round(abs(delta_tau_n(3)));
end


t=0:res:200000;

hirf=(t/tau).^(n-1).*exp(-t/tau)/(tau*factorial(n-1));
hirf=[zeros(1,floor(delta/res)),hirf];
