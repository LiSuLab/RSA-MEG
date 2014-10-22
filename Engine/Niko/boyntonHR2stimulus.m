function subsampledHR=boyntonHR2stimulus(stimDur_ms, TR_ms, nTRs, vis)

HRsamplingPeriod_ms=24000;
if nargin>2
    HRsamplingPeriod_ms=nTRs*TR_ms;
end

if nargin<4
    vis=0;
end

res=1; %one-ms resolution
hirf_ms=boyntonModel(res);

t=0:res:20;

prtRes_ms=stimDur_ms; %protocol resolution in ms

prt=[1, zeros(1,HRsamplingPeriod_ms/prtRes_ms)];

prt_ms=[];

for i=1:size(prt,2)
    prt_ms=[prt_ms, ones(1,prtRes_ms)*prt(i)];
end

HR_ms=conv(prt_ms,hirf_ms);

i=ceil(TR_ms/2):TR_ms:HRsamplingPeriod_ms; %start in the middle of the first scan volume
%i=1:TR_ms:HRsamplingPeriod_ms %start at the beginning of the first scan volume
subsampledHR=HR_ms(i);


if vis
    figure(1);clf;

    subplot(3,1,1);
    plot(prt_ms, 'k', 'LineWidth',3);
    axis([0 HRsamplingPeriod_ms 0 1.2]);
    title('stimulus');

    subplot(3,1,2);
    plot(HR_ms, 'k', 'LineWidth',3);
    axis([0 HRsamplingPeriod_ms 0 max(HR_ms)]);
    title('hemodynamic response [ms]');
    
    subplot(3,1,3);
    plot(subsampledHR, 'k', 'LineWidth',3);
    axis([1 nTRs 0 max(subsampledHR)]);
    title(['hemodynamic response [TR], TR=',num2str(TR_ms),'ms']);
end