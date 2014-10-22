res=1; %one-ms resolution
hirf=boyntonModel(res);

t=0:res:20;

prtRes=400; %protocol resolution in ms

prt1=[1 0 0 0 0 0 0 0 0 0];

prt1ms=[];

for i=1:size(prt1,2)
    prt1ms=[prt1ms, ones(1,prtRes)*prt1(i)];
end

hemoPred1=conv(prt1ms,hirf);

figure(1);clf;hold on;
plot(prt1ms/30,'r');
plot(hemoPred1,'r');

samplingRes=1500;

%i=samplingRes/2:samplingRes:24000 %start in the middle of the first scan volume
i=samplingRes/4:samplingRes:24000 %start in the middle of the first scan volume
%i=1:samplingRes:24000 %start at the beginning of the first scan volume
subsampledHemoPred=hemoPred1(i);

subsampledHemoPred'

figure;
plot(subsampledHemoPred);

%plot(prt2ms/30,'b');
%plot(hemoPred2,'b');

%plot(prt3ms/30,'y');
%plot(hemoPred3,'y');

%plot(prt4ms/30,'c');
%plot(hemoPred4,'c');

%corrcoef([hemoPred1',hemoPred2',hemoPred3',hemoPred4'])