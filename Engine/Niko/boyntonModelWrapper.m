res=1; %one-ms resolution
hirf=boyntonModel(res);

t=0:res:20;

prtRes=500; %protocol resolution in ms

prt1=[1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
prt1=[prt1, prt1];
prt2=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0];
prt2=[prt2, prt2];
prt3=[0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
prt3=[prt3, prt3];
prt4=[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0];
prt4=[prt4, prt4];

prt1ms=[];
prt2ms=[];
prt3ms=[];
prt4ms=[];

for i=1:size(prt1,2)
    prt1ms=[prt1ms, ones(1,prtRes)*prt1(i)];
    prt2ms=[prt2ms, ones(1,prtRes)*prt2(i)];
    prt3ms=[prt3ms, ones(1,prtRes)*prt3(i)];
    prt4ms=[prt4ms, ones(1,prtRes)*prt4(i)];
end

hemoPred1=conv(prt1ms,hirf);
hemoPred2=conv(prt2ms,hirf);
hemoPred3=conv(prt3ms,hirf);
hemoPred4=conv(prt4ms,hirf);

figure(1);clf;hold on;
plot(prt1ms/30,'r');
plot(hemoPred1,'r');

plot(prt2ms/30,'b');
plot(hemoPred2,'b');

plot(prt3ms/30,'y');
plot(hemoPred3,'y');

plot(prt4ms/30,'c');
plot(hemoPred4,'c');

corrcoef([hemoPred1',hemoPred2',hemoPred3',hemoPred4'])