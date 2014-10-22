function axis_Xmax(newXmax)

xlim=get(gca,'XLim');
xlim(2)=newXmax;
set(gca,'XLim',xlim);
