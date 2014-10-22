function axis_Xmin(newXmin)

xlim=get(gca,'XLim');
xlim(1)=newXmin;
set(gca,'XLim',xlim);
