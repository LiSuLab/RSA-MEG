function axis_Ymax(newYmax)

ylim=get(gca,'YLim');
ylim(2)=newYmax;
set(gca,'YLim',ylim);
