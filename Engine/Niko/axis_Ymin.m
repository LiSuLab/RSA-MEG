function axis_Ymin(newYmin)

ylim=get(gca,'YLim');
ylim(1)=newYmin;
set(gca,'YLim',ylim);
