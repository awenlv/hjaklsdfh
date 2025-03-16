
clear;clc;
%% garment factory

% parameter
C = 11.3;%1.003*1.293*6*280*2.8/0.15/3600

R = 0.0202;

yita = 2.8;
Pres = 46;

deltat = 0:0.1:2+eps;%调节时间
ttrans = 3:-0.1:0+eps;%温升范围
[ttran,Deltat] = meshgrid(ttrans,deltat);
Tin1 = 26.65*ones(length(Deltat(:,1)),length(Deltat(1,:)));
Tin2 = Tin1 + Deltat;

Tout = 30;
M = exp(ttran/(R*C));
Preduce = ((ttran-(R*C*log((Tout-Tin1)./(Tout-Tin2)))) <= 0).*((Tout-Tin1)./(yita*R)+Pres)...
    +((ttran-(R*C*log((Tout-Tin1)./(Tout-Tin2)))) > 0).*((M.*Deltat)./(yita*R.*(M-1)));
%% plot
figure
set(gcf,'unit','centimeters','position',[0,0,8,6])
subplot(1,2,1)
meshz(ttran,Deltat,Preduce);
hold on 
a = plot3(ttrans,1*ones(1,31),Preduce(11,:),'b-','LineWidth',2);
hold on 
b = plot3(1.5*ones(1,21),deltat,Preduce(:,16),'r-','LineWidth',2);
% legend([a b],'固定温升范围1℃','固定调节时间1.5h')
xlabel('\fontsize{8}\fontname{Times new roman}duration (h)')
ylabel('\fontsize{8}\fontname{Times new roman}\Delta\itT\rm (℃)');
zlabel('\fontsize{8}\fontname{Times new roman}Power (kW)')
set(gca,'FontName','Times New Roman','FontSize',8)
view(135,30)
box off

hold on
Tout = 30;
M = exp(ttran/(R*C));
Preduce = ((ttran-(R*C*log((Tout-Tin1)./(Tout-Tin2)))) <= 0).*((Tout-Tin1)./(yita*R)+Pres)...
    +((ttran-(R*C*log((Tout-Tin1)./(Tout-Tin2)))) > 0).*((M.*Deltat)./(yita*R.*(M-1)));
%% hotel
%parameter
C = 15.3;%1.003*1.293*3*506*2.8/0.1/3600
R = 0.0118;
yita = 2.8;
Pres = 278;
deltat = 0:0.1:2+eps;%调节时间
ttrans = 3:-0.1:0+eps;%温升范围
[ttran,Deltat] = meshgrid(ttrans,deltat);
Tin1 = 25.5*ones(length(Deltat(:,1)),length(Deltat(1,:)));
Tin2 = Tin1 + Deltat;

Tout = 30;
M = exp(ttran/(R*C));
Preduce = ((ttran-(R*C*log((Tout-Tin1)./(Tout-Tin2)))) <= 0).*((Tout-Tin1)./(yita*R)+Pres)...
    +((ttran-(R*C*log((Tout-Tin1)./(Tout-Tin2)))) > 0).*((M.*Deltat)./(yita*R.*(M-1)));

subplot(1,2,2)
meshz(ttran,Deltat,Preduce);
hold on 
a = plot3(ttrans,1*ones(1,31),Preduce(11,:),'b-','LineWidth',2);
hold on 
b = plot3(1.5*ones(1,21),deltat,Preduce(:,16),'r-','LineWidth',2);
% legend([a b],'固定温升范围1℃','固定调节时间1.5h')
xlabel('\fontsize{8}\fontname{Times new roman}duration (h)')
ylabel('\fontsize{8}\fontname{Times new roman}\Delta\itT\rm (℃)');
zlabel('\fontsize{8}\fontname{Times new roman}Power (kW)')
set(gca,'FontName','Times New Roman','FontSize',8)
view(135,30)
box off