
% The Baseload_Umass_homeG_30min.m is used for load decomposition of residential customer in Western Massachusetts,USA
%% load data¡ª¡ªUmass 1st,Jan,2016--31st,Dec,2016

load HomeGdata30min.mat


%% clustering of march
Pcluster=P3;
temp3=T3;
[idx,C,sumD]=kmeans(Pcluster,4);
[Y,I]=sort(mean(C'));
P3baseloadweekday=Pcluster(find(idx==I(1)),:);
P3baseloadweekend=Pcluster(find(idx==I(3)),:);
P3otherweekday=Pcluster(find(idx==I(2)),:);
P3otherweekend=Pcluster(find(idx==I(4)),:);
weekday3=[find(idx==I(1));find(idx==I(2))];
weekend3=[find(idx==I(3));find(idx==I(4))];

cluster3=zeros(size(idx,1),1);
cluster3(find(idx==I(1)),1)=1;
cluster3(find(idx==I(3)),1)=2;
cluster_3=zeros(size(idx,1),1);
cluster_3([find(idx==I(1));find(idx==I(3))],1)=1;
P3baseloadweekdaytrue=baseload3(find(idx==I(1)),:);
P3baseloadweekendtrue=baseload3(find(idx==I(3)),:);
temp3weekday=temp3(find(idx==I(1)),:);
temp3weekend=temp3(find(idx==I(3)),:);
temp3otherweekday=temp3(find(idx==I(2)),:);
temp3otherweekend=temp3(find(idx==I(4)),:);

figure(4)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp3weekday;temp3weekend],[P3baseloadweekday;P3baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

figure(5)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(P3baseloadweekday','r-')
hold on
plot(P3baseloadweekend','b-')
hold on
plot(P3otherweekday','y-')
hold on
plot(P3otherweekend','m-')
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

figure(6)
set(gcf,'unit','centimeters','position',[0,0,16,12])

plot(P3baseloadweekdaytrue','k-.','LineWidth',1)
hold on
plot(P3baseloadweekendtrue','b-.','LineWidth',1)
hold on
plot(C(I(3),:),'k-','LineWidth',4)
hold on
plot(C(I(1),:),'b-','LineWidth',4)
hold on
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')
%% weekday correction
%% correlation analysis
temp31=reshape([temp3weekday;temp3weekend],size([temp3weekday;temp3weekend],1)*size([temp3weekday;temp3weekend],2),1);
P3baseload1=reshape([P3baseloadweekday;P3baseloadweekend],size([P3baseloadweekday;P3baseloadweekend],1)*size([P3baseloadweekday;P3baseloadweekend],2),1);
%% initial CP
m = 0;
for i = 60:0.2:80
    m = m+1;

        ft = fittype(@(a,b,x)(x>=i).*(a.*x+b));
        [fitresult,goodness] = fit( temp31, P3baseload1, ft,'StartPoint',[0.05 -2]);
        % k0=15; b0=300
       resnorm(1,m) = goodness.sse;%SSE
 
end

[resnormmin3] = min(resnorm);
[i] = find(resnorm==resnormmin3);
CP = 60+(i(1)-1)*0.2;
Ct3=zeros(size(temp3weekday,1),size(temp3weekday,2));
for i=1:size(temp3weekday,1)
    for j=1:size(temp3weekday,2)
        if (P3baseloadweekday(i,j)>=0.6*max(P3baseloadweekday(i,:)))
            Ct3(i,j)=1;
        end

    end
end

Ct31=zeros(size(temp3weekday,1),size(temp3weekday,2));
error31=length(find(abs(Ct31-Ct3)==1))./(size(temp3weekday,1)*size(temp3weekday,2));
Ct31=Ct3;
%% Ct
while error31>=0.001
    
[a,b]=(find(Ct3==1));
if size(P3baseloadweekday,1)>=2
loadloop1=P3baseloadweekday(find(Ct3==1));
temploop1=temp3weekday(find(Ct3==1));
loadloop2=P3baseloadweekday(find(Ct3==0));
temploop2=temp3weekday(find(Ct3==0));

x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P3baseloadweekday,1)==1
loadloop1=P3baseloadweekday(find(Ct3==1));
temploop1=temp3weekday(find(Ct3==1));
loadloop2=P3baseloadweekday(find(Ct3==0));
temploop2=temp3weekday(find(Ct3==0));

x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp3weekday,1),size(temp3weekday,2));
r2=zeros(size(temp3weekday,1),size(temp3weekday,2));
for i=1:size(temp3weekday,1)
    for j=1:size(temp3weekday,2)
if temp3weekday(i,j)>CP
   r1(i,j)=P3baseloadweekday(i,j)-b(2)*temp3weekday(i,j)-b(1);
   r2(i,j)=P3baseloadweekday(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct31(i,j)=1;
   end
end
    end
end
error31=length(find(abs(Ct31-Ct3)==1))./(size(temp3weekday,1)*size(temp3weekday,2));
Ct3=Ct31;
end
if size(P3baseloadweekday,1)==1
P3baseloadweekday2=P3baseloadweekday(find(Ct3==1))';
temp3weekday2=temp3weekday(find(Ct3==1))';
P3baseloadweekday3=P3baseloadweekday(find(Ct3==0))';
temp3weekday3=temp3weekday(find(Ct3==0))';
Ct3weekday=Ct3;
end
if size(P3baseloadweekday,1)~=1
P3baseloadweekday2=P3baseloadweekday(find(Ct3==1));
temp3weekday2=temp3weekday(find(Ct3==1));
P3baseloadweekday3=P3baseloadweekday(find(Ct3==0));
temp3weekday3=temp3weekday(find(Ct3==0));
Ct3weekday=Ct3;
end
%% correction for otherload

for i=1:size(P3baseloadweekday,2)

    if find(Ct3(:,i)==1)
        if length(find(Ct3(:,i)==1))<size(P3baseloadweekday,1)
P3baseloadweekday(find(Ct3(:,i)==1),i)=ones(size(find(Ct3(:,i)==1),1),1)*mean(P3baseloadweekday(setdiff([1:size(P3baseloadweekday,1)]',find(Ct3(:,i)==1)),i));

        end
      if length(find(Ct3(:,i)==1))==size(P3baseloadweekday,1)
          
    P3baseloadweekday(:,i)=ones(size(P3baseloadweekday,1),1)*mean(P3baseloadweekday(find(Ct3==0)))*rand();

      end
    end
    
end

%% weekend correction


Ct3=zeros(size(temp3weekend,1),size(temp3weekend,2));


for i=1:size(temp3weekend,1)
    for j=1:size(temp3weekend,2)
        if (P3baseloadweekend(i,j)>=0.6*max(P3baseloadweekend(i,:)))
            Ct3(i,j)=1;
        end

    end
end

Ct31=zeros(size(temp3weekend,1),size(temp3weekend,2));
error31=length(find(abs(Ct31-Ct3)==1))./(size(temp3weekend,1)*size(temp3weekend,2));
Ct31=Ct3;

%% Ct
while error31>=0.001
    
[a,b]=(find(Ct3==1));
loadloop1=P3baseloadweekend(find(Ct3==1));
temploop1=temp3weekend(find(Ct3==1));
loadloop2=P3baseloadweekend(find(Ct3==0));
temploop2=temp3weekend(find(Ct3==0));

if size(P3baseloadweekend,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P3baseloadweekend,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp3weekend,1),size(temp3weekend,2));
r2=zeros(size(temp3weekend,1),size(temp3weekend,2));
for i=1:size(temp3weekend,1)
    for j=1:size(temp3weekend,2)
if temp3weekend(i,j)>CP
   r1(i,j)=P3baseloadweekend(i,j)-b(2)*temp3weekend(i,j)-b(1);
   r2(i,j)=P3baseloadweekend(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct31(i,j)=1;
   end
end
    end
end
error31=length(find(abs(Ct31-Ct3)==1))./(size(temp3weekend,1)*size(temp3weekend,2));
Ct3=Ct31;
end
if size(P3baseloadweekend,1)==1
P3baseloadweekend2=P3baseloadweekend(find(Ct3==1))';
temp3weekend2=temp3weekend(find(Ct3==1))';
P3baseloadweekend3=P3baseloadweekend(find(Ct3==0))';
temp3weekend3=temp3weekend(find(Ct3==0))';
Ct3weekend=Ct3;
end
if size(P3baseloadweekend,1)~=1
P3baseloadweekend2=P3baseloadweekend(find(Ct3==1));
temp3weekend2=temp3weekend(find(Ct3==1));
P3baseloadweekend3=P3baseloadweekend(find(Ct3==0));
temp3weekend3=temp3weekend(find(Ct3==0));
Ct3weekend=Ct3;
end

%% correction for otherload

for i=1:size(P3baseloadweekend,2)

    if find(Ct3(:,i)==1)
        if length(find(Ct3(:,i)==1))<size(P3baseloadweekend,1)
P3baseloadweekend(find(Ct3(:,i)==1),i)=ones(size(find(Ct3(:,i)==1),1),1)*mean(P3baseloadweekend(setdiff([1:size(P3baseloadweekend,1)]',find(Ct3(:,i)==1)),i));

        end
      if length(find(Ct3(:,i)==1))==size(P3baseloadweekend,1)
          
    P3baseloadweekend(:,i)=ones(size(P3baseloadweekend,1),1)*mean(P3baseloadweekend(find(Ct3==0)))*rand();

      end
    end
    
end

figure(7)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp3weekday;temp3weekend],[P3baseloadweekday;P3baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

figure(8)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(mean(P3baseloadweekdaytrue),'k-.','LineWidth',1)
hold on
plot(mean(P3baseloadweekendtrue),'b-.','LineWidth',1)
hold on
plot(mean(P3baseloadweekday),'k-','LineWidth',2)
hold on
plot(mean(P3baseloadweekend),'b-','LineWidth',2)
hold on
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

%% clustering of April
Pcluster=P4;
temp4=T4;
[idx,C,sumD]=kmeans(Pcluster,4);

[Y,I]=sort(mean(C'));
P4baseloadweekday=Pcluster(find(idx==I(1)),:);
P4baseloadweekend=Pcluster(find(idx==I(3)),:);
P4otherweekday=Pcluster(find(idx==I(2)),:);
P4otherweekend=Pcluster(find(idx==I(4)),:);
cluster4=zeros(size(idx,1),1);
cluster4(find(idx==I(1)),1)=1;
cluster4(find(idx==I(3)),1)=2;
cluster_4=zeros(size(idx,1),1);
cluster_4([find(idx==I(1));find(idx==I(3))],1)=1;
P4baseloadweekdaytrue=baseload4(find(idx==I(1)),:);
P4baseloadweekendtrue=baseload4(find(idx==I(4)),:);
weekday4=[find(idx==I(1));find(idx==I(2))];
weekend4=[find(idx==I(3));find(idx==I(3))];

temp4weekday=temp4(find(idx==I(1)),:);
temp4weekend=temp4(find(idx==I(3)),:);
temp4otherweekday=temp4(find(idx==I(2)),:);
temp4otherweekend=temp4(find(idx==I(4)),:);

figure(7)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp3weekday;temp3weekend],[P3baseloadweekday;P3baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')
%% weekday correction
%% 
temp41=reshape([temp4weekday;temp4weekend],size([temp4weekday;temp4weekend],1)*size([temp4weekday;temp4weekend],2),1);
P4baseload1=reshape([P4baseloadweekday;P4baseloadweekend],size([P4baseloadweekday;P4baseloadweekend],1)*size([P4baseloadweekday;P4baseloadweekend],2),1);
%% CP
m = 0;
for i = 60:0.2:80
    m = m+1;

        ft = fittype(@(a,b,x)(x>=i).*(a.*x+b));
        [fitresult,goodness] = fit( temp41, P4baseload1, ft,'StartPoint',[0.05 -2]);
        % k0=15; b0=300
       resnorm(1,m) = goodness.sse;%SSE
 
end

[resnormmin4] = min(resnorm);
[i] = find(resnorm==resnormmin4);
CP = 60+(i(1)-1)*0.2;

Ct4=zeros(size(temp4weekday,1),size(temp4weekday,2));
for i=1:size(temp4weekday,1)
    for j=1:size(temp4weekday,2)
        if (P4baseloadweekday(i,j)>=0.6*max(P4baseloadweekday(i,:)))
            Ct4(i,j)=1;
        end

    end
end

Ct41=zeros(size(temp4weekday,1),size(temp4weekday,2));
error41=length(find(abs(Ct41-Ct4)==1))./(size(temp4weekday,1)*size(temp4weekday,2));
Ct41=Ct4;
%% Ct
while error41>=0.001
    
[a,b]=(find(Ct4==1));
loadloop1=P4baseloadweekday(find(Ct4==1));
temploop1=temp4weekday(find(Ct4==1));
loadloop2=P4baseloadweekday(find(Ct4==0));
temploop2=temp4weekday(find(Ct4==0));

if size(P4baseloadweekday,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P4baseloadweekday,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp4weekday,1),size(temp4weekday,2));
r2=zeros(size(temp4weekday,1),size(temp4weekday,2));
for i=1:size(temp4weekday,1)
    for j=1:size(temp4weekday,2)
if temp4weekday(i,j)>CP
   r1(i,j)=P4baseloadweekday(i,j)-b(2)*temp4weekday(i,j)-b(1);
   r2(i,j)=P4baseloadweekday(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct41(i,j)=1;
   end
end
    end
end
error41=length(find(abs(Ct41-Ct4)==1))./(size(temp4weekday,1)*size(temp4weekday,2));
Ct4=Ct41;
end
if size(P4baseloadweekday,1)==1
P4baseloadweekday2=P4baseloadweekday(find(Ct4==1))';
temp4weekday2=temp4weekday(find(Ct4==1))';
P4baseloadweekday3=P4baseloadweekday(find(Ct4==0))';
temp4weekday3=temp4weekday(find(Ct4==0))';
Ct4weekday=Ct4;
end

if size(P4baseloadweekday,1)~=1
P4baseloadweekday2=P4baseloadweekday(find(Ct4==1));
temp4weekday2=temp4weekday(find(Ct4==1));
P4baseloadweekday3=P4baseloadweekday(find(Ct4==0));
temp4weekday3=temp4weekday(find(Ct4==0));
Ct4weekday=Ct4;
end

%% correction

for i=1:size(P4baseloadweekday,2)

    if find(Ct4(:,i)==1)
        if length(find(Ct4(:,i)==1))<size(P4baseloadweekday,1)
P4baseloadweekday(find(Ct4(:,i)==1),i)=ones(size(find(Ct4(:,i)==1),1),1)*mean(P4baseloadweekday(setdiff([1:size(P4baseloadweekday,1)]',find(Ct4(:,i)==1)),i));

        end
      if length(find(Ct4(:,i)==1))==size(P4baseloadweekday,1)
          
    P4baseloadweekday(:,i)=ones(size(P4baseloadweekday,1),1)*mean(P4baseloadweekday(find(Ct4==0)))*rand();

      end
    end
    
end

%% weekend correction
Ct4=zeros(size(temp4weekend,1),size(temp4weekend,2));


for i=1:size(temp4weekend,1)
    for j=1:size(temp4weekend,2)
        if (P4baseloadweekend(i,j)>=0.6*max(P4baseloadweekend(i,:)))
            Ct4(i,j)=1;
        end

    end
end

Ct41=zeros(size(temp4weekend,1),size(temp4weekend,2));
error41=length(find(abs(Ct41-Ct4)==1))./(size(temp4weekend,1)*size(temp4weekend,2));
Ct41=Ct4;
%% Ct
while error41>=0.001
    
[a,b]=(find(Ct4==1));
loadloop1=P4baseloadweekend(find(Ct4==1));
temploop1=temp4weekend(find(Ct4==1));
loadloop2=P4baseloadweekend(find(Ct4==0));
temploop2=temp4weekend(find(Ct4==0));

if size(P4baseloadweekend,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P4baseloadweekend,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end
r1=zeros(size(temp4weekend,1),size(temp4weekend,2));
r2=zeros(size(temp4weekend,1),size(temp4weekend,2));
for i=1:size(temp4weekend,1)
    for j=1:size(temp4weekend,2)
if temp4weekend(i,j)>CP
   r1(i,j)=P4baseloadweekend(i,j)-b(2)*temp4weekend(i,j)-b(1);
   r2(i,j)=P4baseloadweekend(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct41(i,j)=1;
   end
end
    end
end
error41=length(find(abs(Ct41-Ct4)==1))./(size(temp4weekend,1)*size(temp4weekend,2));
Ct4=Ct41;
end
if size(P4baseloadweekend,1)==1
P4baseloadweekend2=P4baseloadweekend(find(Ct4==1))';
temp4weekend2=temp4weekend(find(Ct4==1))';
P4baseloadweekend3=P4baseloadweekend(find(Ct4==0))';
temp4weekend3=temp4weekend(find(Ct4==0))';
Ct4weekend=Ct4;
end

if size(P4baseloadweekend,1)~=1
P4baseloadweekend2=P4baseloadweekend(find(Ct4==1));
temp4weekend2=temp4weekend(find(Ct4==1));
P4baseloadweekend3=P4baseloadweekend(find(Ct4==0));
temp4weekend3=temp4weekend(find(Ct4==0));
Ct4weekend=Ct4;
end

%% correction
for i=1:size(P4baseloadweekend,2)

    if find(Ct4(:,i)==1)
        if length(find(Ct4(:,i)==1))<size(P4baseloadweekend,1)
P4baseloadweekend(find(Ct4(:,i)==1),i)=ones(size(find(Ct4(:,i)==1),1),1)*mean(P4baseloadweekend(setdiff([1:size(P4baseloadweekend,1)]',find(Ct4(:,i)==1)),i));

        end
      if length(find(Ct4(:,i)==1))==size(P4baseloadweekend,1)
          
    P4baseloadweekend(:,i)=ones(size(P4baseloadweekend,1),1)*mean(P4baseloadweekend(find(Ct4==0)))*rand();

      end
    end
    
end

figure(8)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp4weekday;temp4weekend],[P4baseloadweekday;P4baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

figure(9)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(mean(P4baseloadweekdaytrue),'k-.','LineWidth',1)
hold on
plot(mean(P4baseloadweekendtrue),'b-.','LineWidth',1)
hold on
plot(mean(P4baseloadweekday),'k-','LineWidth',2)
hold on
plot(mean(P4baseloadweekend),'b-','LineWidth',2)
hold on
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')


%% clustering of May
Pcluster=P5;
temp5=T5;
[idx,C,sumD]=kmeans(Pcluster,4);
[Y,I]=sort(mean(C'));
P5baseloadweekday=Pcluster(find(idx==I(1)),:);
P5baseloadweekend=Pcluster(find(idx==I(3)),:);
P5otherweekday=Pcluster(find(idx==I(2)),:);
P5otherweekend=Pcluster(find(idx==I(4)),:);
cluster5=zeros(size(idx,1),1);
cluster5(find(idx==I(1)),1)=1;
cluster5(find(idx==I(3)),1)=2;
cluster_5=zeros(size(idx,1),1);
cluster_5([find(idx==I(1));find(idx==I(3))],1)=1;
P5baseloadweekdaytrue=baseload5(find(idx==I(1)),:);
P5baseloadweekendtrue=baseload5(find(idx==I(3)),:);
weekday5=[find(idx==I(1));find(idx==I(2))];
weekend5=[find(idx==I(3));find(idx==I(4))];
temp5weekday=temp5(find(idx==I(1)),:);
temp5weekend=temp5(find(idx==I(3)),:);
temp5otherweekday=temp5(find(idx==I(2)),:);
temp5otherweekend=temp5(find(idx==I(4)),:);
figure(10)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp5weekday;temp5weekend],[P5baseloadweekday;P5baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')
%% weekday correction
%% 
temp51=reshape([temp5weekday;temp5weekend],size([temp5weekday;temp5weekend],1)*size([temp5weekday;temp5weekend],2),1);
P5baseload1=reshape([P5baseloadweekday;P5baseloadweekend],size([P5baseloadweekday;P5baseloadweekend],1)*size([P5baseloadweekday;P5baseloadweekend],2),1);
%% CP
m = 0;
for i = 60:0.2:80
    m = m+1;

        ft = fittype(@(a,b,x)(x>=i).*(a.*x+b));
        [fitresult,goodness] = fit( temp51, P5baseload1, ft,'StartPoint',[0.05 -2]);
        % k0=15; b0=500
       resnorm(1,m) = goodness.sse;%SSE
 
end

[resnormmin5] = min(resnorm);
[i] = find(resnorm==resnormmin5);
CP = 60+(i(1)-1)*0.2;
Ct5=zeros(size(temp5weekday,1),size(temp5weekday,2));
for i=1:size(temp5weekday,1)
    for j=1:size(temp5weekday,2)
        if (P5baseloadweekday(i,j)>=0.6*max(P5baseloadweekday(i,:)))
            Ct5(i,j)=1;
        end

    end
end

Ct51=zeros(size(temp5weekday,1),size(temp5weekday,2));
error51=length(find(abs(Ct51-Ct5)==1))./(size(temp5weekday,1)*size(temp5weekday,2));
Ct51=Ct5;
%% Ct
while error51>=0.001
    
[a,b]=(find(Ct5==1));
loadloop1=P5baseloadweekday(find(Ct5==1));
temploop1=temp5weekday(find(Ct5==1));
loadloop2=P5baseloadweekday(find(Ct5==0));
temploop2=temp5weekday(find(Ct5==0));
if size(P5baseloadweekday,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P5baseloadweekday,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp5weekday,1),size(temp5weekday,2));
r2=zeros(size(temp5weekday,1),size(temp5weekday,2));
for i=1:size(temp5weekday,1)
    for j=1:size(temp5weekday,2)
if temp5weekday(i,j)>CP
   r1(i,j)=P5baseloadweekday(i,j)-b(2)*temp5weekday(i,j)-b(1);
   r2(i,j)=P5baseloadweekday(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct51(i,j)=1;
   end
end
    end
end
error51=length(find(abs(Ct51-Ct5)==1))./(size(temp5weekday,1)*size(temp5weekday,2));
Ct5=Ct51;
end
if size(P5baseloadweekday,1)==1
P5baseloadweekday2=P5baseloadweekday(find(Ct5==1))';
temp5weekday2=temp5weekday(find(Ct5==1))';
P5baseloadweekday3=P5baseloadweekday(find(Ct5==0))';
temp5weekday3=temp5weekday(find(Ct5==0))';
Ct5weekday=Ct5;
end

if size(P5baseloadweekday,1)~=1
P5baseloadweekday2=P5baseloadweekday(find(Ct5==1));
temp5weekday2=temp5weekday(find(Ct5==1));
P5baseloadweekday3=P5baseloadweekday(find(Ct5==0));
temp5weekday3=temp5weekday(find(Ct5==0));
Ct5weekday=Ct5;
end
%% correction

for i=1:size(P5baseloadweekday,2)

    if find(Ct5(:,i)==1)
        if length(find(Ct5(:,i)==1))<size(P5baseloadweekday,1)
P5baseloadweekday(find(Ct5(:,i)==1),i)=ones(size(find(Ct5(:,i)==1),1),1)*mean(P5baseloadweekday(setdiff([1:size(P5baseloadweekday,1)]',find(Ct5(:,i)==1)),i));

        end
      if length(find(Ct5(:,i)==1))==size(P5baseloadweekday,1)
          
    P5baseloadweekday(:,i)=ones(size(P5baseloadweekday,1),1)*mean(P5baseloadweekday(find(Ct5==0)))*rand();

      end
    end
    
end

%% weekend correction
Ct5=zeros(size(temp5weekend,1),size(temp5weekend,2));
for i=1:size(temp5weekend,1)
    for j=1:size(temp5weekend,2)
        if (P5baseloadweekend(i,j)>=0.6*max(P5baseloadweekend(i,:)))
            Ct5(i,j)=1;
        end

    end
end

Ct51=zeros(size(temp5weekend,1),size(temp5weekend,2));
error51=length(find(abs(Ct51-Ct5)==1))./(size(temp5weekend,1)*size(temp5weekend,2));
Ct51=Ct5;
%% Ct
while error51>=0.001
    
[a,b]=(find(Ct5==1));
loadloop1=P5baseloadweekend(find(Ct5==1));
temploop1=temp5weekend(find(Ct5==1));
loadloop2=P5baseloadweekend(find(Ct5==0));
temploop2=temp5weekend(find(Ct5==0));
if size(P5baseloadweekend,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P5baseloadweekend,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp5weekend,1),size(temp5weekend,2));
r2=zeros(size(temp5weekend,1),size(temp5weekend,2));
for i=1:size(temp5weekend,1)
    for j=1:size(temp5weekend,2)
if temp5weekend(i,j)>CP
   r1(i,j)=P5baseloadweekend(i,j)-b(2)*temp5weekend(i,j)-b(1);
   r2(i,j)=P5baseloadweekend(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct51(i,j)=1;
   end
end
    end
end
error51=length(find(abs(Ct51-Ct5)==1))./(size(temp5weekend,1)*size(temp5weekend,2));
Ct5=Ct51;
end
if size(P5baseloadweekend,1)==1
P5baseloadweekend2=P5baseloadweekend(find(Ct5==1))';
temp5weekend2=temp5weekend(find(Ct5==1))';
P5baseloadweekend3=P5baseloadweekend(find(Ct5==0))';
temp5weekend3=temp5weekend(find(Ct5==0))';
Ct5weekend=Ct5;
end

if size(P5baseloadweekend,1)~=1
P5baseloadweekend2=P5baseloadweekend(find(Ct5==1));
temp5weekend2=temp5weekend(find(Ct5==1));
P5baseloadweekend3=P5baseloadweekend(find(Ct5==0));
temp5weekend3=temp5weekend(find(Ct5==0));
Ct5weekend=Ct5;
end
%% correction
for i=1:size(P5baseloadweekend,2)

    if find(Ct5(:,i)==1)
        if length(find(Ct5(:,i)==1))<size(P5baseloadweekend,1)
P5baseloadweekend(find(Ct5(:,i)==1),i)=ones(size(find(Ct5(:,i)==1),1),1)*mean(P5baseloadweekend(setdiff([1:size(P5baseloadweekend,1)]',find(Ct5(:,i)==1)),i));

        end
      if length(find(Ct5(:,i)==1))==size(P5baseloadweekend,1)
          
    P5baseloadweekend(:,i)=ones(size(P5baseloadweekend,1),1)*mean(P5baseloadweekend(find(Ct5==0)))*rand();

      end
    end
    
end

figure(11)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp5weekday;temp5weekend],[P5baseloadweekday;P5baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

figure(12)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(mean(P5baseloadweekdaytrue),'k-.','LineWidth',1)
hold on
plot(mean(P5baseloadweekendtrue),'b-.','LineWidth',1)
hold on
plot(mean(P5baseloadweekday),'k-','LineWidth',2)
hold on
plot(mean(P5baseloadweekend),'b-','LineWidth',2)
hold on
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')


%% clustering of October
Pcluster=P10;
temp10=T10;
[idx,C,sumD]=kmeans(Pcluster,4);

[Y,I]=sort(mean(C'));
P10baseloadweekday=Pcluster(find(idx==I(1)),:);
P10baseloadweekend=Pcluster(find(idx==I(3)),:);
P10otherweekday=Pcluster(find(idx==I(2)),:);
P10otherweekend=Pcluster(find(idx==I(4)),:);
cluster10=zeros(size(idx,1),1);
cluster10(find(idx==I(1)),1)=1;
cluster10(find(idx==I(3)),1)=2;
cluster_10=zeros(size(idx,1),1);
cluster_10([find(idx==I(1));find(idx==I(3))],1)=1;
P10baseloadweekdaytrue=baseload10(find(idx==I(1)),:);
P10baseloadweekendtrue=baseload10(find(idx==I(3)),:);
weekday10=[find(idx==I(1));find(idx==I(2))];
weekend10=[find(idx==I(3));find(idx==I(4))];
temp10weekday=temp10(find(idx==I(1)),:);
temp10weekend=temp10(find(idx==I(3)),:);
temp10otherweekday=temp10(find(idx==I(2)),:);
temp10otherweekend=temp10(find(idx==I(4)),:);
figure(13)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp10weekday;temp10weekend],[P10baseloadweekday;P10baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')
%% weekday correction
%% 
temp101=reshape([temp10weekday;temp10weekend],size([temp10weekday;temp10weekend],1)*size([temp10weekday;temp10weekend],2),1);
P10baseload1=reshape([P10baseloadweekday;P10baseloadweekend],size([P10baseloadweekday;P10baseloadweekend],1)*size([P10baseloadweekday;P10baseloadweekend],2),1);
%% CP
m = 0;
for i = 60:0.2:80
    m = m+1;

        ft = fittype(@(a,b,x)(x>=i).*(a.*x+b));
        [fitresult,goodness] = fit( temp101, P10baseload1, ft,'StartPoint',[0.05 -2]);
        % k0=15; b0=1000
       resnorm(1,m) = goodness.sse;%SSE
 
end

[resnormmin10] = min(resnorm);
[i] = find(resnorm==resnormmin10);
CP = 60+(i(1)-1)*0.2;
Ct10=zeros(size(temp10weekday,1),size(temp10weekday,2));
for i=1:size(temp10weekday,1)
    for j=1:size(temp10weekday,2)
        if (P10baseloadweekday(i,j)>=0.6*max(P10baseloadweekday(i,:)))
            Ct10(i,j)=1;
        end

    end
end

Ct101=zeros(size(temp10weekday,1),size(temp10weekday,2));
error101=length(find(abs(Ct101-Ct10)==1))./(size(temp10weekday,1)*size(temp10weekday,2));
Ct101=Ct10;
%% Ct
while error101>=0.001
    
[a,b]=(find(Ct10==1));
loadloop1=P10baseloadweekday(find(Ct10==1));
temploop1=temp10weekday(find(Ct10==1));
loadloop2=P10baseloadweekday(find(Ct10==0));
temploop2=temp10weekday(find(Ct10==0));
if size(P10baseloadweekday,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P10baseloadweekday,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp10weekday,1),size(temp10weekday,2));
r2=zeros(size(temp10weekday,1),size(temp10weekday,2));
for i=1:size(temp10weekday,1)
    for j=1:size(temp10weekday,2)
if temp10weekday(i,j)>CP
   r1(i,j)=P10baseloadweekday(i,j)-b(2)*temp10weekday(i,j)-b(1);
   r2(i,j)=P10baseloadweekday(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct101(i,j)=1;
   end
end
    end
end
error101=length(find(abs(Ct101-Ct10)==1))./(size(temp10weekday,1)*size(temp10weekday,2));
Ct10=Ct101;
end
if size(P10baseloadweekday,1)==1
P10baseloadweekday2=P10baseloadweekday(find(Ct10==1))';
temp10weekday2=temp10weekday(find(Ct10==1))';
P10baseloadweekday3=P10baseloadweekday(find(Ct10==0))';
temp10weekday3=temp10weekday(find(Ct10==0))';
Ct10weekday=Ct10;
end
if size(P10baseloadweekday,1)~=1
P10baseloadweekday2=P10baseloadweekday(find(Ct10==1));
temp10weekday2=temp10weekday(find(Ct10==1));
P10baseloadweekday3=P10baseloadweekday(find(Ct10==0));
temp10weekday3=temp10weekday(find(Ct10==0));
Ct10weekday=Ct10;
end
%% correction

for i=1:size(P10baseloadweekday,2)

    if find(Ct10(:,i)==1)
        if length(find(Ct10(:,i)==1))<size(P10baseloadweekday,1)
P10baseloadweekday(find(Ct10(:,i)==1),i)=ones(size(find(Ct10(:,i)==1),1),1)*mean(P10baseloadweekday(setdiff([1:size(P10baseloadweekday,1)]',find(Ct10(:,i)==1)),i));

        end
      if length(find(Ct10(:,i)==1))==size(P10baseloadweekday,1)
          
    P10baseloadweekday(:,i)=ones(size(P10baseloadweekday,1),1)*mean(P10baseloadweekday(find(Ct10==0)))*rand();

      end
    end
    
end

%% weekend correction
Ct10=zeros(size(temp10weekend,1),size(temp10weekend,2));


for i=1:size(temp10weekend,1)
    for j=1:size(temp10weekend,2)
        if (P10baseloadweekend(i,j)>=0.6*max(P10baseloadweekend(i,:)))
            Ct10(i,j)=1;
        end

    end
end

Ct101=zeros(size(temp10weekend,1),size(temp10weekend,2));
error101=length(find(abs(Ct101-Ct10)==1))./(size(temp10weekend,1)*size(temp10weekend,2));
Ct101=Ct10;
%% Ct
while error101>=0.001
    
[a,b]=(find(Ct10==1));
loadloop1=P10baseloadweekend(find(Ct10==1));
temploop1=temp10weekend(find(Ct10==1));
loadloop2=P10baseloadweekend(find(Ct10==0));
temploop2=temp10weekend(find(Ct10==0));
if size(P10baseloadweekend,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P10baseloadweekend,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp10weekend,1),size(temp10weekend,2));
r2=zeros(size(temp10weekend,1),size(temp10weekend,2));
for i=1:size(temp10weekend,1)
    for j=1:size(temp10weekend,2)
if temp10weekend(i,j)>CP
   r1(i,j)=P10baseloadweekend(i,j)-b(2)*temp10weekend(i,j)-b(1);
   r2(i,j)=P10baseloadweekend(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct101(i,j)=1;
   end
end
    end
end
error101=length(find(abs(Ct101-Ct10)==1))./(size(temp10weekend,1)*size(temp10weekend,2));
Ct10=Ct101;
end
if size(P10baseloadweekend,1)==1
P10baseloadweekend2=P10baseloadweekend(find(Ct10==1))';
temp10weekend2=temp10weekend(find(Ct10==1))';
P10baseloadweekend3=P10baseloadweekend(find(Ct10==0))';
temp10weekend3=temp10weekend(find(Ct10==0))';
Ct10weekend=Ct10;
end

if size(P10baseloadweekend,1)~=1
P10baseloadweekend2=P10baseloadweekend(find(Ct10==1));
temp10weekend2=temp10weekend(find(Ct10==1));
P10baseloadweekend3=P10baseloadweekend(find(Ct10==0));
temp10weekend3=temp10weekend(find(Ct10==0));
Ct10weekend=Ct10;
end
%% correction
for i=1:size(P10baseloadweekend,2)

    if find(Ct10(:,i)==1)
        if length(find(Ct10(:,i)==1))<size(P10baseloadweekend,1)
P10baseloadweekend(find(Ct10(:,i)==1),i)=ones(size(find(Ct10(:,i)==1),1),1)*mean(P10baseloadweekend(setdiff([1:size(P10baseloadweekend,1)]',find(Ct10(:,i)==1)),i));

        end
      if length(find(Ct10(:,i)==1))==size(P10baseloadweekend,1)
          
    P10baseloadweekend(:,i)=ones(size(P10baseloadweekend,1),1)*mean(P10baseloadweekend(find(Ct10==0)))*rand();

      end
    end
    
end
figure(14)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp10weekday;temp10weekend],[P10baseloadweekday;P10baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

figure(15)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(mean(P10baseloadweekdaytrue),'k-.','LineWidth',1)
hold on
plot(mean(P10baseloadweekendtrue),'b-.','LineWidth',1)
hold on
plot(mean(P10baseloadweekday),'k-','LineWidth',2)
hold on
plot(mean(P10baseloadweekend),'b-','LineWidth',2)
hold on
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')





%% clustering of November
Pcluster=P11;
temp11=T11;
[idx,C,sumD]=kmeans(Pcluster,4);

[Y,I]=sort(mean(C'));
P11baseloadweekday=Pcluster(find(idx==I(1)),:);
P11baseloadweekend=Pcluster(find(idx==I(3)),:);
P11otherweekday=Pcluster(find(idx==I(2)),:);
P11otherweekend=Pcluster(find(idx==I(4)),:);
cluster11=zeros(size(idx,1),1);
cluster11(find(idx==I(1)),1)=1;
cluster11(find(idx==I(3)),1)=2;
cluster_11=zeros(size(idx,1),1);
cluster_11([find(idx==I(1));find(idx==I(3))],1)=1;
P11baseloadweekdaytrue=baseload11(find(idx==I(1)),:);
P11baseloadweekendtrue=baseload11(find(idx==I(3)),:);
weekday11=[find(idx==I(1));find(idx==I(2))];
weekend11=[find(idx==I(3));find(idx==I(4))];
temp11weekday=temp11(find(idx==I(1)),:);
temp11weekend=temp11(find(idx==I(3)),:);
temp11otherweekday=temp11(find(idx==I(2)),:);
temp11otherweekend=temp11(find(idx==I(4)),:);
figure(16)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp11weekday;temp11weekend],[P11baseloadweekday;P11baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')
%% weekday correction
%% 
temp111=reshape([temp11weekday;temp11weekend],size([temp11weekday;temp11weekend],1)*size([temp11weekday;temp11weekend],2),1);
P11baseload1=reshape([P11baseloadweekday;P11baseloadweekend],size([P11baseloadweekday;P11baseloadweekend],1)*size([P11baseloadweekday;P11baseloadweekend],2),1);
%% CP
m = 0;
for i = 60:0.2:80
    m = m+1;

        ft = fittype(@(a,b,x)(x>=i).*(a.*x+b));
        [fitresult,goodness] = fit( temp111, P11baseload1, ft,'StartPoint',[0.05 -2]);
        % k0=15; b0=1110
       resnorm(1,m) = goodness.sse;%SSE
 
end

[resnormmin11] = min(resnorm);
[i] = find(resnorm==resnormmin11);
CP = 60+(i(1)-1)*0.2;
Ct11=zeros(size(temp11weekday,1),size(temp11weekday,2));
for i=1:size(temp11weekday,1)
    for j=1:size(temp11weekday,2)
        if (P11baseloadweekday(i,j)>=0.6*max(P11baseloadweekday(i,:)))
            Ct11(i,j)=1;
        end

    end
end

Ct111=zeros(size(temp11weekday,1),size(temp11weekday,2));
error111=length(find(abs(Ct111-Ct11)==1))./(size(temp11weekday,1)*size(temp11weekday,2));
Ct111=Ct11;
%% Ct
while error111>=0.001
    
[a,b]=(find(Ct11==1));
loadloop1=P11baseloadweekday(find(Ct11==1));
temploop1=temp11weekday(find(Ct11==1));
loadloop2=P11baseloadweekday(find(Ct11==0));
temploop2=temp11weekday(find(Ct11==0));
if size(P11baseloadweekday,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P11baseloadweekday,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp11weekday,1),size(temp11weekday,2));
r2=zeros(size(temp11weekday,1),size(temp11weekday,2));
for i=1:size(temp11weekday,1)
    for j=1:size(temp11weekday,2)
if temp11weekday(i,j)>CP
   r1(i,j)=P11baseloadweekday(i,j)-b(2)*temp11weekday(i,j)-b(1);
   r2(i,j)=P11baseloadweekday(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct111(i,j)=1;
   end
end
    end
end
error111=length(find(abs(Ct111-Ct11)==1))./(size(temp11weekday,1)*size(temp11weekday,2));
Ct11=Ct111;
end
if size(P11baseloadweekday,1)==1
P11baseloadweekday2=P11baseloadweekday(find(Ct11==1))';
temp11weekday2=temp11weekday(find(Ct11==1))';
P11baseloadweekday3=P11baseloadweekday(find(Ct11==0))';
temp11weekday3=temp11weekday(find(Ct11==0))';
Ct11weekday=Ct11;
end
if size(P11baseloadweekday,1)~=1
P11baseloadweekday2=P11baseloadweekday(find(Ct11==1));
temp11weekday2=temp11weekday(find(Ct11==1));
P11baseloadweekday3=P11baseloadweekday(find(Ct11==0));
temp11weekday3=temp11weekday(find(Ct11==0));
Ct11weekday=Ct11;
end

%% correction
for i=1:size(P11baseloadweekday,2)

    if find(Ct11(:,i)==1)
        if length(find(Ct11(:,i)==1))<size(P11baseloadweekday,1)
P11baseloadweekday(find(Ct11(:,i)==1),i)=ones(size(find(Ct11(:,i)==1),1),1)*mean(P11baseloadweekday(setdiff([1:size(P11baseloadweekday,1)]',find(Ct11(:,i)==1)),i));

        end
      if length(find(Ct11(:,i)==1))==size(P11baseloadweekday,1)
          
    P11baseloadweekday(:,i)=ones(size(P11baseloadweekday,1),1)*mean(P11baseloadweekday(find(Ct11==0)))*rand();

      end
    end
    
end

%% correction
Ct11=zeros(size(temp11weekend,1),size(temp11weekend,2));


for i=1:size(temp11weekend,1)
    for j=1:size(temp11weekend,2)
        if (P11baseloadweekend(i,j)>=0.6*max(P11baseloadweekend(i,:)))
            Ct11(i,j)=1;
        end

    end
end

Ct111=zeros(size(temp11weekend,1),size(temp11weekend,2));
error111=length(find(abs(Ct111-Ct11)==1))./(size(temp11weekend,1)*size(temp11weekend,2));
Ct111=Ct11;
%% Ct
while error111>=0.001
    
[a,b]=(find(Ct11==1));
loadloop1=P11baseloadweekend(find(Ct11==1));
temploop1=temp11weekend(find(Ct11==1));
loadloop2=P11baseloadweekend(find(Ct11==0));
temploop2=temp11weekend(find(Ct11==0));

if size(P11baseloadweekend,1)>=2
x=temploop1;
Y=loadloop1;
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

if size(P11baseloadweekend,1)==1
x=temploop1';
Y=loadloop1';
X=[ones(size(x,1),1),x];
[b,bint,r,rint,s]=regress(Y,X);
end

r1=zeros(size(temp11weekend,1),size(temp11weekend,2));
r2=zeros(size(temp11weekend,1),size(temp11weekend,2));
for i=1:size(temp11weekend,1)
    for j=1:size(temp11weekend,2)
if temp11weekend(i,j)>CP
   r1(i,j)=P11baseloadweekend(i,j)-b(2)*temp11weekend(i,j)-b(1);
   r2(i,j)=P11baseloadweekend(i,j)-mean(loadloop1);
   if abs(r1(i,j))<abs(r2(i,j))
       Ct111(i,j)=1;
   end
end
    end
end
error111=length(find(abs(Ct111-Ct11)==1))./(size(temp11weekend,1)*size(temp11weekend,2));
Ct11=Ct111;
end
if size(P11baseloadweekend,1)==1
P11baseloadweekend2=P11baseloadweekend(find(Ct11==1))';
temp11weekend2=temp11weekend(find(Ct11==1))';
P11baseloadweekend3=P11baseloadweekend(find(Ct11==0))';
temp11weekend3=temp11weekend(find(Ct11==0))';
Ct11weekend=Ct11;
end
if size(P11baseloadweekend,1)~=1
P11baseloadweekend2=P11baseloadweekend(find(Ct11==1));
temp11weekend2=temp11weekend(find(Ct11==1));
P11baseloadweekend3=P11baseloadweekend(find(Ct11==0));
temp11weekend3=temp11weekend(find(Ct11==0));
Ct11weekend=Ct11;
end
%% correction
for i=1:size(P11baseloadweekend,2)

    if find(Ct11(:,i)==1)
        if length(find(Ct11(:,i)==1))<size(P11baseloadweekend,1)
P11baseloadweekend(find(Ct11(:,i)==1),i)=ones(size(find(Ct11(:,i)==1),1),1)*mean(P11baseloadweekend(setdiff([1:size(P11baseloadweekend,1)]',find(Ct11(:,i)==1)),i));

        end
      if length(find(Ct11(:,i)==1))==size(P11baseloadweekend,1)
          
    P11baseloadweekend(:,i)=ones(size(P11baseloadweekend,1),1)*mean(P11baseloadweekend(find(Ct11==0)))*rand();

      end
    end
    
end

figure(17)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot([temp11weekday;temp11weekend],[P11baseloadweekday;P11baseloadweekend],'o','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

figure(18)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(mean(P11baseloadweekdaytrue),'k-.','LineWidth',1)
hold on
plot(mean(P11baseloadweekendtrue),'b-.','LineWidth',1)
hold on
plot(mean(P11baseloadweekday),'k-','LineWidth',2)
hold on
plot(mean(P11baseloadweekend),'b-','LineWidth',2)
hold on
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')
    
Pbaseloadweekday=[P3baseloadweekday;P4baseloadweekday;P5baseloadweekday;P10baseloadweekday;P11baseloadweekday];
Pbaseloadweekend=[P3baseloadweekend;P4baseloadweekend;P5baseloadweekend;P10baseloadweekend;P11baseloadweekend];
Pbaseloadweekdaytrue=[P3baseloadweekdaytrue;P4baseloadweekdaytrue;P5baseloadweekdaytrue;P10baseloadweekdaytrue;P11baseloadweekdaytrue];
Pbaseloadweekendtrue=[P3baseloadweekendtrue;P4baseloadweekendtrue;P5baseloadweekendtrue;P10baseloadweekendtrue;P11baseloadweekendtrue];

Pbaseloadweekday=max(Pbaseloadweekday,0.001);%weekday baseload
Pbaseloadweekend=max(Pbaseloadweekend,0.001);%weekend baseload

%% accuracy test for baseload 

airmax3=max(Pair3')';
airmax4=max(Pair4')';
airmax5=max(Pair5')';
airmax10=max(Pair10')';
airmax11=max(Pair11')';

standard=mean([median(airmax3),median(airmax4)]);
clustertrue3=zeros(size(cluster3,1),1);
clustertrue4=zeros(size(cluster4,1),1);
clustertrue5=zeros(size(cluster5,1),1);
clustertrue10=zeros(size(cluster10,1),1);
clustertrue11=zeros(size(cluster11,1),1);
clustertrue3(find(airmax3<standard))=1;
clustertrue4(find(airmax4<standard))=1;
clustertrue5(find(airmax5<standard))=1;
clustertrue10(find(airmax10<standard))=1;
clustertrue11(find(airmax11<standard))=1;

% F1-SCORE
[A,~]=confusionmat([clustertrue3;clustertrue4;clustertrue5;clustertrue10;clustertrue11],[cluster_3;cluster_4;cluster_5;cluster_10;cluster_11]);
precise3=A(2,2)/(A(1,2) + A(2,2));
recall3=A(2,2)/(A(2,1) + A(2,2));
F1SCORE=2*precise3*recall3/(precise3 + recall3);
% RMSE
RMSE1=mean(sqrt(mean((Pbaseloadweekdaytrue-Pbaseloadweekday).^2)));
RMSE2=mean(sqrt(mean((mean(Pbaseloadweekdaytrue)-mean(Pbaseloadweekday)).^2)));
% MAPE
MAPE1=mean(mean(abs((Pbaseloadweekdaytrue-Pbaseloadweekday)./Pbaseloadweekdaytrue)))*100;
MAPE2=mean(abs((mean(Pbaseloadweekdaytrue)-mean(Pbaseloadweekday))./mean(Pbaseloadweekdaytrue)))*100;

% NRMSE
NRMSE1=100*mean(sqrt(mean((Pbaseloadweekdaytrue-Pbaseloadweekday).^2))./(max((Pbaseloadweekdaytrue))-min((Pbaseloadweekdaytrue))));
NRMSE2=100*mean(sqrt(mean((mean(Pbaseloadweekdaytrue)-mean(Pbaseloadweekday)).^2)))./(max((mean(Pbaseloadweekdaytrue)))-min((mean(Pbaseloadweekdaytrue))));

figure(19)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(mean(Pbaseloadweekday)','k-.','LineWidth',1)
hold on
plot(mean(Pbaseloadweekend)','b-.','LineWidth',1)
hold on
plot(mean(Pbaseloadweekdaytrue),'k-','LineWidth',4)
hold on
plot(mean(Pbaseloadweekendtrue),'b-','LineWidth',4)
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')
%% distribution test
% % weekday
% [weekday_k ,weekday_j] = test_function(Pbaseloadweekday,0.05);
% % weekend
% [weekend_k ,weekend_j] = test_function(Pbaseloadweekend,0.05);
% 
% [I1,Y1]=sort(weekday_k);
% [I2,Y2]=sort(weekend_k);

%% lognormal distribution
k = 2;%k¦Ò

for i = 1:size(Pbaseloadweekday,2)
    parmhat = lognfit(Pbaseloadweekday(:,i),0.05);
    mu = parmhat(1);
    sigma = parmhat(2);

    lowerweekday(i) = exp(mu-k*sigma);
    higherweekday(i) = exp(mu+k*sigma);
    meanweekday(i)=exp(mu);
end

figure(20)
set(gcf,'unit','centimeters','position',[0,0,16,6])
subplot(1,2,1)

plot(lowerweekday,'r-','LineWidth',4)
hold on
plot(higherweekday,'r-','LineWidth',4)
hold on
plot(meanweekday,'b-','LineWidth',4)
hold on
plot(Pbaseloadweekdaytrue','k:','LineWidth',1)

set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')
hold on
k=2;
for i = 1:size(Pbaseloadweekend,2)
    parmhat = lognfit(Pbaseloadweekend(:,i),0.05);
    mu = parmhat(1);
    sigma = parmhat(2);
    lowerweekend(i) = exp(mu-k*sigma);
    higherweekend(i) = exp(mu+k*sigma);
    meanweekend(i)=exp(mu);
end

subplot(1,2,2)

plot(lowerweekend,'r-','LineWidth',4)
hold on
plot(higherweekend,'r-','LineWidth',4)
hold on
plot(meanweekend,'b-','LineWidth',4)
hold on
plot(Pbaseloadweekendtrue','k:','LineWidth',1)

set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

lognmeanweekend=meanweekend;
lognmeanweekday=meanweekday;

% RMSE
RMSE3=mean(sqrt(mean((mean(Pbaseloadweekdaytrue)-mean(lognmeanweekday)).^2)));
% MAPE
MAPE3=mean(abs((mean(Pbaseloadweekdaytrue)-mean(lognmeanweekday))./mean(Pbaseloadweekdaytrue)))*100;

% NRMSE
NRMSE3=100*mean(sqrt(mean((mean(Pbaseloadweekdaytrue)-mean(lognmeanweekday)).^2)))./(max((mean(Pbaseloadweekdaytrue)))-min((mean(Pbaseloadweekdaytrue))));

figure(21)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(lognmeanweekday','k-.','LineWidth',1)
hold on
plot(lognmeanweekend','b-.','LineWidth',1)
hold on
plot(mean(Pbaseloadweekdaytrue),'k-','LineWidth',4)
hold on
plot(mean(Pbaseloadweekendtrue),'b-','LineWidth',4)
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

%% Generalized extremum distribution

k = 2;%k¦Ò
     
for i = 1:size(Pbaseloadweekday,2)
    phat = gevfit(Pbaseloadweekday(:,i),0.05);
 
    mu =phat(3);
    sigma = phat(2);
    lowerweekday(i) = mu-k*sigma;
    higherweekday(i) = mu+k*sigma;
    meanweekday(i)=mu;
end

figure(22)

set(gcf,'unit','centimeters','position',[0,0,16,6])
subplot(1,2,1)
plot(lowerweekday,'r-','LineWidth',4)
hold on
plot(higherweekday,'r-','LineWidth',4)
hold on
plot(meanweekday,'b-','LineWidth',4)
hold on
plot(Pbaseloadweekdaytrue','k:','LineWidth',1)

set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (15min)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

k=2;
for i = 1:size(Pbaseloadweekend,2)
    phat = gevfit(Pbaseloadweekend(:,i),0.05);
     mu =phat(3);
    sigma = phat(2);
    lowerweekend(i) = mu-k*sigma;
    higherweekend(i) = mu+k*sigma;
    meanweekend(i)=mu;
end

subplot(1,2,2)
plot(lowerweekend,'r-','LineWidth',4)
hold on
plot(higherweekend,'r-','LineWidth',4)
hold on
plot(meanweekend,'b-','LineWidth',4)
hold on
plot(Pbaseloadweekendtrue','k:','LineWidth',1)
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (15min)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

pmeanweekend=meanweekend;
pmeanweekday=meanweekday;

% RMSE
RMSE5=mean(sqrt(mean((mean(Pbaseloadweekdaytrue)-mean(pmeanweekday)).^2)));
% MAPE
MAPE5=mean(abs((mean(Pbaseloadweekdaytrue)-mean(pmeanweekday))./mean(Pbaseloadweekdaytrue)))*100;
% NRMSE
NRMSE5=100*mean(sqrt(mean((mean(Pbaseloadweekdaytrue)-mean(pmeanweekday)).^2)))./(max((mean(Pbaseloadweekdaytrue)))-min((mean(Pbaseloadweekdaytrue))));

figure(23)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(pmeanweekday','k-.','LineWidth',1)
hold on
plot(pmeanweekend','b-.','LineWidth',1)
hold on
plot(mean(Pbaseloadweekdaytrue),'k-','LineWidth',4)
hold on
plot(mean(Pbaseloadweekendtrue),'b-','LineWidth',4)
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

%% weekday|weekend
weekend9_judge=[4;5;6;11;12;13;18;19;20;25;26;27];
weekend8_judge=[1;2;7;8;9;14;15;16;21;22;23;28;29;30];
weekend7_judge=[1;2;3;8;9;10;15;16;17;22;23;24;29;30;31];
weekend6_judge=[3;4;5;10;11;12;17;18;19;24;25;26];

%% ACLs calculation
load9=P9;
air9_es=load9;
for i=1:size(load9,1)
    
air9_es(i,:)=load9(i,:)-lognmeanweekday*(100+30*rand)./100;
end

for i=1:size(weekend9_judge,1)
    air9_es(weekend9_judge(i),:)=load9(weekend9_judge(i),:)-lognmeanweekend*(100+30*rand)./100;
end
load8=P8;
air8_es=load8;
for i=1:size(load8,1)
    
air8_es(i,:)=load8(i,:)-lognmeanweekday*(100+30*rand)./100;
end

for i=1:size(weekend8_judge,1)
    air8_es(weekend8_judge(i),:)=load8(weekend8_judge(i),:)-lognmeanweekend*(100+30*rand)./100;
end
load7=P7;
air7_es=load7;
for i=1:size(load7,1)
    
air7_es(i,:)=load7(i,:)-lognmeanweekday*(100+30*rand)./100;
end

for i=1:size(weekend7_judge,1)
    air7_es(weekend7_judge(i),:)=load7(weekend7_judge(i),:)-lognmeanweekend*(100+30*rand)./100;
end
load6=P6;
air6_es=load6;
for i=1:size(load6,1)
    
air6_es(i,:)=load6(i,:)-lognmeanweekday*(100+30*rand)./100;
end

for i=1:size(weekend6_judge,1)
    air6_es(weekend6_judge(i),:)=load6(weekend6_judge(i),:)-lognmeanweekend*(100+30*rand)./100;
end

air6_es=max(air6_es,0);
air7_es=max(air7_es,0);
air8_es=max(air8_es,0);
air9_es=max(air9_es,0);

air6_es=min(air6_es,1);
air7_es=min(air7_es,1);
air8_es=min(air8_es,1);
air9_es=min(air9_es,1);
air6=Pair6;
air7=Pair7;
air8=Pair8;
air9=Pair9;

%% error analysis
error6=air6_es-Pair6;
error7=air7_es-Pair7;
error8=air8_es-Pair8;
error9=air9_es-Pair9;

figure(24)
set(gcf,'unit','centimeters','position',[0,0,16,12])
plot(error6','k-.','LineWidth',1)
hold on
plot(error7','b-.','LineWidth',1)
hold on
plot(error8','r-.','LineWidth',1)
hold on
plot(error9','y-.','LineWidth',1)
hold on
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Time (h)')
ylabel('\fontsize{10.5}\fontname{Times new roman}Load Power (kW)')

% 6,7,8 
figure(25)
set(gcf,'unit','centimeters','position',[0,0,8,6])
subplot(2,1,1)
plot([air6(1,:) air6(2,:) air6(3,:) air6(4,:) air6(5,:) air6(6,:) air6(7,:)],'k-','LineWidth',1)
hold on
plot([air6_es(1,:) air6_es(2,:) air6_es(3,:) air6_es(4,:) air6_es(5,:) air6_es(6,:) air6_es(7,:)],'r-','LineWidth',1)
hold on
set(gca,'FontName','Times New Roman','FontSize',8)

ylabel('\fontsize{8}\fontname{Times new roman}ACLs (kW)')
xlim([0 180])
subplot(2,1,2)

plot([air7(1,:) air7(2,:) air7(3,:) air7(4,:) air7(5,:) air7(6,:) air7(7,:)],'k-','LineWidth',1)
hold on
plot([air7_es(1,:) air7_es(2,:) air7_es(3,:) air7_es(4,:) air7_es(5,:) air7_es(6,:) air7_es(7,:)],'r-','LineWidth',1)
hold on
set(gca,'FontName','Times New Roman','FontSize',8)

ylabel('\fontsize{8}\fontname{Times new roman}ACLs (kW)')
xlim([0 180])

airsummer_es=[air6_es;air7_es;air8_es;air9_es];
airsummer=[air6;air7;air8;air9];
temp6=T6;
T7=reshape(temprature7,48,31);
T7=T7';
temp7=T7;
temp8=T8;
temp9=T9;
tempsummer=[temp6;temp7;temp8;temp9];

A1=zeros(size(airsummer,1),size(airsummer,2));

for i=1:size(airsummer,1)
    for j=1:size(airsummer,2)
    if airsummer_es(i,j)-airsummer(i,j)>=0.1*max(max(airsummer))
      A1(i,j)=1;  % FP
    end
     if airsummer_es(i,j)-airsummer(i,j)<=-0.1*max(max(airsummer))
      A1(i,j)=2;  % FN
    end
end
end

TP=size(find(A1==0),1);
FP=size(find(A1==1),1);
FN=size(find(A1==2),1);
precise3=TP/(TP + FP);
recall3=TP/(TP + FN);
F1SCORE_hvac=2*precise3*recall3/(precise3 + recall3);

% RMSE
RMSE6=mean(sqrt(mean((airsummer-airsummer_es).^2)));

% NRMSE
NRMSE6=100*mean(sqrt(mean((airsummer-airsummer_es).^2))./(max((airsummer))-min((airsummer))));
% MAPE
% 
airsummer1=airsummer;
airsummer_es1=airsummer_es;

airsummer_es1(find(airsummer1==0))=[];
airsummer1(find(airsummer1==0))=[];

hvac_nan=abs((airsummer1-airsummer_es1)./airsummer1);
hvac_nan(find(isnan(hvac_nan)==1))=0;
MAPE6=mean(mean(hvac_nan))*100;
MAE=mean(mean(abs(airsummer-airsummer_es)')');
result=[F1SCORE_hvac RMSE6 NRMSE6 MAPE6 MAE];

figure(26)
set(gcf,'unit','centimeters','position',[0,0,16,12])

plot(tempsummer,airsummer,'bo','MarkerSize',2);
hold on
set(gca,'FontName','Times New Roman','FontSize',10.5)
xlabel('\fontsize{10.5}\fontname{Times new roman}Outdoor Temperature (¨H)')
ylabel('\fontsize{10.5}\fontname{Times new roman}ACL (kW)')

Ct2=zeros(size(tempsummer,1),size(tempsummer,2));
for i=1:size(tempsummer,1)
    for j=1:size(tempsummer,2)

        if (airsummer_es(i,j)>=0.01)
        Ct2(i,j)=1;
        end
    end
end
%error3=length(find(abs(Ct_true-Ct2)==1))./(size(tempsummer,1)*size(tempsummer,2));

Pother=[P3otherweekday;P3otherweekend;P4otherweekday;P4otherweekend;P5otherweekday;P5otherweekend;P10otherweekday;P10otherweekend;P11otherweekday;P11otherweekend];
Tother=[temp3otherweekday;temp3otherweekend;temp4otherweekday;temp4otherweekend;temp5otherweekday;temp5otherweekend;temp10otherweekday;temp10otherweekend;temp11otherweekday;temp11otherweekend];
Pbaseload1=[P3baseloadweekday3;P3baseloadweekend3;P4baseloadweekday3;P4baseloadweekend3;P5baseloadweekday3;P5baseloadweekend3;P10baseloadweekday3;P10baseloadweekend3;P11baseloadweekday3;P11baseloadweekend3];
Tbaseload1=[temp3weekday3;temp3weekend3;temp4weekday3;temp4weekend3;temp5weekday3;temp5weekend3;temp10weekday3;temp10weekend3;temp11weekday3;temp11weekend3;];
Pbaseload2=[P3baseloadweekday2;P3baseloadweekend2;P4baseloadweekday2;P4baseloadweekend2;P5baseloadweekday2;P5baseloadweekend2;P10baseloadweekday2;P10baseloadweekend2;P11baseloadweekday2;P11baseloadweekend2];
Tbaseload2=[temp3weekday2;temp3weekend2;temp4weekday2;temp4weekend2;temp5weekday2;temp5weekend2;temp10weekday2;temp10weekend2;temp11weekday2;temp11weekend2;];
%% total scatterplot
figure(27)
set(gcf,'unit','centimeters','position',[0,0,8,6])
plot((Tother-32)./1.8,Pother,'Yo','MarkerSize',2);
hold on
plot((Tbaseload1-32)./1.8,Pbaseload1,'go','MarkerSize',2);
hold on
plot((Tbaseload2-32)./1.8,Pbaseload2,'ro','MarkerSize',2);
set(gca,'FontName','Times New Roman','FontSize',8)
xlabel('\fontsize{8}\fontname{Times new roman}Outdoor Temperature (¡æ)')
ylabel('\fontsize{8}\fontname{Times new roman}Load Power (kW)')
hold on


air7_es_new=reshape(air7_es(1:7,:),size(air7_es(1:7,:),1)*size(air7_es(1:7,:),2),1);

air7_new=reshape(air7(1:7,:),size(air7(1:7,:),1)*size(air7(1:7,:),2),1);

figure (28)
set(gcf,'unit','centimeters','position',[0,0,8,3])

plot(air7_new,'k-','LineWidth',1)
hold on
plot(air7_es_new,'r-.','LineWidth',1)
hold on
set(gca,'FontName','Times New Roman','FontSize',8)
ylabel('\fontsize{8}\fontname{Times new roman}ACLs (kW)')
xlabel('\fontsize{8}\fontname{Times new roman}Time (30min)')
ylim([0 1.5])
xlim([0 350])
hold on
axes('position',[0.2,0.7,0.4,0.2]);
hold on
plot(air7_new(140:240,1),'k-','LineWidth',1);
hold on
plot(air7_es_new(140:240,1),'r-.','LineWidth',1);
set(gca,'FontName','Times New Roman','FontSize',4)
ylabel('\fontsize{4}\fontname{Times new roman}ACLs (kW)')
xlim([0 100])
set(gca,'xtick',[140:20:240])
set(gca,'xticklabel',{'140','160','180','200','220','240'})