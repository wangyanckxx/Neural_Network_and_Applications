% TSP Solving by Hopfield Neural Network

function TSP_hopfield()
clear all;
close all;

% step 1
A=1.5;
D=1;
u0=0.02;
step=0.01;

% step 2
N=50;
citys=load('8.txt');
citys
Initial_Length=Initial_RouteLength(citys);        % �����ʼ·������

DistanceCity=dist(citys,citys');

% step 3
u=2*rand(N,N)-1;
U=0.5*u0*log(N-1)+u;
V=(1+tanh(U/u0))/2;

for k=1:1:2000
    times(k)=k;
    
%     step 4
    dU=DeltaU(V,DistanceCity,A,D);
    
%     step 5
    U=U+dU*step;
    
%     step 6
    V=(1+tanh(U/u0))/2;
    
%     step 7 ������������
    E=Energy(V,DistanceCity,A,D);
    Ep(k)=E;
    
%     step 8 ���·���Ϸ���
    [V1,CheckR]=RouteCheck(V);
end

% step 9
if (CheckR==0)
    Final_E=Energy(V1,DistanceCity,A,D);
    Final_Length=Final_RouteLength(V1,citys);       % ��������·������
    disp('��������');k
    disp('Ѱ��·������');V1
    disp('��������������');Final_E
    disp('��ʼ·�̣�');Initial_Length
    disp('���·�̣�');Final_Length
    PlotR(V1,citys);    % Ѱ��·����ͼ
else
    disp('Ѱ��·����Ч');
end



figure(2);
plot(times,Ep,'r');
title('Energy Function Change');
xlabel('k');
ylabel('E');

web -browser http://www.ilovematlab.cn/thread-44738-1-1.html
