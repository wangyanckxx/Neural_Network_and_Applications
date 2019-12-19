I1 = imread('5.png');
I1=rgb2gray(I1);
I2 = imread('6.png');
I2=rgb2gray(I2);
I3 = imread('7.png');
I3=rgb2gray(I3);
I4 = imread('8.png');
I4=rgb2gray(I4);
[h,w] = size(I1);

s1=reshape(I1,[1,h*w]);
s2=reshape(I2,[1,h*w]);
s3=reshape(I3,[1,h*w]);
s4=reshape(I4,[1,h*w]);

s=[s1;s2;s3;s4];
sig=double(s);
Aorig=rand(3,size(sig,1));
mixedsig=Aorig*sig;

MixedS_bak=mixedsig;
[~,c] = size(mixedsig);
mixedsig = mixedsig - mean(mixedsig')'*ones(1,c);
mixedsig_cov = cov(mixedsig');
[E,D] = eig(mixedsig_cov);
Q = inv(sqrt(D))*(E)';
mixedsig_white = Q*mixedsig;
lsl = cov(mixedsig_white');

X = mixedsig_white;
[VariableNum,SampleNum]=size(X); 
numofIC=VariableNum;                        
B=zeros(numofIC,VariableNum);               
for r=1:numofIC                             
    i=1;maxIterationsNum=150;               
    b=2*(rand(numofIC,1)-0.5);                   
    b=b/norm(b);                            
    while i<=maxIterationsNum+1 
        if i == maxIterationsNum           
            fprintf('\n第%d分量在%d次迭代内并不收敛', r,maxIterationsNum); 
            break; 
        end 
        bOld=b;                            
        u=1; 
        t=X'*b; 
        g=t.^3; 
        dg=3*t.^2; 
        b=((1-u)*t'*g*b+u*X*g)/SampleNum-mean(dg)*b;                                      
        b=b-B*B'*b;                         
        b=b/norm(b);  
        if abs(abs(b'*bOld)-1)<1e-10        
             B(:,r)=b;                     
             break; 
         end 
        i=i+1;         
    end 
end 

ICAedS=B'*Q*MixedS_bak;                  
ICAedS_bak=ICAedS; 
ICAedS=abs(55*ICAedS); 

imwrite([I1;I2;I3;I4],'./3幅观察图像结果/原始灰度图.jpg')
mixeds1=reshape(mixedsig(1,:),[h,w]);
mixeds2=reshape(mixedsig(2,:),[h,w]);
mixeds3=reshape(mixedsig(3,:),[h,w]);
imwrite([uint8(round(mixeds1));uint8(round(mixeds2));uint8(round(mixeds3));...
    ],'./3幅观察图像结果/随机混合图.jpg')
icas1=reshape(ICAedS(1,:),[h,w]);
icas2=reshape(ICAedS(2,:),[h,w]);
icas3=reshape(ICAedS(3,:),[h,w]);
imwrite([uint8(round(icas1));uint8(round(icas2));uint8(round(icas3));...
    ],'./3幅观察图像结果/分离效果图.jpg')