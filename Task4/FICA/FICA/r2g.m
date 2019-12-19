B='C:/Users/kuang/Desktop/tmp.png';
A=imread(B);
imshow(A);
imfinfo(B);
C=rgb2gray(A);       %处理彩色图片
imshow(C);
str0='C:/Users/kuang/Desktop/'
str1='test_gray';
str2='.bmp';
save_path=[str0,str1,str2];  %保存图片路径
imwrite(C,save_path);
