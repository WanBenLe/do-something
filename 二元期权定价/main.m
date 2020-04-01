% fid=fopen('BJAWS_20150531213500.TXT');
% i = 1;
% while 1
%     tline = fgetl(fid);
%     
%     if ~ischar(tline ), 
%         break;
%     end
%     B{i} = strsplit(tline,' ');
%     i = i+1;
% %     disp(tline) 
% end
% fclose(fid);

%% A存储最终结果
clear
clc







file=dir('F:\D\科研数据\北京气象数据\黄焕春\样例程序\*.txt');
j = 1;
for n=1:length(file)
    fid=fopen(file(n).name);
    i = 1;
    while 1
        tline = fgetl(fid);

        if ~ischar(tline ), 
            break;
        end
        B{i,j} = strsplit(tline,' ');
        i = i+1;
    %     disp(tline) 
    end
    j = j+1;
    fclose(fid);
end 

% str2double(B{1,1}{1,3})
[m,n] = size(B);
for i = 1:m
    lena = 1;
    num = 0;
    s = 0;
    for j = 1:n
        [m1,n1] = size(B{i,j});
        if(n1>lena)
            lena = n1;
        end
    end
    for k = 4:lena
        
        for j = 1:n
            [m1,n1] = size(B{i,j});
            if(k<n1)
                if isnan(str2double(B{i,j}{1,k}))
                    s = s;
                else
                    num = num+1;
                    s = s +str2double(B{i,j}{1,k});
                end
            end
            
        end
        A(i,k-3) = s/num;
    end
end

