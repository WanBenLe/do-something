[data,~,b]=tblread('alldata.csv',',');
data=[table(b) array2table(data)];
data{:,2}=floor(data{:,2}/100);
data=[data array2table(floor(data{:,2}/100))];
data=table2cell(data);
a=all(cellfun(@(x) isempty(x),data),1);
data(a(:,1),:)=[];
a=all(cellfun(@(x) isempty(x),data),3);
data(a(:,3),:)=[];
data=cell2table(data);
data(isnan(table2array(data(:,7))),:)=[];
%年平均ROE
yearlist=unique(data{:,8},'rows');
roe_all=zeros(size(yearlist,1),1);
for i = 1:size(yearlist,1)
    temp=data(data{:,8}==yearlist(i),:);
    roe=mean(temp{:,4});
    roe_all(i,:)=roe;
end



%月平均收益率
monthlist=unique(data{:,2},'rows');
monthretun=zeros(size(monthlist,1),2);
for i = 1:size(monthlist,1)
    temp=data(data{:,2}==monthlist(i),:);
    returnx=nanmean(temp{:,3});
    monthretun(i,:)=[monthlist(i),returnx];
end
monthretun=[floor(monthretun(:,1)/100),monthretun];

%月收益率-ROE
rsubroe=zeros(size(monthretun,1),1);
for i = 1:size(monthretun,1)
    rsubroe(i)=monthretun(i,3)-roe_all(monthretun(i,1)==yearlist);
end
monthretun(:,3)=rsubroe;

%计算SMBHML
KOOO=cell(50,10);
SMBHML=zeros(size(yearlist,1),2);
for i = 1:size(yearlist,1)
    date=yearlist(i)*100+12;
    temp=data(data{:,2}==date,:);
    
    x50=floor(size(temp,1)*0.5);
    x30=floor(size(temp,1)*0.3);
    x70=floor(size(temp,1)*0.7);
    %市值和账面市值比的分组股票
    temp = sortrows(temp,'data6','descend');
    M1=temp(1:x50,1);
    M2=temp((x50+1):end,1);
    temp = sortrows(temp,'data7','descend');
    BP1=temp(1:x30,1);
    BP2=temp((x30+1):(x70-1),1);
    BP3=temp((x70+1):end,1);

    %计算SMB和HML
    SL=union(M2,BP1);
    
    SLR =dex(data,SL,i,yearlist);
    SM=union(M2,BP2);
    SMR =dex(data,SM,i,yearlist);
    SH=union(M2,BP3);
    SHR =dex(data,SH,i,yearlist);
    BL=union(M1,BP1);
    BLR =dex(data,BL,i,yearlist);
    BM=union(M1,BP2);
    BMR =dex(data,BM,i,yearlist);
    BH=union(M1,BP3);
    BHR =dex(data,BH,i,yearlist);
    
    KOOO(1:size(BH,1),i)=table2cell(BH);
    SMB=(SLR+SMR+SHR)/3-(BLR+BMR+BHR)/3;
    HML=(SHR+BHR)/2-(SLR+BLR)/2;
    SMBHML(i,:)=[SMB,HML];
end


%月平均收益率
monthlist=unique(data{:,2},'rows');
monthretunend=zeros(size(monthlist,1),2);
for i = 1:size(monthlist,1)
    yearsss=floor(monthlist(i,:)/100);
    inx=find(yearsss==yearlist);
    oho=KOOO(:,inx);
    oho(cellfun(@isempty,oho))=[];
    temp=data(((data{:,2}==monthlist(i))&ismember(data{:,1},oho(:,1))),:);
    returnx=nanmean(temp{:,3});
    monthretunend(i,:)=[monthlist(i),returnx];
end
monthretun=[floor(monthretunend(:,1)/100),monthretunend];




SMBHML=[yearlist,SMBHML];
%月SMBHML
SMBHMLm=zeros(size(monthretun,1),2);
for i = 1:size(monthretun,1)
    SMBHMLm(i,:)=SMBHML((monthretun(i,1)==SMBHML(:,1)),2:3);
end
allmonthretun=[monthretun,SMBHMLm];

[data,~,b]=tblread('data.csv',',');
data=[table(b) array2table(data)];
data{:,2}=floor(data{:,2}/100);
data=[data array2table(floor(data{:,2}/100))];
data=table2cell(data);
a=all(cellfun(@(x) isempty(x),data),1);
data(a(:,1),:)=[];
a=all(cellfun(@(x) isempty(x),data),1);
data(a(:,1),:)=[];
data=cell2table(data);
data(isnan(table2array(data(:,7))),:)=[];
%年平均ROE
yearlist=unique(data{:,8},'rows');
roe_all=zeros(size(yearlist,1),1);
for i = 1:size(yearlist,1)
    temp=data(data{:,8}==yearlist(i),:);
    roe=mean(temp{:,4});
    roe_all(i,:)=roe;
end



%月平均收益率
monthlist=unique(data{:,2},'rows');
monthretun=zeros(size(monthlist,1),2);
for i = 1:size(monthlist,1)
    temp=data(data{:,2}==monthlist(i),:);
    returnx=nanmean(temp{:,3});
    monthretun(i,:)=[monthlist(i),returnx];
end
monthretun=[floor(monthretun(:,1)/100),monthretun];

%月收益率-ROE
rsubroe=zeros(size(monthretun,1),1);
for i = 1:size(monthretun,1)
    rsubroe(i)=monthretun(i,3)-roe_all(monthretun(i,1)==yearlist);
end
monthretun(:,3)=rsubroe;

alldata=[monthretun(:,2:3),allmonthretun(:,3:5)];
alldata(:,3)=alldata(:,3)/100;
xlswrite('res.xls',alldata);
'a'
function bxR =dex(ax,bx,cx,dx)
    temp=ax((ismember(ax(:,1),bx)) & (ax{:,8}==dx(cx)),:);
    bxR=0;
    for j =1:size(bx,1)
        temp1=temp(ismember(temp(:,1),bx(cx,1)),:);
        bxR=bxR+temp1{end,5}/temp1{1,5}-1;
    end    
    bxR=bxR/size(bx,1); 
end
