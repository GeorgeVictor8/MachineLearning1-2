clear all
clear
clc
ds = datastore('house_prices_data_training_data.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',18000);
T = read(ds);
TrainingSet=T(1:1080,:);
size(T);
Alpha=.01;

m=length(T{:,1});

U0training=T{1:1080,2};
U0training=(U0training-mean(U0training))./std(U0training);

U0CV=T{1081:1500,2};
U0CV=(U0CV-mean(U0CV))./std(U0CV); %Normalizing output


UTraining=T{1:1080,6:10};
UCrossValidation=T{1081:1800,6:10};

LengthTraining = length(UTraining);

LengthCrossValidation = length(UCrossValidation);

z=length(UTraining);

XTraining=[ones(z,1) UTraining UTraining.^2];

XCrossValidation=[ones(length(UCrossValidation),1) UCrossValidation UCrossValidation.^2];

nTraining=length(XTraining(1,:));
for w=2:nTraining
    if max(abs(XTraining(:,w)))~=0
    XTraining(:,w)=(XTraining(:,w)-mean((XTraining(:,w))))./std(XTraining(:,w));
    end
end
nCV=length(XCrossValidation(1,:));
for w=2:nCV
    if max(abs(XCrossValidation(:,w)))~=0
    XCrossValidation(:,w)=(XCrossValidation(:,w)-mean((XCrossValidation(:,w))))./std(XCrossValidation(:,w));
    end
end

YTraining=TrainingSet{:,3}/mean(TrainingSet{:,3});
YCrossValidation=TrainingSet{11,3}/mean(TrainingSet{11,3});

ThetaTraining=zeros(nTraining,1);
ThetaCrossValidation=zeros(nCV,1);
% for k=1:500
% 
% ETrain(k)=(1/(2*m))*sum((XTrain*ThetaTrain-Y).^2);
% ECV(k)=(1/(2*m))*sum((XCV*ThetaCV-Y).^2);
% end
k=1;
R=1;
ETraining=[];
ECrossValidation=[];
while R==1
Alpha=Alpha*1;
ThetaTraining=ThetaTraining-(Alpha/m)*XTraining'*(XTraining*ThetaTraining-YTraining);
ThetaCrossValidation=ThetaCrossValidation-(Alpha/m)*XCrossValidation'*(XCrossValidation*ThetaCrossValidation-YCrossValidation);

ETraining=(1/(2*m))*sum((XTraining*ThetaTraining-YTraining).^2);
ETraining=[ETraining;ETraining];
ECrossValidation(k)=(1/(2*m))*sum((XCrossValidation*ThetaCrossValidation-YCrossValidation).^2);
ECrossValidation=[ECrossValidation;ECrossValidation];
 k=k+1
if ETraining(k-1)-ETraining(k)<0
   
    break
end 
q=(ETraining(k-1)-ETraining(k))./ETraining(k-1);
if q <.000001
    R=0;
end
end

figure (1)
plot(k,ECrossValidation,'black')
hold on
plot(k,ETraining,'red')
legend('CV','Train')
title('House Price')
ylabel('Cost Fun')
xlabel('Iter')

