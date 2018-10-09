function [data,mx,mn]=normalise(data,mx,mn)
if(nargin==1)
mx=max(data(:,1:size(data,2)));
mn=min(data(:,1:size(data,2)));
end

val = mx-mn;
lenC = size(data,2);
for i=1:lenC
    if val(i) == 0
        val(i) = 0.000000001;
    end
    data(:,i)=(data(:,i)-mn(i))/val(i);
end