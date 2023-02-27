% Data Processing for Engineers and Scientists
%
% Array flattening demo

a = 1:24;
a2= reshape( a, 12, 2);
a3= reshape( a, 2, 3, 4);

disp(a2)

for i=1:2
    disp(reshape(a3(i,:,:), 3,4))
end
