clc; clear; close all;

data = readtable('tah_synthetic_healthy.csv');

t = data.timestamp;
inlet_p1 = data.inlet_pressure_1;
inlet_p2 = data.inlet_pressure_2;
outlet_p1 = data.outlet_pressure_1;
outlet_p2 = data.outlet_pressure_2;
flow = data.flow_rate;
temp = data.temperature;

n = 1000;

figure;

subplot(3,2,1); plot(t(1:n), inlet_p1(1:n)); title('Inlet Pressure 1'); ylabel('mmHg')
subplot(3,2,2); plot(t(1:n), inlet_p2(1:n)); title('Inlet Pressure 2'); ylabel('mmHg')
subplot(3,2,3); plot(t(1:n), outlet_p1(1:n)); title('Outlet Pressure 1'); ylabel('mmHg')
subplot(3,2,4); plot(t(1:n), outlet_p2(1:n)); title('Outlet Pressure 2'); ylabel('mmHg')
subplot(3,2,5); plot(t(1:n), flow(1:n)); title('Flow Rate'); ylabel('L/min')
subplot(3,2,6); plot(t(1:n), temp(1:n)); title('Temperature'); ylabel('deg C')

% basic stats to confirm values are within expected ranges
signals = [inlet_p1, inlet_p2, outlet_p1, outlet_p2, flow, temp];
names   = {'inlet_p1','inlet_p2','outlet_p1','outlet_p2','flow','temp'};

fprintf('%-15s %8s %8s %8s %8s\n','Sensor','Mean','Std','Min','Max')
for i = 1:length(names)
    fprintf('%-15s %8.3f %8.3f %8.3f %8.3f\n', names{i}, ...
        mean(signals(:,i)), std(signals(:,i)), min(signals(:,i)), max(signals(:,i)))
end