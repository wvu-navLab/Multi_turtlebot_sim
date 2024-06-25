clear
clc
close all

% Read the CSV file
data = readtable('imu_data.csv');

% Extract relevant columns
time = data.time;  % Assuming 'time' is a column in your table
linear_acceleration_z = data.linear_acceleration_z;  % Example column names
angular_velocity_x = data.angular_velocity_x;

Fs = 50; %Sampling rate
t0 = 1/Fs;

theta = cumsum(angular_velocity_x, 1)*t0;

maxNumM = 100;
L = size(theta, 1);
maxM = 2.^floor(log2(L/2));
m = logspace(log10(1), log10(maxM), maxNumM).';
m = ceil(m); % m must be an integer.
m = unique(m); % Remove duplicates.
tau = m*t0;
avar = zeros(numel(m), 1);
for i = 1:numel(m)
    mi = m(i);
    avar(i,:) = sum( ...
        (theta(1+2*mi:L) - 2*theta(1+mi:L-mi) + theta(1:L-2*mi)).^2, 1);
end
avar = avar ./ (2*tau.^2 .* (L - 2*m));

adev = sqrt(avar)*206265;

figure
loglog(tau, adev)
title('Allan Deviation')
xlabel('\tau');
ylabel('\sigma(\tau)')
grid on
axis equal