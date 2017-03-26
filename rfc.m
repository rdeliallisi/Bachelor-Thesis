% Clear previous workspace
clear all;
clc;

% Set the rendom number seed
seed = 14;
rng(seed);

% Input, reservoir, output dimensions
res_size = 100;
expand_size = 1000;
in_dim = 1;
out_dim = 1;

% Training/testing/washout size
train_size = 300;
washout_size = 100;
test_size = 100;

% Scaling parameters
gf_scale = 1;
w_in_scale = 1;
bias_scale = 0.3;

% Regularizer
reg = 1e-2;

% Leaking rate
a = 1;

% Randomly initialize g and g from a uniform distribution(-1,1)
g = 2 * rand(res_size, expand_size) - 1;
f = 2 * rand(expand_size, res_size) - 1;

% Compute scaling parameters for g and f
prod = g * f;
spec_rad_prod = max(abs(eigs(prod)));
real_scale = sqrt(gf_scale / spec_rad_prod);

% Scale G and F
g = real_scale .* g;
f = real_scale .* f;

% Compute and scale w_in
w_in = 2 * rand(res_size, in_dim) - 1;
w_in = w_in_scale .* w_in;

% Compute and scale the bias
bias = 2 * rand(res_size, 1) - 1;
bias = bias_scale .* bias;

% Set teaching pattern generating functions(more than 1 pattern is required
% for this script to work)

% First pattern
pattprofile = [0 0 3 2 5]; % the integer-periodic pattern profile, will be
                           % normalized to range [-1 1]
pattLength = size(pattprofile,2);
maxVal = max(pattprofile); minVal = min(pattprofile);
pattprofile = 2 * (pattprofile - minVal) / (maxVal - minVal) - 1;
pattern{1} = @(n) (pattprofile(mod(n, pattLength)+1));

% Set pattern number and appertures
patt_number = 1;
apperture = [100];

% Containers for training d, w_out and rfcs.
d_z = [];
d_p = [];
w_r = [];
w_p = [];
c_z_patt = [];
c_z = cell(1, patt_number);
internal_training = [];

% Initialize reservoir state and expanded state
r = zeros(res_size, 1);
z = zeros(expand_size, 1);

% Run the network on all patterns
for i = 1:patt_number
    % Get current pattern generator
    curr_patt = pattern{i};
    for j = 1:train_size
        r_old = r;
        p = curr_patt(j);
        r = (1-a) * r + a * tanh(g * z + w_in * p + bias);
        z = f * r;
        if j > washout_size
            d_z = [d_z z_old];
            d_p = [d_p (w_in * p)];
            w_r = [w_r r_old];
            w_p = [w_p p];
            c_z_patt = [c_z_patt z];
        end;
        z_old = z;
    end;
    c_z{i} = c_z_patt;
end;

% Compute d using ridge regression
d = inv(d_z * d_z' + reg * eye(expand_size)) * d_z * d_p';
d = d';

mean_abs_d = mean(mean(abs(d)));

% Compute w_out using ridge regression
w_out = inv(w_r * w_r' + reg * eye(res_size)) * w_r * w_p';
w_out = w_out';

mean_abss_w_out = mean(abs(w_out));

% Compute the random feature conceptors
c = cell(1, patt_number);
for i = 1:patt_number
    z_square = c_z{i}.^2;
    square_mean = mean(z_square, 2);
    c{i} = square_mean .* (square_mean + apperture(i)^-2).^-1;
end;

% Test run with 1 pattern only

y = zeros(test_size, 1);

internal_testing = zeros(res_size, test_size);

for j = 1:test_size
    r = (1 - a) * r + a * tanh(g * z + d * z + bias);
    internal_testing(:, j) = r;
    y(j) = w_out * r;
    z = c{1} .* (f * r);
end;

correct_output = zeros(test_size, 1);
for i = 1:test_size
    correct_output(i) = pattern{1}(i);
end;

figure(1);
subplot(3,1,1);
plot(w_r(1:10, 1:test_size)');
subplot(3,1,2);
plot(internal_testing(1:10, :)');
subplot(3,1,3);
plot(1:test_size, correct_output, 1:test_size, y);

