from numpy import *;
from matplotlib.pyplot import *;
import scipy.linalg;
from math import sin;

# Input, reservoir, output size
res_size = 20;
in_size = 0;
out_size = 1;

# Training data size and washout time
test_size = 50;
train_size = 300;
washout_size = 100;

# Scaling parameters
w_scale = 0.98;
w_back_scale = 1.01;
bias_scale = 0;
reg = 1e-7;

# Compute w
random.seed(0);
w_0 = random.uniform(-1.0, 1.0, (res_size, res_size));
spectral_rad_w_0 = max(abs(linalg.eig(w_0)[0]));
w = w_0 * 1.0/spectral_rad_w_0 * w_scale;

# Compute w_back
w_back = random.uniform(-1.0, 1.0, (res_size, 1)) * w_back_scale;

# Compute bias
bias = random.normal(size = (res_size, 1)) * bias_scale;

# Teaching signal
d = [0.5 * sin(i / 4) for i in range(0, train_size + test_size + 1)];

# Drive network using teaching signal
m = zeros((train_size - washout_size, res_size));
t = resize(d[washout_size:train_size], (train_size - washout_size, 1));
# Reservoir
x = zeros((res_size, 1));
internal_training = [];
for i in range(train_size):
	x_old = x;
	x = tanh(dot(w, x) + dot(w_back, d[i]) + bias);
	internal_training.append(x[1:5,0]);
	if(i >= washout_size):
		m[i - washout_size] = x_old[:,0];

# w_out = dot(linalg.pinv(m), t);
w_out = dot(linalg.inv(dot(m.T,m) + reg * eye(res_size)), dot(m.T, t));
print("Abs Mean of w_out",mean(abs(w_out)));

y = zeros(test_size);
u = d[train_size]
internal_generating = [];
for i in range(test_size):
	x = tanh(dot(w, x) + dot(w_back, u) + bias);
	internal_generating.append(x[1:5,0]);
	y[i] = dot(w_out.T, x);
	u = y[i];
print("MSE =", 
	sum(square(d[train_size + 1 : train_size + test_size + 1] - y[0:test_size]))
	/ test_size);

figure(1);
subplot(3, 1, 1);
plot(internal_training);
subplot(3, 1, 2);
plot(internal_generating);
subplot(3, 1, 3);
plot(y);
plot(d[train_size + 1 : train_size + test_size + 1]);
show();
