clear all

% initialize random generator
randn('seed',1);
rand('seed',1);

% size of the problem
N = 1e6;

% create vectors
a = rand(N,1); % \in [0,1]
b = 1 - a;

%% CPU
% now a+b should be equal to vector of ones
cpu_aplusb = a+b;
cpu_avgaplusb = sum(cpu_aplusb)/N; % = 1


%% GPU
% prepare device
g = gpuDevice(1);
reset(g);

% transfer data to gpu
gpu_a = gpuArray(a);  
gpu_b = gpuArray(b);

% see if variables are on CPU or GPU
%class(a)
%class(gpu_a)

% compute
gpu_aplusb = gpu_a + gpu_b;
gpu_avgaplusb = sum(gpu_aplusb)/N; % = 1

% results will be automagically trasfered to CPU if necessary,
% but maybe it is better to do it manually
%cpu_avgaplusb = gather(gpu_avgaplusb);


%% GPU using kernels
% please compile "mykernel.cu" in folder CudaFiles using
% (in Linux terminal, not in Matlab)
% cd CudaFiles
% nvcc -ptx mykernel.cu
% and this produces "mykernel.ptx" which is used in following code

addpath('CudaFiles')

% define kernel as matlab function
kernel = parallel.gpu.CUDAKernel( 'mykernel.ptx', 'mykernel.cu' );

% compute optimal nuber of threads and grid
kernel.ThreadBlockSize = kernel.MaxThreadsPerBlock;
kernel.GridSize = ceil(N/kernel.MaxThreadsPerBlock);

% allocate storage for results 
gpu2_aplusb = gpuArray(zeros(size(a))); 

% compute sum of the vectors using own cuda kernel:
% (see that the first argument of kernel (double *aplusb) is not const)
[gpu2_aplusb] = feval( kernel, gpu2_aplusb, gpu_a, gpu_b, N );
          
% compute the sum (I am too lazy to write kernels for this, sorry :)
gpu2_avgaplusb = sum(gpu2_aplusb)/N; % = 1

% check the results
disp(['cpu : ' num2str(cpu_avgaplusb)])
disp(['gpu : ' num2str(gpu_avgaplusb)])
disp(['gpu2: ' num2str(gpu2_avgaplusb)])
