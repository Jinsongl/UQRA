%% Simulation Data
deg = 2;
filename = sprintf('RM3_2Hem%01d_train.mat', deg);
batch_size = 2;
ith_batch  = 1;
get_batch_data(batch_size, ith_batch, filename);
if ismac
    data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/RM3/Data';
elseif isunix
    data_dir = '/home/jinsong/Documents/MUSE_UQ_DATA/UQRA_Examples/RM3/Data';
elseif ispc
    data_dir = '/Volumes/GoogleDrive/My Drive/MUSE_UQ_DATA/UQRA_Examples/RM3/Data';
else
    disp('Platform not supported')
end



simu = simulationClass();               % Initialize Simulation Class
simu.simMechanicsFile = 'RM3.slx';      % Location of Simulink Model File
simu.solver = 'ode4';                   % simu.solver = 'ode4' for fixed step & simu.solver = 'ode45' for variable step 
% simu.mode = 'rapid-accelerator';      % Specify Simulation Mode ('normal','accelerator','rapid-accelerator')
simu.explorer  ='on';                  % Turn SimMechanics Explorer (on/off)
simu.startTime = 0;                     % Simulation Start Time [s]
simu.rampTime  = 40;                   % Wave Ramp Time Length [s]
simu.endTime   = 400;                  % Simulation End Time [s]
simu.dt        = 0.1; 							% Simulation time-step [s]
simu.mcrCaseFile = 'batch_data.mat';

%% Wave Information 
% % Regular Waves  
% waves = waveClass('regularCIC');        % Initialize Wave Class and Specify Type 
% waves.H = 1.5;                          % Wave Height [m]
% waves.T = 8;                            % Wave Period [s]
% Irregular Waves
waves = waveClass('irregular');
waves.H = 1.5;                          % Wave Height [m]
waves.T = 8;                            % Wave Period [s]
waves.spectrumType = 'JS';
waves.phaseSeed = 0;
%% Body Data
% Float
body(1) = bodyClass('./hydroData/rm3.h5');      
body(1).geometryFile = './geometry/float.stl';    
body(1).mass = 'equilibrium';                   
body(1).momOfInertia = [20907301 21306090.66 37085481.11];  

% Spar/Plate
body(2) = bodyClass('./hydroData/rm3.h5'); 
body(2).geometryFile = './geometry/plate.stl'; 
body(2).mass = 'equilibrium';                   
body(2).momOfInertia = [94419614.57 94407091.24 28542224.82];

%% PTO and Constraint Parameters
% Floating (3DOF) Joint
constraint(1) = constraintClass('Constraint1'); %Create Constraint Variable and Set Constraint Name
constraint(1).loc = [0 0 0];            % Constraint Location [m]

% Translational PTO
pto(1) = ptoClass('PTO1');           	% Initialize PTO Class for PTO1
pto(1).k = 0;                           % PTO Stiffness [N/m]
pto(1).c=1200000;                       % PTO Damping [N/(m/s)]
pto(1).loc = [0 0 0];                  	% PTO Location [m]
