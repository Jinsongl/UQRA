function [headers, maxima] = get_maxima(data)

headers = cell(1,1000);
maxima  = zeros(1,1000);

i = 1;
maxima(i)  = max(abs(data.waves.H));
headers{i} = 'WaveHs'; i = i + 1;
maxima(i)  = max(abs(data.waves.T));
headers{i} = 'WaveTp'; i = i + 1;
maxima(i)  = max(abs(data.output.wave.elevation(4001:end)));
headers{i} = 'WaveElv'; i = i + 1;

bodynames = {'Float', 'Spar'};
for ibody = 1:2
    bodyname = bodynames{ibody};
    maxima(i) = max(abs(data.output.bodies(ibody).position(4001:end,1)));
    headers{i}= strcat(bodyname,'Surge'); i = i + 1;
    maxima(i) = max(abs(data.output.bodies(ibody).position(4001:end,3)));
    headers{i}= strcat(bodyname,'Heave'); i = i + 1;
    maxima(i) = max(abs(data.output.bodies(ibody).position(4001:end,5)));
    headers{i}= strcat(bodyname,'Pitch'); i = i + 1;

    maxima(i) = max(abs(data.output.bodies(ibody).velocity(4001:end,1)));
    headers{i}= strcat(bodyname,'Surge_Vel'); i = i + 1;
    maxima(i) = max(abs(data.output.bodies(ibody).velocity(4001:end,3)));
    headers{i}= strcat(bodyname,'Heave_Vel'); i = i + 1;
    maxima(i) = max(abs(data.output.bodies(ibody).velocity(4001:end,5)));
    headers{i}= strcat(bodyname,'Pitch_Vel'); i = i + 1;

    maxima(i) = max(abs(data.output.bodies(ibody).acceleration(4001:end,1)));
    headers{i}= strcat(bodyname,'Surge_Acl'); i = i + 1;
    maxima(i) = max(abs(data.output.bodies(ibody).acceleration(4001:end,3)));
    headers{i}= strcat(bodyname,'Heave_Acl'); i = i + 1;
    maxima(i) = max(abs(data.output.bodies(ibody).acceleration(4001:end,5)));
    headers{i}= strcat(bodyname,'Pitch_Acl'); i = i + 1;

    maxima(i) = max(abs(data.output.bodies(ibody).forceTotal(4001:end,1)));
    headers{i}= strcat(bodyname,'Surge_ForceTotal'); i = i + 1;
    maxima(i) = max(abs(data.output.bodies(ibody).forceTotal(4001:end,3)));
    headers{i}= strcat(bodyname,'Heave_ForceTotal'); i = i + 1;
    maxima(i) = max(abs(data.output.bodies(ibody).forceTotal(4001:end,5)));
    headers{i}= strcat(bodyname,'Pitch_ForceTotal'); i = i + 1;
end
bodyname = 'PTO';
maxima(i) = max(abs(data.output.ptos.position(4001:end,3)));
headers{i}= strcat(bodyname,'Heave'); i = i + 1;

maxima(i) = max(abs(data.output.ptos.velocity(4001:end,3)));
headers{i}= strcat(bodyname,'Heave_Vel'); i = i + 1;

maxima(i) = max(abs(data.output.ptos.acceleration(4001:end,3)));
headers{i}= strcat(bodyname,'Heave_Acl'); i = i + 1;

maxima(i) = max(abs(data.output.ptos.forceTotal(4001:end,1)));
headers{i}= strcat(bodyname,'Surge_ForceTotal'); i = i + 1;
maxima(i) = max(abs(data.output.ptos.forceTotal(4001:end,2)));
headers{i}= strcat(bodyname,'Sway_ForceTotal'); i = i + 1;
maxima(i) = max(abs(data.output.ptos.forceTotal(4001:end,3)));
headers{i}= strcat(bodyname,'Heave_ForceTotal'); i = i + 1;
maxima(i) = max(abs(data.output.ptos.forceTotal(4001:end,4)));
headers{i}= strcat(bodyname,'Roll_ForceTotal'); i = i + 1;
maxima(i) = max(abs(data.output.ptos.forceTotal(4001:end,5)));
headers{i}= strcat(bodyname,'Pitch_ForceTotal'); i = i + 1;
maxima(i) = max(abs(data.output.ptos.forceTotal(4001:end,6)));
headers{i}= strcat(bodyname,'Yaw_ForceTotal'); i = i + 1;

headers = headers(1:i-1);
maxima  = maxima(1:i-1);
% return headers, maxima;



