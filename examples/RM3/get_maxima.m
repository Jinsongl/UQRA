function [headers, maxima] = get_maxima(data)

headers = cell(1,1000);
maxima  = zeros(1,1000);

i = 1;
maxima(i)  = max(data.waves.H);
headers{i} = 'WaveHs'; i = i + 1;
maxima(i)  = max(data.waves.T);
headers{i} = 'WaveTp'; i = i + 1;
maxima(i)  = max(data.output.wave.elevation);
headers{i} = 'WaveElv'; i = i + 1;

bodynames = {'Float', 'Spar'};
for ibody = 1:2
    bodyname = bodynames{ibody};
    maxima(i) = max(data.output.bodies(ibody).position(:,1));
    headers{i}= strcat(bodyname,'Surge'); i = i + 1;
    maxima(i) = max(data.output.bodies(ibody).position(:,3));
    headers{i}= strcat(bodyname,'Heave'); i = i + 1;
    maxima(i) = max(data.output.bodies(ibody).position(:,5));
    headers{i}= strcat(bodyname,'Pitch'); i = i + 1;

    maxima(i) = max(data.output.bodies(ibody).velocity(:,1));
    headers{i}= strcat(bodyname,'Surge_Vel'); i = i + 1;
    maxima(i) = max(data.output.bodies(ibody).velocity(:,3));
    headers{i}= strcat(bodyname,'Heave_Vel'); i = i + 1;
    maxima(i) = max(data.output.bodies(ibody).velocity(:,5));
    headers{i}= strcat(bodyname,'Pitch_Vel'); i = i + 1;

    maxima(i) = max(data.output.bodies(ibody).acceleration(:,1));
    headers{i}= strcat(bodyname,'Surge_Acl'); i = i + 1;
    maxima(i) = max(data.output.bodies(ibody).acceleration(:,3));
    headers{i}= strcat(bodyname,'Heave_Acl'); i = i + 1;
    maxima(i) = max(data.output.bodies(ibody).acceleration(:,5));
    headers{i}= strcat(bodyname,'Pitch_Acl'); i = i + 1;

    maxima(i) = max(data.output.bodies(ibody).forceTotal(:,1));
    headers{i}= strcat(bodyname,'Surge_ForceTotal'); i = i + 1;
    maxima(i) = max(data.output.bodies(ibody).forceTotal(:,3));
    headers{i}= strcat(bodyname,'Heave_ForceTotal'); i = i + 1;
    maxima(i) = max(data.output.bodies(ibody).forceTotal(:,5));
    headers{i}= strcat(bodyname,'Pitch_ForceTotal'); i = i + 1;
end
bodyname = 'PTO';
maxima(i) = max(data.output.ptos.position(:,3));
headers{i}= strcat(bodyname,'Heave'); i = i + 1;

maxima(i) = max(data.output.ptos.velocity(:,3));
headers{i}= strcat(bodyname,'Heave_Vel'); i = i + 1;

maxima(i) = max(data.output.ptos.acceleration(:,3));
headers{i}= strcat(bodyname,'Heave_Acl'); i = i + 1;

maxima(i) = max(data.output.ptos.forceTotal(:,1));
headers{i}= strcat(bodyname,'Surge_ForceTotal'); i = i + 1;
maxima(i) = max(data.output.ptos.forceTotal(:,2));
headers{i}= strcat(bodyname,'Sway_ForceTotal'); i = i + 1;
maxima(i) = max(data.output.ptos.forceTotal(:,3));
headers{i}= strcat(bodyname,'Heave_ForceTotal'); i = i + 1;
maxima(i) = max(data.output.ptos.forceTotal(:,4));
headers{i}= strcat(bodyname,'Roll_ForceTotal'); i = i + 1;
maxima(i) = max(data.output.ptos.forceTotal(:,5));
headers{i}= strcat(bodyname,'Pitch_ForceTotal'); i = i + 1;
maxima(i) = max(data.output.ptos.forceTotal(:,6));
headers{i}= strcat(bodyname,'Yaw_ForceTotal'); i = i + 1;

headers = headers(1:i-1);
maxima  = maxima(1:i-1);
% return headers, maxima;



