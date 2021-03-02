%Example of user input MATLAB file for MCR post processing
%filename = ['savedData',sprintf('%03d', imcr),'.mat'];
filename = sprintf('RM3_2Hem%01d_SS%01d_R%01d.mat', deg, imcr+batch_size*ith_batch, waves.phaseSeed);
filename = fullfile(data_dir, filename);
mcr.Avgpower(imcr) = mean(output.ptos.powerInternalMechanics(2000:end,3));
mcr.CPTO(imcr)  = pto(1).c;
data.output = output;
data.waves  = waves;
[headers, maxima] = get_maxima(data);
save (filename, 'mcr','waves', 'headers', 'maxima');


%% Plot Power Matrix

% if imcr == length(mcr.cases);
%     H = mcr.cases(:,1);
%     T = mcr.cases(:,2);
%     c = mcr.cases(:,3);
%     P = mcr.Avgpower';
% 
%     % Damping = 1200000
%     figure
%     mat = [mcr.Avgpower(2) mcr.Avgpower(4);...      % Create Power Matrix
%         mcr.Avgpower(1) mcr.Avgpower(3)];
%     imagesc(mat);                                   % Create a colored plot of the matrix values
%     colormap parula
%     caxis([min(P) max(P)])
% 
%     textStrings = num2str(mat(:),'%0.2f');          % Create strings from the matrix values
%     textStrings = strtrim(cellstr(textStrings));    % Remove any space padding
%     [x,y] = meshgrid(1:2);                          % Create x and y coordinates for the strings
%     hStrings = text(x(:),y(:),textStrings(:),...    % Plot the strings
%                     'HorizontalAlignment','center');
%     midValue = mean(get(gca,'CLim'));               % Get the middle value of the color range
%     textColors = repmat(mat(:) < midValue,1,3);     % Choose white or black for the text color
%     set(hStrings,{'Color'},num2cell(textColors,2)); % Change the text colors
%     colorbar
%     set(gca,'XTick',1:2,...                         % Change the axes tick marks
%             'XTickLabel',{'1.5','2.5'},...          % and tick labels
%             'YTick',1:2,...
%             'YTickLabel',{'8','6'},...
%             'TickLength',[0 0]);
%     xlabel('Period [s]')
%     ylabel('Height [m]')
%     title(['Power Matrix for Damping = ' num2str(c(1)) ' [N/m/s]'])
% 
% 
%     % Damping = 2400000
%     figure
%     mat = [mcr.Avgpower(6) mcr.Avgpower(8);...      % Create Power Matrix
%         mcr.Avgpower(5) mcr.Avgpower(7)];
%     imagesc(mat);                                   % Create a colored plot of the matrix values
%     colormap default
%     caxis([min(P) max(P)])
% 
%     textStrings = num2str(mat(:),'%0.2f');          % Create strings from the matrix values
%     textStrings = strtrim(cellstr(textStrings));    % Remove any space padding
%     [x,y] = meshgrid(1:2);                          % Create x and y coordinates for the strings
%     hStrings = text(x(:),y(:),textStrings(:),...    % Plot the strings
%                     'HorizontalAlignment','center');
%     midValue = mean(get(gca,'CLim'));               % Get the middle value of the color range
%     textColors = repmat(mat(:) < midValue,1,3);     % Choose white or black for the text color
%     set(hStrings,{'Color'},num2cell(textColors,2)); % Change the text colors
%     colorbar
%     set(gca,'XTick',1:2,...                         % Change the axes tick marks
%             'XTickLabel',{'1.5','2.5'},...          % and tick labels
%             'YTick',1:2,...
%             'YTickLabel',{'8','6'},...
%             'TickLength',[0 0]);
%     xlabel('Period [s]')
%     ylabel('Height [m]')
%     title(['Power Matrix for Damping = ' num2str(c(end)) ' [N/m/s]'])
% end
