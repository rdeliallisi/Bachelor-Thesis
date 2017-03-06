function par = mcplotframeHJ1(d, n, p, proj, xRootShifts, yRootShifts)
% Plots frames of motion capture data.
%
% syntax
% par = mcplotframe(d, n);
% par = mcplotframe(d, n, p);
% par = mcplotframe(d, n, p, proj);
%
% input parameters
% d: MoCap data structure
% n: vector containing the numbers of the frames to be plotted
% p: animpar structure
% proj: projection used: 0 = orthographic (default), 1 = perspective
%
% output
% par: animpar structure used for plotting the frames (if color strings were used, they will converted to RGB triplets)
%
% examples
% par = mcplotframe(d, 1);
% mcplotframe(d, 500:10:600, par, 1);
%
% comments
% If the animpar structure is not given, the function calls
% mcinitanimpar and sets the .limits field of the animpar structure
% automatically so that all the markers fit into all frames.
% If the par.pers field (perspective projection) is not given, it is created internally for backwards
% compatibility. For explanation of the par.pers field, see help mcinitanimpar
%
% see also
% mcanimate, mcinitanimpar
%
% � Part of the Motion Capture Toolbox, Copyright �2008,
% University of Jyvaskyla, Finland

if nargin<4
    proj=0;
end

if nargin<3
    p = mcinitanimpar;
    if nargin==1 || ~isnumeric(n) %output fix [BB20111031]
        disp([10, 'Error: please set frame number(s) you want to plot.', 10]);
        return
    end
end

for k=1:length(n)
    n1=n(k);
    if n1>d.nFrames
        w1=sprintf('Error: indicated frame(s) (%d) exceeds number of frames in data (%d).', max(n), d.nFrames);
        disp([10, w1, 10]);
        return
    end
end


if p.animate && not(strcmp(p.folder, '0'))
    mkdir(p.folder)
    currdir = cd; % store current directory
    cd(p.folder)
end

%for compatibility
if ~isfield(p, 'pers')
    p.pers.c=[0 -3000 0];
    p.pers.th=[0 0 0];
    p.pers.e=[0 -2000 0];
end

par=p;

% color management (compatibility): convert old string color definition into num array [BB20111031]
if ischar(p.colors)
    colors=NaN(5,3);
    for k=1:5
        colors(k,:)=lookup_l(p.colors(k));
    end
end

bgcol=colors(1,:); % set background color

if isfield(p,'markercolors') && ~isempty(p.markercolors) %field and colors are set
    if ischar(p.markercolors) % but in string format
        mcol=NaN(d.nMarkers,3);
        for k=1:size(p.markercolors,2)
            mcol(k,:)=lookup_l(p.markercolors(k));%convert to num array
        end
    else
        mcol=p.markercolors; %field and colors are set in num format already
    end
    if d.nMarkers > length(p.markercolors)
        for k=length(p.markercolors)+1:d.nMarkers
            mcol(k,:)=colors(2,:);
        end
    end
else %no field or/and empty
    mcol=repmat(colors(2,:),d.nMarkers,1);
end

if isfield(p,'conncolors') && ~isempty(p.conncolors) %field and colors are set
    if ischar(p.conncolors) % but in string format
        ccol=NaN(size(p.conn,1),3);
        for k=1:size(p.conncolors,2)
            ccol(k,:)=lookup_l(p.conncolors(k));%convert to num array
        end
    else
        ccol=p.conncolors; %field and colors are set in num format already
    end
    if size(p.conn,1) > length(p.conncolors)
        for k=length(p.conncolors)+1:size(p.conn,1)
            ccol(k,:)=colors(3,:);
        end
    end
else %no field or/and empty
    ccol=repmat(colors(3,:),size(p.conn,1),1);
end

%BBFIX 20120404: mcmerge problems
if p.trl~=0
    tcol=repmat(colors(4,:),d.nMarkers,1);
    if isfield(p,'tracecolors') && ~isempty(p.tracecolors) %(field and) tracecolors are set
        if ischar(p.tracecolors) % but in string format
            for k=1:size(p.tracecolors,2)
                tcol(k,:)=lookup_l(p.tracecolors(k));%convert to num array
            end
        else
            tcol(1:size(p.tracecolors,1),:)=p.tracecolors; %tracecolors in num format already
        end
    end
    if isempty(p.trm)
        p.trm=1:d.nMarkers; %plot all traces if trm is empty
    else
        if length(p.trm)<d.nMarkers
            %sort p.tracecolors/tcol1 according to p.trm
            %increase p.trm to have same length as d.nMarkers and fill it up with NaNs
            tmp=repmat(colors(4,:),d.nMarkers,1);
            tmp1=nan(d.nMarkers,1)';
            for k=1:length(p.trm)
                tmp(p.trm(k),:)=tcol(k,:);
                tmp1(p.trm(k))=p.trm(k);
            end
            tcol=tmp;
            p.trm=tmp1;
        end
    end
else
    tcol=[];
    p.trm=[];
end
p.tracecolors=tcol;

if ~isempty(p.trm)%created problem when merging and translating...
    p.trm=p.trm(:);
end

%warning if trace length is empty, but marker trace vector is set
if (p.trl==0 && ~isempty(p.trm)) || (p.trl==0 && ~isempty(p.tracecolors))
    disp([10, 'Warning: Please set trace length (trl) in your animation parameters in order to plot traces.', 10])
end

%if trace length is set, but vector indicating the markers to be traced is empty, all markers will be trace
if p.trl~=0 && isempty(p.trm)
    disp([10, 'Note: All markers traced.', 10])
    p.trm=1:d.nMarkers;
end


%BBFIX20120404: mcmerge problems
if p.showmnum==1
    ncol=repmat(colors(5,:),d.nMarkers,1);
    if isfield(p,'numbercolors') && ~isempty(p.numbercolors) %(field and) numbercolors are set
        if ischar(p.numbercolors) % but in string format
            for k=1:size(p.numbercolors,2)
                ncol(k,:)=lookup_l(p.numbercolors(k));%convert to num array
            end
        else
            ncol(1:size(p.numbercolors,1),:)=p.numbercolors; %numbercolors in num format already
        end
    end
    if isempty(p.numbers)
        p.numbers=1:d.nMarkers; %plot all markers if numbers is empty
    else
        if length(p.numbers)<d.nMarkers
            %sort p1.numbercolors/ncol1 according to p1.numbers
            %increase p1.numbers to have same length as d1.nMarkers and fill it up with NaNs
            tmp=repmat(colors(5,:),d.nMarkers,1);
            tmp1=nan(d.nMarkers,1)';
            for k=1:length(p.numbers)
                tmp(p.numbers(k),:)=ncol(k,:);
                tmp1(p.numbers(k))=p.numbers(k);
            end
            ncol=tmp;
            p.numbers=tmp1;
        end
    end
else
    ncol=[];
    p.numbers=[]; %fill up p1.numbers with nan
end


par.colors=colors;
par.markercolors=mcol;
par.conncolors=ccol;
par.tracecolors=tcol;
par.numbercolors=ncol;

par.numbers=p.numbers;


%fill up trace widths (twidth) to have same length as traced markers (trm) / nMarkers
if p.trl~=0
    if length(p.twidth)<d.nMarkers
        twidth=nan(d.nMarkers,1)';
        i=1;
        for k=1:length(p.trm)
            if isnan(p.trm(k))
            else
                twidth(k)=p.twidth(i);
                if i<length(p.twidth)
                    i=i+1;
                end
            end
        end
        p.twidth=twidth;
    end
end

%individual widths for connectors and traces
%fill up cwidth
if length(p.cwidth)<size(p.conn,1)
    cl=length(p.cwidth);
    cwidth=nan(size(p.conn,1),1);
    cwidth(1:length(p.cwidth))=p.cwidth;
    for k=cl+1:length(cwidth)
        if isnan(cwidth(k))
            cwidth(k)=cwidth(cl);%fill up with last given value
        end
    end
elseif length(p.cwidth)>size(p.conn,1)%if cwidth is longer than conn
    cwidth=p.cwidth(1:size(p.conn,1));
else
    cwidth=p.cwidth;
end
p.cwidth=cwidth;


if isfield(p,'twidth') && length(p.twidth)>1
    p.twidth=p.twidth;
else
    if ~isempty(p.trm)
        p.twidth=repmat(p.twidth(1),1,length(p.trm));
    end
end



az=p.az;
el=p.el;

d1 = mcrotate(d, az, [0 0 1], [0 0 0]);
d2 = mcrotate(d1, el, [1 0 0], [0 0 0]);

% bottom outer rectangle points (added HJ)
botScaling = p.botWidth / 2;
botpoints = botScaling *[-1 -1 0 1 -1 0 1 1 0 -1 1 0 -1 -1 0 ]; % outer frame
botpoints = mcrotate(botpoints, az, [0 0 1], [0 0 0]);
botpoints = mcrotate(botpoints, el, [1 0 0], [0 0 0]);
% bottom moving gridlines
xShifts = cumsum(xRootShifts);
yShifts = cumsum(yRootShifts);
nGridlines = p.nGridlines;
gridlinesX = cell(1,nGridlines);
gridlinesY = cell(1,nGridlines);
for i = 1:nGridlines
    gridlinesX{i} = repmat(botScaling * ...
        [-1+(i-1)*2/nGridlines -1 0 -1+(i-1)*2/nGridlines 1 0], length(n),1);
    gridlinesX{i}(2:end,[1 4]) = ...
        gridlinesX{i}(2:end,[1 4]) - repmat(xShifts,1,2);
    gridlinesX{i}(2:end,[1 4]) = ...
        mod(gridlinesX{i}(2:end,[1 4]) + botScaling, 2*botScaling) - botScaling;
    gridlinesX{i} = mcrotate(gridlinesX{i}, az, [0 0 1], [0 0 0]);
    gridlinesX{i} = mcrotate(gridlinesX{i}, el, [1 0 0], [0 0 0]);
    gridlinesY{i} = repmat(botScaling * ...
        [-1 -1+(i-1)*2/nGridlines  0 1 -1+(i-1)*2/nGridlines 0], length(n),1);
    gridlinesY{i}(2:end,[2 5]) = ...
        gridlinesY{i}(2:end,[2 5]) - repmat(yShifts,1,2);
    gridlinesY{i}(2:end,[2 5]) = ...
        mod(gridlinesY{i}(2:end,[2 5]) + botScaling, 2*botScaling) - botScaling;
    gridlinesY{i} = mcrotate(gridlinesY{i}, az, [0 0 1], [0 0 0]);
    gridlinesY{i} = mcrotate(gridlinesY{i}, el, [1 0 0], [0 0 0]);
end


if proj==0 % orthographic projection
    x=d2.data(n,1:3:end);
    y=d2.data(n,2:3:end);
    z=d2.data(n,3:3:end);
    botx = botpoints(1,1:3:end);
    boty = botpoints(1,2:3:end);
    botz = botpoints(1,3:3:end);
    
else % perspective projection
    if ~isfield(p,'pers') % for backward compatibility, use default values
        p.pers.c=[0 -4000 0];
        p.pers.th=[0 0 0];
        p.pers.e=[0 -2000 0];
    end
    
    th=180*p.pers.th/pi;
    rot1=[1 0 0; 0 cos(th(1)) -sin(th(1)); 0 sin(th(1)) cos(th(1))];
    rot2=[cos(th(2)) 0 sin(th(2)); 0 1 0; -sin(th(2)) 0 cos(th(2))];
    rot3=[cos(th(3)) -sin(th(3)) 0; sin(th(3)) cos(th(3)) 0; 0 0 1];
    rot123 = rot1*rot2*rot3;
    dd=zeros(size(d2.data(n,:)));
    botptsPers = zeros(size(botpoints));
    % gridlines1Pers = zeros(size(gridlines1));
    %p.pers.e(2)=min(min(d.data(n,2:3:end)));
    for k=1:d.nMarkers
        dd(:,3*k+(-2:0))=(rot123*(d2.data(n,3*k+(-2:0))' - ...
            repmat(p.pers.c',1,length(n))))';
    end
    nBotPoints = size(botpoints,2)/3;
    for k=1:nBotPoints
        botptsPers(1,3*k+(-2:0)) = ...
            (rot123*(botpoints(1,3*k+(-2:0))'-repmat(p.pers.c',1,1)))';
    end
    
    for i = 1:nGridlines
        gridlinesXPers{1,i} = zeros(size(gridlinesX{i}));
        for k=1:2
            gridlinesXPers{1,i}(:,3*k+(-2:0)) = ...
                (rot123*(gridlinesX{i}(:,3*k+(-2:0))' - ...
                repmat(p.pers.c',1,length(n))))';
        end
        gridlinesXx{1,i} =...
            -(gridlinesXPers{1,i}(:,1:3:end)-...
            repmat(p.pers.e(1),length(n),2)).*(p.pers.e(2)./gridlinesXPers{1,i}(:,2:3:end));
        gridlinesXz{1,i} =...
            -(gridlinesXPers{1,i}(:,3:3:end)-...
            repmat(p.pers.e(3),length(n),2)).*(p.pers.e(2)./gridlinesXPers{1,i}(:,2:3:end));
        
        gridlinesYPers{1,i} = zeros(size(gridlinesY{i}));
        for k=1:2
            gridlinesYPers{1,i}(:,3*k+(-2:0)) = ...
                (rot123*(gridlinesY{i}(:,3*k+(-2:0))' - ...
                repmat(p.pers.c',1,length(n))))';
        end
        gridlinesYx{1,i} =...
            -(gridlinesYPers{1,i}(:,1:3:end)-...
            repmat(p.pers.e(1),length(n),2)).*(p.pers.e(2)./gridlinesYPers{1,i}(:,2:3:end));
        gridlinesYz{1,i} =...
            -(gridlinesYPers{1,i}(:,3:3:end)-...
            repmat(p.pers.e(3),length(n),2)).*(p.pers.e(2)./gridlinesYPers{1,i}(:,2:3:end));
        
    end
    
    
    
    % make closest marker to be on the projection plan
    dd(:,2:3:end)=dd(:,2:3:end)-min(min(dd(:,2:3:end)))+p.pers.e(2)-p.pers.c(2);
    x=-(dd(:,1:3:end)-repmat(p.pers.e(1),length(n),d.nMarkers)).*(p.pers.e(2)./dd(:,2:3:end));
    y=(p.pers.e(2)-p.pers.c(2))./dd(:,2:3:end); % used for marker size scaling
    z=-(dd(:,3:3:end)-repmat(p.pers.e(3),length(n),d.nMarkers)).*(p.pers.e(2)./dd(:,2:3:end));
    
    botx =...
        -(botptsPers(:,1:3:end)-...
        repmat(p.pers.e(1),1,nBotPoints)).*(p.pers.e(2)./botptsPers(:,2:3:end));
    botz =...
        -(botptsPers(:,3:3:end)-...
        repmat(p.pers.e(3),1,nBotPoints)).*(p.pers.e(2)./botptsPers(:,2:3:end));
    
    %     gridlines1x =...
    %         -(gridlines1Pers(:,1:3:end)-...
    %         repmat(p.pers.e(1),length(n),2)).*(p.pers.e(2)./gridlines1Pers(:,2:3:end));
    %     gridlines1z =...
    %         -(gridlines1Pers(:,3:3:end)-...
    %         repmat(p.pers.e(3),length(n),2)).*(p.pers.e(2)./gridlines1Pers(:,2:3:end));
    %
end


if isempty(p.limits)
    % find ranges of coordinates
    tmp=x(:); maxx=max(tmp(find(~isnan(tmp))));
    minx=min(tmp(find(~isnan(tmp))));
    tmp=y(:); maxy=max(tmp(find(~isnan(tmp))));
    miny=min(tmp(find(~isnan(tmp))));
    tmp=z(:); maxz=max(tmp(find(~isnan(tmp))));
    minz=min(tmp(find(~isnan(tmp))));
    midx = (maxx+minx)/2;
    midy = (maxy+miny)/2;
    midz = (maxz+minz)/2;
    
    scrratio = p.scrsize(1)/p.scrsize(2);
    range = max([(maxx-minx)/scrratio, maxz-minz, maxy - miny])/2;
    zrange = (maxy-miny)/2;
    % axis limits for plot
    p.limits = [midx-scrratio*1.2*range ...
        midx+scrratio*1.2*range ...
        midy-1.2*range ...
        midy+1.2*range ...
        midz-1.2*range ...
        midz+1.2*range];
end
minxx = p.limits(1);
maxxx = p.limits(2);
minyy = p.limits(3);
maxyy = p.limits(4);
minzz = p.limits(5);
maxzz = p.limits(6);

if p.animate % HJ: in animate case, set figure and axes outside main loop
    figure(p.fignr); clf;
    set(gcf, 'WindowStyle','normal');
    set(gcf,'Position',[50 50 p.scrsize(1) p.scrsize(2)]) ; % DVD: w=720 h=420
    set(gcf, 'color', bgcol);
    view(0,90);
    colormap([ones(64,1) zeros(64,1) zeros(64,1)]);
end


for k=1:size(x,1) % main loop
    if p.animate
        clf;
        axes('position', [0 0 1 1], 'XLim', [minxx maxxx], 'YLim', [minzz maxzz]);
        hold on;
    else
        figure(p.fignr); clf;
        set(gcf, 'WindowStyle','normal');
        set(gcf,'Position',[50 50 p.scrsize(1) p.scrsize(2)]) ; % DVD: w=720 h=420
        axes('position', [0 0 1 1]); hold on
        set(gcf, 'color', bgcol);
        view(0,90);
        colormap([ones(64,1) zeros(64,1) zeros(64,1)]);
    end
    
    % plot bottom and moving gridlines (added HJ)
    for i = 1:nGridlines
        plot(gridlinesXx{1,i}(k,:), gridlinesXz{1,i}(k,:), 'g');
        plot(gridlinesYx{1,i}(k,:), gridlinesYz{1,i}(k,:), 'g');
    end
    plot(botx, botz, 'b');
    
    %     gridlinesXx{1,i}
    %     plot(gridlines1x(k,:), gridlines1z(k,:) , 'k');
    
%     % plot marker-to-marker connections
%     if ~isempty(p.conn)
%         for m=1:size(p.conn,1)
%             %if x(k,p.conn(m,1))*x(k,p.conn(m,2))~=0
%             if isfinite(x(k,p.conn(m,1))*x(k,p.conn(m,2)))
%                 plot([x(k,p.conn(m,1)) x(k,p.conn(m,2))], [z(k,p.conn(m,1)) z(k,p.conn(m,2))], '-','Color',ccol(m,:),'LineWidth', p.cwidth(m));
%             end
%         end
%     end
    
    % plot midpoint-to-midpoint connections
    if ~isempty(p.conn2)
        for m=1:size(p.conn2,1)
            %if x(k,p.conn2(m,1))*x(k,p.conn2(m,2))*x(k,p.conn2(m,3))*x(k,p.conn2(m,4))~=0
            if isfinite(x(k,p.conn2(m,1))*x(k,p.conn2(m,2))*x(k,p.conn2(m,3))*x(k,p.conn2(m,4)))
                tmpx1 = (x(k,p.conn2(m,1))+x(k,p.conn2(m,2)))/2;
                tmpx2 = (x(k,p.conn2(m,3))+x(k,p.conn2(m,4)))/2;
                tmpy1 = (z(k,p.conn2(m,1))+z(k,p.conn2(m,2)))/2;
                tmpy2 = (z(k,p.conn2(m,3))+z(k,p.conn2(m,4)))/2;
                plot([tmpx1 tmpx2], [tmpy1 tmpy2], '-','Color',ccol(m,:),'LineWidth', p.cwidth(m));
            end
        end
    end
    
    % plot traces if animation
    if p.animate && p.trl~=0
        %         trml=sort(p.trm);%BB: sorting marker traces
        trlen = round(p.fps * p.trl);
        start=max(1,k-trlen);
        ind = start-1+find(~isnan(x(start:k)));
        for m=1:length(p.trm)
            if isnan(p.trm(m)) %BBFIX 20120404 mcmerge adaption - NaN traces not plotted
            else
                plot(x(ind,m),z(ind,m),'-','Color',tcol(m,:),'Linewidth',p.twidth(m));
            end
        end
    end
    
    
    % plot markers
    for m=1:size(x,2)
        %if x(k,m)~=0 & ~isnan(x(k,m)) % if marker visible
        if isfinite(x(k,m)) % if marker visible
            if proj==0 % orthographic projection
                plot(x(k,m),z(k,m),'o','MarkerSize',p.msize(min(m,length(p.msize))),'MarkerEdgeColor',mcol(m,:),'MarkerFaceColor',mcol(m,:))
            else % perspective projection
                %                plot(x(k,m),z(k,m),[mcol(m) 'o'],'MarkerSize',p.msize(min(m,length(p.msize))),'MarkerFaceColor',mcol(m))
                if m == 16
                    plot(x(k,m),z(k,m),'o',...
                        'MarkerSize',round(y(k,m)*p.msize(min(m,length(p.msize)))),...
                        'MarkerEdgeColor',mcol(m,:),...
                        'MarkerFaceColor','g');
                elseif m == 20
                    plot(x(k,m),z(k,m),'o',...
                        'MarkerSize',round(y(k,m)*p.msize(min(m,length(p.msize)))),...
                        'MarkerEdgeColor',mcol(m,:),...
                        'MarkerFaceColor','r');
                    
                else
                    plot(x(k,m),z(k,m),'o',...
                        'MarkerSize',round(y(k,m)*p.msize(min(m,length(p.msize)))),...
                        'MarkerEdgeColor',mcol(m,:),...
                        'MarkerFaceColor',mcol(m,:));
                end
            end
            if p.showmnum
                if isempty(p.numbers)
                    h=text(x(k,m),z(k,m)+50,num2str(m));
                    set(h,'FontSize',16);
                    set(h,'Color',ncol(m,:))
                else
                    %if ismember(m, p.numbers) %FIX BB20120326: redefinining numbers plotting (mcmerge)
                    if m<=length(p.numbers)
                        if isnan(p.numbers(m)) %NaN numbers not plotted
                        else h=text(x(k,m),z(k,m)+50,num2str(p.numbers(m)));
                            set(h,'FontSize',16);
                            set(h,'Color',ncol(m,:))
                        end
                    end
                end
            end
            axis off
        end
%         if p.animate
%             axis off
%         else
%             axis([minxx maxxx minzz maxzz]); axis off;
%         end
    end
    
     % plot marker-to-marker connections
    if ~isempty(p.conn)
        for m=1:size(p.conn,1)
            %if x(k,p.conn(m,1))*x(k,p.conn(m,2))~=0
            if isfinite(x(k,p.conn(m,1))*x(k,p.conn(m,2)))
                plot([x(k,p.conn(m,1)) x(k,p.conn(m,2))], [z(k,p.conn(m,1)) z(k,p.conn(m,2))], '-','Color',ccol(m,:),'LineWidth', p.cwidth(m));
            end
        end
        if p.animate
            axis off
        else
            axis([minxx maxxx minzz maxzz]); axis off;
        end
    end
    
    if p.showfnum
        h=text(minxx+0.95*(maxxx-minxx), minzz+0.97*(maxzz-minzz), num2str(k),...
            'HorizontalAlignment','Right','FontSize',12,'FontWeight','bold');
        set(h,'Color',colors(5,:))
    end
    
    
    %  line([minxx, maxxx],[0 0]);
    
    
    
    if isfield(d,'mus')
        cColors = ['b', 'g', 'r', 'c', 'm'];
        nC = size(d.mus, 1); % nr of conceptors
        % plot baselines
        cWidth = (maxxx - minxx) / (4 * nC);
        cLeftmost = minxx + (maxxx - minxx) / 30;
        cBaseHight = minzz + 9*(maxzz - minzz) / 10;
        for c = 1:nC
            cColIndex = mod(c, length(cColors));
            if cColIndex == 0
                cColIndex = length(cColors);
            end
            cCol = cColors(cColIndex);
            line(cLeftmost + [(c-1)*cWidth, c*cWidth], ...
                [cBaseHight cBaseHight], ...
                'Color', cCol);
            if d.mus(c,k) > 0
                rectangle('Position', ...
                    [cLeftmost + (c-1)*cWidth, cBaseHight, ...
                    cWidth, d.mus(c,k) * (maxzz - minzz)/10], ...
                    'FaceColor', cCol, 'LineStyle','none');
            end
        end
        
    end
    %     %plot some copywrite text or anything else - BB20121102
    %     text(maxxx-250, 0, '�', 'FontSize', 40, 'color', [.7 .7 .7]);
    %     text(maxxx-300, 0, {'Birgitta Burger', 'Jyv�skyl� Univ.', 'Finland'});
    %     text(minxx+40, minzz+0.97*(maxzz-minzz), 'High Sub-Band 2 Flux', 'FontSize', 16, 'FontWeight', 'bold');
    %     text(minxx+70, minzz+0.97*(maxzz-minzz)-75, {'high speed of head'}, 'FontSize', 12, 'FontWeight', 'bold');
    
    drawnow
    hold off
    if  p.animate && not(strcmp(p.folder, '0'))
        fn=['frame', sprintf('%0.4d',k),'.png'];
        imwrite(frame2im(getframe),fn,'png');
    end
    
end

if p.animate && not(strcmp(p.folder, '0'))
    close
    cd(currdir);
end

return;


function colorar=lookup_l(colorstr)
if strcmp(colorstr, 'k')
    colorar=[0 0 0];
elseif strcmp(colorstr, 'w')
    colorar=[1 1 1];
elseif strcmp(colorstr, 'r')
    colorar=[1 0 0];
elseif strcmp(colorstr, 'g')
    colorar=[0 1 0];
elseif strcmp(colorstr, 'b')
    colorar=[0 0 1];
elseif strcmp(colorstr, 'y')
    colorar=[1 1 0];
elseif strcmp(colorstr, 'm')
    colorar=[1 0 1];
elseif strcmp(colorstr, 'c')
    colorar=[0 1 1];
end

return;



