% Phase 1: dynamic topology G(t) + delay-only Dijkstra baseline.
% Phase 2: add queue/load-aware link cost and compare with delay-only Dijkstra.
% Phase 3: replace global source-to-destination routing with local next-hop
%          decisions, action masks and loop prevention.
%
% This script is a research-process prototype, not the final MAPPO/CTDE method.
% It matches the modeling variables in the README:
%   G(t) = (V, E(t), X(t))
%   x_i(t)  = [q_i(t), Q_i^max, service_i(t)]
%   x_ij(t) = [d_ij(t), C_ij(t), r_ij(t), rho_ij(t), p_ij^out(t), T_rem_ij(t)]

clear; clc; close all;

cfg = defaultConfig();
outDir = fullfile(pwd, "outputs");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

fprintf("LEO routing phase 1-3 MATLAB prototype\n");
fprintf("Output directory: %s\n\n", outDir);

%% Phase 1: dynamic topology + delay-only Dijkstra
phase1Table = phase1_dynamicTopologyDijkstra(cfg);
writetable(phase1Table, fullfile(outDir, "phase1_dynamic_topology_dijkstra.csv"));

disp("=== Phase 1: first several dynamic-topology Dijkstra records ===");
disp(phase1Table(1:min(8, height(phase1Table)), :));

%% Phase 2: delay-only Dijkstra vs queue/load-aware Dijkstra
metricsDelay = runRoutingPolicy(cfg, "B0_delay_only_dijkstra", "delay_only");
metricsQueue = runRoutingPolicy(cfg, "B1_queue_load_dijkstra", "queue_load");
phase2Metrics = struct2table([metricsDelay; metricsQueue]);
writetable(phase2Metrics, fullfile(outDir, "phase2_policy_metrics.csv"));

disp("=== Phase 2: policy comparison ===");
disp(phase2Metrics);

%% Phase 3: local next-hop decision + action mask + loop prevention
metricsLocal = runLocalNextHopPolicy(cfg, "B2_local_next_hop_masked", "queue_load");
phase3Metrics = struct2table(metricsLocal);
writetable(phase3Metrics, fullfile(outDir, "phase3_local_next_hop_metrics.csv"));

phaseAllMetrics = struct2table([metricsDelay; metricsQueue; metricsLocal]);
writetable(phaseAllMetrics, fullfile(outDir, "phase123_policy_metrics.csv"));

fprintf("=== Phase 3: local next-hop policy with action mask and loop prevention ===\n");
disp(phase3Metrics);

%% Plot comparison
fig = figure("Name", "Phase 1-3 policy comparison", "Color", "w");
tiledlayout(2, 3, "Padding", "compact", "TileSpacing", "compact");

labels = categorical(phaseAllMetrics.policy);
labels = reordercats(labels, phaseAllMetrics.policy);

nexttile;
bar(labels, phaseAllMetrics.avgDelayMs);
ylabel("Average delay (ms)");
title("Average end-to-end delay");
grid on;

nexttile;
bar(labels, phaseAllMetrics.p95DelayMs);
ylabel("P95 delay (ms)");
title("Tail delay");
grid on;

nexttile;
bar(labels, phaseAllMetrics.maxQueue);
ylabel("Max queue");
title("Maximum queue length");
grid on;

nexttile;
bar(labels, phaseAllMetrics.dropRate);
ylabel("Drop rate");
title("Packet drop rate");
grid on;

nexttile;
bar(labels, phaseAllMetrics.avgHops);
ylabel("Average hops");
title("Path length");
grid on;

nexttile;
bar(labels, phaseAllMetrics.loopDropRate);
ylabel("Loop/action-mask drop rate");
title("Loop prevention pressure");
grid on;

exportgraphics(fig, fullfile(outDir, "phase123_policy_comparison.png"), "Resolution", 200);

fprintf("\nDone. Generated files:\n");
fprintf("  %s\n", fullfile(outDir, "phase1_dynamic_topology_dijkstra.csv"));
fprintf("  %s\n", fullfile(outDir, "phase2_policy_metrics.csv"));
fprintf("  %s\n", fullfile(outDir, "phase3_local_next_hop_metrics.csv"));
fprintf("  %s\n", fullfile(outDir, "phase123_policy_metrics.csv"));
fprintf("  %s\n", fullfile(outDir, "phase123_policy_comparison.png"));

%% Local functions

function cfg = defaultConfig()
    cfg.nPlanes = 6;
    cfg.satsPerPlane = 8;
    cfg.timeSlots = 30;

    cfg.qMax = 60;                 % Q_i^max
    cfg.servicePacketsPerSlot = 4; % service_i(t)
    cfg.capacityMbps = 100;        % C_ij(t)
    cfg.dRefMs = 20;               % reference delay for normalization

    cfg.betaQ = 1.2;               % queue cost coefficient
    cfg.betaRho = 0.8;             % load cost coefficient
    cfg.loadDecay = 0.55;
    cfg.packetDemandMbps = 1.0;
    cfg.maxLocalHops = 14;            % TTL for local next-hop routing
    cfg.betaProgress = 0.7;           % destination-progress coefficient
    cfg.betaTRem = 0.4;               % remaining-link-time coefficient

    cfg.randomSeed = 7;
end

function n = numSats(cfg)
    n = cfg.nPlanes * cfg.satsPerPlane;
end

function id = satId(plane, pos, cfg)
    % 1-based satellite id.
    id = (plane - 1) * cfg.satsPerPlane + pos;
end

function [plane, pos] = decodeSat(id, cfg)
    plane = floor((id - 1) / cfg.satsPerPlane) + 1;
    pos = mod(id - 1, cfg.satsPerPlane) + 1;
end

function delay = sameOrbitDelayMs(t, plane, pos, cfg)
    t0 = t - 1;
    delay = 8.0 + 1.5 * sin(2 * pi * (t0 + pos) / cfg.timeSlots);
end

function delay = crossPlaneDelayMs(t, plane, pos, cfg)
    t0 = t - 1;
    delay = 12.0 + 2.0 * cos(2 * pi * (t0 + plane + pos) / cfg.timeSlots);
end

function ok = crossPlaneAvailable(t, plane, pos, cfg)
    % A toy seam/polar interruption model.
    % It makes E(t) change over time without needing external orbit files.
    t0 = t - 1;
    phase = mod(t0 + pos - 1, cfg.timeSlots) / cfg.timeSlots;
    seamLink = (plane == cfg.nPlanes);
    ok = true;
    if seamLink && phase >= 0.45 && phase <= 0.65
        ok = false;
    end
end

function trem = remainingTimeSeconds(t, isCross, plane, pos, cfg)
    if ~isCross
        trem = 999.0;
        return;
    end

    t0 = t - 1;
    phaseStep = mod(t0 + pos - 1, cfg.timeSlots);
    startStep = floor(0.45 * cfg.timeSlots);
    if phaseStep < startStep
        trem = startStep - phaseStep;
    else
        trem = cfg.timeSlots - phaseStep + startStep;
    end
    trem = max(1.0, double(trem));
end

function G = buildTopology(t, queues, usedRate, cfg)
    % Build directed graph. Each physical ISL is represented by two directed arcs.
    n = numSats(cfg);
    src = [];
    dst = [];
    delay = [];
    capacity = [];
    used = [];
    rho = [];
    reliability = [];
    pout = [];
    trem = [];
    isCrossList = [];

    function addArc(u, v, delayMs, isCross, plane, pos)
        r = usedRate(u, v);
        rhoVal = min(1.0, r / cfg.capacityMbps);
        relVal = max(0.80, 0.995 - 0.12 * rhoVal);
        src(end + 1, 1) = u;
        dst(end + 1, 1) = v;
        delay(end + 1, 1) = delayMs;
        capacity(end + 1, 1) = cfg.capacityMbps;
        used(end + 1, 1) = r;
        rho(end + 1, 1) = rhoVal;
        reliability(end + 1, 1) = relVal;
        pout(end + 1, 1) = 1.0 - relVal;
        trem(end + 1, 1) = remainingTimeSeconds(t, isCross, plane, pos, cfg);
        isCrossList(end + 1, 1) = double(isCross);
    end

    function addUndirectedLink(u, v, delayMs, isCross, plane, pos)
        addArc(u, v, delayMs, isCross, plane, pos);
        addArc(v, u, delayMs, isCross, plane, pos);
    end

    for p = 1:cfg.nPlanes
        for s = 1:cfg.satsPerPlane
            u = satId(p, s, cfg);

            % Same-orbit link.
            nextPos = mod(s, cfg.satsPerPlane) + 1;
            vSame = satId(p, nextPos, cfg);
            addUndirectedLink(u, vSame, sameOrbitDelayMs(t, p, s, cfg), false, p, s);

            % Cross-plane link.
            nextPlane = mod(p, cfg.nPlanes) + 1;
            if crossPlaneAvailable(t, p, s, cfg)
                vCross = satId(nextPlane, s, cfg);
                addUndirectedLink(u, vCross, crossPlaneDelayMs(t, p, s, cfg), true, p, s);
            end
        end
    end

    G = digraph(src, dst, delay, n);
    G.Edges.DelayMs = delay;
    G.Edges.CapacityMbps = capacity;
    G.Edges.UsedRateMbps = used;
    G.Edges.Rho = rho;
    G.Edges.Reliability = reliability;
    G.Edges.POut = pout;
    G.Edges.TRem = trem;
    G.Edges.IsCross = isCrossList;

    G.Nodes.Queue = queues(:);
    G.Nodes.QMax = cfg.qMax * ones(n, 1);
    G.Nodes.Service = cfg.servicePacketsPerSlot * ones(n, 1);
end

function T = phase1_dynamicTopologyDijkstra(cfg)
    n = numSats(cfg);
    queues = zeros(n, 1);
    usedRate = zeros(n, n);
    srcSat = satId(1, 1, cfg);
    dstSat = satId(ceil(cfg.nPlanes / 2), ceil(cfg.satsPerPlane / 2), cfg);

    time = [];
    numEdges = [];
    hops = [];
    delayMs = [];
    pathText = strings(0, 1);

    for t = 1:cfg.timeSlots
        G = buildTopology(t, queues, usedRate, cfg);
        [path, dist] = shortestpath(G, srcSat, dstSat);
        time(end + 1, 1) = t;
        numEdges(end + 1, 1) = numedges(G);

        if isempty(path) || isinf(dist)
            hops(end + 1, 1) = NaN;
            delayMs(end + 1, 1) = Inf;
            pathText(end + 1, 1) = "no-path";
        else
            hops(end + 1, 1) = numel(path) - 1;
            delayMs(end + 1, 1) = pathDelayMs(G, path);
            pathText(end + 1, 1) = join(string(path), "->");
        end
    end

    T = table(time, numEdges, hops, delayMs, pathText, ...
        'VariableNames', {'timeSlot', 'numDirectedEdges', 'hops', 'delayMs', 'path'});
end

function packets = generatePackets(t, cfg, hotspotDst)
    % Columns: [src, dst, createdTime, demandMbps]
    n = numSats(cfg);
    packets = [];

    % Background traffic.
    for k = 1:10
        src = randi(n);
        dst = randi(n);
        if src ~= dst
            packets(end + 1, :) = [src, dst, t, cfg.packetDemandMbps]; %#ok<AGROW>
        end
    end

    % Hotspot traffic to make queue pressure visible.
    for k = 1:18
        src = randi(n);
        if src ~= hotspotDst
            packets(end + 1, :) = [src, hotspotDst, t, cfg.packetDemandMbps]; %#ok<AGROW>
        end
    end
end

function G2 = setRoutingWeight(G, cfg, mode)
    G2 = G;
    if mode == "delay_only"
        G2.Edges.Weight = G2.Edges.DelayMs;
    elseif mode == "queue_load"
        toNode = G2.Edges.EndNodes(:, 2);
        qNorm = G2.Nodes.Queue(toNode) ./ cfg.qMax;
        G2.Edges.Weight = G2.Edges.DelayMs ...
            + cfg.betaQ * cfg.dRefMs * qNorm ...
            + cfg.betaRho * cfg.dRefMs * G2.Edges.Rho;
    else
        error("Unknown weight mode: %s", mode);
    end
end

function metric = runRoutingPolicy(cfg, policyName, weightMode)
    rng(cfg.randomSeed);
    n = numSats(cfg);
    queues = zeros(n, 1);
    usedRate = zeros(n, n);
    hotspotDst = satId(ceil(cfg.nPlanes / 2), ceil(cfg.satsPerPlane / 2), cfg);

    delivered = 0;
    dropped = 0;
    delays = [];
    hopList = [];
    maxQueueTrace = [];

    for t = 1:cfg.timeSlots
        packets = generatePackets(t, cfg, hotspotDst);

        for p = 1:size(packets, 1)
            % Refresh G(t) before each routing decision so queue/load-aware
            % routing can use the latest local state in this time slot.
            G = buildTopology(t, queues, usedRate, cfg);
            GRoute = setRoutingWeight(G, cfg, weightMode);

            src = packets(p, 1);
            dst = packets(p, 2);
            demand = packets(p, 4);

            [path, dist] = shortestpath(GRoute, src, dst);
            if isempty(path) || isinf(dist)
                dropped = dropped + 1;
                continue;
            end

            [ok, queues, usedRate] = applyPathLoad(G, path, queues, usedRate, demand, cfg);
            if ok
                delivered = delivered + 1;
                delays(end + 1, 1) = estimateE2EDelayMs(G, path, cfg); %#ok<AGROW>
                hopList(end + 1, 1) = numel(path) - 1; %#ok<AGROW>
            else
                dropped = dropped + 1;
            end
        end

        queues = max(0, queues - cfg.servicePacketsPerSlot);
        usedRate = usedRate * cfg.loadDecay;
        maxQueueTrace(end + 1, 1) = max(queues); %#ok<AGROW>
    end

    total = delivered + dropped;
    if isempty(delays)
        avgDelay = Inf;
        p95Delay = Inf;
        avgHops = Inf;
    else
        avgDelay = mean(delays);
        p95Delay = percentileValue(delays, 95);
        avgHops = mean(hopList);
    end

    metric = struct();
    metric.policy = string(policyName);
    metric.delivered = delivered;
    metric.dropped = dropped;
    metric.dropRate = dropped / max(1, total);
    metric.avgDelayMs = avgDelay;
    metric.p95DelayMs = p95Delay;
    metric.avgHops = avgHops;
    metric.maxQueue = max(maxQueueTrace);
    metric.loopDrops = 0;
    metric.ttlDrops = 0;
    metric.loopDropRate = 0;
end

function metric = runLocalNextHopPolicy(cfg, policyName, weightMode)
    rng(cfg.randomSeed);
    n = numSats(cfg);
    queues = zeros(n, 1);
    usedRate = zeros(n, n);
    hotspotDst = satId(ceil(cfg.nPlanes / 2), ceil(cfg.satsPerPlane / 2), cfg);

    delivered = 0;
    dropped = 0;
    loopDrops = 0;
    ttlDrops = 0;
    delays = [];
    hopList = [];
    maxQueueTrace = [];

    for t = 1:cfg.timeSlots
        packets = generatePackets(t, cfg, hotspotDst);

        for p = 1:size(packets, 1)
            src = packets(p, 1);
            dst = packets(p, 2);
            demand = packets(p, 4);

            [status, path, GLast] = routeByLocalNextHop(t, src, dst, queues, usedRate, cfg, weightMode);
            if status ~= "ok"
                dropped = dropped + 1;
                if status == "loop_mask_blocked"
                    loopDrops = loopDrops + 1;
                elseif status == "ttl_exceeded"
                    ttlDrops = ttlDrops + 1;
                end
                continue;
            end

            [ok, queues, usedRate] = applyPathLoad(GLast, path, queues, usedRate, demand, cfg);
            if ok
                delivered = delivered + 1;
                delays(end + 1, 1) = estimateE2EDelayMs(GLast, path, cfg); %#ok<AGROW>
                hopList(end + 1, 1) = numel(path) - 1; %#ok<AGROW>
            else
                dropped = dropped + 1;
            end
        end

        queues = max(0, queues - cfg.servicePacketsPerSlot);
        usedRate = usedRate * cfg.loadDecay;
        maxQueueTrace(end + 1, 1) = max(queues); %#ok<AGROW>
    end

    total = delivered + dropped;
    if isempty(delays)
        avgDelay = Inf;
        p95Delay = Inf;
        avgHops = Inf;
    else
        avgDelay = mean(delays);
        p95Delay = percentileValue(delays, 95);
        avgHops = mean(hopList);
    end

    metric = struct();
    metric.policy = string(policyName);
    metric.delivered = delivered;
    metric.dropped = dropped;
    metric.dropRate = dropped / max(1, total);
    metric.avgDelayMs = avgDelay;
    metric.p95DelayMs = p95Delay;
    metric.avgHops = avgHops;
    metric.maxQueue = max(maxQueueTrace);
    metric.loopDrops = loopDrops;
    metric.ttlDrops = ttlDrops;
    metric.loopDropRate = (loopDrops + ttlDrops) / max(1, total);
end

function [status, path, GLast] = routeByLocalNextHop(t, src, dst, queues, usedRate, cfg, weightMode)
    current = src;
    visited = false(numSats(cfg), 1);
    visited(current) = true;
    path = current;
    GLast = buildTopology(t, queues, usedRate, cfg);

    for hop = 1:cfg.maxLocalHops
        if current == dst
            status = "ok";
            return;
        end

        GLast = buildTopology(t, queues, usedRate, cfg);
        nbrs = successors(GLast, current);
        if isempty(nbrs)
            status = "no_neighbor";
            return;
        end

        % Action mask: do not go to nodes already visited by this packet.
        % This is the hand-written version of the mask later passed to MAPPO.
        candidateNbrs = nbrs(~visited(nbrs));
        if isempty(candidateNbrs)
            status = "loop_mask_blocked";
            return;
        end

        nextHop = selectLocalNextHop(GLast, current, dst, candidateNbrs, cfg, weightMode);
        current = nextHop;
        path(end + 1) = current; %#ok<AGROW>
        visited(current) = true;
    end

    if current == dst
        status = "ok";
    else
        status = "ttl_exceeded";
    end
end

function nextHop = selectLocalNextHop(G, current, dst, candidateNbrs, cfg, weightMode)
    bestCost = Inf;
    nextHop = candidateNbrs(1);

    for k = 1:numel(candidateNbrs)
        v = candidateNbrs(k);
        e = findedge(G, current, v);
        if e == 0
            continue;
        end

        localCost = localEdgeCost(G, e, v, cfg, weightMode);
        progressCost = cfg.betaProgress * cfg.dRefMs * topologyDistanceToDestination(v, dst, cfg);
        tremPenalty = cfg.betaTRem * cfg.dRefMs / max(1.0, G.Edges.TRem(e));
        totalCost = localCost + progressCost + tremPenalty;

        if totalCost < bestCost
            bestCost = totalCost;
            nextHop = v;
        end
    end
end

function cost = localEdgeCost(G, edgeIdx, toNode, cfg, weightMode)
    cost = G.Edges.DelayMs(edgeIdx);
    if weightMode == "queue_load"
        qNorm = G.Nodes.Queue(toNode) / cfg.qMax;
        cost = cost ...
            + cfg.betaQ * cfg.dRefMs * qNorm ...
            + cfg.betaRho * cfg.dRefMs * G.Edges.Rho(edgeIdx);
    elseif weightMode ~= "delay_only"
        error("Unknown local weight mode: %s", weightMode);
    end
end

function d = topologyDistanceToDestination(node, dst, cfg)
    [p1, s1] = decodeSat(node, cfg);
    [p2, s2] = decodeSat(dst, cfg);
    planeDist = abs(p1 - p2);
    planeDist = min(planeDist, cfg.nPlanes - planeDist);
    slotDist = abs(s1 - s2);
    slotDist = min(slotDist, cfg.satsPerPlane - slotDist);
    d = planeDist + slotDist;
end

function [ok, queues, usedRate] = applyPathLoad(G, path, queues, usedRate, demand, cfg)
    ok = true;

    % Queue pressure is added to relay nodes.
    for k = 2:(numel(path) - 1)
        v = path(k);
        queues(v) = queues(v) + 1;
        if queues(v) > cfg.qMax
            ok = false;
            return;
        end
    end

    % Link load is updated on both directions of the physical ISL.
    for k = 1:(numel(path) - 1)
        u = path(k);
        v = path(k + 1);
        if findedge(G, u, v) == 0
            ok = false;
            return;
        end
        usedRate(u, v) = usedRate(u, v) + demand;
        usedRate(v, u) = usedRate(v, u) + demand;
    end
end

function d = pathDelayMs(G, path)
    d = 0;
    for k = 1:(numel(path) - 1)
        e = findedge(G, path(k), path(k + 1));
        d = d + G.Edges.DelayMs(e);
    end
end

function d = estimateE2EDelayMs(G, path, cfg)
    propagation = pathDelayMs(G, path);
    queueDelay = 0;
    loadDelay = 0;

    for k = 2:numel(path)
        v = path(k);
        queueDelay = queueDelay + G.Nodes.Queue(v) / cfg.servicePacketsPerSlot;
    end

    for k = 1:(numel(path) - 1)
        e = findedge(G, path(k), path(k + 1));
        loadDelay = loadDelay + G.Edges.Rho(e) * cfg.dRefMs;
    end

    d = propagation + queueDelay + loadDelay;
end

function y = percentileValue(x, p)
    x = sort(x(:));
    if isempty(x)
        y = NaN;
        return;
    end
    idx = 1 + (numel(x) - 1) * p / 100;
    lo = floor(idx);
    hi = ceil(idx);
    if lo == hi
        y = x(lo);
    else
        y = x(lo) + (idx - lo) * (x(hi) - x(lo));
    end
end
