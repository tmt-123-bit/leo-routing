%% Preliminary MATLAB prototype for the proposed LEO routing idea
%
% This is early-stage research code, not the final MAPPO/CTDE training code.
% It implements:
%   1) Dynamic topology G(t) = (V, E(t), X(t))
%   2) Delay-only Dijkstra baseline
%   3) Queue/load-aware routing
%   4) Link-lifetime/reliability action mask
%   5) Local observation z_ij and local reward logs
%   6) Control-overhead and computation-overhead estimates

clear; clc; close all;

cfg = defaultConfig();
outDir = fullfile(pwd, "outputs");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

policies = [
    makePolicy("P0_delay_only", false, false, false, false, false)
    makePolicy("P1_queue_load", true,  false, false, false, false)
    makePolicy("P2_queue_load_lifetime_mask", true, true,  false, false, false)
    makePolicy("P3_local_full", true, true,  true,  true,  true)
    makePolicy("A1_local_no_queue_load", false, true, true,  true,  true)
    makePolicy("A2_local_no_lifetime_mask", true, false, true,  true,  true)
    makePolicy("A3_local_no_reliability_risk", true, true,  false, true,  true)
    makePolicy("A4_local_no_progress", true, true,  true,  false, true)
];

fprintf("Preliminary LEO routing prototype\n");
fprintf("N = %d satellites, T = %d slots\n", cfg.nSats, cfg.timeSlots);
fprintf("Output directory: %s\n\n", outDir);

metrics = repmat(emptyMetric(), numel(policies), 1);
allLogs = table();

for p = 1:numel(policies)
    fprintf("Running %s ...\n", policies(p).name);
    [metrics(p), logs] = runPolicy(cfg, policies(p));
    allLogs = [allLogs; logs]; %#ok<AGROW>
end

metricsTable = struct2table(metrics);
writetable(metricsTable, fullfile(outDir, "policy_metrics.csv"));
writetable(allLogs, fullfile(outDir, "time_slot_logs.csv"));
plotMetrics(metricsTable, outDir);

disp("=== Policy metrics ===");
disp(metricsTable);

fprintf("\nGenerated files:\n");
fprintf("  %s\n", fullfile(outDir, "policy_metrics.csv"));
fprintf("  %s\n", fullfile(outDir, "time_slot_logs.csv"));
fprintf("  %s\n", fullfile(outDir, "policy_comparison.png"));

%% Configuration and policy

function cfg = defaultConfig()
    cfg.nPlanes = 4;
    cfg.satsPerPlane = 6;
    cfg.nSats = cfg.nPlanes * cfg.satsPerPlane;
    cfg.timeSlots = 30;
    cfg.maxLocalHops = 12;

    cfg.qMax = 45;
    cfg.servicePacketsPerSlot = 3;
    cfg.capacityMbps = 100;
    cfg.packetDemandMbps = 1.0;
    cfg.dRefMs = 20;

    cfg.betaQueue = 1.20;
    cfg.betaLoad = 0.80;
    cfg.betaRisk = 10.0;
    cfg.betaLifetime = 8.0;
    cfg.betaProgress = 3.0;

    cfg.tSafe = 3.0;
    cfg.rMin = 0.86;
    cfg.bMinMbps = 5.0;
    cfg.loadDecay = 0.55;
    cfg.randomSeed = 11;

    cfg.helloPeriodSlots = 1;
    cfg.bytesNodeId = 2;
    cfg.bytesTimestamp = 4;
    cfg.bytesQueue = 2;
    cfg.bytesLoad = 2;
    cfg.bytesReliability = 2;
    cfg.bytesTRem = 2;

    cfg.zDim = 7;
    cfg.hidden1 = 32;
    cfg.hidden2 = 16;
end

function policy = makePolicy(name, useQueueLoad, useLifetimeMask, useReliabilityRisk, useProgress, useLocalNextHop)
    policy.name = string(name);
    policy.useQueueLoad = useQueueLoad;
    policy.useLifetimeMask = useLifetimeMask;
    policy.useReliabilityRisk = useReliabilityRisk;
    policy.useProgress = useProgress;
    policy.useLocalNextHop = useLocalNextHop;
end

function m = emptyMetric()
    m.policy = "";
    m.delivered = 0;
    m.dropped = 0;
    m.dropRate = 0;
    m.avgDelayMs = 0;
    m.p95DelayMs = 0;
    m.avgHops = 0;
    m.maxQueue = 0;
    m.avgJainLoad = 0;
    m.rerouteRiskEvents = 0;
    m.loopDrops = 0;
    m.ttlDrops = 0;
    m.controlBytes = 0;
    m.estimatedFLOPs = 0;
    m.invalidActionRatio = 0;
    m.avgLocalReward = 0;
end

%% Main simulation

function [metric, logs] = runPolicy(cfg, policy)
    rng(cfg.randomSeed);
    queues = zeros(cfg.nSats, 1);
    usedRate = zeros(cfg.nSats, cfg.nSats);
    age = zeros(cfg.nSats, 1);
    hotspotDst = satId(ceil(cfg.nPlanes / 2), ceil(cfg.satsPerPlane / 2), cfg);

    delivered = 0;
    dropped = 0;
    delays = [];
    hops = [];
    rewards = [];
    maxQueueTrace = [];
    jainTrace = [];
    riskEvents = 0;
    loopDrops = 0;
    ttlDrops = 0;
    controlBytes = 0;
    estimatedFLOPs = 0;
    invalidActions = 0;
    candidateActions = 0;

    logRows = [];

    for t = 1:cfg.timeSlots
        packets = generatePackets(t, cfg, hotspotDst);
        slotDelivered = 0;
        slotDropped = 0;
        slotInvalid = 0;
        slotCandidate = 0;
        slotRewards = [];

        for k = 1:size(packets, 1)
            src = packets(k, 1);
            dst = packets(k, 2);
            demand = packets(k, 4);

            X = buildTopologyState(t, queues, usedRate, age, cfg);
            estimatedFLOPs = estimatedFLOPs + estimateDecisionFLOPs(sum(isfinite(X.delay(src, :))), cfg);

            % Construct local observation for the first hop. Later MAPPO/CTDE
            % Actor can use this type of per-neighbor feature with an action mask.
            z = localObservationZ(X, queues, src, dst, age, cfg); %#ok<NASGU>

            if policy.useLocalNextHop
                [path, routeStatus, localInvalid, localCandidate] = routeLocalNextHop(t, src, dst, queues, usedRate, age, cfg, policy);
                invalidActions = invalidActions + localInvalid;
                candidateActions = candidateActions + localCandidate;
                slotInvalid = slotInvalid + localInvalid;
                slotCandidate = slotCandidate + localCandidate;
                if routeStatus ~= "ok"
                    dropped = dropped + 1;
                    slotDropped = slotDropped + 1;
                    if routeStatus == "loop_blocked"
                        loopDrops = loopDrops + 1;
                    elseif routeStatus == "ttl_exceeded"
                        ttlDrops = ttlDrops + 1;
                    end
                    continue;
                end
            else
                [W, invalidCount, candidateCount] = buildRoutingWeights(X, queues, cfg, policy, dst);
                invalidActions = invalidActions + invalidCount;
                candidateActions = candidateActions + candidateCount;
                slotInvalid = slotInvalid + invalidCount;
                slotCandidate = slotCandidate + candidateCount;

                [path, dist] = dijkstraMatrix(W, src, dst);
                if isempty(path) || isinf(dist)
                    dropped = dropped + 1;
                    slotDropped = slotDropped + 1;
                    continue;
                end
            end

            rLocal = localReward(X, queues, path, cfg);
            rewards(end + 1, 1) = rLocal; %#ok<AGROW>
            slotRewards(end + 1, 1) = rLocal; %#ok<AGROW>

            if hasRiskyFirstHop(X, path, cfg)
                riskEvents = riskEvents + 1;
            end

            [ok, queues, usedRate] = applyPathLoad(X, path, queues, usedRate, demand, cfg);
            if ok
                delivered = delivered + 1;
                slotDelivered = slotDelivered + 1;
                delays(end + 1, 1) = estimateE2EDelayMs(X, queues, path, cfg); %#ok<AGROW>
                hops(end + 1, 1) = numel(path) - 1; %#ok<AGROW>
            else
                dropped = dropped + 1;
                slotDropped = slotDropped + 1;
            end
        end

        queues = max(0, queues - cfg.servicePacketsPerSlot);
        usedRate = usedRate * cfg.loadDecay;

        age = age + 1;
        if mod(t, cfg.helloPeriodSlots) == 0
            Xhello = buildTopologyState(t, queues, usedRate, age, cfg);
            controlBytes = controlBytes + estimateHelloBytes(Xhello, cfg, policy);
            age(:) = 0;
        end

        Xlog = buildTopologyState(t, queues, usedRate, age, cfg);
        maxQueueTrace(end + 1, 1) = max(queues); %#ok<AGROW>
        jainTrace(end + 1, 1) = jainLoadIndex(Xlog.rho); %#ok<AGROW>
        if isempty(slotRewards)
            avgSlotReward = NaN;
        else
            avgSlotReward = mean(slotRewards);
        end

        logRows(end + 1, :) = [ ...
            t, policyIndex(policy.name), slotDelivered, slotDropped, ...
            max(queues), mean(queues), jainLoadIndex(Xlog.rho), ...
            slotInvalid, slotCandidate, avgSlotReward ...
        ]; %#ok<AGROW>
    end

    total = delivered + dropped;
    metric = emptyMetric();
    metric.policy = policy.name;
    metric.delivered = delivered;
    metric.dropped = dropped;
    metric.dropRate = dropped / max(1, total);
    metric.avgDelayMs = meanOrInf(delays);
    metric.p95DelayMs = percentileOrInf(delays, 95);
    metric.avgHops = meanOrInf(hops);
    metric.maxQueue = max(maxQueueTrace);
    metric.avgJainLoad = mean(jainTrace);
    metric.rerouteRiskEvents = riskEvents;
    metric.loopDrops = loopDrops;
    metric.ttlDrops = ttlDrops;
    metric.controlBytes = controlBytes;
    metric.estimatedFLOPs = estimatedFLOPs;
    metric.invalidActionRatio = invalidActions / max(1, candidateActions);
    metric.avgLocalReward = meanOrInf(rewards);

    logs = array2table(logRows, 'VariableNames', { ...
        'timeSlot', 'policyId', 'delivered', 'dropped', 'maxQueue', ...
        'avgQueue', 'jainLoad', 'invalidActions', 'candidateActions', 'avgLocalReward'});
    logs.policy = repmat(policy.name, height(logs), 1);
    logs = movevars(logs, 'policy', 'After', 'policyId');
end

%% Topology and state

function X = buildTopologyState(t, queues, usedRate, age, cfg)
    n = cfg.nSats;
    infMat = inf(n, n);
    X.delay = infMat;
    X.capacity = zeros(n, n);
    X.usedRate = zeros(n, n);
    X.rho = zeros(n, n);
    X.reliability = zeros(n, n);
    X.pout = ones(n, n);
    X.trem = zeros(n, n);
    X.ageTo = zeros(n, n);
    X.available = false(n, n);

    for p = 1:cfg.nPlanes
        for s = 1:cfg.satsPerPlane
            u = satId(p, s, cfg);

            vSame = satId(p, mod(s, cfg.satsPerPlane) + 1, cfg);
            X = addUndirectedLink(X, u, vSame, sameOrbitDelayMs(t, p, s, cfg), false, p, s, usedRate, age, cfg);

            nextPlane = mod(p, cfg.nPlanes) + 1;
            if crossPlaneAvailable(t, p, s, cfg)
                vCross = satId(nextPlane, s, cfg);
                X = addUndirectedLink(X, u, vCross, crossPlaneDelayMs(t, p, s, cfg), true, p, s, usedRate, age, cfg);
            end
        end
    end

    X.queues = queues;
end

function X = addUndirectedLink(X, u, v, delayMs, isCross, plane, pos, usedRate, age, cfg)
    X = addDirectedLink(X, u, v, delayMs, isCross, plane, pos, usedRate, age, cfg);
    X = addDirectedLink(X, v, u, delayMs, isCross, plane, pos, usedRate, age, cfg);
end

function X = addDirectedLink(X, u, v, delayMs, isCross, plane, pos, usedRate, age, cfg)
    r = usedRate(u, v);
    rhoVal = min(1.0, r / cfg.capacityMbps);
    relVal = max(0.80, 0.995 - 0.12 * rhoVal);

    X.delay(u, v) = delayMs;
    X.capacity(u, v) = cfg.capacityMbps;
    X.usedRate(u, v) = r;
    X.rho(u, v) = rhoVal;
    X.reliability(u, v) = relVal;
    X.pout(u, v) = 1.0 - relVal;
    X.trem(u, v) = remainingTimeSecondsForLink(isCross, plane, pos, cfg);
    X.ageTo(u, v) = age(v);
    X.available(u, v) = true;
end

function trem = remainingTimeSecondsForLink(isCross, plane, pos, cfg)
    % Static-ish local estimate used by this toy topology. Cross-plane links
    % have shorter remaining time near the seam, same-plane links are stable.
    if ~isCross
        trem = 999.0;
    else
        trem = 2.0 + mod(plane + pos, 8);
    end
end

function id = satId(plane, pos, cfg)
    id = (plane - 1) * cfg.satsPerPlane + pos;
end

function [plane, pos] = decodeSat(id, cfg)
    plane = floor((id - 1) / cfg.satsPerPlane) + 1;
    pos = mod(id - 1, cfg.satsPerPlane) + 1;
end

function delay = sameOrbitDelayMs(t, plane, pos, cfg)
    delay = 8.0 + 1.5 * sin(2 * pi * (t + pos) / cfg.timeSlots);
end

function delay = crossPlaneDelayMs(t, plane, pos, cfg)
    delay = 12.0 + 2.0 * cos(2 * pi * (t + plane + pos) / cfg.timeSlots);
end

function ok = crossPlaneAvailable(t, plane, pos, cfg)
    phase = mod(t + pos - 2, cfg.timeSlots) / cfg.timeSlots;
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
    phaseStep = mod(t + pos - 2, cfg.timeSlots);
    startStep = floor(0.45 * cfg.timeSlots);
    if phaseStep < startStep
        trem = startStep - phaseStep;
    else
        trem = cfg.timeSlots - phaseStep + startStep;
    end
    trem = max(1.0, double(trem));
end

%% Routing logic

function [W, invalidCount, candidateCount] = buildRoutingWeights(X, queues, cfg, policy, dstNode)
    W = X.delay;
    candidate = isfinite(X.delay);
    invalid = false(size(W));

    [rows, cols] = find(candidate);
    for k = 1:numel(rows)
        u = rows(k);
        v = cols(k);

        if policy.useQueueLoad
            qNorm = queues(v) / cfg.qMax;
            W(u, v) = W(u, v) + cfg.betaQueue * cfg.dRefMs * qNorm ...
                + cfg.betaLoad * cfg.dRefMs * X.rho(u, v);
        end

        if policy.useReliabilityRisk
            risk = 1.0 - X.reliability(u, v);
            W(u, v) = W(u, v) + cfg.betaRisk * cfg.dRefMs * risk;
        end

        if policy.useProgress
            prog = progressValue(u, v, dstNode, cfg);
            W(u, v) = W(u, v) - cfg.betaProgress * cfg.dRefMs * prog;
        end

        if policy.useLifetimeMask
            remainingBandwidth = X.capacity(u, v) - X.usedRate(u, v);
            if X.trem(u, v) < cfg.tSafe || X.reliability(u, v) < cfg.rMin || remainingBandwidth < cfg.bMinMbps
                invalid(u, v) = true;
            else
                lifetimePenalty = cfg.tSafe / max(cfg.tSafe, X.trem(u, v));
                W(u, v) = W(u, v) + cfg.betaLifetime * cfg.dRefMs * lifetimePenalty;
            end
        end
    end

    W(invalid) = Inf;
    invalidCount = sum(invalid(:));
    candidateCount = sum(candidate(:));
end

function [path, status, invalidTotal, candidateTotal] = routeLocalNextHop(t, src, dst, queues, usedRate, age, cfg, policy)
    current = src;
    visited = false(cfg.nSats, 1);
    visited(current) = true;
    path = current;
    invalidTotal = 0;
    candidateTotal = 0;

    for hop = 1:cfg.maxLocalHops
        if current == dst
            status = "ok";
            return;
        end

        X = buildTopologyState(t, queues, usedRate, age, cfg);
        candidates = find(isfinite(X.delay(current, :)));
        candidateTotal = candidateTotal + numel(candidates);
        if isempty(candidates)
            status = "no_neighbor";
            return;
        end

        [nextHop, ok, invalidCount] = selectLocalNextHop(X, queues, current, dst, visited, cfg, policy);
        invalidTotal = invalidTotal + invalidCount;
        if ~ok
            status = "loop_blocked";
            return;
        end

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

function [nextHop, ok, invalidCount] = selectLocalNextHop(X, queues, current, dstNode, visited, cfg, policy)
    candidates = find(isfinite(X.delay(current, :)));
    invalid = false(size(candidates));
    costs = inf(size(candidates));

    for idx = 1:numel(candidates)
        j = candidates(idx);
        if visited(j)
            invalid(idx) = true;
            continue;
        end
        if ~actionFeasible(X, queues, current, j, cfg, policy)
            invalid(idx) = true;
            continue;
        end
        costs(idx) = localNextHopCost(X, queues, current, j, dstNode, cfg, policy);
    end

    invalidCount = sum(invalid);
    validIdx = find(~invalid);
    if isempty(validIdx)
        nextHop = current;
        ok = false;
        return;
    end

    [~, bestRelIdx] = min(costs(validIdx));
    nextHop = candidates(validIdx(bestRelIdx));
    ok = true;
end

function feasible = actionFeasible(X, queues, u, v, cfg, policy)
    feasible = X.available(u, v);
    if ~feasible
        return;
    end
    if policy.useLifetimeMask
        remainingBandwidth = X.capacity(u, v) - X.usedRate(u, v);
        if X.trem(u, v) < cfg.tSafe || X.reliability(u, v) < cfg.rMin || remainingBandwidth < cfg.bMinMbps
            feasible = false;
            return;
        end
    end
    if policy.useQueueLoad && queues(v) >= cfg.qMax
        feasible = false;
        return;
    end
end

function c = localNextHopCost(X, queues, u, v, dstNode, cfg, policy)
    c = X.delay(u, v);
    if policy.useQueueLoad
        c = c + cfg.betaQueue * cfg.dRefMs * (queues(v) / cfg.qMax) ...
            + cfg.betaLoad * cfg.dRefMs * X.rho(u, v);
    end
    if policy.useReliabilityRisk
        c = c + cfg.betaRisk * cfg.dRefMs * (1.0 - X.reliability(u, v));
    end
    if policy.useLifetimeMask
        lifetimePenalty = cfg.tSafe / max(cfg.tSafe, X.trem(u, v));
        c = c + cfg.betaLifetime * cfg.dRefMs * lifetimePenalty;
    end
    if policy.useProgress
        c = c - cfg.betaProgress * cfg.dRefMs * progressValue(u, v, dstNode, cfg);
    end
end

function [path, dist] = dijkstraMatrix(W, src, dst)
    n = size(W, 1);
    visited = false(n, 1);
    distVec = inf(n, 1);
    prev = zeros(n, 1);
    distVec(src) = 0;

    for iter = 1:n
        masked = distVec;
        masked(visited) = inf;
        [best, u] = min(masked);
        if isinf(best)
            break;
        end
        if u == dst
            break;
        end
        visited(u) = true;
        neighbors = find(isfinite(W(u, :)));
        for v = neighbors
            alt = distVec(u) + W(u, v);
            if alt < distVec(v)
                distVec(v) = alt;
                prev(v) = u;
            end
        end
    end

    dist = distVec(dst);
    if isinf(dist)
        path = [];
        return;
    end

    path = dst;
    u = dst;
    while u ~= src
        u = prev(u);
        if u == 0
            path = [];
            dist = inf;
            return;
        end
        path = [u, path]; %#ok<AGROW>
    end
end

function packets = generatePackets(t, cfg, hotspotDst)
    packets = [];
    for k = 1:6
        src = randi(cfg.nSats);
        dst = randi(cfg.nSats);
        if src ~= dst
            packets(end + 1, :) = [src, dst, t, cfg.packetDemandMbps]; %#ok<AGROW>
        end
    end
    for k = 1:12
        src = randi(cfg.nSats);
        if src ~= hotspotDst
            packets(end + 1, :) = [src, hotspotDst, t, cfg.packetDemandMbps]; %#ok<AGROW>
        end
    end
end

function [ok, queues, usedRate] = applyPathLoad(X, path, queues, usedRate, demand, cfg)
    ok = true;
    for k = 2:(numel(path) - 1)
        v = path(k);
        queues(v) = queues(v) + 1;
        if queues(v) > cfg.qMax
            ok = false;
            return;
        end
    end
    for k = 1:(numel(path) - 1)
        u = path(k);
        v = path(k + 1);
        if ~X.available(u, v)
            ok = false;
            return;
        end
        usedRate(u, v) = usedRate(u, v) + demand;
        usedRate(v, u) = usedRate(v, u) + demand;
    end
end

%% Observation, reward, and metrics

function z = localObservationZ(X, queues, sat, dstNode, age, cfg)
    neighbors = find(isfinite(X.delay(sat, :)));
    z = zeros(numel(neighbors), 8);
    for k = 1:numel(neighbors)
        j = neighbors(k);
        z(k, :) = [ ...
            j, ...
            queues(j) / cfg.qMax, ...
            X.delay(sat, j) / cfg.dRefMs, ...
            X.rho(sat, j), ...
            X.reliability(sat, j), ...
            min(1.0, X.trem(sat, j) / cfg.tSafe), ...
            age(j) / max(1, cfg.helloPeriodSlots), ...
            progressValue(sat, j, dstNode, cfg) ...
        ];
    end
end

function r = localReward(X, queues, path, cfg)
    if numel(path) < 2
        r = -10;
        return;
    end
    u = path(1);
    v = path(2);
    delayNorm = X.delay(u, v) / cfg.dRefMs;
    queueNorm = queues(v) / cfg.qMax;
    risk = 1.0 - X.reliability(u, v);
    lifetimePenalty = cfg.tSafe / max(cfg.tSafe, X.trem(u, v));
    progress = progressValue(u, v, path(end), cfg);
    r = -delayNorm - queueNorm - risk - lifetimePenalty + 0.5 * progress;
end

function prog = progressValue(u, v, dstNode, cfg)
    [pu, su] = decodeSat(u, cfg);
    [pv, sv] = decodeSat(v, cfg);
    [pd, sd] = decodeSat(dstNode, cfg);
    distU = torusDistance(pu, su, pd, sd, cfg);
    distV = torusDistance(pv, sv, pd, sd, cfg);
    prog = (distU - distV) / max(1.0, cfg.nPlanes + cfg.satsPerPlane);
end

function d = torusDistance(p1, s1, p2, s2, cfg)
    dp = abs(p1 - p2);
    dp = min(dp, cfg.nPlanes - dp);
    ds = abs(s1 - s2);
    ds = min(ds, cfg.satsPerPlane - ds);
    d = dp + ds;
end

function risky = hasRiskyFirstHop(X, path, cfg)
    risky = false;
    if numel(path) < 2
        return;
    end
    u = path(1);
    v = path(2);
    if X.trem(u, v) < cfg.tSafe || X.reliability(u, v) < cfg.rMin
        risky = true;
    end
end

function d = estimateE2EDelayMs(X, queues, path, cfg)
    propagation = 0;
    queueDelay = 0;
    loadDelay = 0;
    for k = 1:(numel(path) - 1)
        u = path(k);
        v = path(k + 1);
        propagation = propagation + X.delay(u, v);
        loadDelay = loadDelay + X.rho(u, v) * cfg.dRefMs;
        queueDelay = queueDelay + queues(v) / cfg.servicePacketsPerSlot;
    end
    d = propagation + queueDelay + loadDelay;
end

function bytes = estimateHelloBytes(X, cfg, policy)
    nEdges = sum(isfinite(X.delay(:)));
    fields = cfg.bytesNodeId + cfg.bytesTimestamp + cfg.bytesQueue;
    if policy.useQueueLoad
        fields = fields + cfg.bytesLoad;
    end
    if policy.useReliabilityRisk
        fields = fields + cfg.bytesReliability;
    end
    if policy.useLifetimeMask
        fields = fields + cfg.bytesTRem;
    end
    bytes = nEdges * fields;
end

function flops = estimateDecisionFLOPs(k, cfg)
    macsPerNeighbor = cfg.zDim * cfg.hidden1 + cfg.hidden1 * cfg.hidden2 + cfg.hidden2;
    flops = 2 * k * macsPerNeighbor;
end

function j = jainLoadIndex(rho)
    vals = rho(isfinite(rho) & rho > 0);
    if isempty(vals) || sum(vals.^2) == 0
        j = 1.0;
    else
        j = (sum(vals)^2) / (numel(vals) * sum(vals.^2));
    end
end

%% Small helpers

function id = policyIndex(name)
    switch string(name)
        case "P0_delay_only"
            id = 0;
        case "P1_queue_load"
            id = 1;
        case "P2_queue_load_lifetime_mask"
            id = 2;
        case "P3_local_full"
            id = 3;
        case "A1_local_no_queue_load"
            id = 4;
        case "A2_local_no_lifetime_mask"
            id = 5;
        case "A3_local_no_reliability_risk"
            id = 6;
        case "A4_local_no_progress"
            id = 7;
        otherwise
            id = 99;
    end
end

function v = meanOrInf(x)
    if isempty(x)
        v = Inf;
    else
        v = mean(x);
    end
end

function y = percentileOrInf(x, p)
    if isempty(x)
        y = Inf;
        return;
    end
    x = sort(x(:));
    idx = 1 + (numel(x) - 1) * p / 100;
    lo = floor(idx);
    hi = ceil(idx);
    if lo == hi
        y = x(lo);
    else
        y = x(lo) + (idx - lo) * (x(hi) - x(lo));
    end
end

function plotMetrics(metricsTable, outDir)
    fig = figure("Color", "w", "Name", "Preliminary LEO routing comparison");
    tiledlayout(2, 3, "Padding", "compact", "TileSpacing", "compact");
    labels = categorical(metricsTable.policy);
    labels = reordercats(labels, metricsTable.policy);

    nexttile;
    bar(labels, metricsTable.avgDelayMs);
    title("Average delay");
    ylabel("ms");
    grid on;

    nexttile;
    bar(labels, metricsTable.p95DelayMs);
    title("P95 delay");
    ylabel("ms");
    grid on;

    nexttile;
    bar(labels, metricsTable.maxQueue);
    title("Max queue");
    ylabel("packets");
    grid on;

    nexttile;
    bar(labels, metricsTable.avgJainLoad);
    title("Jain load index");
    ylim([0 1]);
    grid on;

    nexttile;
    bar(labels, metricsTable.controlBytes);
    title("Control overhead");
    ylabel("bytes");
    grid on;

    nexttile;
    bar(labels, metricsTable.estimatedFLOPs);
    title("Estimated FLOPs");
    ylabel("FLOPs");
    grid on;

    exportgraphics(fig, fullfile(outDir, "policy_comparison.png"), "Resolution", 200);
end
