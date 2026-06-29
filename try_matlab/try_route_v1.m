%% try_route_v1
% 先把论文里的前期想法跑起来：
% 1) LEO 拓扑会随时间变；
% 2) 只按传播时延走最短路，会把流量挤到一些短路径上；
% 3) 加一点队列/负载信息后，看最大队列和尾时延会不会好一点；
% 4) 再试一下 T_rem 过滤，看能不能避开快断链路。
%
% 这不是最终 MAPPO 版本，只是前面试 baseline 和变量是否有用。

clear; clc; close all;

cfg = setup0();
outDir = fullfile(pwd, "tmp_out");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

runs = [
    pol("dijkstra_delay", false, false, false, false)
    pol("add_q_load", true,  false, false, false)
    pol("add_life_mask", true, true,  false, false)
    pol("try_risk_prog", true, true,  true,  true)
];

allStat = repmat(blankStat(), numel(runs), 1);
allLog = table();

for ii = 1:numel(runs)
    fprintf("running %s ...\n", runs(ii).name);
    [allStat(ii), oneLog] = oneRun(cfg, runs(ii));
    allLog = [allLog; oneLog]; %#ok<AGROW>
end

statTab = struct2table(allStat);
writetable(statTab, fullfile(outDir, "stat.csv"));
writetable(allLog, fullfile(outDir, "slot_log.csv"));
drawPic(statTab, outDir);

disp(statTab);
fprintf("\nfiles are in: %s\n", outDir);

%% config

function cfg = setup0()
    cfg.nPlane = 4;
    cfg.perPlane = 6;
    cfg.n = cfg.nPlane * cfg.perPlane;
    cfg.T = 30;

    cfg.qMax = 45;
    cfg.service = 3;
    cfg.cap = 100;
    cfg.pktRate = 1.0;
    cfg.dRef = 20;

    cfg.bq = 1.20;
    cfg.bl = 0.80;
    cfg.brisk = 10.0;
    cfg.blife = 8.0;
    cfg.bprog = 3.0;

    cfg.tSafe = 3.0;
    cfg.rMin = 0.86;
    cfg.bMin = 5.0;
    cfg.decay = 0.55;
    cfg.seed = 11;

    % hello 包粗略估算用，先别太当真，后面再细化
    cfg.helloT = 1;
    cfg.byteId = 2;
    cfg.byteTs = 4;
    cfg.byteQ = 2;
    cfg.byteLoad = 2;
    cfg.byteRel = 2;
    cfg.byteLife = 2;

    % 粗略估计一下如果以后换成小 Actor，每次打分大概多少 FLOPs
    cfg.zDim = 7;
    cfg.h1 = 32;
    cfg.h2 = 16;
end

function p = pol(name, qload, maskLife, risk, prog)
    p.name = string(name);
    p.qload = qload;
    p.maskLife = maskLife;
    p.risk = risk;
    p.prog = prog;
end

function s = blankStat()
    s.name = "";
    s.ok = 0;
    s.drop = 0;
    s.dropRate = 0;
    s.avgDelay = 0;
    s.p95Delay = 0;
    s.avgHop = 0;
    s.maxQ = 0;
    s.jain = 0;
    s.badFirstHop = 0;
    s.ctrlBytes = 0;
    s.flops = 0;
    s.maskRatio = 0;
    s.rLocal = 0;
end

%% one experiment

function [s, logTab] = oneRun(cfg, p)
    rng(cfg.seed);
    q = zeros(cfg.n, 1);
    used = zeros(cfg.n, cfg.n);
    age = zeros(cfg.n, 1);
    hotDst = sid(ceil(cfg.nPlane / 2), ceil(cfg.perPlane / 2), cfg);

    ok = 0; drop = 0;
    delays = [];
    hops = [];
    rewards = [];
    maxQs = [];
    jains = [];
    badHop = 0;
    ctrl = 0;
    flops = 0;
    masked = 0;
    cand = 0;
    rows = [];

    for t = 1:cfg.T
        pkts = makePkts(t, cfg, hotDst);
        okSlot = 0; dropSlot = 0;
        maskSlot = 0; candSlot = 0;
        rSlot = [];

        for kk = 1:size(pkts, 1)
            src = pkts(kk, 1);
            dst = pkts(kk, 2);
            dem = pkts(kk, 4);

            X = makeTopo(t, q, used, age, cfg);
            [W, m1, c1] = makeWeight(X, q, cfg, p, dst);

            masked = masked + m1;
            cand = cand + c1;
            maskSlot = maskSlot + m1;
            candSlot = candSlot + c1;
            flops = flops + roughFlops(sum(isfinite(X.d(src, :))), cfg);

            % 先把局部观测也算出来，后面接 Actor 会用
            z = obsZ(X, q, src, dst, age, cfg); %#ok<NASGU>

            path = myDijkstra(W, src, dst);
            if isempty(path)
                drop = drop + 1;
                dropSlot = dropSlot + 1;
                continue;
            end

            rr = localR(X, q, path, cfg);
            rewards(end + 1, 1) = rr; %#ok<AGROW>
            rSlot(end + 1, 1) = rr; %#ok<AGROW>

            if riskyHop(X, path, cfg)
                badHop = badHop + 1;
            end

            [pass, q, used] = putLoad(X, path, q, used, dem, cfg);
            if pass
                ok = ok + 1;
                okSlot = okSlot + 1;
                delays(end + 1, 1) = e2eDelay(X, q, path, cfg); %#ok<AGROW>
                hops(end + 1, 1) = numel(path) - 1; %#ok<AGROW>
            else
                drop = drop + 1;
                dropSlot = dropSlot + 1;
            end
        end

        q = max(0, q - cfg.service);
        used = used * cfg.decay;

        age = age + 1;
        if mod(t, cfg.helloT) == 0
            Xh = makeTopo(t, q, used, age, cfg);
            ctrl = ctrl + helloBytes(Xh, cfg, p);
            age(:) = 0;
        end

        Xlog = makeTopo(t, q, used, age, cfg);
        maxQs(end + 1, 1) = max(q); %#ok<AGROW>
        jains(end + 1, 1) = jainIdx(Xlog.rho); %#ok<AGROW>
        if isempty(rSlot)
            rMean = NaN;
        else
            rMean = mean(rSlot);
        end

        rows(end + 1, :) = [t, pId(p.name), okSlot, dropSlot, max(q), mean(q), ...
            jainIdx(Xlog.rho), maskSlot, candSlot, rMean]; %#ok<AGROW>
    end

    total = ok + drop;
    s = blankStat();
    s.name = p.name;
    s.ok = ok;
    s.drop = drop;
    s.dropRate = drop / max(1, total);
    s.avgDelay = mean2(delays);
    s.p95Delay = pct2(delays, 95);
    s.avgHop = mean2(hops);
    s.maxQ = max(maxQs);
    s.jain = mean(jains);
    s.badFirstHop = badHop;
    s.ctrlBytes = ctrl;
    s.flops = flops;
    s.maskRatio = masked / max(1, cand);
    s.rLocal = mean2(rewards);

    logTab = array2table(rows, 'VariableNames', ...
        {'t','pid','ok','drop','maxQ','avgQ','jain','masked','cand','rLocal'});
    logTab.name = repmat(p.name, height(logTab), 1);
    logTab = movevars(logTab, 'name', 'After', 'pid');
end

%% topology

function X = makeTopo(t, q, used, age, cfg)
    n = cfg.n;
    X.d = inf(n, n);
    X.cap = zeros(n, n);
    X.used = zeros(n, n);
    X.rho = zeros(n, n);
    X.rel = zeros(n, n);
    X.pout = ones(n, n);
    X.trem = zeros(n, n);
    X.ageTo = zeros(n, n);
    X.ok = false(n, n);
    X.q = q;

    for p = 1:cfg.nPlane
        for s = 1:cfg.perPlane
            u = sid(p, s, cfg);
            v1 = sid(p, mod(s, cfg.perPlane) + 1, cfg);
            X = addLink(X, u, v1, dSame(t, p, s, cfg), false, p, s, used, age, cfg);

            p2 = mod(p, cfg.nPlane) + 1;
            if crossOK(t, p, s, cfg)
                v2 = sid(p2, s, cfg);
                X = addLink(X, u, v2, dCross(t, p, s, cfg), true, p, s, used, age, cfg);
            end
        end
    end
end

function X = addLink(X, u, v, d, isCross, plane, pos, used, age, cfg)
    X = addArc(X, u, v, d, isCross, plane, pos, used, age, cfg);
    X = addArc(X, v, u, d, isCross, plane, pos, used, age, cfg);
end

function X = addArc(X, u, v, d, isCross, plane, pos, used, age, cfg)
    r = used(u, v);
    rho = min(1.0, r / cfg.cap);
    rel = max(0.80, 0.995 - 0.12 * rho);
    X.d(u, v) = d;
    X.cap(u, v) = cfg.cap;
    X.used(u, v) = r;
    X.rho(u, v) = rho;
    X.rel(u, v) = rel;
    X.pout(u, v) = 1 - rel;
    X.trem(u, v) = lifeLeft(isCross, plane, pos);
    X.ageTo(u, v) = age(v);
    X.ok(u, v) = true;
end

function id = sid(p, s, cfg)
    id = (p - 1) * cfg.perPlane + s;
end

function [p, s] = sidBack(id, cfg)
    p = floor((id - 1) / cfg.perPlane) + 1;
    s = mod(id - 1, cfg.perPlane) + 1;
end

function d = dSame(t, ~, s, cfg)
    d = 8.0 + 1.5 * sin(2 * pi * (t + s) / cfg.T);
end

function d = dCross(t, p, s, cfg)
    d = 12.0 + 2.0 * cos(2 * pi * (t + p + s) / cfg.T);
end

function ok = crossOK(t, p, s, cfg)
    ph = mod(t + s - 2, cfg.T) / cfg.T;
    ok = true;
    if p == cfg.nPlane && ph >= 0.45 && ph <= 0.65
        ok = false;
    end
end

function left = lifeLeft(isCross, p, s)
    if ~isCross
        left = 999;
    else
        left = 2 + mod(p + s, 8);
    end
end

%% routing

function [W, mCnt, cCnt] = makeWeight(X, q, cfg, p, dst)
    W = X.d;
    cand = isfinite(W);
    mask = false(size(W));
    [rr, cc] = find(cand);

    for ii = 1:numel(rr)
        u = rr(ii);
        v = cc(ii);

        if p.qload
            W(u, v) = W(u, v) + cfg.bq * cfg.dRef * (q(v) / cfg.qMax) ...
                + cfg.bl * cfg.dRef * X.rho(u, v);
        end

        if p.risk
            W(u, v) = W(u, v) + cfg.brisk * cfg.dRef * (1 - X.rel(u, v));
        end

        if p.prog
            W(u, v) = W(u, v) - cfg.bprog * cfg.dRef * prog(u, v, dst, cfg);
        end

        if p.maskLife
            leftBw = X.cap(u, v) - X.used(u, v);
            if X.trem(u, v) < cfg.tSafe || X.rel(u, v) < cfg.rMin || leftBw < cfg.bMin
                mask(u, v) = true;
            else
                W(u, v) = W(u, v) + cfg.blife * cfg.dRef * cfg.tSafe / max(cfg.tSafe, X.trem(u, v));
            end
        end
    end

    W(mask) = inf;
    mCnt = sum(mask(:));
    cCnt = sum(cand(:));
end

function path = myDijkstra(W, src, dst)
    n = size(W, 1);
    seen = false(n, 1);
    dist = inf(n, 1);
    pre = zeros(n, 1);
    dist(src) = 0;

    for k = 1:n
        tmp = dist;
        tmp(seen) = inf;
        [best, u] = min(tmp);
        if isinf(best), break; end
        if u == dst, break; end
        seen(u) = true;
        nb = find(isfinite(W(u, :)));
        for v = nb
            alt = dist(u) + W(u, v);
            if alt < dist(v)
                dist(v) = alt;
                pre(v) = u;
            end
        end
    end

    if isinf(dist(dst))
        path = [];
        return;
    end

    path = dst;
    u = dst;
    while u ~= src
        u = pre(u);
        if u == 0
            path = [];
            return;
        end
        path = [u, path]; %#ok<AGROW>
    end
end

function pkts = makePkts(t, cfg, hotDst)
    pkts = [];
    for k = 1:6
        src = randi(cfg.n);
        dst = randi(cfg.n);
        if src ~= dst
            pkts(end + 1, :) = [src, dst, t, cfg.pktRate]; %#ok<AGROW>
        end
    end
    for k = 1:12
        src = randi(cfg.n);
        if src ~= hotDst
            pkts(end + 1, :) = [src, hotDst, t, cfg.pktRate]; %#ok<AGROW>
        end
    end
end

function [pass, q, used] = putLoad(X, path, q, used, dem, cfg)
    pass = true;
    for k = 2:(numel(path) - 1)
        v = path(k);
        q(v) = q(v) + 1;
        if q(v) > cfg.qMax
            pass = false;
            return;
        end
    end
    for k = 1:(numel(path) - 1)
        u = path(k);
        v = path(k + 1);
        if ~X.ok(u, v)
            pass = false;
            return;
        end
        used(u, v) = used(u, v) + dem;
        used(v, u) = used(v, u) + dem;
    end
end

%% obs/reward/log helpers

function z = obsZ(X, q, i, dst, age, cfg)
    nb = find(isfinite(X.d(i, :)));
    z = zeros(numel(nb), 8);
    for k = 1:numel(nb)
        j = nb(k);
        z(k, :) = [j, q(j)/cfg.qMax, X.d(i,j)/cfg.dRef, X.rho(i,j), ...
            X.rel(i,j), min(1, X.trem(i,j)/cfg.tSafe), ...
            age(j)/max(1,cfg.helloT), prog(i,j,dst,cfg)];
    end
end

function r = localR(X, q, path, cfg)
    if numel(path) < 2
        r = -10;
        return;
    end
    u = path(1);
    v = path(2);
    r = -X.d(u,v)/cfg.dRef ...
        - q(v)/cfg.qMax ...
        - (1 - X.rel(u,v)) ...
        - cfg.tSafe/max(cfg.tSafe, X.trem(u,v)) ...
        + 0.5 * prog(u,v,path(end),cfg);
end

function val = prog(u, v, dst, cfg)
    [pu, su] = sidBack(u, cfg);
    [pv, sv] = sidBack(v, cfg);
    [pd, sd] = sidBack(dst, cfg);
    du = torusDist(pu, su, pd, sd, cfg);
    dv = torusDist(pv, sv, pd, sd, cfg);
    val = (du - dv) / max(1, cfg.nPlane + cfg.perPlane);
end

function d = torusDist(p1, s1, p2, s2, cfg)
    dp = abs(p1 - p2);
    dp = min(dp, cfg.nPlane - dp);
    ds = abs(s1 - s2);
    ds = min(ds, cfg.perPlane - ds);
    d = dp + ds;
end

function b = riskyHop(X, path, cfg)
    b = false;
    if numel(path) < 2, return; end
    u = path(1);
    v = path(2);
    if X.trem(u,v) < cfg.tSafe || X.rel(u,v) < cfg.rMin
        b = true;
    end
end

function d = e2eDelay(X, q, path, cfg)
    d = 0;
    for k = 1:(numel(path) - 1)
        u = path(k);
        v = path(k + 1);
        d = d + X.d(u,v) + q(v)/cfg.service + X.rho(u,v)*cfg.dRef;
    end
end

function bytes = helloBytes(X, cfg, p)
    nEdge = sum(isfinite(X.d(:)));
    one = cfg.byteId + cfg.byteTs + cfg.byteQ;
    if p.qload, one = one + cfg.byteLoad; end
    if p.risk, one = one + cfg.byteRel; end
    if p.maskLife, one = one + cfg.byteLife; end
    bytes = nEdge * one;
end

function f = roughFlops(k, cfg)
    mac = cfg.zDim * cfg.h1 + cfg.h1 * cfg.h2 + cfg.h2;
    f = 2 * k * mac;
end

function j = jainIdx(rho)
    x = rho(isfinite(rho) & rho > 0);
    if isempty(x) || sum(x.^2) == 0
        j = 1;
    else
        j = (sum(x)^2) / (numel(x) * sum(x.^2));
    end
end

function id = pId(name)
    switch string(name)
        case "dijkstra_delay"
            id = 0;
        case "add_q_load"
            id = 1;
        case "add_life_mask"
            id = 2;
        otherwise
            id = 3;
    end
end

function v = mean2(x)
    if isempty(x), v = inf; else, v = mean(x); end
end

function y = pct2(x, p)
    if isempty(x)
        y = inf;
        return;
    end
    x = sort(x(:));
    idx = 1 + (numel(x) - 1) * p / 100;
    lo = floor(idx); hi = ceil(idx);
    if lo == hi
        y = x(lo);
    else
        y = x(lo) + (idx - lo) * (x(hi) - x(lo));
    end
end

function drawPic(T, outDir)
    fig = figure("Color", "w", "Name", "routing tries");
    tiledlayout(2, 3, "Padding", "compact", "TileSpacing", "compact");
    labs = categorical(T.name);
    labs = reordercats(labs, T.name);

    nexttile; bar(labs, T.avgDelay); title("avg delay"); ylabel("ms"); grid on;
    nexttile; bar(labs, T.p95Delay); title("p95 delay"); ylabel("ms"); grid on;
    nexttile; bar(labs, T.maxQ); title("max queue"); grid on;
    nexttile; bar(labs, T.jain); title("Jain load"); ylim([0 1]); grid on;
    nexttile; bar(labs, T.ctrlBytes); title("hello bytes"); grid on;
    nexttile; bar(labs, T.flops); title("rough FLOPs"); grid on;

    exportgraphics(fig, fullfile(outDir, "quick_plot.png"), "Resolution", 200);
end
