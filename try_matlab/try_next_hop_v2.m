%% try_next_hop_v2
% 这一版按老师说的往后走一点：
% 1) Dijkstra 只当全局 baseline；
% 2) 正式试逐跳本地下一跳决策；
% 3) 每个包带 visited 集合，先把环路问题压住；
% 4) 把后面接 MAPPO 要用的 obs/action/reward 轨迹先导出来。
%
% 这里还没有自己写 MAPPO。后面如果接学习算法，优先接官方/成熟库，
% 例如 marlbenchmark/on-policy、cleanmarl 或 BenchMARL。

clear; clc; close all;

cfg = setupV2();
outDir = fullfile(pwd, "tmp_out_v2");
if ~exist(outDir, "dir")
    mkdir(outDir);
end

runs = [
    pol("global_dijkstra",       "global", false, false, false, false)
    pol("local_delay_noloop",    "local",  false, false, false, true)
    pol("local_q_life_noloop",   "local",  true,  true,  true,  true)
    pol("local_q_life_export",   "local",  true,  true,  true,  true)
];

allStat = repmat(blankStat(), numel(runs), 1);
allSlot = table();
allDecision = table();

for ii = 1:numel(runs)
    fprintf("running %s ...\n", runs(ii).name);
    [allStat(ii), slotTab, decTab] = oneRun(cfg, runs(ii));
    allSlot = [allSlot; slotTab]; %#ok<AGROW>
    allDecision = [allDecision; decTab]; %#ok<AGROW>
end

statTab = struct2table(allStat);
writetable(statTab, fullfile(outDir, "stat_v2.csv"));
writetable(allSlot, fullfile(outDir, "slot_log_v2.csv"));
writetable(allDecision, fullfile(outDir, "mappo_rollout_like_v2.csv"));
drawPic(statTab, outDir);

disp(statTab);
fprintf("\nfiles are in: %s\n", outDir);

%% config

function cfg = setupV2()
    cfg.nPlane = 4;
    cfg.perPlane = 6;
    cfg.n = cfg.nPlane * cfg.perPlane;
    cfg.T = 35;
    cfg.maxHop = 24;

    cfg.qMax = 45;
    cfg.service = 3;
    cfg.cap = 100;
    cfg.pktRate = 1.0;
    cfg.pktBytes = 1200;
    cfg.dRef = 20;

    cfg.wDelay = 1.00;
    cfg.wQ = 1.25;
    cfg.wLoad = 0.85;
    cfg.wRisk = 1.20;
    cfg.wLife = 0.55;
    cfg.wProg = 3.00;

    cfg.tSafe = 3.0;
    cfg.rMin = 0.86;
    cfg.bMin = 5.0;
    cfg.decay = 0.55;
    cfg.seed = 12;

    % 先把通信开销预算记下来，不在这一版里强行调参。
    cfg.ctrlBudget = 0.05;  % control bytes / data bytes
    cfg.helloT = 1;
    cfg.byteId = 2;
    cfg.byteTs = 4;
    cfg.byteQ = 2;
    cfg.byteLoad = 2;
    cfg.byteRel = 2;
    cfg.byteLife = 2;

    % 后面换成小 Actor 时，用这个粗算一次本地推理量。
    cfg.zDim = 7;
    cfg.h1 = 32;
    cfg.h2 = 16;
end

function p = pol(name, mode, qload, maskLife, risk, prog)
    p.name = string(name);
    p.mode = string(mode);
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
    s.noCandidate = 0;
    s.maxHopDrop = 0;
    s.badFirstHop = 0;
    s.ctrlBytes = 0;
    s.dataBytes = 0;
    s.ctrlRatio = 0;
    s.overBudget = 0;
    s.flops = 0;
    s.maskRatio = 0;
    s.rLocal = 0;
end

%% one experiment

function [s, slotTab, decTab] = oneRun(cfg, p)
    rng(cfg.seed);
    q = zeros(cfg.n, 1);
    used = zeros(cfg.n, cfg.n);
    age = zeros(cfg.n, 1);
    hotDst = sid(ceil(cfg.nPlane / 2), ceil(cfg.perPlane / 2), cfg);

    ok = 0; drop = 0;
    noCandidate = 0; maxHopDrop = 0; badHop = 0;
    ctrl = 0; flops = 0; masked = 0; cand = 0;
    delays = []; hops = []; rewards = []; maxQs = []; jains = [];
    slotRows = []; decRows = [];

    for t = 1:cfg.T
        pkts = makePkts(t, cfg, hotDst);
        okSlot = 0; dropSlot = 0; maskSlot = 0; candSlot = 0; rSlot = [];

        for kk = 1:size(pkts, 1)
            src = pkts(kk, 1);
            dst = pkts(kk, 2);
            dem = pkts(kk, 4);

            X = makeTopo(t, q, used, age, cfg);
            if p.mode == "global"
                [path, m1, c1, dRows, failCode, f1] = routeGlobal(X, q, src, dst, age, cfg, p, t);
            else
                [path, m1, c1, dRows, failCode, f1] = routeLocal(X, q, src, dst, age, cfg, p, t);
            end

            masked = masked + m1;
            cand = cand + c1;
            maskSlot = maskSlot + m1;
            candSlot = candSlot + c1;
            flops = flops + f1;
            decRows = [decRows; dRows]; %#ok<AGROW>

            if failCode == 2
                noCandidate = noCandidate + 1;
            elseif failCode == 3
                maxHopDrop = maxHopDrop + 1;
            end

            if isempty(path) || path(end) ~= dst
                drop = drop + 1;
                dropSlot = dropSlot + 1;
                continue;
            end

            if riskyFirstHop(X, path, cfg)
                badHop = badHop + 1;
            end

            rr = localR(X, q, path, cfg);
            rewards(end + 1, 1) = rr; %#ok<AGROW>
            rSlot(end + 1, 1) = rr; %#ok<AGROW>

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
        slotRows(end + 1, :) = [t, pId(p.name), okSlot, dropSlot, max(q), mean(q), ...
            jainIdx(Xlog.rho), maskSlot, candSlot, mean2(rSlot)]; %#ok<AGROW>
    end

    total = ok + drop;
    dataBytes = ok * cfg.pktBytes;

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
    s.noCandidate = noCandidate;
    s.maxHopDrop = maxHopDrop;
    s.badFirstHop = badHop;
    s.ctrlBytes = ctrl;
    s.dataBytes = dataBytes;
    s.ctrlRatio = ctrl / max(1, dataBytes);
    s.overBudget = double(s.ctrlRatio > cfg.ctrlBudget);
    s.flops = flops;
    s.maskRatio = masked / max(1, cand);
    s.rLocal = mean2(rewards);

    slotTab = array2table(slotRows, 'VariableNames', ...
        {'t','pid','ok','drop','maxQ','avgQ','jain','masked','cand','rLocal'});
    slotTab.name = repmat(p.name, height(slotTab), 1);
    slotTab = movevars(slotTab, 'name', 'After', 'pid');

    decTab = array2table(decRows, 'VariableNames', ...
        {'t','pid','src','dst','cur','act','hop','nCand','masked','qAct', ...
        'delayAct','rhoAct','relAct','tremAct','reward','done','failCode'});
    decTab.name = repmat(p.name, height(decTab), 1);
    decTab = movevars(decTab, 'name', 'After', 'pid');
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

function [path, mCnt, cCnt, rows, failCode, flops] = routeGlobal(X, q, src, dst, age, cfg, p, t)
    [W, mCnt, cCnt] = makeWeight(X, q, cfg, p, dst);
    path = myDijkstra(W, src, dst);
    flops = cfg.n * cfg.n;

    if isempty(path)
        rows = decRow(t, p, src, dst, src, 0, 0, cCnt, mCnt, NaN, NaN, NaN, NaN, NaN, -10, 0, 1);
        failCode = 1;
        return;
    end

    act = path(2);
    [qAct, dAct, rhoAct, relAct, tremAct] = edgeInfo(X, q, src, act);
    r = hopReward(X, q, src, act, dst, cfg, false);
    rows = decRow(t, p, src, dst, src, act, 1, cCnt, mCnt, qAct, dAct, rhoAct, relAct, tremAct, r, act == dst, 0);
    failCode = 0;
    %#ok<NASGU>
    age = age;
end

function [path, mCnt, cCnt, rows, failCode, flops] = routeLocal(X, q, src, dst, age, cfg, p, t)
    visited = false(cfg.n, 1);
    visited(src) = true;
    path = src;
    rows = [];
    mCnt = 0;
    cCnt = 0;
    flops = 0;
    failCode = 0;

    cur = src;
    for h = 1:cfg.maxHop
        if cur == dst
            return;
        end

        [act, info] = chooseHop(X, q, cur, dst, visited, age, cfg, p);
        mCnt = mCnt + info.masked;
        cCnt = cCnt + info.total;
        flops = flops + roughFlops(max(1, info.total), cfg);

        if act == 0
            rows = [rows; decRow(t, p, src, dst, cur, 0, h, info.nCand, info.masked, ...
                NaN, NaN, NaN, NaN, NaN, -10, 0, 2)]; %#ok<AGROW>
            failCode = 2;
            path = [];
            return;
        end

        r = hopReward(X, q, cur, act, dst, cfg, visited(act));
        [qAct, dAct, rhoAct, relAct, tremAct] = edgeInfo(X, q, cur, act);
        rows = [rows; decRow(t, p, src, dst, cur, act, h, info.nCand, info.masked, ...
            qAct, dAct, rhoAct, relAct, tremAct, r, act == dst, 0)]; %#ok<AGROW>

        if visited(act) && act ~= dst
            failCode = 5;
            path = [];
            return;
        end

        path = [path, act]; %#ok<AGROW>
        if act == dst
            return;
        end

        visited(act) = true;
        cur = act;
    end

    failCode = 3;
    path = [];
end

function [act, info] = chooseHop(X, q, u, dst, visited, age, cfg, p)
    nb = find(isfinite(X.d(u, :)));
    total = numel(nb);
    bad = false(size(nb));

    for k = 1:numel(nb)
        v = nb(k);
        leftBw = X.cap(u, v) - X.used(u, v);
        if visited(v) && v ~= dst
            bad(k) = true;
        end
        if ~X.ok(u, v)
            bad(k) = true;
        end
        if p.maskLife && (X.trem(u, v) < cfg.tSafe || X.rel(u, v) < cfg.rMin || leftBw < cfg.bMin)
            bad(k) = true;
        end
    end

    cand = nb(~bad);
    info.total = total;
    info.masked = sum(bad);
    info.nCand = numel(cand);

    if isempty(cand)
        act = 0;
        return;
    end

    cost = zeros(numel(cand), 1);
    for k = 1:numel(cand)
        v = cand(k);
        one = cfg.wDelay * X.d(u, v) / cfg.dRef;
        if p.qload
            one = one + cfg.wQ * q(v) / cfg.qMax + cfg.wLoad * X.rho(u, v);
        end
        if p.risk
            one = one + cfg.wRisk * (1 - X.rel(u, v));
        end
        if p.maskLife
            one = one + cfg.wLife * cfg.tSafe / max(cfg.tSafe, X.trem(u, v));
        end
        if p.prog
            one = one - cfg.wProg * prog(u, v, dst, cfg);
        end

        % 邻居状态太旧时轻微加惩罚，算是给后面通信预算实验留入口。
        one = one + 0.05 * age(v) / max(1, cfg.helloT);
        cost(k) = one;
    end

    [~, ix] = min(cost);
    act = cand(ix);
end

function [W, mCnt, cCnt] = makeWeight(X, q, cfg, p, dst)
    W = X.d;
    cand = isfinite(W);
    mask = false(size(W));
    [rr, cc] = find(cand);

    for ii = 1:numel(rr)
        u = rr(ii);
        v = cc(ii);

        if p.qload
            W(u, v) = W(u, v) + cfg.wQ * cfg.dRef * (q(v) / cfg.qMax) ...
                + cfg.wLoad * cfg.dRef * X.rho(u, v);
        end

        if p.risk
            W(u, v) = W(u, v) + cfg.wRisk * cfg.dRef * (1 - X.rel(u, v));
        end

        if p.prog
            W(u, v) = W(u, v) - cfg.wProg * cfg.dRef * prog(u, v, dst, cfg);
        end

        if p.maskLife
            leftBw = X.cap(u, v) - X.used(u, v);
            if X.trem(u, v) < cfg.tSafe || X.rel(u, v) < cfg.rMin || leftBw < cfg.bMin
                mask(u, v) = true;
            else
                W(u, v) = W(u, v) + cfg.wLife * cfg.dRef * cfg.tSafe / max(cfg.tSafe, X.trem(u, v));
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
    qNew = q;
    usedNew = used;

    for k = 2:(numel(path) - 1)
        v = path(k);
        qNew(v) = qNew(v) + 1;
        if qNew(v) > cfg.qMax
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
        usedNew(u, v) = usedNew(u, v) + dem;
        usedNew(v, u) = usedNew(v, u) + dem;
    end

    q = qNew;
    used = usedNew;
end

%% obs/reward/log helpers

function r = hopReward(X, q, u, v, dst, cfg, wasVisited)
    if v == 0
        r = -10;
        return;
    end
    r = - X.d(u, v) / cfg.dRef ...
        - cfg.wQ * q(v) / cfg.qMax ...
        - cfg.wLoad * X.rho(u, v) ...
        - cfg.wRisk * (1 - X.rel(u, v)) ...
        - cfg.wLife * cfg.tSafe / max(cfg.tSafe, X.trem(u, v)) ...
        + cfg.wProg * prog(u, v, dst, cfg);
    if v == dst
        r = r + 3;
    end
    if wasVisited
        r = r - 5;
    end
end

function r = localR(X, q, path, cfg)
    if numel(path) < 2
        r = -10;
        return;
    end
    rs = zeros(numel(path) - 1, 1);
    for k = 1:(numel(path) - 1)
        rs(k) = hopReward(X, q, path(k), path(k + 1), path(end), cfg, false);
    end
    r = mean(rs);
end

function [qAct, dAct, rhoAct, relAct, tremAct] = edgeInfo(X, q, u, v)
    qAct = q(v);
    dAct = X.d(u, v);
    rhoAct = X.rho(u, v);
    relAct = X.rel(u, v);
    tremAct = X.trem(u, v);
end

function row = decRow(t, p, src, dst, cur, act, hop, nCand, masked, qAct, dAct, rhoAct, relAct, tremAct, reward, done, failCode)
    row = [t, pId(p.name), src, dst, cur, act, hop, nCand, masked, ...
        qAct, dAct, rhoAct, relAct, tremAct, reward, double(done), failCode];
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

function b = riskyFirstHop(X, path, cfg)
    b = false;
    if numel(path) < 2, return; end
    u = path(1);
    v = path(2);
    if X.trem(u, v) < cfg.tSafe || X.rel(u, v) < cfg.rMin
        b = true;
    end
end

function d = e2eDelay(X, q, path, cfg)
    d = 0;
    for k = 1:(numel(path) - 1)
        u = path(k);
        v = path(k + 1);
        d = d + X.d(u, v) + q(v) / cfg.service + X.rho(u, v) * cfg.dRef;
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
    if isempty(x) || sum(x .^ 2) == 0
        j = 1;
    else
        j = (sum(x) ^ 2) / (numel(x) * sum(x .^ 2));
    end
end

function id = pId(name)
    switch string(name)
        case "global_dijkstra"
            id = 0;
        case "local_delay_noloop"
            id = 1;
        case "local_q_life_noloop"
            id = 2;
        otherwise
            id = 3;
    end
end

function v = mean2(x)
    if isempty(x)
        v = inf;
    else
        v = mean(x, 'omitnan');
    end
end

function y = pct2(x, p)
    if isempty(x)
        y = inf;
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

function drawPic(T, outDir)
    fig = figure("Color", "w", "Name", "next hop tries", "Position", [100 100 1400 900]);
    tiledlayout(2, 3, "Padding", "loose", "TileSpacing", "compact");
    x = 1:height(T);
    labs = ["gDij","lDelay","lQLife","export"];

    nexttile; bar(x, T.avgDelay); title("avg delay"); ylabel("ms"); grid on; fixX(x, labs);
    nexttile; bar(x, T.p95Delay); title("p95 delay"); ylabel("ms"); grid on; fixX(x, labs);
    nexttile; bar(x, T.dropRate); title("drop rate"); grid on; fixX(x, labs);
    nexttile; bar(x, T.maxQ); title("max queue"); grid on; fixX(x, labs);
    nexttile; bar(x, T.ctrlRatio); yline(0.05, "--"); title("ctrl/data"); grid on; fixX(x, labs);
    nexttile; bar(x, T.flops); title("rough FLOPs"); grid on; fixX(x, labs);

    exportgraphics(fig, fullfile(outDir, "quick_plot_v2.png"), "Resolution", 200);
end

function fixX(x, labs)
    ax = gca;
    ax.XTick = x;
    ax.XTickLabel = labs;
    ax.FontSize = 9;
    xlim([0.5, numel(x) + 0.5]);
end
