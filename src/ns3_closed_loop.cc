/*
 * CLOSED-LOOP ns-3 validation: the routing policy decides inside the ns-3
 * event loop (companion server: src/ns3_closed_loop_server.py).
 *
 * Unlike leo-validation.cc (which replays precomputed source-routed paths),
 * here NO path is known in advance. ns-3 runs the real data plane (FIFO
 * drop-tail device queues, byte-serialised P2P links, propagation delay) and
 * at every slot boundary it ships each node's RAW state — local queue length,
 * head-of-line packet fields (dst/class/hops/visited/prev/created), per-node
 * arrival counts, and per-directed-link transmissions — to the policy server
 * over TCP. The server reconstructs the training env's candidate features
 * from that state and replies with next-hop decisions, which ns-3 executes.
 * Queues/bandwidth/contention the policy sees are therefore ns-3's OWN, i.e.
 * a true closed loop; the feature formulas are the env's (parity by
 * construction).
 *
 * Env service semantics mirrored exactly (leo_multiagent_env.step):
 *   - one decision per node per slot, for its HOL packet only (FIFO);
 *   - hold (action 0) is only offered when no feasible route exists;
 *   - link capacity: at most --link-capacity forwards per directed link per
 *     slot (by packet id); excess stay queued (blocked, not dropped);
 *   - arrival: hop_count++; delivery at dst takes precedence; else TTL
 *     (hops >= max-hops) drop; else dest-queue-full drop; else enqueue;
 *   - deadline: local_slot - created + 1 >= deadline[class] -> dropped,
 *     checked at each boundary after arrivals; delivery only counts within
 *     deadline;
 *   - source admission: dropped (source_queue_overflow) if src queue full.
 *
 * Build: copy to ~/ns-3.48/scratch/leo-closed-loop.cc && ./ns3 build
 * Run:   ./ns3 run "scratch/leo-closed-loop --input=...packets.csv
 *         --output=...csv --bridge-host=<server-ip> --bridge-port=7341
 *         --policy-name=mappo"
 * The input is the SAME packets CSV as replay mode (path column ignored), so
 * traffic is bit-identical across replay and closed loop.
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <deque>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LeoClosedLoop");

namespace {
constexpr uint32_t N_PLANES = 4;
constexpr uint32_t N_SPP = 6;
constexpr uint32_t N_NODES = N_PLANES * N_SPP;   // 24
constexpr uint16_t LEO_PROTO = 0x9999;
constexpr double INTRA_DELAY_MS = 8.0;
constexpr double CROSS_DELAY_MS = 12.0;
} // namespace

// ---- on-wire header (no path: next hop decided per slot) ----
class CLHeader : public Header
{
public:
  uint32_t pktId = 0;
  uint32_t finalDst = 0;
  uint32_t srcNode = 0;
  uint64_t sendTimeNs = 0;
  uint8_t cls = 0;
  uint8_t episode = 0;
  uint8_t createdSlot = 0;   // episode-local creation slot
  uint8_t hops = 0;          // hops completed at arrival
  uint8_t prevNode = 0;      // 0 = none (fresh at source)
  uint32_t visitedMask = 0;  // bit(node-1) for every node touched incl. src

  CLHeader() = default;
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("CLHeader").SetParent<Header>().SetGroupName("Leo")
                         .AddConstructor<CLHeader>();
    return tid;
  }
  TypeId GetInstanceTypeId() const override { return GetTypeId(); }
  void Print(std::ostream& os) const override { os << "cl[" << pktId << "]"; }
  uint32_t GetSerializedSize() const override { return 4+4+4+8+1+1+1+1+1+4; }
  void Serialize(Buffer::Iterator s) const override {
    s.WriteU32(pktId); s.WriteU32(finalDst); s.WriteU32(srcNode);
    s.WriteU64(sendTimeNs); s.WriteU8(cls); s.WriteU8(episode);
    s.WriteU8(createdSlot); s.WriteU8(hops); s.WriteU8(prevNode);
    s.WriteU32(visitedMask);
  }
  uint32_t Deserialize(Buffer::Iterator s) override {
    pktId = s.ReadU32(); finalDst = s.ReadU32(); srcNode = s.ReadU32();
    sendTimeNs = s.ReadU64(); cls = s.ReadU8(); episode = s.ReadU8();
    createdSlot = s.ReadU8(); hops = s.ReadU8(); prevNode = s.ReadU8();
    visitedMask = s.ReadU32();
    return GetSerializedSize();
  }
};

struct QPkt {                       // in-queue copy of the packet state
  uint32_t pktId, finalDst, srcNode;
  uint8_t cls, episode, createdSlot, hops, prevNode;
  uint32_t visitedMask;
  uint64_t sendNs;
};

struct Injection {
  uint32_t gslot, gid, src, dst, cls, episode, createdSlot;
};

struct Sim {
  std::map<uint32_t, Ptr<Node>> nodes;
  std::map<std::pair<uint32_t,uint32_t>, Ptr<PointToPointNetDevice>> devOf;
  std::string output, policyName;
  uint64_t sent=0, delivered=0, deviceQueueDrops=0;
  uint64_t deadlineDrops=0, ttlDrops=0, queueDrops=0, sourceDrops=0, blocked=0;
  uint64_t holds=0, decisions=0;
  std::map<uint32_t,std::string> dropReason;   // global pktId -> reason
  std::map<uint32_t,double> deliveredDelayMs;
  std::set<uint32_t> allSent;
  std::vector<double> delaysMs;
  std::map<std::pair<uint32_t,uint32_t>,uint64_t> linkTx;      // directed totals
  std::map<std::pair<uint32_t,uint32_t>,uint64_t> snapshotTx;  // last boundary
  std::map<uint32_t,uint32_t> incoming;       // arrivals since last boundary
  std::map<uint32_t,std::deque<QPkt>> nodeQ;  // per-node FIFO (env semantics)
  std::vector<Injection> injects;
  double slotSec = 1.0;
  uint32_t episodeSlots = 30, linkCapacity = 3, nodeQCap = 45, maxHops = 12;
  uint32_t deadlineSlots[3] = {30,12,20};
  int sock = -1;
} g;

// structural neighbor list per node in FIXED order (sorted ids): action slot i
// (1-based) == i-th entry. Both the report and the execution use this mapping.
static std::map<uint32_t, std::vector<uint32_t>> nbrsOf;

static std::pair<uint32_t,uint32_t> satPlanePos(uint32_t id) {
  return {(id-1)/N_SPP + 1, (id-1)%N_SPP + 1};
}
static uint32_t satId(uint32_t pl, uint32_t po) { return (pl-1)*N_SPP + po; }
static bool isCross(uint32_t u, uint32_t v) {
  return satPlanePos(u).first != satPlanePos(v).first;
}
static uint32_t GlobalSlot() {
  int64_t s = (int64_t)std::floor(Simulator::Now().GetSeconds()/g.slotSec + 1e-9);
  return (uint32_t)std::max<int64_t>(s, 1);
}
static bool DeadlineExceeded(uint32_t localSlot, uint32_t created, uint32_t cls) {
  return (int64_t)localSlot - (int64_t)created + 1
       >= (int64_t)g.deadlineSlots[cls % 3];
}

// ---- bridge socket helpers (blocking; sim time is frozen at boundaries) ----
static void BridgeSend(const std::string& line) {
  std::string buf = line + "\n";
  const char* p = buf.data(); size_t left = buf.size();
  while (left) {
    ssize_t n = ::send(g.sock, p, left, 0);
    if (n <= 0) { std::cerr << "bridge send failed\n"; std::exit(2); }
    p += n; left -= (size_t)n;
  }
}
static std::string BridgeRecvLine() {
  std::string line; char c;
  while (true) {
    ssize_t n = ::recv(g.sock, &c, 1, 0);
    if (n <= 0) { std::cerr << "bridge recv failed\n"; std::exit(2); }
    if (c == '\n') break;
    if (c != '\r') line += c;
  }
  return line;
}

static void PhyTxDirected(uint32_t u, uint32_t v, Ptr<const Packet>) {
  g.linkTx[{u,v}]++;
}

// arrival: delivery takes precedence; then TTL; then dest-queue-full; else FIFO
void OnRx(Ptr<Node> node, uint32_t nodeId, Ptr<NetDevice>, Ptr<const Packet> pckt,
          uint16_t, Address const&, Address const&, NetDevice::PacketType)
{
  (void)node;
  Ptr<Packet> p = pckt->Copy();
  CLHeader h;
  p->RemoveHeader(h);
  uint32_t gslot = GlobalSlot();
  uint32_t lslot = gslot - h.episode * g.episodeSlots;
  uint32_t hops = (uint32_t)h.hops + 1;        // arrival completes the hop
  if (nodeId == h.finalDst) {
    if (!DeadlineExceeded(lslot, h.createdSlot, h.cls)) {
      g.delivered++;
      double dMs = (Simulator::Now().GetNanoSeconds()-(int64_t)h.sendTimeNs)/1e6;
      g.delaysMs.push_back(dMs);
      g.deliveredDelayMs[h.pktId] = dMs;
    } else {
      g.deadlineDrops++;
      g.dropReason[h.pktId] = "deadline_exceeded";
    }
    return;
  }
  if (hops >= g.maxHops) {
    g.ttlDrops++; g.dropReason[h.pktId] = "ttl_exceeded"; return;
  }
  if (g.nodeQ[nodeId].size() >= g.nodeQCap) {
    g.queueDrops++; g.dropReason[h.pktId] = "queue_overflow"; return;
  }
  QPkt q{h.pktId, h.finalDst, h.srcNode, h.cls, h.episode, h.createdSlot,
         (uint8_t)hops, (uint8_t)nodeId, h.visitedMask | (1u << (nodeId-1)),
         h.sendTimeNs};
  g.nodeQ[nodeId].push_back(q);
  g.incoming[nodeId]++;
}

// one slot: expire deadlines, report state, get decisions, execute service
void SlotBoundary(uint32_t gslot)
{
  // 1) inject this slot's new packets (env: created before decisions)
  for (const auto& inj : g.injects) {
    if (inj.gslot != gslot || inj.src == inj.dst) continue;
    g.allSent.insert(inj.gid);
    if (g.nodeQ[inj.src].size() >= g.nodeQCap) {
      g.sourceDrops++; g.dropReason[inj.gid] = "source_queue_overflow";
      continue;
    }
    g.sent++;
    QPkt q{inj.gid, inj.dst, inj.src, (uint8_t)inj.cls, (uint8_t)inj.episode,
           (uint8_t)inj.createdSlot, 0, 0, 1u << (inj.src-1),
           Simulator::Now().GetNanoSeconds()};
    g.nodeQ[inj.src].push_back(q);
  }

  // 2) expire deadlines (env: after arrivals, before next decisions)
  for (auto& [node, q] : g.nodeQ) {
    (void)node;
    std::deque<QPkt> keep;
    for (auto& pkt : q) {
      uint32_t lslot = gslot - pkt.episode * g.episodeSlots;
      if (DeadlineExceeded(lslot, pkt.createdSlot, pkt.cls)) {
        g.deadlineDrops++; g.dropReason[pkt.pktId] = "deadline_exceeded";
      } else keep.push_back(pkt);
    }
    q.swap(keep);
  }

  // 3) which episode is active? (episodes staggered; at most one)
  int activeEp = -1;
  for (const auto& inj : g.injects) {
    uint32_t epStart = inj.episode * g.episodeSlots + 1;
    if (gslot >= epStart && gslot < epStart + g.episodeSlots) {
      activeEp = (int)inj.episode; break;
    }
  }

  if (activeEp >= 0) {
    uint32_t lslot = gslot - (uint32_t)activeEp * g.episodeSlots;

    // directed TX deltas since last boundary (accepted forwards last slot)
    std::map<std::pair<uint32_t,uint32_t>,uint32_t> txDelta;
    for (auto& [k, total] : g.linkTx) {
      uint64_t prev = g.snapshotTx.count(k) ? g.snapshotTx[k] : 0;
      if (total > prev) txDelta[k] = (uint32_t)(total - prev);
    }
    g.snapshotTx = g.linkTx;

    // ---- report ----
    std::ostringstream head;
    head << "SLOT " << activeEp << " " << lslot << " " << N_NODES
         << " " << txDelta.size();
    BridgeSend(head.str());
    for (uint32_t id = 1; id <= N_NODES; ++id) {
      auto& q = g.nodeQ[id];
      std::ostringstream al;
      if (!q.empty()) {
        const QPkt& h = q.front();
        al << "AG " << id << " " << q.size() << " " << g.incoming[id] << " "
           << h.pktId << " " << h.finalDst << " " << (int)h.cls << " "
           << (int)h.hops << " " << (int)h.prevNode << " "
           << (int)h.createdSlot << " " << h.visitedMask;
      } else {
        al << "AG " << id << " " << q.size() << " " << g.incoming[id]
           << " -1 0 0 0 0 0 0";
      }
      al << " 0";                               // slot 0: hold placeholder
      const auto& nbrs = nbrsOf[id];
      for (uint32_t i = 0; i < 6; ++i) al << " " << (i < nbrs.size() ? nbrs[i] : 0);
      BridgeSend(al.str());
    }
    for (auto& [k, n] : txDelta) {
      std::ostringstream tl;
      tl << "TX " << k.first << " " << k.second << " " << n;
      BridgeSend(tl.str());
    }

    // ---- decisions ----
    std::istringstream as(BridgeRecvLine());    // "ACT <n>"
    std::string tag; int nd = 0; as >> tag >> nd;
    std::map<uint32_t,int> actionOf;
    for (int i = 0; i < nd; ++i) {
      std::istringstream ds(BridgeRecvLine());
      uint32_t node; int actslot, nh;
      ds >> node >> actslot >> nh;
      (void)nh;                                  // server maps slot->neighbor too
      actionOf[node] = actslot;
      if (actslot > 0) g.decisions++; else g.holds++;
    }

    // ---- execute with link-capacity resolution (by packet id) ----
    struct Prop { uint32_t node, nh, pktId; QPkt pkt; };
    std::map<std::pair<uint32_t,uint32_t>, std::vector<Prop>> byLink;
    for (auto& [node, actslot] : actionOf) {
      if (actslot <= 0) continue;
      const auto& q = g.nodeQ[node];
      if (q.empty()) continue;
      const auto& nbrs = nbrsOf[node];
      if ((size_t)actslot > nbrs.size()) continue;
      uint32_t nh = nbrs[actslot - 1];
      byLink[{node, nh}].push_back({node, nh, q.front().pktId, q.front()});
    }
    for (auto& [link, props] : byLink) {
      (void)link;
      std::sort(props.begin(), props.end(),
                [](const Prop& a, const Prop& b){ return a.pktId < b.pktId; });
      size_t acc = 0;
      for (auto& pr : props) {
        if (acc++ >= g.linkCapacity) { g.blocked++; continue; }  // stays queued
        auto& q = g.nodeQ[pr.node];
        if (q.empty() || q.front().pktId != pr.pktId) continue;
        q.pop_front();
        CLHeader h;
        h.pktId = pr.pkt.pktId; h.finalDst = pr.pkt.finalDst;
        h.srcNode = pr.pkt.srcNode; h.sendTimeNs = pr.pkt.sendNs;
        h.cls = pr.pkt.cls; h.episode = pr.pkt.episode;
        h.createdSlot = pr.pkt.createdSlot; h.hops = pr.pkt.hops + 1;
        h.prevNode = pr.pkt.prevNode; h.visitedMask = pr.pkt.visitedMask;
        Ptr<Packet> p = Create<Packet>(1500 - h.GetSerializedSize());
        p->AddHeader(h);
        auto dev = g.devOf[{pr.node, pr.nh}];
        bool ok = dev->Send(p, dev->GetBroadcast(), LEO_PROTO);
        if (!ok) { g.deviceQueueDrops++; g.dropReason[pr.pktId] = "device_queue_full"; }
      }
    }
  }
  g.incoming.clear();

  double now = Simulator::Now().GetSeconds();
  double next = (std::floor(now/g.slotSec + 1e-9) + 1.0) * g.slotSec;
  Simulator::Schedule(Seconds(std::max(next - now, 1e-9)),
                      &SlotBoundary, (uint32_t)(next/g.slotSec + 1e-9));
}

int main(int argc, char* argv[])
{
  std::string inputFile, outputFile, host = "127.0.0.1", policyName = "mappo";
  std::string deadlineStr = "30,12,20";
  double slotSec = 1.0, intraMs = INTRA_DELAY_MS, crossMs = CROSS_DELAY_MS;
  uint32_t bwKbps = 36, qsize = 64, episodeSlots = 30, linkCap = 3;
  uint32_t nodeQCap = 45, maxHops = 12, port = 7341;
  CommandLine cmd;
  cmd.AddValue("input", "packets CSV (traffic only; path column ignored)", inputFile);
  cmd.AddValue("output", "per-packet results csv", outputFile);
  cmd.AddValue("policy-name", "mappo|dijkstra (server-side provider)", policyName);
  cmd.AddValue("bridge-host", "policy server host", host);
  cmd.AddValue("bridge-port", "policy server port", port);
  cmd.AddValue("bw-kbps", "ISL data rate kb/s", bwKbps);
  cmd.AddValue("qsize-pkts", "device drop-tail queue (packets)", qsize);
  cmd.AddValue("slot-sec", "env slot -> ns-3 seconds", slotSec);
  cmd.AddValue("episode-slots", "env slots per episode", episodeSlots);
  cmd.AddValue("link-capacity", "forwards per directed link per slot", linkCap);
  cmd.AddValue("node-qsize", "per-node local queue cap (env max_queue_packets)", nodeQCap);
  cmd.AddValue("max-hops", "env max_local_hops (TTL)", maxHops);
  cmd.AddValue("deadline-slots", "per-class deadlines", deadlineStr);
  cmd.AddValue("intra-ms", "intra-plane delay (ms)", intraMs);
  cmd.AddValue("cross-ms", "cross-plane delay (ms)", crossMs);
  cmd.Parse(argc, argv);
  if (inputFile.empty() || outputFile.empty()) {
    std::cerr << "need --input and --output\n"; return 1;
  }
  g.slotSec = slotSec; g.episodeSlots = episodeSlots;
  g.linkCapacity = linkCap; g.nodeQCap = nodeQCap; g.maxHops = maxHops;
  g.policyName = policyName; g.output = outputFile;
  {
    std::stringstream ss(deadlineStr); std::string cell; uint32_t i = 0;
    while (std::getline(ss, cell, ',') && i < 3)
      g.deadlineSlots[i++] = (uint32_t)std::stoul(cell);
  }

  // ---- connect to the policy server ----
  g.sock = ::socket(AF_INET, SOCK_STREAM, 0);
  sockaddr_in addr{}; addr.sin_family = AF_INET;
  addr.sin_port = htons((uint16_t)port);
  ::inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
  if (::connect(g.sock, (sockaddr*)&addr, sizeof addr) != 0) {
    std::cerr << "cannot connect bridge " << host << ":" << port << "\n"; return 1;
  }
  {
    std::ostringstream hs;
    hs << "HELLO " << N_NODES << " 7 " << policyName;
    BridgeSend(hs.str());
    std::cerr << "bridge: " << BridgeRecvLine() << "\n";
  }

  // ---- topology (identical physical model to leo-validation) ----
  NodeContainer all; all.Create(N_NODES);
  for (uint32_t i = 1; i <= N_NODES; ++i) g.nodes[i] = all.Get(i-1);
  PointToPointHelper p2p;
  p2p.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize",
               StringValue(std::to_string(qsize) + "p"));
  std::set<std::pair<uint32_t,uint32_t>> edges;
  for (uint32_t id = 1; id <= N_NODES; ++id) {
    auto [pl, po] = satPlanePos(id);
    uint32_t sameOrbit = satId(pl, po % N_SPP + 1);
    uint32_t nextPlane = satId(pl % N_PLANES + 1, po);
    edges.insert({std::min(id,sameOrbit), std::max(id,sameOrbit)});
    edges.insert({std::min(id,nextPlane), std::max(id,nextPlane)});
  }
  for (auto [u, v] : edges) {
    bool cross = isCross(u, v);
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(cross ? crossMs : intraMs)));
    p2p.SetDeviceAttribute("DataRate",
                           DataRateValue(DataRate(std::to_string(bwKbps) + "kb/s")));
    NetDeviceContainer nd = p2p.Install(all.Get(u-1), all.Get(v-1));
    auto du = DynamicCast<PointToPointNetDevice>(nd.Get(0));
    auto dv = DynamicCast<PointToPointNetDevice>(nd.Get(1));
    g.devOf[{u,v}] = du;
    g.devOf[{v,u}] = dv;
  }
  // fixed neighbor order per node (sorted ids) — shared by report + execution
  for (uint32_t id = 1; id <= N_NODES; ++id) {
    for (uint32_t other = 1; other <= N_NODES; ++other)
      if (g.devOf.count({id, other})) nbrsOf[id].push_back(other);
    std::sort(nbrsOf[id].begin(), nbrsOf[id].end());
  }
  for (uint32_t i = 1; i <= N_NODES; ++i) {
    Ptr<Node> n = g.nodes[i];
    n->RegisterProtocolHandler(MakeBoundCallback(&OnRx, n, (uint32_t)i),
                               0 /* wildcard */, nullptr, /*promisc=*/false);
  }
  for (auto& [k, dev] : g.devOf)
    dev->TraceConnectWithoutContext(
        "PhyTxBegin", MakeBoundCallback(&PhyTxDirected, k.first, k.second));

  // ---- traffic (same CSV as replay; only traffic columns used) ----
  std::ifstream in(inputFile);
  if (!in) { std::cerr << "cannot open " << inputFile << "\n"; return 1; }
  std::string line; std::getline(in, line);   // header
  uint32_t maxGslot = 0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::vector<std::string> f; std::stringstream ss(line); std::string cell;
    while (std::getline(ss, cell, ',')) f.push_back(cell);
    if (f.size() < 6) continue;
    uint32_t ep = std::stoul(f[0]), pid = std::stoul(f[1]);
    uint32_t src = std::stoul(f[2]), dst = std::stoul(f[3]);
    uint32_t cls = std::stoul(f[4]), created = std::stoul(f[5]);
    uint32_t gslot = created + ep * episodeSlots;
    g.injects.push_back({gslot, ep * 1000000u + pid, src, dst, cls, ep, created});
    maxGslot = std::max(maxGslot, gslot);
  }
  std::cerr << "closed loop: " << g.injects.size() << " packets, horizon "
            << maxGslot << " slots\n";

  Simulator::Schedule(Seconds(slotSec), &SlotBoundary, 1u);
  Simulator::Stop(Seconds((maxGslot + 10) * slotSec + 5.0));
  Simulator::Run();

  // ---- per-packet results ----
  std::ofstream out(outputFile);
  out << "packet_id,delivered,delay_ms,drop_reason\n";
  for (uint32_t pid : g.allSent)
    out << pid << "," << (g.deliveredDelayMs.count(pid) ? 1 : 0) << ","
        << (g.deliveredDelayMs.count(pid) ? g.deliveredDelayMs[pid] : -1.0)
        << "," << (g.dropReason.count(pid) ? g.dropReason[pid] : "backlog")
        << "\n";

  std::sort(g.delaysMs.begin(), g.delaysMs.end());
  auto pct = [&](double q)->double {
    return g.delaysMs.empty() ? -1.0
      : g.delaysMs[(size_t)(q*(g.delaysMs.size()-1))];
  };
  double mean = g.delaysMs.empty() ? -1
    : std::accumulate(g.delaysMs.begin(), g.delaysMs.end(), 0.0)/g.delaysMs.size();
  double imb = 0.0;
  if (!g.linkTx.empty()) {
    double s=0, sq=0; uint64_t n=0;
    for (auto& [_,c] : g.linkTx) { s+=c; sq+=(double)c*c; n++; }
    double m = s/n;
    imb = m>0 ? std::sqrt(std::max(0.0, sq/n - m*m))/m : 0.0;
  }
  std::cout << "RESULT,policy=" << g.policyName
            << ",closed_loop=1"
            << ",sent=" << g.sent
            << ",delivered=" << g.delivered
            << ",delivery_ratio=" << (g.sent ? (double)g.delivered/g.sent : 0.0)
            << ",mean_delay_ms=" << mean
            << ",p50_delay_ms=" << pct(0.50)
            << ",p95_delay_ms=" << pct(0.95)
            << ",deadline_drops=" << g.deadlineDrops
            << ",ttl_drops=" << g.ttlDrops
            << ",queue_drops=" << g.queueDrops
            << ",source_drops=" << g.sourceDrops
            << ",device_queue_drops=" << g.deviceQueueDrops
            << ",blocked_by_link_capacity=" << g.blocked
            << ",holds=" << g.holds
            << ",decisions=" << g.decisions
            << ",load_imbalance=" << imb
            << ",links_active=" << g.linkTx.size() << "\n";
  ::close(g.sock);
  Simulator::Destroy();
  return 0;
}
