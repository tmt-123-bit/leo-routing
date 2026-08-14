/*
 * ns-3 packet-level validation for the LEO MAPPO routing study.
 *
 * Replays per-packet source-routed paths (extracted from the slot-based env by
 * ns3_trace_extractor.py) through an ns-3 point-to-point torus mesh with real
 * FIFO drop-tail queues, byte bandwidth, and propagation delay. Both MAPPO and
 * Dijkstra traces run on the IDENTICAL physical model + traffic, so the only
 * variable is the path choice -> directly tests whether MAPPO's congestion-
 * awareness + fairness advantage survives at packet fidelity.
 *
 * Works at the PointToPointNetDevice layer (no IP stack): each node registers a
 * promiscuous protocol handler; packets carry a custom LeoHeader with the full
 * node path and a next-hop index. On receipt, a node either delivers (final hop)
 * or forwards out the device toward the next node; if that device's drop-tail
 * queue is full, Send() returns false -> packet dropped (the packet-level
 * realism we are validating against).
 *
 * DYNAMIC-TOPOLOGY MODE (pass --links=links_<policy>.csv): links go down per the
 * env's real per-slot break schedule (episode,slot,u,v = DOWN directed link). A
 * packet that reaches a node whose outgoing link is currently down WAITS at that
 * node and retries at the next slot boundary (matches the env, where a broken
 * link simply isn't actionable that slot and waiting costs deadline). A packet
 * whose age reaches its class deadline (env rule: slot - created + 1 >= deadline
 * -> drop) is dropped as deadline_exceeded; delivery also only counts within the
 * deadline. Without --links the run is fully static (original behavior; only the
 * 3 extra header bytes differ, which shifts delays by <1 ms).
 *
 * Build:  ./ns3 build   (this file lives in scratch/)
 * Run:    ./ns3 run "scratch/leo-validation --input=...packets_mappo.csv
 *                     --output=...ns3_result_mappo.csv --bw-mbps=10 --qsize-pkts=64"
 *
 * Output: a per-packet CSV (packet_id, delivered, delay_ms, queue_drops_at_fwd)
 *         plus a one-line summary on stdout.
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <tuple>
#include <cmath>
#include <algorithm>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("LeoValidation");

namespace {
constexpr uint32_t N_PLANES = 4;
constexpr uint32_t N_SPP = 6;            // sats per plane
constexpr uint32_t N_NODES = N_PLANES * N_SPP;  // 24
constexpr uint16_t LEO_PROTO = 0x9999;   // custom ethertype for our packets
constexpr uint32_t MAXHOPS = 32;
constexpr double INTRA_DELAY_MS = 8.0;
constexpr double CROSS_DELAY_MS = 12.0;
} // namespace

// ---- custom source-routing header ----
class LeoHeader : public Header
{
public:
  uint32_t pktId = 0;
  uint32_t finalDst = 0;
  uint64_t sendTimeNs = 0;
  uint8_t nextIdx = 0;        // index of the node that should next receive
  uint8_t pathLen = 0;
  uint8_t cls = 0;            // traffic class (indexes deadlineSlots)
  uint8_t episode = 0;        // episode id (slot clock offset)
  uint8_t createdSlot = 0;    // 1-based env slot the packet was created in
  uint32_t path[MAXHOPS] = {0};

  LeoHeader() = default;
  static TypeId GetTypeId() {
    static TypeId tid = TypeId("LeoHeader")
      .SetParent<Header>()
      .SetGroupName("Leo")
      .AddConstructor<LeoHeader>();
    return tid;
  }
  TypeId GetInstanceTypeId() const override { return GetTypeId(); }
  void Print(std::ostream& os) const override { os << "leo[" << pktId << "]"; }
  uint32_t GetSerializedSize() const override {
    return 4 + 4 + 8 + 1 + 1 + 1 + 1 + 1 + 4 * MAXHOPS;
  }
  void Serialize(Buffer::Iterator start) const override {
    start.WriteU32(pktId);
    start.WriteU32(finalDst);
    start.WriteU64(sendTimeNs);
    start.WriteU8(nextIdx);
    start.WriteU8(pathLen);
    start.WriteU8(cls);
    start.WriteU8(episode);
    start.WriteU8(createdSlot);
    for (uint32_t i = 0; i < MAXHOPS; ++i) start.WriteU32(path[i]);
  }
  uint32_t Deserialize(Buffer::Iterator start) override {
    pktId = start.ReadU32();
    finalDst = start.ReadU32();
    sendTimeNs = start.ReadU64();
    nextIdx = start.ReadU8();
    pathLen = start.ReadU8();
    cls = start.ReadU8();
    episode = start.ReadU8();
    createdSlot = start.ReadU8();
    for (uint32_t i = 0; i < MAXHOPS; ++i) path[i] = start.ReadU32();
    return GetSerializedSize();
  }
};

// ---- global sim state (simple; single run) ----
struct Sim {
  // node id (1-based) -> Node + per-neighbor device map
  std::map<uint32_t, Ptr<Node>> nodes;
  // (u,v) -> device on node u facing v
  std::map<std::pair<uint32_t,uint32_t>, Ptr<PointToPointNetDevice>> devOf;
  std::string output;          // per-packet results csv path
  std::string policyName;
  // metrics
  uint64_t sent = 0, delivered = 0, queueDrops = 0, macTx = 0, onRx = 0;
  uint64_t arrived = 0, deadlineDrops = 0, waitEvents = 0, stranded = 0, partialEnd = 0;
  std::vector<double> delaysMs;        // delivered delays
  std::vector<uint32_t> deliveredPkts; // pkt ids delivered
  std::vector<std::pair<uint32_t,double>> perPkt; // (pktId, delayMs) delivered; undelivered added later
  // per-link TX counts: edge key -> count
  std::map<std::pair<uint32_t,uint32_t>, uint64_t> linkTx;
  // undelivered tracking: pktId -> sent?
  std::set<uint32_t> allSent;
  // dynamic-topology mode state
  bool dynamic = false;
  double slotSec = 1.0;
  uint32_t episodeSlots = 30;
  uint32_t deadlineSlots[3] = {30, 12, 20};  // per traffic class
  // (episode, slot, u, v) -> link is DOWN that slot
  std::set<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>> downLinks;
} g;

static std::pair<uint32_t,uint32_t> satPlanePos(uint32_t id) {
  return {(id - 1) / N_SPP + 1, (id - 1) % N_SPP + 1};
}
static uint32_t satId(uint32_t plane, uint32_t pos) {
  return (plane - 1) * N_SPP + pos;
}
static bool isCross(uint32_t u, uint32_t v) {
  return satPlanePos(u).first != satPlanePos(v).first;
}

// ---- dynamic-topology helpers ----
// 1-based env slot a packet is in at wall-clock time t (episodes are staggered
// by episodeSlots so their traffic windows don't overlap).
static uint32_t CurSlot(const LeoHeader& h)
{
  int64_t s = (int64_t)std::floor(Simulator::Now().GetSeconds() / g.slotSec + 1e-9);
  s -= (int64_t)h.episode * (int64_t)g.episodeSlots;
  return (uint32_t)std::max<int64_t>(s, 1);
}
// env deadline rule (leo_multiagent_env._expire_deadline_packets): a packet is
// dropped once slot - created_slot + 1 >= deadline[class]; usable through
// slot created + deadline - 2.
static bool DeadlineExceeded(uint32_t slot, const LeoHeader& h)
{
  if (!g.dynamic) return false;
  return (int64_t)slot - (int64_t)h.createdSlot + 1 >= (int64_t)g.deadlineSlots[h.cls % 3];
}
static bool LinkDown(const LeoHeader& h, uint32_t slot, uint32_t u, uint32_t v)
{
  return g.downLinks.count({h.episode, slot, u, v}) > 0;
}

// shared send path for BOTH injection and forwarding: transmit p (headerless,
// header carried in h) from nodeId toward h.path[h.nextIdx]. In dynamic mode a
// currently-down link makes the packet WAIT at the node and retry at the next
// slot boundary (link may heal); waits are bounded by the class deadline.
static void AttemptSend(uint32_t nodeId, LeoHeader h, Ptr<Packet> p)
{
  uint32_t nxt = h.path[h.nextIdx];
  auto it = g.devOf.find({nodeId, nxt});
  if (it == g.devOf.end()) { g.stranded++; return; }   // no such link
  if (g.dynamic) {
    uint32_t s = CurSlot(h);
    if (DeadlineExceeded(s, h)) { g.deadlineDrops++; return; }
    if (LinkDown(h, s, nodeId, nxt)) {
      uint32_t ns = s + 1;                             // retry slot
      if (DeadlineExceeded(ns, h)) { g.deadlineDrops++; return; }
      g.waitEvents++;
      double now = Simulator::Now().GetSeconds();
      double nextBoundary = (std::floor(now / g.slotSec + 1e-9) + 1.0) * g.slotSec;
      Simulator::Schedule(Seconds(std::max(nextBoundary - now, 1e-9)),
                          &AttemptSend, nodeId, h, p);
      return;
    }
  }
  p->AddHeader(h);
  Ptr<PointToPointNetDevice> outDev = it->second;
  bool ok = outDev->Send(p, outDev->GetBroadcast(), LEO_PROTO);
  if (!ok) g.queueDrops++;
}

// promiscuous protocol handler: receives every LEO_PROTO packet on a node
void OnRx(Ptr<Node> node, uint32_t nodeId,
          Ptr<NetDevice> dev, Ptr<const Packet> pckt,
          uint16_t proto, Address const&, Address const&, NetDevice::PacketType)
{
  (void)proto;  // protocol number is mangled by Ppp EtherToPpp; identify via LeoHeader
  g.onRx++;
  Ptr<Packet> p = pckt->Copy();
  LeoHeader h;
  p->RemoveHeader(h);
  // h.nextIdx is the index of THIS receiving node in the path
  if (h.nextIdx >= h.pathLen) return;  // malformed
  uint32_t here = h.path[h.nextIdx];
  if (here != nodeId) return;          // not for us (shouldn't happen on p2p)
  // delivered? Two conditions must hold: this is the last node of the path AND
  // it is the actual final destination. Env-dropped packets carry PARTIAL paths
  // (visited stops mid-network) -> they must NOT count as delivered just because
  // they walked off the end of their partial path.
  if (h.nextIdx == h.pathLen - 1) {
    if (here != h.finalDst) { g.partialEnd++; return; }  // partial path ran out
    g.arrived++;
    if (!DeadlineExceeded(CurSlot(h), h)) {
      g.delivered++;
      double dMs = (Simulator::Now().GetNanoSeconds() - (int64_t)h.sendTimeNs) / 1e6;
      g.delaysMs.push_back(dMs);
      g.perPkt.emplace_back(h.pktId, dMs);
    } else {
      g.deadlineDrops++;
    }
    return;
  }
  // forward toward path[nextIdx+1]
  h.nextIdx = h.nextIdx + 1;
  AttemptSend(nodeId, h, p);
}

void PhyTxCallback(uint32_t u, uint32_t v, Ptr<const Packet>) {
  g.linkTx[{std::min(u,v), std::max(u,v)}]++;
}

// free function (MakeBoundCallback needs a function pointer, not a lambda)
void PhyTxLink(uint32_t a, uint32_t b, Ptr<const Packet>) {
  g.linkTx[{std::min(a, b), std::max(a, b)}]++;
}
void MacTxInc(Ptr<const Packet>) { g.macTx++; }

// inject one packet at its source toward path[1]
void InjectPacket(uint32_t pktId, uint32_t episode, uint32_t cls, uint32_t createdSlot,
                  uint32_t src, uint32_t dst, std::vector<uint32_t> path)
{
  if (path.size() < 2) return;             // src==dst / unrouted: not a routing decision
  uint32_t firstHop = path[1];
  if (g.devOf.find({src, firstHop}) == g.devOf.end()) return;  // source lacks that link
  g.sent++;
  g.allSent.insert(pktId);
  LeoHeader h;
  h.pktId = pktId;
  // TRUE destination (from the CSV), NOT path.back(): env-dropped packets carry
  // partial paths, and path.back() would be whatever mid-network node they
  // stopped at -> delivery must only count at the real dst.
  h.finalDst = dst;
  h.sendTimeNs = Simulator::Now().GetNanoSeconds();
  h.nextIdx = 1;
  h.cls = (uint8_t)cls;
  h.episode = (uint8_t)episode;
  h.createdSlot = (uint8_t)createdSlot;
  h.pathLen = (uint8_t)std::min((size_t)MAXHOPS, path.size());
  for (uint32_t i = 0; i < h.pathLen; ++i) h.path[i] = path[i];
  Ptr<Packet> p = Create<Packet>(1500 - h.GetSerializedSize());
  AttemptSend(src, h, p);
}

// split "a b c" -> ints
std::vector<uint32_t> parseIntList(const std::string& s) {
  std::vector<uint32_t> out; std::istringstream iss(s); uint32_t x;
  while (iss >> x) out.push_back(x);
  return out;
}

int main(int argc, char* argv[])
{
  std::string inputFile, outputFile, linksFile, deadlineStr = "30,12,20", policyName = "mappo";
  double slotSec = 1.0, intraMs = INTRA_DELAY_MS, crossMs = CROSS_DELAY_MS;
  uint32_t bwKbps = 36, qsize = 64, pktBytes = 1500, episodeSlots = 30;
  CommandLine cmd;
  cmd.AddValue("input", "packets_<policy>.csv from extractor", inputFile);
  cmd.AddValue("output", "per-packet results csv path", outputFile);
  cmd.AddValue("policy-name", "label", policyName);
  cmd.AddValue("bw-kbps", "ISL data rate (kb/s); 36 ~= env 3-pkt/slot capacity", bwKbps);
  cmd.AddValue("qsize-pkts", "drop-tail queue size (packets)", qsize);
  cmd.AddValue("slot-sec", "env slot -> ns-3 seconds", slotSec);
  cmd.AddValue("episode-slots", "env slots per episode (stagger offset)", episodeSlots);
  cmd.AddValue("intra-ms", "intra-plane propagation delay (ms)", intraMs);
  cmd.AddValue("cross-ms", "cross-plane propagation delay (ms)", crossMs);
  cmd.AddValue("links", "links_<policy>.csv (episode,slot,u,v DOWN entries); "
               "absent = fully static topology (original behavior)", linksFile);
  cmd.AddValue("deadline-slots", "per-class deadline in slots, comma-separated "
               "(env packet_class_deadlines, e.g. 30,12,20)", deadlineStr);
  cmd.Parse(argc, argv);
  if (inputFile.empty() || outputFile.empty()) {
    std::cerr << "need --input and --output\n"; return 1;
  }

  // ---- dynamic-topology mode state ----
  g.slotSec = slotSec;
  g.episodeSlots = episodeSlots;
  {
    std::stringstream ss(deadlineStr); std::string cell;
    uint32_t i = 0;
    while (std::getline(ss, cell, ',') && i < 3) {
      g.deadlineSlots[i++] = (uint32_t)std::stoul(cell);
    }
  }
  if (!linksFile.empty()) {
    g.dynamic = true;
    std::ifstream lin(linksFile);
    if (!lin) { std::cerr << "cannot open " << linksFile << "\n"; return 1; }
    std::string lline; std::getline(lin, lline); // header
    while (std::getline(lin, lline)) {
      if (lline.empty()) continue;
      std::vector<std::string> f; std::stringstream ls(lline); std::string cell;
      while (std::getline(ls, cell, ',')) f.push_back(cell);
      if (f.size() < 4) continue;
      g.downLinks.insert({std::stoul(f[0]), std::stoul(f[1]), std::stoul(f[2]), std::stoul(f[3])});
    }
    std::cerr << "dynamic mode: " << g.downLinks.size() << " down-link slot entries from "
              << linksFile << "\n";
  }

  // ---- build the 24-node torus mesh ----
  NodeContainer all;
  all.Create(N_NODES);
  for (uint32_t i = 1; i <= N_NODES; ++i) g.nodes[i] = all.Get(i - 1);

  PointToPointHelper p2p;
  p2p.SetQueue("ns3::DropTailQueue<Packet>", "MaxSize",
               StringValue(std::to_string(qsize) + "p"));
  std::set<std::pair<uint32_t,uint32_t>> edges;
  for (uint32_t id = 1; id <= N_NODES; ++id) {
    auto [pl, po] = satPlanePos(id);
    uint32_t sameOrbit = satId(pl, po % N_SPP + 1);
    uint32_t nextPlane = satId(pl % N_PLANES + 1, po);
    edges.insert({std::min(id, sameOrbit), std::max(id, sameOrbit)});
    edges.insert({std::min(id, nextPlane), std::max(id, nextPlane)});
  }
  for (auto [u, v] : edges) {
    bool cross = isCross(u, v);
    p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(cross ? crossMs : intraMs)));
    p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate(std::to_string(bwKbps) + "kb/s")));
    NetDeviceContainer nd = p2p.Install(all.Get(u - 1), all.Get(v - 1));
    auto du = DynamicCast<PointToPointNetDevice>(nd.Get(0));
    auto dv = DynamicCast<PointToPointNetDevice>(nd.Get(1));
    g.devOf[{u, v}] = du;
    g.devOf[{v, u}] = dv;
    // per-link TX trace (count transmissions) + per-device MacTx counter
    du->TraceConnectWithoutContext("PhyTxBegin", MakeBoundCallback(&PhyTxLink, u, v));
    dv->TraceConnectWithoutContext("PhyTxBegin", MakeBoundCallback(&PhyTxLink, v, u));
    du->TraceConnectWithoutContext("MacTx", MakeCallback(&MacTxInc));
    dv->TraceConnectWithoutContext("MacTx", MakeCallback(&MacTxInc));
  }

  // ---- register the forwarding handler on every node ----
  // promisc=false: Node::AddDevice wires the device receive to
  // NonPromiscReceiveFromDevice, which only dispatches promisc=false handlers.
  // P2P delivers every packet as PACKET_HOST, so non-promisc catches all.
  for (uint32_t i = 1; i <= N_NODES; ++i) {
    Ptr<Node> n = g.nodes[i];
    n->RegisterProtocolHandler(MakeBoundCallback(&OnRx, n, (uint32_t)i),
                               0 /* wildcard: match all protocols */, nullptr, /*promisc=*/false);
  }

  // ---- read the packet trace + schedule injections ----
  std::ifstream in(inputFile);
  if (!in) { std::cerr << "cannot open " << inputFile << "\n"; return 1; }
  std::string line; std::getline(in, line); // header
  uint64_t npkts = 0;
  double maxInjectT = 0.0;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    // CSV: episode,packet_id,src,dst,traffic_class,created_slot,delivered,delivery_slot,
    //      hop_count,delay_slots,delay_ms,path,drop_reason
    std::vector<std::string> f; std::stringstream ss(line); std::string cell;
    // path is field 11 and may be the last before drop_reason; split on commas but
    // path/drop_reason have no commas, so a simple comma split is fine.
    while (std::getline(ss, cell, ',')) f.push_back(cell);
    if (f.size() < 12) continue;
    uint32_t episode = std::stoul(f[0]);
    uint32_t pktId = std::stoul(f[1]);
    // packet_id is per-episode (0..~191), NOT globally unique across episodes ->
    // encode as a global id so the per-packet output set doesn't dedup-collapse
    // 880 packets (5 episodes) down to ~192. Figure decodes ep = gid/1e6.
    uint32_t gid = episode * 1000000u + pktId;
    uint32_t src = std::stoul(f[2]);
    uint32_t dst = std::stoul(f[3]);
    uint32_t cls = std::stoul(f[4]);
    uint32_t createdSlot = std::stoul(f[5]);
    std::vector<uint32_t> path = parseIntList(f[11]);
    // stagger episodes so their traffic windows do not overlap
    double t = (createdSlot + episode * episodeSlots) * slotSec;
    Simulator::Schedule(Seconds(t), &InjectPacket, gid, episode, cls, createdSlot, src, dst, path);
    npkts++;
    maxInjectT = std::max(maxInjectT, t);
  }
  std::cerr << "scheduled " << npkts << " packet injections from " << inputFile << "\n";

  g.output = outputFile;
  g.policyName = policyName;

  // ---- run (stop well after last injection + a generous drain window so
  //       in-flight packets either deliver or queue-overflow, not get cut off) ----
  double drainSec = 60.0;
  double runSec = npkts ? (maxInjectT + drainSec) : 10.0;
  Simulator::Stop(Seconds(runSec));
  Simulator::Run();

  // ---- write per-packet results ----
  std::ofstream out(outputFile);
  out << "packet_id,delivered,delay_ms\n";
  std::map<uint32_t,double> del;
  for (auto& [pid, d] : g.perPkt) del[pid] = d;
  for (uint32_t pid : g.allSent)
    out << pid << "," << (del.count(pid) ? 1 : 0) << ","
        << (del.count(pid) ? del[pid] : -1.0) << "\n";

  // ---- summary on stdout (single CSV line) ----
  std::sort(g.delaysMs.begin(), g.delaysMs.end());
  auto pct = [&](double q)->double {
    if (g.delaysMs.empty()) return -1.0;
    return g.delaysMs[(size_t)(q * (g.delaysMs.size() - 1))];
  };
  double mean = g.delaysMs.empty() ? -1
      : std::accumulate(g.delaysMs.begin(), g.delaysMs.end(), 0.0) / g.delaysMs.size();
  // load imbalance = std/mean over per-link TX counts
  double imb = 0.0;
  if (!g.linkTx.empty()) {
    double s = 0, sq = 0; uint64_t n = 0;
    for (auto& [_, c] : g.linkTx) { s += c; sq += (double)c * c; n++; }
    double m = s / n;
    imb = (m > 0) ? std::sqrt(std::max(0.0, sq / n - m * m)) / m : 0.0;
  }
  double delRatio = g.sent ? (double)g.delivered / g.sent : 0.0;
  std::cout << "RESULT,policy=" << policyName
            << ",sent=" << g.sent
            << ",delivered=" << g.delivered
            << ",delivery_ratio=" << delRatio
            << ",queue_drops=" << g.queueDrops
            << ",mac_tx=" << g.macTx
            << ",on_rx=" << g.onRx
            << ",mean_delay_ms=" << mean
            << ",p50_delay_ms=" << pct(0.50)
            << ",p95_delay_ms=" << pct(0.95)
            << ",load_imbalance=" << imb
            << ",links_active=" << g.linkTx.size()
            << ",dynamic=" << (g.dynamic ? 1 : 0)
            << ",arrived=" << g.arrived
            << ",deadline_drops=" << g.deadlineDrops
            << ",wait_events=" << g.waitEvents
            << ",stranded=" << g.stranded
            << ",partial_end=" << g.partialEnd << "\n";

  Simulator::Destroy();
  return 0;
}
