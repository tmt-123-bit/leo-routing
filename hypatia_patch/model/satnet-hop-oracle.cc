#include "satnet-hop-oracle.h"
#include "ns3/ipv4.h"
#include "ns3/ipv4-arbiter-routing.h"
#include "arbiter-satnet.h"

namespace ns3 {

SatnetHopOracle* SatnetHopOracle::m_instance = 0;

SatnetHopOracle*
SatnetHopOracle::Get ()
{
    if (m_instance == 0) {
        m_instance = new SatnetHopOracle ();
    }
    return m_instance;
}

void
SatnetHopOracle::Install (NodeContainer nodes)
{
    SatnetHopOracle* instance = Get ();
    instance->m_arbiters.clear ();
    for (uint32_t i = 0; i < nodes.GetN (); i++) {
        Ptr<Ipv4> ipv4 = nodes.Get (i)->GetObject<Ipv4> ();
        NS_ABORT_MSG_IF (ipv4 == 0, "Node has no Ipv4");
        Ptr<Ipv4ArbiterRouting> arbiterRouting =
            ipv4->GetRoutingProtocol ()->GetObject<Ipv4ArbiterRouting> ();
        NS_ABORT_MSG_IF (arbiterRouting == 0, "Node has no Ipv4ArbiterRouting");
        instance->m_arbiters.push_back (arbiterRouting->GetArbiter ());
    }
    instance->m_installed = true;
}

int64_t
SatnetHopOracle::HopDistance (int32_t from, int32_t dst)
{
    NS_ABORT_MSG_IF (!m_installed, "SatnetHopOracle::Install must be called first");
    if (from == dst) {
        return 0;
    }
    if (from < 0 || dst < 0 || (uint32_t) from >= m_arbiters.size ()
            || (uint32_t) dst >= m_arbiters.size ()) {
        return -1;
    }
    ArbiterSatnet* arbiter = dynamic_cast<ArbiterSatnet*> (PeekPointer (m_arbiters[from]));
    if (arbiter == 0) {
        return -1;
    }
    int64_t hops = 0;
    int32_t current = from;
    int64_t guard = 4 * (int64_t) m_arbiters.size () + 8;  // cycle guard
    while (current != dst) {
        std::tuple<int32_t, int32_t, int32_t> decision
            = arbiter->TopologySatelliteNetworkDecide (current, dst,
                                                       Ptr<const Packet> (), Ipv4Header (), false);
        int32_t next = std::get<0> (decision);
        if (next < 0 || next == current) {
            return -1;  // undefined next hop
        }
        hops++;
        if (hops > guard) {
            return -1;  // cycle
        }
        current = next;
        arbiter = dynamic_cast<ArbiterSatnet*> (PeekPointer (m_arbiters[current]));
        if (arbiter == 0) {
            return -1;
        }
    }
    return hops;
}

} // namespace ns3
