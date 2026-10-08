/*
 * Hop-distance oracle over the installed per-time-interval forwarding
 * tables of Hypatia (ArbiterSingleForward state loaded from the
 * satgenpy-generated fstate files).
 *
 * The hop distance from a satellite to a destination node is obtained by
 * chasing the next-hop entries (node -> next_hop(node, dst)) until the
 * destination is reached, with a cycle guard. Because the forwarding
 * state is refreshed by ArbiterSingleForwardHelper at every interval
 * boundary, distances automatically follow the currently installed
 * state. No memoization across simulation time is performed: a chase is
 * at most tens of hops and only runs on enqueue events.
 */

#ifndef SATNET_HOP_ORACLE_H
#define SATNET_HOP_ORACLE_H

#include <vector>
#include "ns3/node-container.h"
#include "ns3/arbiter.h"

namespace ns3 {

class SatnetHopOracle
{
public:
    // Singleton access
    static SatnetHopOracle* Get ();

    // Collect the arbiter of every node (call once after arbiters are installed)
    static void Install (NodeContainer nodes);

    // Number of hops from node with id 'from' to node with id 'dst'
    // following the currently installed forwarding tables.
    // Returns -1 if unreachable / undefined.
    int64_t HopDistance (int32_t from, int32_t dst);

private:
    SatnetHopOracle () {}
    static SatnetHopOracle* m_instance;

    std::vector<Ptr<Arbiter> > m_arbiters;   // index = node id
    bool m_installed = false;
};

} // namespace ns3

#endif /* SATNET_HOP_ORACLE_H */
