/*
 * DeadlineUdpScheduler: installs DeadlineUdpApplication endpoints and
 * registers the deadline bursts listed in the run directory's
 * deadline_udp_burst_schedule.csv:
 *
 *   burst_id,start_time_ns,from_node_id,to_node_id,
 *   target_rate_mbps,duration_ns,deadline_ms
 *
 * The destination address is resolved exactly like the basic-sim UDP
 * burst scheduler (first IPv4 interface address of the target node).
 */

#ifndef DEADLINE_UDP_SCHEDULER_H
#define DEADLINE_UDP_SCHEDULER_H

#include "ns3/basic-simulation.h"
#include "ns3/node-container.h"
#include "ns3/ipv4.h"
#include "ns3/inet-socket-address.h"

namespace ns3 {

class DeadlineUdpScheduler
{
public:
    DeadlineUdpScheduler (Ptr<BasicSimulation> basicSimulation, NodeContainer nodes);
};

} // namespace ns3

#endif /* DEADLINE_UDP_SCHEDULER_H */
