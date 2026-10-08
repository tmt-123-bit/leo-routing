/*
 * Deadline UDP application: end-to-end UDP bursts whose packets carry a
 * DeadlineTag (destination node id + absolute deadline). The receiver
 * logs, for every arriving packet, its send time, arrival time and
 * deadline to logs_ns3/deadline_udp_packets.csv so that on-time delivery
 * rates can be computed per burst.
 */

#ifndef DEADLINE_UDP_APPLICATION_H
#define DEADLINE_UDP_APPLICATION_H

#include <vector>
#include <tuple>
#include <map>
#include <fstream>
#include "ns3/application.h"
#include "ns3/inet-socket-address.h"
#include "ns3/socket.h"
#include "ns3/udp-socket.h"
#include "ns3/simulator.h"
#include "deadline-tag.h"

namespace ns3 {

class DeadlineUdpApplication : public Application
{
public:
    static TypeId GetTypeId (void);
    DeadlineUdpApplication ();
    virtual ~DeadlineUdpApplication ();

    void RegisterOutgoingBurst (
            uint32_t burst_id,
            int32_t from_node_id,
            int32_t to_node_id,
            InetSocketAddress targetAddress,
            int64_t start_time_ns,
            int64_t duration_ns,
            double target_rate_megabit_per_s,
            int64_t deadline_offset_ns,
            uint8_t traffic_class
    );

    void SetLoggingPath (std::string path);

protected:
    virtual void StartApplication (void);
    virtual void StopApplication (void);

private:
    void BurstSendOut (size_t internal_burst_idx);
    void HandleRead (Ptr<Socket> socket);

    struct BurstInfo {
        uint32_t burst_id;
        int32_t from_node_id;
        int32_t to_node_id;
        InetSocketAddress target;
        int64_t start_time_ns;
        int64_t duration_ns;
        double target_rate_mbps;
        int64_t deadline_offset_ns;   // per-packet deadline = send time + offset
        uint8_t traffic_class;
        uint64_t sent;
        BurstInfo ()
            : target (InetSocketAddress (Ipv4Address::GetAny (), 0))
        {}
    };

    std::vector<BurstInfo> m_outgoing;
    std::map<uint32_t, std::vector<std::tuple<uint64_t, uint64_t, uint64_t, int> > > m_arrivals;
    Ptr<Socket> m_socket;
    uint16_t m_port;
    std::string m_logPath;
};

} // namespace ns3

#endif /* DEADLINE_UDP_APPLICATION_H */
