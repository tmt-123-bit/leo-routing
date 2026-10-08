#include "deadline-udp-application.h"
#include "ns3/socket-factory.h"
#include "ns3/packet.h"
#include "ns3/uinteger.h"
#include "ns3/log.h"

#include <cstdio>
#include <map>
#include <inttypes.h>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("DeadlineUdpApplication");

NS_OBJECT_ENSURE_REGISTERED (DeadlineUdpApplication);

TypeId
DeadlineUdpApplication::GetTypeId (void)
{
    static TypeId tid = TypeId ("ns3::DeadlineUdpApplication")
        .SetParent<Application> ()
        .SetGroupName ("SatelliteNetwork")
        .AddConstructor<DeadlineUdpApplication> ()
        .AddAttribute ("Port",
                       "The destination UDP port of the outgoing bursts",
                       UintegerValue (1726),
                       MakeUintegerAccessor (&DeadlineUdpApplication::m_port),
                       MakeUintegerChecker<uint16_t> ())
    ;
    return tid;
}

DeadlineUdpApplication::DeadlineUdpApplication ()
{
    m_socket = 0;
    m_port = 1726;
}

DeadlineUdpApplication::~DeadlineUdpApplication ()
{}

void
DeadlineUdpApplication::SetLoggingPath (std::string path)
{
    m_logPath = path;
}

void
DeadlineUdpApplication::RegisterOutgoingBurst (
        uint32_t burst_id,
        int32_t from_node_id,
        int32_t to_node_id,
        InetSocketAddress targetAddress,
        int64_t start_time_ns,
        int64_t duration_ns,
        double target_rate_megabit_per_s,
        int64_t deadline_offset_ns,
        uint8_t traffic_class
) {
    BurstInfo info;
    info.burst_id = burst_id;
    info.from_node_id = from_node_id;
    info.to_node_id = to_node_id;
    info.target = targetAddress;
    info.start_time_ns = start_time_ns;
    info.duration_ns = duration_ns;
    info.target_rate_mbps = target_rate_megabit_per_s;
    info.deadline_offset_ns = deadline_offset_ns;
    info.traffic_class = traffic_class;
    info.sent = 0;
    m_outgoing.push_back (info);
}

void
DeadlineUdpApplication::StartApplication (void)
{
    if (m_socket == 0) {
        TypeId tid = TypeId::LookupByName ("ns3::UdpSocketFactory");
        m_socket = Socket::CreateSocket (GetNode (), tid);
        InetSocketAddress local = InetSocketAddress (Ipv4Address::GetAny (), m_port);
        if (m_socket->Bind (local) == -1) {
            NS_FATAL_ERROR ("DeadlineUdpApplication: failed to bind socket");
        }
    }
    m_socket->SetRecvCallback (MakeCallback (&DeadlineUdpApplication::HandleRead, this));

    for (size_t i = 0; i < m_outgoing.size (); i++) {
        Simulator::Schedule (NanoSeconds (m_outgoing[i].start_time_ns),
                             &DeadlineUdpApplication::BurstSendOut, this, i);
    }
}

void
DeadlineUdpApplication::BurstSendOut (size_t internal_burst_idx)
{
    BurstInfo& info = m_outgoing[internal_burst_idx];
    uint64_t now_ns = (uint64_t) Simulator::Now ().GetNanoSeconds ();

    Ptr<Packet> p = Create<Packet> (1500);
    DeadlineTag tag ((uint32_t) info.to_node_id,
                     now_ns + (uint64_t) info.deadline_offset_ns,
                     info.burst_id, (uint32_t) info.sent, now_ns,
                     info.traffic_class);
    p->AddPacketTag (tag);
    info.sent++;

    m_socket->SendTo (p, 0, info.target);

    uint64_t packet_gap = (uint64_t) std::ceil (1500.0 / (info.target_rate_mbps / 8000.0));
    if (now_ns + packet_gap < (uint64_t) (info.start_time_ns + info.duration_ns)) {
        Simulator::Schedule (NanoSeconds (packet_gap),
                             &DeadlineUdpApplication::BurstSendOut, this, internal_burst_idx);
    }
}

void
DeadlineUdpApplication::HandleRead (Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;
    while ((packet = socket->RecvFrom (from))) {
        DeadlineTag tag;
        if (packet->PeekPacketTag (tag)) {
            uint64_t arrival = (uint64_t) Simulator::Now ().GetNanoSeconds ();
            int on_time = arrival <= tag.GetDeadlineNs () ? 1 : 0;
            m_arrivals[tag.GetBurstId ()].push_back (
                std::make_tuple (tag.GetSendNs (), arrival, tag.GetDeadlineNs (), on_time));
        }
    }
}

void
DeadlineUdpApplication::StopApplication (void)
{
    if (m_socket != 0) {
        m_socket->Close ();
        m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
        m_socket = 0;
    }

    if (!m_logPath.empty ()) {
        // Sent side: per burst
        FILE* sent_f = fopen ((m_logPath + "/deadline_udp_sent.csv").c_str (), "a");
        if (sent_f != 0) {
            fprintf (sent_f, "node_id,burst_id,sent_packets\n");
            for (size_t i = 0; i < m_outgoing.size (); i++) {
                fprintf (sent_f, "%d,%u,%" PRIu64 "\n",
                         m_outgoing[i].from_node_id, m_outgoing[i].burst_id, m_outgoing[i].sent);
            }
            fclose (sent_f);
        }
        // Received side: per packet
        if (!m_arrivals.empty ()) {
            FILE* recv_f = fopen ((m_logPath + "/deadline_udp_packets.csv").c_str (), "a");
            if (recv_f != 0) {
                fprintf (recv_f, "burst_id,seq_send_ns,arrival_ns,deadline_ns,on_time\n");
                for (auto const& kv : m_arrivals) {
                    for (auto const& row : kv.second) {
                        fprintf (recv_f, "%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d\n",
                                 kv.first, std::get<0> (row), std::get<1> (row),
                                 std::get<2> (row), std::get<3> (row));
                    }
                }
                fclose (recv_f);
            }
        }
    }
}

} // namespace ns3
